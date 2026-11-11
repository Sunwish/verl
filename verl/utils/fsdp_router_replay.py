# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict full-sequence R3 for the Qwen3 MoE NPU FSDP2 path (SP=1).

Targets are indexed by model layer and travel with the nested micro-batch.
They remain alive until backward, including activation recomputation, finishes.
There is deliberately no process-global replay slot: actor and reference models
can coexist in the same worker without sharing replay state.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)


def r3_route_fingerprint(routes: torch.Tensor) -> torch.Tensor:
    """Order-sensitive int64 diagnostic signature, invariant to transport dtype.

    This is a diagnostic checksum, not a cryptographic content hash. Token,
    layer, and top-k slot all participate so a token shift changes the result.
    """
    positions = torch.arange(1, routes.numel() + 1, dtype=torch.int64, device=routes.device)
    weights = positions.square().remainder(2147483647)
    return (routes.reshape(-1).to(torch.int64) * weights).sum()


def route_alignment_diagnostics(targets, native, mask, offsets) -> dict:
    """Compare expert *sets*, including nearby rows without crossing sequences.

    token_delta=d compares capture[t] to native[t+d]. This only diagnoses a
    possible shift; consecutive tokens can legitimately share expert sets.
    Run on CPU outside forward/checkpoint/graph capture.
    """
    targets, native = targets.long(), native.long()
    target_sets, native_sets = targets.sort(-1).values, native.sort(-1).values
    selected = mask.bool()
    equal = target_sets == native_sets
    overlap = (targets.unsqueeze(-1) == native.unsqueeze(-2)).any(-1).float().mean(-1)
    result = {
        "captured_tokens": int(selected.sum()),
        "set_equal_fraction_per_layer": equal.all(-1)[selected].float().mean(0).tolist(),
        "set_overlap_per_layer": overlap[selected].mean(0).tolist(),
        "token_delta_set_overlap": {},
    }
    for delta in (-2, -1, 0, 1, 2):
        matches, count = 0, 0
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            lo, hi = max(start, start - delta), min(end, end - delta)
            if lo >= hi:
                continue
            valid = selected[lo:hi]
            if valid.any():
                a, b = targets[lo:hi][valid], native[lo + delta : hi + delta][valid]
                matches += (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum((0, 2))
                count += a.shape[0] * targets.shape[-1]
        result["token_delta_set_overlap"][delta] = (matches.float() / count).tolist() if count else []
    result["layer_delta_set_overlap"] = {}
    for delta in (-1, 1):
        lo, hi = max(0, -delta), min(targets.shape[1], targets.shape[1] - delta)
        if lo < hi:
            a, b = targets[selected, lo:hi], native[selected, lo + delta : hi + delta]
            result["layer_delta_set_overlap"][delta] = (
                (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean((0, 2)).tolist()
            )
    return result


def log_r3_probability_comparison(replay_log_probs, native_log_probs, rollout_log_probs, response_mask, rank, batch):
    """Log same-token numerical evidence; never substitute the PPO old policy."""
    from verl.utils.debug.metrics import pearson_correlation_coefficient

    mask = response_mask.detach().cpu().bool()
    values = [p.detach().float().cpu() for p in (replay_log_probs, native_log_probs, rollout_log_probs)]
    if not mask.any():
        raise RuntimeError("[R3][PROB_COMPARE] diagnostic has no valid response tokens")
    summary = {}
    for name, logs in zip(("replay", "native", "rollout"), values, strict=True):
        if logs.shape != mask.shape or not torch.isfinite(logs[mask]).all():
            raise RuntimeError(f"[R3][PROB_COMPARE] invalid {name} log-probability tensor")
        summary[f"{name}_mean_log_prob"] = logs[mask].mean().item()
    for name, a, b in (("replay_rollout", 0, 2), ("native_rollout", 1, 2), ("replay_native", 0, 1)):
        summary[f"{name}_pearson"] = pearson_correlation_coefficient(values[a].exp(), values[b].exp(), mask)
        summary[f"{name}_mean_abs_prob_diff"] = (values[a].exp() - values[b].exp())[mask].abs().mean().item()
        summary[f"{name}_mean_abs_log_prob_diff"] = (values[a] - values[b])[mask].abs().mean().item()
    logger.warning("[R3][PROB_COMPARE] rank=%s batch=%s valid_tokens=%s %s", rank, batch, mask.sum().item(), summary)
    return summary


def validate_vllm_r3_routes(
    routes: torch.Tensor,
    prompt_length: int,
    response_length: int,
    num_layers: int,
    top_k: int,
    num_experts: int,
    request_id: str,
) -> None:
    """Validate CPU capture before padding/transport; never accept zero placeholders.

    vLLM captures input-token routes: the last generated token has not run a
    forward, so a completed request contains P+R-1 routing rows, including prefill.
    """
    expected = (prompt_length + response_length - 1, num_layers, top_k)
    if response_length < 1 or tuple(routes.shape) != expected:
        raise RuntimeError(
            f"[R3][CAPTURE_INVALID] request={request_id} shape={tuple(routes.shape)} expected={expected} "
            f"prompt={prompt_length} response={response_length}; require full prefill+decode input-token routing"
        )
    if routes.is_floating_point() or routes.dtype == torch.bool:
        raise RuntimeError(f"[R3][CAPTURE_INVALID] request={request_id} expert dtype={routes.dtype}")
    invalid = (routes < 0) | (routes >= num_experts)
    ordered = routes.sort(dim=-1).values
    duplicate = (ordered[..., 1:] == ordered[..., :-1]).any(dim=-1)
    if invalid.any() or duplicate.any():
        raise RuntimeError(
            f"[R3][CAPTURE_INVALID] request={request_id} invalid_ids={invalid.sum().item()} "
            f"duplicate_rows={duplicate.sum().item()} bad_ids={invalid.nonzero()[:8].tolist()} "
            f"duplicate_token_layers={duplicate.nonzero()[:8].tolist()} "
            f"expert_range=[0,{num_experts}); possible incomplete TP gather or stale capture buffer"
        )
    checksum = (routes.to(torch.int64) * torch.arange(1, num_layers + 1).view(1, -1, 1)).sum().item()
    logger.warning(
        "[R3][CAPTURE] request=%s prompt=%s response=%s shape=%s input_token_routing=True "
        "last_generated_token_uncaptured=True invalid_ids=0 duplicate_rows=0 route_checksum=%s "
        "fingerprint=%s status=PASS",
        request_id,
        prompt_length,
        response_length,
        tuple(routes.shape),
        checksum,
        r3_route_fingerprint(routes).item(),
    )


class FSDPRouterReplay:
    """Bind validated rollout expert IDs to Qwen3 model layers and audit their use."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int,
        rank: int,
        checkpointing: bool,
        debug_native_compare: bool = False,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.checkpointing = checkpointing
        self.active = False
        self._targets: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._batch_id = 0
        self._microbatch = 0
        self._backward = False
        self._phase = "inactive"
        self.debug_native_compare = debug_native_compare
        self.diagnosing_native = False
        self._moe_diagnostic = None

    def install(self, model: nn.Module) -> None:
        """Attach to the exact NPU MoE entry point before FSDP wraps the modules."""
        blocks = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeSparseMoeBlock"]
        if len(blocks) != self.num_layers:
            raise ValueError(f"[R3] found {len(blocks)} Qwen3 MoE blocks, expected {self.num_layers}")
        for layer, block in enumerate(blocks):
            if getattr(block.forward, "__name__", "") != "qwen3_moe_sparse_moe_block_forward_npu":
                raise ValueError(f"[R3] layer={layer} is not using verl's Qwen3 NPU MoE forward")
            block._fsdp_r3_replay = (self, layer)
        logger.warning(
            "[R3][INSTALL] rank=%s layers=%s experts=%s top_k=%s checkpointing=%s scope=qwen3_moe/fsdp2/npu/sp1",
            self.rank,
            self.num_layers,
            self.num_experts,
            self.top_k,
            self.checkpointing,
        )

    @contextmanager
    def batch(self, forward_only: bool) -> Iterator[None]:
        """Keep targets active through forward and backward; clear on every exit."""
        if self.active:
            raise RuntimeError("[R3] nested replay batches are unsupported")
        self.active = True
        self._phase = "old_log_prob" if forward_only else "actor_update"
        self._batch_id += 1
        self._microbatch = 0
        self._totals = None
        self._forward_total = 0
        self._backward_total = 0
        try:
            yield
            if self._microbatch == 0:
                raise RuntimeError("[R3] replay batch completed without any routed micro-batches")
            slots, native_diff, replay_diff, checksum = self._totals.cpu().tolist()
            logger.warning(
                "[R3][VERIFY] rank=%s phase=%s batch=%s microbatches=%s layers=%s "
                "forward_calls=%s recompute_calls=%s replayed_slots=%s native_diff_slots=%s "
                "replay_mismatch_slots=%s target_checksum=%s coverage=1.000000 status=PASS",
                self.rank,
                self._phase,
                self._batch_id,
                self._microbatch,
                self.num_layers,
                self._forward_total,
                self._backward_total,
                slots,
                native_diff,
                replay_diff,
                checksum,
            )
        except Exception:
            logger.warning(
                "[R3][FAILED] rank=%s phase=%s batch=%s microbatch=%s backward=%s forward_layers=%s "
                "recompute_layers=%s",
                self.rank,
                self._phase,
                self._batch_id,
                self._microbatch,
                self._backward,
                getattr(self, "_forward_calls", None),
                getattr(self, "_backward_calls", None),
                exc_info=True,
            )
            raise
        finally:
            self.active = False
            self.diagnosing_native = False
            self._targets = self._mask = None
            self._totals = None
            self._backward = False
            self._forward_indices = {}
            self._layer_diffs = []
            self._diagnostic_indices = {}
            self._moe_diagnostic = None

    def set_microbatch(self, input_ids: torch.Tensor, routes: torch.Tensor | None, response_mask: torch.Tensor) -> None:
        """Validate nested alignment and prepare all captured input positions.

        Lossless synthetic padding samples use native routing. Real sequences
        replay every input except their final token, which vLLM never evaluated.
        """
        if not self.active:
            raise RuntimeError("[R3] set_microbatch outside an active batch")
        if routes is None or not routes.is_nested or not input_ids.is_nested:
            raise RuntimeError("[R3] require nested input_ids and routed_experts; capture or transport dropped routing")
        if not torch.equal(input_ids.offsets(), routes.offsets()):
            raise RuntimeError("[R3] input_ids/routed_experts offsets differ after dynamic micro-batch packing")
        targets = routes.values().to(dtype=torch.int64)
        if routes.dtype.is_floating_point or routes.dtype == torch.bool:
            raise RuntimeError(f"[R3] expert IDs must be integers, got {routes.dtype}")
        expected = (input_ids.values().numel(), self.num_layers, self.top_k)
        if tuple(targets.shape) != expected:
            raise RuntimeError(f"[R3] routed_experts shape={tuple(targets.shape)} expected={expected}")
        lengths = input_ids.offsets().diff()
        if (lengths < 1).any():
            raise RuntimeError("[R3] empty input sequence in replay micro-batch")
        real_samples = response_mask.bool().any(dim=-1)
        if real_samples.shape != lengths.shape:
            raise RuntimeError("[R3] response_mask batch dimension does not match nested input_ids")
        mask = torch.repeat_interleave(real_samples, lengths)
        mask[input_ids.offsets()[1:] - 1] = False
        selected = targets[mask]
        invalid = (selected < 0) | (selected >= self.num_experts)
        ordered = selected.sort(dim=-1).values
        duplicates = (ordered[..., 1:] == ordered[..., :-1]).any(dim=-1)
        if invalid.any() or duplicates.any():
            raise RuntimeError(
                f"[R3] rank={self.rank} phase={self._phase} microbatch={self._microbatch} "
                f"invalid_ids={invalid.sum().item()} duplicate_rows={duplicates.sum().item()} "
                f"bad_ids={invalid.nonzero()[:8].tolist()} "
                f"duplicate_token_layers={duplicates.nonzero()[:8].tolist()}; refuse partial/zero/stale routing"
            )
        self._targets, self._mask = targets, mask
        self._forward_calls = [0] * self.num_layers
        self._backward_calls = [0] * self.num_layers
        self._forward_indices: dict[int, torch.Tensor] = {}
        self._backward = False
        self._stats = torch.zeros(4, dtype=torch.int64, device=targets.device)
        self._diagnostic_indices = {}
        self._moe_diagnostic = None
        self._layer_diffs = []
        self._microbatch += 1
        lengths_cpu = lengths.cpu().tolist()
        self._diagnostic_offsets = input_ids.offsets().cpu().tolist() if self.debug_native_compare else None
        # Audit signatures on CPU, both to keep the signature arithmetic identical
        # to rollout validation and avoid relying on NPU int64 pow/remainder kernels.
        # Copy the narrow transport dtype once rather than the expanded int64 targets.
        routes_cpu = routes.values().detach().cpu()
        real_samples_cpu = real_samples.cpu()
        fingerprints = torch.stack([r3_route_fingerprint(piece[:-1]) for piece in routes_cpu.split(lengths_cpu)])
        # Every real sample is logged on every rank. The audit tool matches these
        # against rollout CAPTURE fingerprints after padding, DP redistribution,
        # sample shuffling, and dynamic micro-batch packing.
        logger.warning(
            "[R3][TRANSPORT] rank=%s phase=%s batch=%s microbatch=%s captured_lengths=%s "
            "sample_fingerprints=%s synthetic_samples=%s",
            self.rank,
            self._phase,
            self._batch_id,
            self._microbatch,
            [length - 1 for length, real in zip(lengths_cpu, real_samples_cpu.tolist(), strict=True) if real],
            fingerprints[real_samples_cpu].tolist(),
            (~real_samples_cpu).sum().item(),
        )
        if self.rank == 0 and self._microbatch == 1:
            logger.warning(
                "[R3][TARGET] rank=%s phase=%s batch=%s microbatch=%s lengths=%s captured_tokens=%s "
                "native_only_tokens=%s first_token_layer0=%s",
                self.rank,
                self._phase,
                self._batch_id,
                self._microbatch,
                lengths_cpu,
                mask.sum().item(),
                (~mask).sum().item(),
                selected[:1, 0].cpu().tolist(),
            )

    def route(
        self,
        layer: int,
        router_logits: torch.Tensor,
        native_indices: torch.Tensor,
        norm_topk_prob: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actual dispatch indices and differentiable current-policy weights."""
        if self._targets is None or self._mask is None:
            raise RuntimeError(f"[R3] layer={layer} executed without micro-batch targets")
        target = self._targets[:, layer]
        if native_indices.shape != target.shape or router_logits.shape != (target.shape[0], self.num_experts):
            raise RuntimeError(
                f"[R3] layer={layer} logits={tuple(router_logits.shape)} native={tuple(native_indices.shape)} "
                f"target={tuple(target.shape)}; token order/layout changed inside the model"
            )
        if self._backward:
            # Even the uncaptured final token must retain its original native
            # choice: rerouting it can change GMM saved-tensor shapes during
            # checkpoint replay, despite its logits carrying no response loss.
            indices = self._forward_indices[layer]
        else:
            indices = torch.where(self._mask.unsqueeze(-1), target, native_indices)
            self._forward_indices[layer] = indices.detach().clone()
            if self.compare_native_this_microbatch:
                self._diagnostic_indices[layer] = native_indices.detach().clone()
        weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32).gather(-1, indices)
        if norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        calls = self._backward_calls if self._backward else self._forward_calls
        calls[layer] += 1
        mask = self._mask.unsqueeze(-1)
        native_diff = ((native_indices != target) & mask).sum()
        replay_diff = ((indices != target) & mask).sum()
        checksum = (target * mask * (layer + 1)).sum()
        if not self._backward:
            self._stats += torch.stack((mask.sum() * self.top_k, native_diff, replay_diff, checksum)).detach()
            self._layer_diffs.append(native_diff.detach())
        else:
            self._stats[2] += replay_diff.detach()
        return indices, weights

    @property
    def compare_native_this_microbatch(self) -> bool:
        return self.debug_native_compare and self._phase == "old_log_prob" and self._microbatch == 1

    def moe_diagnostic(self, layer: int):
        """Observe only the first MoE in the paired no-grad diagnostic forwards."""
        if layer != 0 or not self.compare_native_this_microbatch or self._backward:
            return None
        if not (self.active or self.diagnosing_native):
            return None
        if self._moe_diagnostic is None:
            from verl.utils.fsdp_router_replay_debug import R3MoEDiagnostic

            self._moe_diagnostic = R3MoEDiagnostic(self.rank, self._batch_id, self._mask, self._diagnostic_offsets)
        return self._moe_diagnostic

    def record_native(self, layer: int, indices: torch.Tensor) -> None:
        """Observe an independent native forward without overriding its routing."""
        if not self.diagnosing_native or layer in self._diagnostic_indices:
            raise RuntimeError(f"[R3][NATIVE_COMPARE] unexpected native layer={layer}")
        self._diagnostic_indices[layer] = indices.detach().clone()

    @contextmanager
    def native_diagnostic(self) -> Iterator[None]:
        """Run a no-grad control on identical inputs and weights, preserving replay state."""
        if not self.active or self._backward or not self.compare_native_this_microbatch:
            raise RuntimeError("[R3][NATIVE_COMPARE] diagnostic outside first old-log-prob micro-batch")
        replay_native = self._diagnostic_indices
        self._diagnostic_indices = {}
        self.active = False
        self.diagnosing_native = True
        try:
            with torch.no_grad():
                yield
            if set(self._diagnostic_indices) != set(range(self.num_layers)):
                raise RuntimeError("[R3][NATIVE_COMPARE] native forward did not observe every layer")
            targets = self._targets.detach().cpu()
            mask = self._mask.cpu()
            offsets = self._diagnostic_offsets
            for path, observed in (("replay_hidden", replay_native), ("native_hidden", self._diagnostic_indices)):
                native = torch.stack([observed[layer] for layer in range(self.num_layers)], dim=1).cpu()
                stats = route_alignment_diagnostics(targets, native, mask, offsets)
                logger.warning(
                    "[R3][ROUTE_COMPARE] rank=%s batch=%s microbatch=%s path=%s %s",
                    self.rank,
                    self._batch_id,
                    self._microbatch,
                    path,
                    stats,
                )
        finally:
            self.active = True
            self.diagnosing_native = False
            self._diagnostic_indices = {}
            self._moe_diagnostic = None

    def mark_backward(self) -> None:
        """Mark checkpoint re-entries as recomputation, retaining this micro-batch's targets."""
        if self._forward_calls != [1] * self.num_layers:
            raise RuntimeError(f"[R3] missing/repeated forward layer calls: {self._forward_calls}")
        self._backward = True

    def finish_microbatch(self, forward_only: bool) -> None:
        """Verify every layer was reached and summarize without per-layer host syncs."""
        if self._forward_calls != [1] * self.num_layers:
            raise RuntimeError(f"[R3] missing/repeated forward layer calls: {self._forward_calls}")
        if not forward_only and self.checkpointing and any(count < 1 for count in self._backward_calls):
            raise RuntimeError(f"[R3] missing checkpoint recomputation calls: {self._backward_calls}")
        if self._stats[2].item() != 0:
            raise RuntimeError("[R3] actual dispatch indices differ from replay targets")
        if self.rank == 0 and self._microbatch == 1:
            logger.warning(
                "[R3][LAYERS] rank=%s phase=%s batch=%s microbatch=%s native_diff_slots_per_layer=%s "
                "forward_calls=%s recompute_calls=%s",
                self.rank,
                self._phase,
                self._batch_id,
                self._microbatch,
                torch.stack(self._layer_diffs).cpu().tolist(),
                self._forward_calls,
                self._backward_calls,
            )
        self._totals = self._stats.clone() if self._totals is None else self._totals + self._stats
        self._forward_total += sum(self._forward_calls)
        self._backward_total += sum(self._backward_calls)
        self._targets = self._mask = None
        self._forward_indices.clear()
        self._diagnostic_indices.clear()
        self._moe_diagnostic = None
