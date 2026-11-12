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
"""Bounded, detached first-MoE diagnostics for the opt-in R3 native comparison.

CPU references do not reuse the device gather, histogram, or unpermute kernels.
Expert intermediates are sampled by token and aligned by expert ID across the
two real forwards. No expert weights or full expanded activation are copied.
"""

import torch

from verl.utils.fsdp_router_replay_logging import r3_warning


def tensor_difference(actual, expected):
    """Numerical evidence, without imposing an arbitrary NPU accuracy threshold."""
    a, b = actual.float(), expected.float()
    if a.shape != b.shape:
        raise RuntimeError(f"[R3][MOE_DEBUG] comparison shape mismatch: {a.shape} vs {b.shape}")
    if not a.numel():
        return {"elements": 0}
    diff = (a - b).abs()
    rms = b.square().mean().sqrt().item()
    return {
        "elements": a.numel(),
        "nonfinite_actual": int((~a.isfinite()).sum()),
        "nonfinite_reference": int((~b.isfinite()).sum()),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "relative_rms": (a - b).square().mean().sqrt().item() / max(rms, 1e-30),
        "reference_rms": rms,
    }


def routing_weight_reference(logits, indices, normalize):
    """CPU FP32 oracle for current-policy expert weights (including slot order)."""
    weights = logits.float().softmax(-1).gather(-1, indices.long())
    return weights / weights.sum(-1, keepdim=True) if normalize else weights


def dispatch_diagnostics(indices, row_map, histogram, num_experts):
    """Validate CANN's inverse map: map[token*K+slot] -> expert-buffer row.

    A forward argsort is NOT this API's row map. Check all rows, including
    experts with no tokens, before using the map to sample device activations.
    """
    flat, mapping = indices.long().flatten(), row_map.long().flatten()
    counts = torch.bincount(flat, minlength=num_experts)
    histogram = histogram.flatten().double()
    if histogram.shape != counts.shape:
        raise RuntimeError(f"[R3][MOE_DEBUG] histogram shape={histogram.shape}, expected={counts.shape}")
    valid_map = mapping.shape == flat.shape and torch.equal(mapping.sort().values, torch.arange(flat.numel()))
    result = {
        "row_map_is_permutation": valid_map,
        "histogram_mismatch_experts": int((histogram != counts).sum()),
        "histogram_max_abs": (histogram - counts).abs().max().item(),
        "empty_experts": int((counts == 0).sum()),
        "dispatch_rows": flat.numel(),
    }
    if valid_map:
        expert_at_row = torch.empty_like(flat).scatter_(0, mapping, flat)
        result["expert_group_mismatch_rows"] = int((expert_at_row != flat.sort().values).sum())
    return result


def _sample_rows(tensor, rows):
    # Detach before indexing so even a caller with grad enabled retains no graph.
    return tensor.detach().index_select(0, rows.to(tensor.device)).cpu()


def _spread(indices, limit):
    if indices.numel() <= limit:
        return indices
    return indices[torch.linspace(0, indices.numel() - 1, limit).long()]


class R3MoEDiagnostic:
    """One first-layer replay/native pair, owned and cleared by its micro-batch."""

    def __init__(self, rank, batch, mask, offsets):
        self.rank, self.batch = rank, batch
        self.mask, self.offsets = mask.detach().cpu().bool(), offsets
        self.snapshots = {}

    def _log(self, tag, path, stats):
        r3_warning(
            "[R3][%s] rank=%s batch=%s microbatch=1 layer=0 path=%s %s",
            tag,
            self.rank,
            self.batch,
            path,
            stats,
        )

    @torch.no_grad()
    def routing(
        self,
        path,
        hidden,
        logits,
        native_ids,
        native_weights,
        ids,
        weights,
        normalize,
        *,
        gate_first_return,
        gate_class,
        logits_source,
    ):
        ids_cpu, native_cpu = ids.detach().cpu().long(), native_ids.detach().cpu().long()
        logits_cpu, weights_cpu = logits.detach().cpu(), weights.detach().cpu()
        gate_return_cpu = gate_first_return.detach().float().cpu()
        reference = routing_weight_reference(logits_cpu, ids_cpu, normalize)
        same_set = ids_cpu.sort(-1).values.eq(native_cpu.sort(-1).values).all(-1)
        if path == "replay":
            # Cover unchanged sets, changed sets, native-only rows and sequence
            # boundaries. At most 32 tokens, with deterministic spread over T.
            boundaries = torch.tensor(self.offsets[:-1] + [end - 1 for end in self.offsets[1:]])
            tokens = torch.cat(
                [
                    _spread((same_set & self.mask).nonzero().flatten(), 8),
                    _spread((~same_set & self.mask).nonzero().flatten(), 8),
                    _spread((~self.mask).nonzero().flatten(), 8),
                    _spread(boundaries.unique(), 8),
                ]
            ).unique(sorted=True)
            self.tokens = tokens
        elif path != "native" or "replay" not in self.snapshots:
            raise RuntimeError("[R3][MOE_DEBUG] native observation without replay snapshot")
        tokens = self.tokens
        self.snapshots[path] = {
            "ids": ids_cpu[tokens],
            "weights": weights_cpu[tokens].to(hidden.dtype),
            "input": _sample_rows(hidden, tokens),
            "logits": logits_cpu[tokens],
        }
        stats = {
            "tokens": ids_cpu.shape[0],
            "top_k": ids_cpu.shape[-1],
            "experts": logits.shape[-1],
            "normalize": normalize,
            "gate_class": gate_class,
            "router_logits_source": logits_source,
            "gate_first_return_row_sum_min": gate_return_cpu.sum(-1).min().item(),
            "gate_first_return_row_sum_max": gate_return_cpu.sum(-1).max().item(),
            "gate_first_return_vs_raw_logits": tensor_difference(gate_return_cpu, logits_cpu),
            "gate_first_return_vs_raw_softmax": tensor_difference(gate_return_cpu, logits_cpu.float().softmax(-1)),
            "hidden_dtype": str(hidden.dtype),
            "logits_dtype": str(logits.dtype),
            "weights_precast_dtype": str(weights.dtype),
            "ids_stride": list(ids.stride()),
            "weights_stride": list(weights.stride()),
            "hidden_stride": list(hidden.stride()),
            "same_set_captured_tokens": int((same_set & self.mask).sum()),
            "sample_token_rows": tokens.tolist(),
            "selected_weights_cpu_ref": tensor_difference(weights_cpu, reference),
            "selected_cast_weights_cpu_ref": tensor_difference(
                weights_cpu.to(hidden.dtype), reference.to(hidden.dtype)
            ),
            "native_weights_cpu_ref": tensor_difference(
                native_weights.detach().cpu(), routing_weight_reference(logits_cpu, native_cpu, normalize)
            ),
            "selected_weight_sum_min": weights_cpu.float().sum(-1).min().item(),
            "selected_weight_sum_max": weights_cpu.float().sum(-1).max().item(),
        }
        self._log("MOE_WEIGHTS", path, stats)

    @torch.no_grad()
    def dispatch(self, path, hidden, ids, permuted, row_map, histogram, num_experts):
        mapping = row_map.detach().cpu().long().flatten()
        stats = dispatch_diagnostics(ids.detach().cpu(), mapping, histogram.detach().cpu(), num_experts)
        stats.update(
            row_map_dtype=str(row_map.dtype),
            row_map_shape=list(row_map.shape),
            row_map_stride=list(row_map.stride()),
            permuted_shape=list(permuted.shape),
            permuted_stride=list(permuted.stride()),
            histogram_dtype=str(histogram.dtype),
        )
        if not stats["row_map_is_permutation"]:
            self._log("MOE_DISPATCH", path, stats)
            raise RuntimeError("[R3][MOE_DEBUG] invalid permute row map; cannot sample expert rows safely")
        rows = mapping.view(-1, ids.shape[-1])[self.tokens].flatten()
        self.snapshots[path]["rows"] = rows
        selected_input = _sample_rows(permuted, rows).view(len(self.tokens), ids.shape[-1], -1)
        self.snapshots[path]["permuted"] = selected_input
        stats["sample_permuted_input_ref"] = tensor_difference(
            selected_input, self.snapshots[path]["input"].unsqueeze(1).expand_as(selected_input)
        )
        stats["integer_invariants"] = (
            "PASS" if stats["histogram_mismatch_experts"] == stats["expert_group_mismatch_rows"] == 0 else "FAIL"
        )
        self._log("MOE_DISPATCH", path, stats)

    @torch.no_grad()
    def outputs(self, path, up, gate, activation, down, output, cast_weights, qat):
        snap = self.snapshots[path]
        rows, top_k = snap.pop("rows"), snap["ids"].shape[-1]
        for name, tensor in (("up", up), ("gate", gate), ("activation", activation), ("down", down)):
            snap[name] = _sample_rows(tensor, rows).view(len(self.tokens), top_k, -1)
        snap["output"] = _sample_rows(output, self.tokens)
        actual_weights = _sample_rows(cast_weights, self.tokens)
        # FP32 accumulation and native-dtype arithmetic bracket expected BF16
        # rounding. Neither reference invokes npu_moe_token_unpermute.
        fp32_merge = (snap["down"].float() * actual_weights.float().unsqueeze(-1)).sum(1)
        dtype_merge = (snap["down"] * actual_weights.unsqueeze(-1)).sum(1)
        self._log(
            "MOE_MERGE",
            path,
            {
                "qat": qat,
                "sample_tokens": len(self.tokens),
                "cast_weights_dtype": str(cast_weights.dtype),
                "cast_weights_stride": list(cast_weights.stride()),
                "down_dtype": str(down.dtype),
                "down_shape": list(down.shape),
                "down_stride": list(down.stride()),
                "output_dtype": str(output.dtype),
                "cast_weights_vs_observed_precast": tensor_difference(actual_weights, snap["weights"]),
                "unpermute_vs_cpu_fp32_merge": tensor_difference(snap["output"], fp32_merge),
                "unpermute_vs_cpu_dtype_merge": tensor_difference(snap["output"], dtype_merge),
                "swiglu_vs_cpu": tensor_difference(
                    snap["activation"], torch.nn.functional.silu(snap["gate"].float()) * snap["up"].float()
                ),
            },
        )
        if path == "native":
            self.compare()

    def compare(self):
        replay, native = self.snapshots["replay"], self.snapshots["native"]
        # An arbitrary slot permutation must not make identical experts appear
        # different. Shared-expert comparisons also work on changed-set tokens.
        token, rslot, nslot = (replay["ids"].unsqueeze(-1) == native["ids"].unsqueeze(-2)).nonzero(as_tuple=True)
        same = replay["ids"].sort(-1).values.eq(native["ids"].sort(-1).values).all(-1)
        stats = {
            "sample_tokens": len(self.tokens),
            "sample_same_set_tokens": int(same.sum()),
            "sample_same_set_different_order_tokens": int((same & (replay["ids"] != native["ids"]).any(-1)).sum()),
            "shared_token_expert_pairs": len(token),
            "input": tensor_difference(replay["input"], native["input"]),
            "logits": tensor_difference(replay["logits"], native["logits"]),
            "same_set_output": tensor_difference(replay["output"][same], native["output"][same]),
            "changed_set_output": tensor_difference(replay["output"][~same], native["output"][~same]),
            "shared_experts": {},
        }
        for name in ("weights", "permuted", "up", "gate", "activation", "down"):
            stats["shared_experts"][name] = tensor_difference(replay[name][token, rslot], native[name][token, nslot])
        # Changed expert sets legitimately change the normalization denominator.
        # Restrict this weight comparison to identical sets.
        keep = same[token]
        stats["same_set_weights"] = tensor_difference(
            replay["weights"][token[keep], rslot[keep]], native["weights"][token[keep], nslot[keep]]
        )
        stats["examples"] = [
            {
                "token": int(self.tokens[i]),
                "same_set": bool(same[i]),
                "replay_ids": replay["ids"][i].tolist(),
                "native_ids": native["ids"][i].tolist(),
                "replay_weights": replay["weights"][i].float().tolist(),
                "native_weights": native["weights"][i].float().tolist(),
                "replay_output_head": replay["output"][i, :4].float().tolist(),
                "native_output_head": native["output"][i, :4].float().tolist(),
            }
            for i in range(min(2, len(self.tokens)))
        ]
        self._log("MOE_COMPARE", "replay_vs_native", stats)
