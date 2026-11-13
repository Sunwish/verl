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
"""Run the production Qwen3 NPU forward with CPU kernel substitutes.

Only hardware kernels are replaced. The forward functions are compiled from
their source AST to avoid importing torch_npu and globally patching Transformers
in a CPU test process. The QAT autograd and checkpoint implementations are real.
"""

import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from verl.utils.fsdp_router_replay import (
    FSDPRouterReplay,
    log_r3_probability_comparison,
    r3_route_fingerprint,
    route_alignment_diagnostics,
    validate_vllm_r3_routes,
)
from verl.utils.qat.mxfp_linear import MXFPQATConfig
from verl.utils.qat.mxfp_moe import (
    attach_mxfp_expert_qat,
    get_mxfp_expert_qat_runtime_config,
    mxfp_grouped_matmul,
    prepare_mxfp_fprop_input,
)
from verl.workers.utils.padding import left_right_2_no_padding


def _gmm(x, weight, counts):
    return torch.cat([part @ w for part, w in zip(x.split([int(c) for c in counts]), weight, strict=True)])


def _qat_gmm(x, weight, group_list, group_list_type, *, split_item=2, group_type=0):
    counts = group_list.tolist()
    if group_list_type == 0:
        counts = [c - (counts[i - 1] if i else 0) for i, c in enumerate(counts)]
    if split_item == 2:
        return _gmm(x, weight, counts)
    pieces_x = x.split([int(c) for c in counts], dim=1)
    pieces_w = weight.split([int(c) for c in counts])
    return torch.stack([a @ b for a, b in zip(pieces_x, pieces_w, strict=True)])


@pytest.fixture
def block_class(monkeypatch):
    # Ascend's histc accepts integer expert IDs; CPU PyTorch requires floating point.
    cpu_histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda x, **kwargs: cpu_histc(x.float(), **kwargs))

    def permute(x, indices):
        order = indices.flatten().argsort(stable=True)
        # CANN returns the inverse map: token-slot -> sorted expert-buffer row.
        return x[order // indices.shape[-1]], order.argsort()

    def unpermute(x, mapping, probs):
        restored = x.index_select(0, mapping)
        return (restored.view(*probs.shape, -1) * probs.unsqueeze(-1)).sum(dim=1)

    path = Path(__file__).parents[2] / "verl/models/transformers/npu_patch.py"
    names = {
        "_qwen3_sparse_moe_uses_legacy_block_api",
        "_qwen3_sparse_moe_routed_forward_npu",
        "qwen3_moe_sparse_moe_block_forward_npu",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "torch_npu": SimpleNamespace(
            npu_moe_token_permute=permute,
            npu_moe_token_unpermute=unpermute,
            npu_swiglu=lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
        ),
        "NPUGmmFunction": SimpleNamespace(apply=_gmm),
        "get_mxfp_expert_qat_runtime_config": get_mxfp_expert_qat_runtime_config,
        "mxfp_grouped_matmul": mxfp_grouped_matmul,
        "prepare_mxfp_fprop_input": prepare_mxfp_fprop_input,
    }
    exec(compile(tree, str(path), "exec"), ns)
    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", _qat_gmm)

    def init(self, num_experts=4, top_k=2):
        nn.Module.__init__(self)
        self.gate = nn.Linear(32, num_experts, bias=False)
        self.top_k, self.num_experts, self.norm_topk_prob = top_k, num_experts, True
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(num_experts, 64, 32) * 0.08)
        self.experts.down_proj = nn.Parameter(torch.randn(num_experts, 32, 32) * 0.08)

    return type(
        "Qwen3MoeSparseMoeBlock",
        (nn.Module,),
        {"__init__": init, "forward": ns["qwen3_moe_sparse_moe_block_forward_npu"]},
    )


def _nested_inputs(lengths, layers=2):
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    ids = torch.nested.nested_tensor_from_jagged(torch.arange(sum(lengths)), offsets)
    targets = (torch.arange(sum(lengths) * layers).view(-1, layers, 1) + torch.tensor([2, 3])) % 4
    routes = torch.nested.nested_tensor_from_jagged(targets, offsets)
    return ids, routes, torch.ones(len(lengths), 3)


def _oracle(block, x, indices):
    weights = block.gate(x).float().softmax(-1).gather(-1, indices)
    if block.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    outputs = []
    for token, chosen in zip(x, indices, strict=True):
        experts = []
        for expert in chosen:
            gate, up = F.linear(token, block.experts.gate_up_proj[expert]).chunk(2)
            experts.append(F.linear(F.silu(gate) * up, block.experts.down_proj[expert]))
        outputs.append(torch.stack(experts))
    return (torch.stack(outputs) * weights.to(x.dtype).unsqueeze(-1)).sum(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate_api", ["legacy", "tuple_logits", "tuple_probs"])
def test_production_dispatch_and_gate_expert_gradients_match_oracle(block_class, dtype, gate_api, caplog):
    torch.manual_seed(27)
    block = block_class().to(dtype)
    oracle = copy.deepcopy(block)
    if gate_api != "legacy":

        class TupleGate(nn.Module):
            def __init__(self, gate):
                super().__init__()
                self.weight = gate.weight
                self.top_k, self.norm_topk_prob = 2, True

            def forward(self, x):
                logits = F.linear(x, self.weight)
                probs = logits.float().softmax(-1)
                weights, indices = probs.topk(2)
                # Transformers 5.0/5.1 returns probabilities in tuple[0],
                # whereas later releases return the pre-softmax logits.
                first = probs if gate_api == "tuple_probs" else logits
                return first, (weights / weights.sum(-1, keepdim=True)).to(first.dtype), indices

        block.gate = TupleGate(block.gate)
    replay = FSDPRouterReplay(1, 4, 2, 0, False)
    replay.install(block)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    x = torch.randn(7, 32, dtype=dtype, requires_grad=True)
    oracle_x = x.detach().clone().requires_grad_()
    native = oracle.gate(oracle_x).float().topk(2).indices
    expected_indices = routes.values()[:, 0].clone()
    expected_indices[[3, 6]] = native[[3, 6]]
    expected = _oracle(oracle, oracle_x, expected_indices)
    with caplog.at_level(logging.WARNING), replay.batch(forward_only=False):
        replay.set_microbatch(ids, routes, mask)
        output = block(x)
        actual = output[0] if isinstance(output, tuple) else output
        replay.mark_backward()
        actual.float().square().sum().backward()
        replay.finish_microbatch(False)
    expected.float().square().sum().backward()
    tolerance = {"atol": 0.008, "rtol": 0.03} if dtype == torch.bfloat16 else {"atol": 2e-6, "rtol": 1e-4}
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(x.grad, oracle_x.grad, **tolerance)
    for (_, p), (_, expected_p) in zip(block.named_parameters(), oracle.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, expected_p.grad, **tolerance)
    assert block.gate.weight.grad.abs().sum() > 0
    assert "replay_mismatch_slots=0" in caplog.text
    assert not replay.active and replay._targets is None and not replay._forward_indices


@pytest.mark.parametrize("qat", [False, True])
def test_nonreentrant_checkpoint_and_dynamic_microbatches(block_class, qat, caplog):
    torch.manual_seed(14)
    model = nn.ModuleList([block_class(), block_class()])
    if qat:
        config = MXFPQATConfig(
            mode="mxfp_w4a8",
            quant_backend="torch",
            fake_quant_targets=("fprop", "dgrad", "wgrad"),
            backward_reduction_axis_fake_quant=True,
            rotation_enable=True,
            rotation_targets=("fprop", "dgrad"),
            rotation_block_size=32,
            rotation_seed=7,
        )
        assert attach_mxfp_expert_qat(model, config) == 2
    plain_model = copy.deepcopy(model)
    replay = FSDPRouterReplay(2, 4, 2, 0, True)
    plain = FSDPRouterReplay(2, 4, 2, 0, False)
    replay.install(model)
    plain.install(plain_model)
    for lengths in ([4, 3], [2], [5, 2, 3]):
        ids, routes, mask = _nested_inputs(lengths)
        original = torch.randn(sum(lengths), 32)
        results = []
        for controller, blocks, use_checkpoint in ((replay, model, True), (plain, plain_model, False)):
            with caplog.at_level(logging.WARNING), controller.batch(False):
                controller.set_microbatch(ids, routes, mask)
                x = original.clone().requires_grad_()
                for block in blocks:
                    x = checkpoint(block, x, use_reentrant=False)[0] if use_checkpoint else block(x)[0]
                controller.mark_backward()
                x.square().sum().backward()
                controller.finish_microbatch(False)
                results.append(x.detach())
        torch.testing.assert_close(*results)
    for p, q in zip(model.parameters(), plain_model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)
        assert torch.isfinite(p.grad).all()
    assert "recompute_calls=2" in caplog.text


def test_padding_transport_reorder_and_synthetic_samples(monkeypatch):
    def unpad_input(x, mask):
        indices = mask.flatten().nonzero().flatten()
        offsets = F.pad(mask.sum(-1).cumsum(0), (1, 0))
        return x.flatten(0, 1)[indices], indices, offsets, int(mask.sum(-1).max())

    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", unpad_input)
    monkeypatch.setattr("verl.workers.utils.padding.index_first_axis", lambda x, indices: x[indices])
    # Prompt padding on the left, response padding on the right; route row P+R-1 is uncaptured.
    attention = torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    routes = torch.zeros(3, 6, 2, 2, dtype=torch.int64)
    for i, positions in enumerate(([1, 2, 3], [0, 1, 2, 3, 4], [])):
        for position in positions:
            routes[i, position] = torch.tensor([[position % 4, (position + 1) % 4], [2, 3]])
    data = TensorDict(
        {
            "input_ids": torch.arange(18).view(3, 6),
            "position_ids": torch.arange(6).expand(3, -1),
            "attention_mask": attention,
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [0, 0, 0]]),
            "routed_experts": routes,
        },
        batch_size=[3],
    )
    nested = left_right_2_no_padding(data)
    # Same TensorDict indexing that dynamic packing uses.
    from verl.utils.tensordict_utils import index_select_tensor_dict

    reordered = index_select_tensor_dict(nested, [1, 0, 2])
    controller = FSDPRouterReplay(2, 4, 2, 1, False)
    with controller.batch(True):
        controller.set_microbatch(reordered["input_ids"], reordered["routed_experts"], reordered["response_mask"])
        logits = torch.randn(12, 4)
        native = logits.topk(2).indices
        actual, _ = controller.route(0, logits, native, True)
        torch.testing.assert_close(actual[:5], routes[1, :5, 0])
        torch.testing.assert_close(actual[6:9], routes[0, 1:4, 0])
        torch.testing.assert_close(actual[[5, 9, 10, 11]], native[[5, 9, 10, 11]])
        controller.route(1, logits, native, True)
        controller.finish_microbatch(True)


@pytest.mark.parametrize("damage", ["missing", "offset", "duplicate", "range", "float"])
def test_bad_microbatch_fails_and_cleans_up(damage):
    controller = FSDPRouterReplay(2, 4, 2, 1, False)
    ids, routes, mask = _nested_inputs([4, 3])
    if damage == "missing":
        routes = None
    elif damage == "offset":
        routes = torch.nested.nested_tensor_from_jagged(routes.values(), torch.tensor([0, 3, 7]))
    elif damage == "float":
        routes = routes.float()
    else:
        routes.values()[0, 0, 0] = routes.values()[0, 0, 1] if damage == "duplicate" else 4
    with pytest.raises(RuntimeError), controller.batch(True):
        controller.set_microbatch(ids, routes, mask)
    assert not controller.active and controller._targets is None and not controller._forward_indices


def test_missing_layer_and_missing_recompute_are_errors():
    ids, routes, mask = _nested_inputs([4])
    controller = FSDPRouterReplay(2, 4, 2, 1, True)
    logits = torch.randn(4, 4)
    with pytest.raises(RuntimeError, match="forward layer"), controller.batch(False):
        controller.set_microbatch(ids, routes, mask)
        controller.route(0, logits, logits.topk(2).indices, True)
        controller.mark_backward()
    with pytest.raises(RuntimeError, match="recomputation"), controller.batch(False):
        controller.set_microbatch(ids, routes, mask)
        for layer in range(2):
            controller.route(layer, logits, logits.topk(2).indices, True)
        controller.mark_backward()
        controller.finish_microbatch(False)


def test_uncaptured_final_native_route_is_frozen_for_recompute():
    ids, routes, mask = _nested_inputs([4], layers=1)
    controller = FSDPRouterReplay(1, 4, 2, 1, True)
    logits = torch.randn(4, 4)
    with controller.batch(False):
        controller.set_microbatch(ids, routes, mask)
        first, _ = controller.route(0, logits, logits.topk(2).indices, True)
        controller.mark_backward()
        second, _ = controller.route(0, -logits, (-logits).topk(2).indices, True)
        torch.testing.assert_close(first, second)
        controller.finish_microbatch(False)


@pytest.mark.parametrize("damage", [None, "missing_prompt", "extra_last", "zero", "negative", "range", "float"])
def test_capture_validation(damage, caplog):
    _, nested, _ = _nested_inputs([6])
    routes = nested.values()[:-1].clone()
    if damage == "missing_prompt":
        routes = routes[-2:]
    elif damage == "extra_last":
        routes = nested.values()
    elif damage == "zero":
        routes.zero_()
    elif damage in ("negative", "range"):
        routes[0, 0, 0] = -1 if damage == "negative" else 4
    elif damage == "float":
        routes = routes.float()
    if damage is None:
        with caplog.at_level(logging.WARNING):
            validate_vllm_r3_routes(routes, 3, 3, 2, 2, 4, "cpu-test")
        assert "[R3][CAPTURE]" in caplog.text
    else:
        with pytest.raises(RuntimeError, match="CAPTURE_INVALID"):
            validate_vllm_r3_routes(routes, 3, 3, 2, 2, 4, "cpu-test")


def test_reference_model_is_unmodified(block_class):
    actor, reference = block_class(), block_class()
    replay = FSDPRouterReplay(1, 4, 2, 0, False)
    replay.install(actor)
    assert not hasattr(reference, "_fsdp_r3_replay")
    x = torch.randn(3, 32)
    torch.testing.assert_close(reference(x)[0], _oracle(reference, x, reference.gate(x).topk(2).indices))


def test_fingerprint_detects_token_layer_slot_reordering():
    routes = torch.arange(5 * 2 * 2).view(5, 2, 2) % 7
    fingerprint = r3_route_fingerprint(routes)
    assert fingerprint == r3_route_fingerprint(routes.to(torch.uint8))
    for dimension in range(3):
        assert fingerprint != r3_route_fingerprint(routes.flip(dimension))


def test_alignment_diagnostics_ignore_slot_order_and_detect_token_and_layer_shifts():
    torch.manual_seed(281)
    native = torch.stack([torch.randperm(128)[:8] for _ in range(20 * 5)]).view(20, 5, 8)
    mask = torch.ones(20, dtype=torch.bool)
    mask[[9, 19]] = False
    offsets = [0, 10, 20]
    slots = route_alignment_diagnostics(native.flip(-1), native, mask, offsets)
    assert slots["set_equal_fraction_per_layer"] == [1] * 5
    shifted = native.clone()
    for start, end in ((0, 10), (10, 20)):
        shifted[start : end - 1] = native[start + 1 : end]
    token = route_alignment_diagnostics(shifted, native, mask, offsets)
    assert token["token_delta_set_overlap"][1] == [1] * 5
    assert max(token["token_delta_set_overlap"][0]) < 0.2
    layer = route_alignment_diagnostics(native.roll(-1, 1), native, mask, offsets)
    assert layer["layer_delta_set_overlap"][1] == [1] * 4


def test_alignment_diagnostics_never_compare_across_sequence_boundaries():
    # A global shift would match at token 1 -> 2, which crosses samples.
    native = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]])
    targets = torch.tensor([[[2, 3]], [[4, 5]], [[6, 7]], [[0, 1]]])
    result = route_alignment_diagnostics(targets, native, torch.ones(4, dtype=torch.bool), [0, 2, 4])
    assert result["token_delta_set_overlap"][1] == [1]
    short = route_alignment_diagnostics(targets, native, torch.ones(4, dtype=torch.bool), [0, 1, 2, 3, 4])
    assert short["token_delta_set_overlap"][2] == []
    assert short["token_delta_set_overlap"][-2] == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_diagnostic_is_independent_and_preserves_replay_outputs_and_counters(block_class, dtype, caplog):
    torch.manual_seed(39)
    block = block_class().to(dtype)
    reference = copy.deepcopy(block)
    x = torch.randn(7, 32, dtype=dtype)
    native_expected = reference(x)[0]
    replay = FSDPRouterReplay(1, 4, 2, 1, False, debug_native_compare=True)
    replay.install(block)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    with caplog.at_level("WARNING"), replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        output = block(x)[0]
        cached_indices = replay._forward_indices[0].clone()
        stats = replay._stats.clone()
        with replay.native_diagnostic():
            native_actual = block(x)[0]
            assert replay.diagnosing_native and not replay.active
        torch.testing.assert_close(native_actual, native_expected)
        assert replay.active and not replay.diagnosing_native
        assert replay._forward_calls == [1]
        assert replay._backward_calls == [0]
        torch.testing.assert_close(replay._stats, stats)
        torch.testing.assert_close(replay._forward_indices[0], cached_indices)
        torch.testing.assert_close(output, _oracle(reference, x, cached_indices))
        replay.finish_microbatch(True)
    assert "path=native_hidden" in caplog.text and "path=replay_hidden" in caplog.text
    assert "forward_calls=1 recompute_calls=0" in caplog.text
    assert not replay._diagnostic_indices


def test_replay_native_expert_ids_with_different_slot_order_preserves_output(block_class):
    torch.manual_seed(30)
    block = block_class().to(torch.bfloat16)
    reference = copy.deepcopy(block)
    x = torch.randn(7, 32, dtype=torch.bfloat16)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    indices = reference.gate(x).float().softmax(-1).topk(2).indices.flip(-1)
    routes = torch.nested.nested_tensor_from_jagged(indices.unsqueeze(1), ids.offsets())
    replay = FSDPRouterReplay(1, 4, 2, 0, False)
    replay.install(block)
    expected = reference(x)[0]
    with replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        actual = block(x)[0]
        replay.finish_microbatch(True)
    torch.testing.assert_close(actual, expected)


def test_probability_comparison_uses_valid_tokens_and_keeps_inputs_unchanged(caplog):
    native = torch.tensor([[0.01, 0.2, 0.3, 0.9, 0.8, 0.5]]).log()
    replay = torch.tensor([[0.9, 0.01, 0.8, 0.2, 0.3, 0.5]]).log()
    rollout = native.clone()
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    rollout[0, -1] = float("nan")
    before = replay.clone()
    with caplog.at_level("WARNING"):
        stats = log_r3_probability_comparison(replay, native, rollout, mask, rank=1, batch=1)
    assert stats["native_rollout_pearson"] == pytest.approx(1)
    assert stats["replay_rollout_pearson"] < 0
    assert stats["native_rollout_mean_abs_prob_diff"] == 0
    torch.testing.assert_close(replay, before)
    assert "[R3][PROB_COMPARE] rank=1 batch=1 valid_tokens=5" in caplog.text


def test_native_diagnostic_failure_cleans_up_and_propagates(block_class):
    block = block_class()
    replay = FSDPRouterReplay(1, 4, 2, 1, False, debug_native_compare=True)
    replay.install(block)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    with pytest.raises(RuntimeError, match="native failure"), replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        block(torch.randn(7, 32))
        with replay.native_diagnostic():
            raise RuntimeError("native failure")
    assert not replay.active and not replay.diagnosing_native
    assert replay._targets is None and not replay._forward_indices and not replay._diagnostic_indices
    assert replay._moe_diagnostic is None


def _moe_log(caplog, tag, path):
    messages = [record.getMessage() for record in caplog.records if f"[R3][{tag}]" in record.getMessage()]
    message = next(message for message in messages if f"path={path} " in message)
    return ast.literal_eval(message[message.index("{") :])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate_api", ["legacy", "tuple_logits", "tuple_probs"])
@pytest.mark.parametrize("qat", [False, True])
def test_top8_128_experts_diagnostics_and_slot_permutation(block_class, dtype, gate_api, qat, caplog):
    torch.manual_seed(154)
    block = block_class(num_experts=128, top_k=8).to(dtype)
    if gate_api != "legacy":

        class TupleGate(nn.Module):
            def __init__(self, gate):
                super().__init__()
                self.weight = gate.weight
                self.top_k, self.norm_topk_prob = 8, True

            def forward(self, x):
                logits = F.linear(x, self.weight)
                probs = logits.float().softmax(-1)
                weights, indices = probs.topk(self.top_k)
                first = probs if gate_api == "tuple_probs" else logits
                return first, (weights / weights.sum(-1, keepdim=True)).to(first.dtype), indices

        block.gate = TupleGate(block.gate)
    if qat:
        config = MXFPQATConfig(
            mode="mxfp_w4a8",
            quant_backend="torch",
            fake_quant_targets=("fprop", "dgrad", "wgrad"),
            backward_reduction_axis_fake_quant=True,
            rotation_enable=True,
            rotation_targets=("fprop", "dgrad"),
            rotation_block_size=32,
            rotation_seed=7,
        )
        assert attach_mxfp_expert_qat(block, config) == 1
    reference = copy.deepcopy(block)
    x = torch.randn(17, 32, dtype=dtype)
    native_ids = F.linear(x, block.gate.weight).float().softmax(-1).topk(8).indices
    ids, _, mask = _nested_inputs([9, 8], layers=1)
    routes = torch.nested.nested_tensor_from_jagged(native_ids.flip(-1).unsqueeze(1), ids.offsets())
    replay = FSDPRouterReplay(1, 128, 8, 2, False, debug_native_compare=True)
    replay.install(block)
    with caplog.at_level("WARNING"), torch.no_grad(), replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        replay_out = block(x)
        replay_out = replay_out[0] if gate_api == "legacy" else replay_out
        with replay.native_diagnostic():
            native_out = block(x)
            native_out = native_out[0] if gate_api == "legacy" else native_out
        assert replay._moe_diagnostic is None
        replay.finish_microbatch(True)
    expected = reference(x)
    expected = expected[0] if gate_api == "legacy" else expected
    # BF16 summation can change by one ulp when the top-8 slots are reordered.
    torch.testing.assert_close(replay_out, native_out, atol=0.001, rtol=0.02)
    torch.testing.assert_close(native_out, expected)
    for path in ("replay", "native"):
        dispatch = _moe_log(caplog, "MOE_DISPATCH", path)
        assert dispatch["integer_invariants"] == "PASS"
        assert dispatch["sample_permuted_input_ref"]["max_abs"] == 0
        weights = _moe_log(caplog, "MOE_WEIGHTS", path)
        assert weights["top_k"] == 8 and weights["experts"] == 128
        source = "gate_linear_return" if gate_api == "legacy" else "explicit_gate_linear"
        assert weights["router_logits_source"] == source
        assert weights["selected_weights_cpu_ref"]["max_abs"] < (0.002 if dtype == torch.bfloat16 else 1e-6)
        assert weights["native_weights_cpu_ref"]["max_abs"] < (0.002 if dtype == torch.bfloat16 else 1e-6)
        if gate_api == "tuple_probs":
            assert weights["gate_first_return_vs_raw_softmax"]["max_abs"] < 1e-6
            assert weights["gate_first_return_vs_raw_logits"]["max_abs"] > 0.1
        merge = _moe_log(caplog, "MOE_MERGE", path)
        assert merge["cast_weights_vs_observed_precast"]["max_abs"] == 0
        assert merge["unpermute_vs_cpu_dtype_merge"]["max_abs"] == 0
        assert merge["qat"] == qat
    pair = _moe_log(caplog, "MOE_COMPARE", "replay_vs_native")
    assert pair["sample_same_set_different_order_tokens"] > 0
    assert pair["sample_same_set_tokens"] == pair["sample_tokens"]
    for stage in ("permuted", "up", "gate", "activation", "down"):
        assert pair["shared_experts"][stage]["max_abs"] == 0
    assert pair["input"]["max_abs"] == pair["logits"]["max_abs"] == 0
    assert pair["same_set_weights"]["max_abs"] < 1e-6


@pytest.mark.parametrize("first_is_probs", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tuple_router_contract_checkpoint_forward_and_gradients(block_class, first_is_probs, normalize, dtype):
    torch.manual_seed(647)
    base = block_class(num_experts=128, top_k=8).to(dtype)
    base.norm_topk_prob = normalize
    oracle = copy.deepcopy(base)

    class TupleGate(nn.Module):
        def __init__(self, gate):
            super().__init__()
            self.weight = gate.weight
            self.top_k, self.norm_topk_prob = 8, normalize

        def forward(self, x):
            raw = F.linear(x, self.weight)
            probs = raw.float().softmax(-1)
            weights, selected = probs.topk(self.top_k)
            if self.norm_topk_prob:
                weights = weights / weights.sum(-1, keepdim=True)
            first = probs if first_is_probs else raw
            return first, weights.to(first.dtype), selected

    base.gate = TupleGate(base.gate)
    ids, _, mask = _nested_inputs([5, 4], layers=1)
    targets = torch.stack([torch.randperm(128)[:8] for _ in range(9)])
    routes = torch.nested.nested_tensor_from_jagged(targets.unsqueeze(1), ids.offsets())
    x = torch.randn(9, 32, dtype=dtype)
    oracle_x = x.detach().clone().requires_grad_()
    chosen = targets.clone()
    chosen[[4, 8]] = oracle.gate(oracle_x).float().softmax(-1).topk(8).indices[[4, 8]]
    expected = _oracle(oracle, oracle_x, chosen)
    expected.float().square().sum().backward()
    tolerance = {"atol": 0.008, "rtol": 0.03} if dtype == torch.bfloat16 else {"atol": 2e-6, "rtol": 1e-4}
    for checkpointing in (False, True):
        block = copy.deepcopy(base)
        replay = FSDPRouterReplay(1, 128, 8, 0, checkpointing)
        replay.install(block)
        actual_x = x.detach().clone().requires_grad_()
        with replay.batch(False):
            replay.set_microbatch(ids, routes, mask)
            actual = checkpoint(block, actual_x, use_reentrant=False) if checkpointing else block(actual_x)
            replay.mark_backward()
            actual.float().square().sum().backward()
            replay.finish_microbatch(False)
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(actual_x.grad, oracle_x.grad, **tolerance)
        for (_, param), (_, reference) in zip(block.named_parameters(), oracle.named_parameters(), strict=True):
            torch.testing.assert_close(param.grad, reference.grad, **tolerance)
        assert block.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("damage", ["weights", "permuted", "histogram", "row_map", "gmm", "merge"])
def test_moe_diagnostics_locate_injected_kernel_errors(block_class, monkeypatch, caplog, damage):
    torch.manual_seed(81)
    block = block_class(num_experts=128, top_k=8)
    x = torch.randn(7, 32)
    ids, _, mask = _nested_inputs([4, 3], layers=1)
    native = block.gate(x).float().softmax(-1).topk(8).indices
    routes = torch.nested.nested_tensor_from_jagged(native.flip(-1).unsqueeze(1), ids.offsets())
    replay = FSDPRouterReplay(1, 128, 8, 1, False, debug_native_compare=True)
    replay.install(block)
    namespace = block.forward.__func__.__globals__
    kernels = namespace["torch_npu"]
    if damage == "weights":
        original = replay.route

        def bad_weights(*args):
            experts, weights = original(*args)
            return experts, weights.roll(1, -1)

        monkeypatch.setattr(replay, "route", bad_weights)
    elif damage in ("permuted", "row_map"):
        original = kernels.npu_moe_token_permute

        def bad_permute(*args):
            tokens, mapping = original(*args)
            return (tokens.roll(1, 0), mapping) if damage == "permuted" else (tokens, mapping.roll(1))

        monkeypatch.setattr(kernels, "npu_moe_token_permute", bad_permute)
    elif damage == "histogram":
        original = torch.histc
        monkeypatch.setattr(torch, "histc", lambda *args, **kwargs: original(*args, **kwargs).roll(1))
    elif damage == "gmm":
        original = namespace["NPUGmmFunction"].apply
        monkeypatch.setattr(
            namespace["NPUGmmFunction"], "apply", lambda *args: original(*args) * (1.3 if replay.active else 1)
        )
    else:
        original = kernels.npu_moe_token_unpermute
        monkeypatch.setattr(kernels, "npu_moe_token_unpermute", lambda *args, **kwargs: original(*args, **kwargs) + 1)
    with caplog.at_level("WARNING"), replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        block(x)
        with replay.native_diagnostic():
            block(x)
        replay.finish_microbatch(True)
    if damage == "weights":
        assert _moe_log(caplog, "MOE_WEIGHTS", "replay")["selected_weights_cpu_ref"]["max_abs"] > 0.001
    elif damage == "permuted":
        assert _moe_log(caplog, "MOE_DISPATCH", "replay")["sample_permuted_input_ref"]["max_abs"] > 1
    elif damage == "histogram":
        assert _moe_log(caplog, "MOE_DISPATCH", "replay")["histogram_mismatch_experts"] > 0
    elif damage == "row_map":
        assert _moe_log(caplog, "MOE_DISPATCH", "replay")["expert_group_mismatch_rows"] > 0
    elif damage == "gmm":
        pair = _moe_log(caplog, "MOE_COMPARE", "replay_vs_native")
        assert pair["shared_experts"]["permuted"]["max_abs"] == 0
        assert pair["shared_experts"]["up"]["relative_rms"] == pytest.approx(0.3)
    else:
        assert _moe_log(caplog, "MOE_MERGE", "replay")["unpermute_vs_cpu_fp32_merge"]["mean_abs"] > 0.99


def test_dispatch_oracle_rejects_forward_sort_and_duplicate_maps():
    from verl.utils.fsdp_router_replay_debug import dispatch_diagnostics

    ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
    forward = ids.flatten().argsort(stable=True)
    inverse = forward.argsort()
    histogram = torch.bincount(ids.flatten(), minlength=4)
    assert dispatch_diagnostics(ids, inverse, histogram, 4)["expert_group_mismatch_rows"] == 0
    assert dispatch_diagnostics(ids, forward, histogram, 4)["expert_group_mismatch_rows"] > 0
    invalid = inverse.clone()
    invalid[0] = invalid[1]
    assert not dispatch_diagnostics(ids, invalid, histogram, 4)["row_map_is_permutation"]
    assert dispatch_diagnostics(ids, inverse, histogram.float() + 0.1, 4)["histogram_mismatch_experts"] == 4


def test_invalid_row_map_fails_with_diagnostics_and_cleans_state(block_class, monkeypatch, caplog):
    block = block_class()
    replay = FSDPRouterReplay(1, 4, 2, 1, False, debug_native_compare=True)
    replay.install(block)
    kernels = block.forward.__func__.__globals__["torch_npu"]
    original = kernels.npu_moe_token_permute

    def bad_permute(*args):
        tokens, mapping = original(*args)
        mapping[0] = mapping[1]
        return tokens, mapping

    monkeypatch.setattr(kernels, "npu_moe_token_permute", bad_permute)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="invalid permute row map"), replay.batch(True):
        replay.set_microbatch(ids, routes, mask)
        block(torch.randn(7, 32))
    assert not _moe_log(caplog, "MOE_DISPATCH", "replay")["row_map_is_permutation"]
    assert replay._moe_diagnostic is None and not replay.active and replay._targets is None


@pytest.mark.parametrize("enabled,forward_only", [(False, True), (True, False)])
def test_moe_diagnostics_do_not_run_when_disabled_or_during_actor_update(block_class, enabled, forward_only, caplog):
    block = block_class()
    replay = FSDPRouterReplay(1, 4, 2, 1, False, debug_native_compare=enabled)
    replay.install(block)
    ids, routes, mask = _nested_inputs([4, 3], layers=1)
    with caplog.at_level("WARNING"), replay.batch(forward_only):
        replay.set_microbatch(ids, routes, mask)
        assert replay.moe_diagnostic(0) is None
        assert replay.moe_diagnostic(1) is None
        block(torch.randn(7, 32))
        replay.finish_microbatch(forward_only)
    assert "[R3][MOE_" not in caplog.text
