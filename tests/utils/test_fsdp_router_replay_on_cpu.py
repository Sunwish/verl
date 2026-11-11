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

from verl.utils.fsdp_router_replay import FSDPRouterReplay, r3_route_fingerprint, validate_vllm_r3_routes
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
        return x[order // indices.shape[-1]], order

    def unpermute(x, order, probs):
        restored = torch.zeros_like(x).index_copy(0, order, x)
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

    def init(self):
        nn.Module.__init__(self)
        self.gate = nn.Linear(32, 4, bias=False)
        self.top_k, self.num_experts, self.norm_topk_prob = 2, 4, True
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(4, 64, 32) * 0.08)
        self.experts.down_proj = nn.Parameter(torch.randn(4, 32, 32) * 0.08)

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
@pytest.mark.parametrize("new_gate_api", [False, True])
def test_production_dispatch_and_gate_expert_gradients_match_oracle(block_class, dtype, new_gate_api, caplog):
    torch.manual_seed(27)
    block = block_class().to(dtype)
    oracle = copy.deepcopy(block)
    if new_gate_api:

        class TupleGate(nn.Module):
            def __init__(self, gate):
                super().__init__()
                self.weight = gate.weight
                self.top_k, self.norm_topk_prob = 2, True

            def forward(self, x):
                logits = F.linear(x, self.weight)
                weights, indices = logits.float().softmax(-1).topk(2)
                return logits, weights / weights.sum(-1, keepdim=True), indices

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
