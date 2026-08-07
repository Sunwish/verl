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

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

from verl.utils.qat.core import QATConfig, apply_qat, invalidate_all_scales
from verl.utils.qat.linear import QATLinear, QATMode
from verl.utils.qat.mxfp8_experts import MXFP8QATExperts
from verl.utils.qat.mxfp8_linear import (
    MXFP8QATLinear,
    _round_mxfp8_scaled_abs,
    flush_mxfp8_probe,
    mxfp8_probe_step_context,
    quantize_mxfp8_tensor,
    reset_mxfp8_probe,
)
from verl.utils.qat.mxfp8_rotation import MXFP8RotationConfig, apply_mxfp8_block_rotation


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 16, bias=False)
        self.lm_head = nn.Linear(32, 16, bias=False)


class _ProbeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.q_proj(x)


class _ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ProbeLayer(), _ProbeLayer()])

    def forward(self, x):
        return self.layers[0](x) + self.layers[1](x)


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.hidden_dim = 32
        self.intermediate_dim = 32
        self.gate_up_proj = nn.Parameter(torch.randn(2, 64, 32, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.randn(2, 32, 32, dtype=torch.bfloat16))
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(32, 2, bias=False, dtype=torch.bfloat16)
        self.mlp.experts = _PackedExperts()
        self.mlp.shared_expert = _SharedExpert()
        self.mlp.shared_expert_gate = nn.Linear(32, 1, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        hidden_states = x.reshape(-1, 32)
        router_logits = self.mlp.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, 1, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        routed = self.mlp.experts(hidden_states, selected_experts, routing_weights)
        shared = torch.sigmoid(self.mlp.shared_expert_gate(hidden_states)) * self.mlp.shared_expert(hidden_states)
        dense = self.dense(hidden_states)
        return (routed + shared + dense).reshape_as(x)


class _MoeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MoeLayer()])

    def forward(self, x):
        return self.layers[0](x)


@pytest.fixture(autouse=True)
def _reset_mxfp8_probe_state():
    reset_mxfp8_probe()
    yield
    reset_mxfp8_probe()


@pytest.mark.parametrize("mode", ["w8a16_mxfp8", "w8a8_mxfp8"])
def test_apply_qat_mxfp8_replaces_eligible_linear_layers(mode, caplog):
    caplog.set_level(logging.WARNING)
    model = _TinyModel()

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode=mode,
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="hash",
        ),
    )

    assert isinstance(model.proj, MXFP8QATLinear)
    assert model.proj.mode == QATMode(mode)
    assert model.proj.mxfp8_quant_backend == "torch"
    assert model.proj.mxfp8_rounding_mode == "hash"
    assert isinstance(model.lm_head, nn.Linear)
    assert any(
        "MXFP8 QAT applied to training model" in record.message
        and f"mode={mode}" in record.message
        and "quant_backend=torch" in record.message
        and "rounding_mode=hash" in record.message
        for record in caplog.records
    )


def test_mxfp8_block_rotation_preserves_linear_equivalence():
    cfg = MXFP8RotationConfig(enable=True, block_size=32, seed=11)
    x = torch.randn(3, 64, dtype=torch.float32)
    weight = torch.randn(5, 64, dtype=torch.float32)

    rotated_x = apply_mxfp8_block_rotation(x, cfg)
    rotated_w = apply_mxfp8_block_rotation(weight, cfg)

    baseline = F.linear(x, weight)
    rotated = F.linear(rotated_x, rotated_w)

    assert torch.allclose(rotated, baseline, atol=1e-5, rtol=1e-5)


def test_mxfp8_qat_probe_mode_aggregates_by_step_layer_index_and_type(tmp_path):
    model = _ProbeModel()
    output_path = tmp_path / "mxfp8_probe.jsonl"

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_probe_quant_error=True,
            mxfp8_probe_quant_error_output_path=str(output_path),
        ),
    )

    assert isinstance(model.layers[0].q_proj, MXFP8QATLinear)
    assert model.layers[0].q_proj.mxfp8_probe_quant_error is True
    assert model.layers[0].q_proj._mxfp8_layer_type == "q_proj"
    assert model.layers[0].q_proj._mxfp8_layer_index == 0
    assert model.layers[1].q_proj._mxfp8_layer_type == "q_proj"
    assert model.layers[1].q_proj._mxfp8_layer_index == 1

    with torch.no_grad():
        model.layers[0].q_proj.weight.copy_(torch.linspace(-2.0, 2.0, steps=16 * 32, dtype=torch.bfloat16).view(16, 32))
        model.layers[1].q_proj.weight.copy_(torch.linspace(-1.0, 3.0, steps=16 * 32, dtype=torch.bfloat16).view(16, 32))

    x = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.bfloat16).view(2, 32)
    baseline = F.linear(x, model.layers[0].q_proj.weight) + F.linear(x, model.layers[1].q_proj.weight)

    with torch.no_grad():
        with mxfp8_probe_step_context([17, 17]):
            out = model(x)
        with mxfp8_probe_step_context(17):
            out_alt = model(-x)
        with mxfp8_probe_step_context(None):
            out_null = model(x)

    assert torch.allclose(out, baseline)
    assert torch.allclose(
        out_alt, F.linear(-x, model.layers[0].q_proj.weight) + F.linear(-x, model.layers[1].q_proj.weight)
    )
    assert torch.allclose(out_null, baseline)
    flush_mxfp8_probe()
    assert output_path.exists()
    reset_mxfp8_probe()

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) == 4
    assert {
        (record["step"], record["error_type"], record["layer_index"], record["layer_type"]) for record in records
    } == {
        (17, "weight", 0, "q_proj"),
        (17, "activation", 0, "q_proj"),
        (17, "weight", 1, "q_proj"),
        (17, "activation", 1, "q_proj"),
    }
    for record in records:
        assert "layer_name" not in record
        assert record["layer_type"] == "q_proj"
        assert record["layer_index"] in {0, 1}
        assert record["error_metric"] == "relative_abs"


def test_mxfp8_qat_probe_mode_requires_output_path():
    with pytest.raises(ValueError, match="mxfp8_probe_quant_error_output_path"):
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_probe_quant_error=True,
        )


def test_apply_qat_w4a16_still_uses_nvfp4_qat_linear():
    model = _TinyModel()

    apply_qat(model, QATConfig(enable=True, mode="w4a16", group_size=16))

    assert isinstance(model.proj, QATLinear)


def test_apply_qat_mxfp8_requires_group_size_32():
    model = _TinyModel()

    with pytest.raises(ValueError, match="group_size=32"):
        apply_qat(model, QATConfig(enable=True, mode="w8a8_mxfp8", group_size=16))


def test_apply_qat_mxfp8_experts_respects_subswitch_and_ignore_patterns():
    model = _MoeModel()

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
        ),
    )

    assert isinstance(model.layers[0].dense, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.experts, _PackedExperts)
    assert isinstance(model.layers[0].mlp.gate, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert.gate_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert.up_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert.down_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert_gate, nn.Linear)

    model = _MoeModel()
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
            experts={"enable": True},
        ),
    )

    assert isinstance(model.layers[0].dense, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.experts, MXFP8QATExperts)
    assert isinstance(model.layers[0].mlp.gate, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert.gate_proj, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.shared_expert.up_proj, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.shared_expert.down_proj, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.shared_expert_gate, nn.Linear)

    model = _MoeModel()
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
            experts={"enable": True},
            ignore_patterns=["re:.*mlp.experts.gate_up_proj$", "re:.*shared_expert.up_proj$"],
        ),
    )

    assert isinstance(model.layers[0].mlp.experts, MXFP8QATExperts)
    assert model.layers[0].mlp.experts._quantize_gate_up_proj is False
    assert model.layers[0].mlp.experts._quantize_down_proj is True
    assert isinstance(model.layers[0].mlp.shared_expert.gate_proj, MXFP8QATLinear)
    assert isinstance(model.layers[0].mlp.shared_expert.up_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.shared_expert.down_proj, MXFP8QATLinear)


def test_mxfp8_qat_expert_wrapper_runs_forward_and_invalidate_all_scales():
    model = _MoeModel()
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
            experts={"enable": True},
        ),
    )

    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    out = model(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert model.layers[0].mlp.experts._last_gate_up_scale is not None
    assert model.layers[0].mlp.experts._last_down_scale is not None
    assert model.layers[0].mlp.experts._last_gate_up_input_scale is not None
    assert model.layers[0].mlp.experts._last_down_input_scale is not None

    invalidate_all_scales(model)

    assert model.layers[0].mlp.experts._last_gate_up_scale is None
    assert model.layers[0].mlp.experts._last_down_scale is None
    assert model.layers[0].mlp.experts._last_gate_up_input_scale is None
    assert model.layers[0].mlp.experts._last_down_input_scale is None


def test_mxfp8_qat_expert_npu_weight_cache_reuses_qdq_without_probe(monkeypatch):
    model = _MoeModel()
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a16_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
            experts={"enable": True},
        ),
    )

    call_count = 0
    original_quantize = quantize_mxfp8_tensor

    def counting_quantize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp8_experts.quantize_mxfp8_tensor", counting_quantize)

    experts = model.layers[0].mlp.experts
    experts.get_npu_sparse_block_weights(torch.bfloat16)
    assert call_count == 2

    experts.get_npu_sparse_block_weights(torch.bfloat16)
    assert call_count == 2

    invalidate_all_scales(model)

    experts.get_npu_sparse_block_weights(torch.bfloat16)
    assert call_count == 4


def test_mxfp8_qat_expert_probe_splits_gate_up_proj_into_gate_and_up(tmp_path):
    model = _MoeModel()
    output_path = tmp_path / "mxfp8_expert_probe.jsonl"
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="torch",
            mxfp8_rounding_mode="round",
            mxfp8_probe_quant_error=True,
            mxfp8_probe_quant_error_output_path=str(output_path),
            experts={"enable": True},
        ),
    )

    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        with mxfp8_probe_step_context(23):
            _ = model(x)
    flush_mxfp8_probe()
    reset_mxfp8_probe()

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert all(record["error_metric"] == "relative_abs" for record in records)
    layer_types = {(record["error_type"], record["layer_type"]) for record in records}
    assert ("weight", "gate_proj") in layer_types
    assert ("weight", "up_proj") in layer_types
    assert ("weight", "down_proj") in layer_types
    assert ("activation", "gate_proj") in layer_types
    assert ("activation", "up_proj") in layer_types
    assert ("activation", "down_proj") in layer_types
    assert ("weight", "gate_up_proj") not in layer_types
    assert ("activation", "gate_up_proj") not in layer_types


def test_mxfp8_qat_linear_rejects_unknown_quant_backend():
    with pytest.raises(ValueError, match="Unsupported MXFP8 quant backend"):
        MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="rotate")


def test_mxfp8_qat_linear_rejects_unknown_rounding_mode():
    with pytest.raises(ValueError, match="Unsupported MXFP8 rounding mode"):
        MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="torch", mxfp8_rounding_mode="nearest")


def test_mxfp8_qat_config_rejects_stochastic_rounding_with_npu_backend():
    with pytest.raises(ValueError, match="mxfp8_quant_backend='torch'"):
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            mxfp8_quant_backend="npu",
            mxfp8_rounding_mode="random",
        )


def test_round_mxfp8_scaled_abs_rint_uses_bankers_rounding():
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.49, 4.5, 4.51], dtype=torch.float32)

    rounded = _round_mxfp8_scaled_abs(values, "rint")

    assert torch.equal(rounded, torch.tensor([0.0, 2.0, 2.0, 4.0, 4.0, 4.0, 5.0], dtype=torch.float32))



def test_quantize_mxfp8_torch_defaults_to_rint_rounding_on_ties():
    tensor = torch.zeros((1, 32), dtype=torch.float32)
    tensor[0, 0] = 1.0
    tensor[0, 1] = 1.0625 / 256.0

    default_q, default_scale = quantize_mxfp8_tensor(tensor, quant_backend="torch")
    rint_q, rint_scale = quantize_mxfp8_tensor(tensor, quant_backend="torch", rounding_mode="rint")
    round_q, round_scale = quantize_mxfp8_tensor(tensor, quant_backend="torch", rounding_mode="round")

    assert torch.equal(default_q, rint_q)
    assert torch.equal(default_scale, rint_scale)
    assert torch.equal(default_scale, round_scale)
    assert default_q.to(torch.float32)[0, 1].item() == 1.0
    assert round_q.to(torch.float32)[0, 1].item() == 1.125


def test_w8a16_mxfp8_qat_linear_preserves_high_precision_matmul_toggle():
    linear = MXFP8QATLinear(
        32, 8, bias=True, mode=QATMode.W8A16_MXFP8, mxfp8_quant_backend="torch", dtype=torch.bfloat16
    )
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-3.25, 3.25, steps=8 * 32, dtype=torch.bfloat16).view(8, 32))
        linear.bias.zero_()

    x = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.bfloat16).view(2, 32)
    expected = F.linear(x, linear.weight, linear.bias)

    linear.fake_quant_enabled = False
    disabled_out = linear(x)
    assert torch.allclose(disabled_out, expected)

    linear.fake_quant_enabled = True
    enabled_out = linear(x)
    assert enabled_out.shape == expected.shape
    assert enabled_out.dtype == expected.dtype
    assert linear._last_weight_scale is not None
    assert linear._last_input_scale is None
    assert not torch.allclose(enabled_out, expected)


def test_w8a8_mxfp8_qat_linear_fake_quantizes_activation():
    linear = MXFP8QATLinear(
        32, 8, bias=True, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="torch", dtype=torch.bfloat16
    )
    w8a16_linear = MXFP8QATLinear(
        32, 8, bias=True, mode=QATMode.W8A16_MXFP8, mxfp8_quant_backend="torch", dtype=torch.bfloat16
    )
    with torch.no_grad():
        weight = torch.linspace(-2.0, 2.0, steps=8 * 32, dtype=torch.bfloat16).view(8, 32)
        bias = torch.linspace(-0.25, 0.25, steps=8, dtype=torch.bfloat16)
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
        w8a16_linear.weight.copy_(weight)
        w8a16_linear.bias.copy_(bias)

    x = torch.linspace(-1.25, 1.75, steps=64, dtype=torch.bfloat16).view(2, 32)

    w8a8_out = linear(x)
    w8a16_out = w8a16_linear(x)

    assert w8a8_out.shape == w8a16_out.shape
    assert w8a8_out.dtype == w8a16_out.dtype
    assert linear._last_weight_scale is not None
    assert linear._last_input_scale is not None
    assert not torch.allclose(w8a8_out, w8a16_out)


def test_w8a8_mxfp8_qat_linear_supports_3d_activation():
    linear = MXFP8QATLinear(
        32, 8, bias=False, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="torch", dtype=torch.bfloat16
    )
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-2.0, 2.0, steps=8 * 32, dtype=torch.bfloat16).view(8, 32))

    x = torch.linspace(-1.25, 1.75, steps=2 * 3 * 32, dtype=torch.bfloat16).view(2, 3, 32)

    out = linear(x)

    assert out.shape == (2, 3, 8)
    assert out.dtype == x.dtype
    assert linear._last_input_scale is not None
    assert linear._last_input_scale.shape == (2, 3, 1)


def test_invalidate_all_scales_handles_nvfp4_and_mxfp8_modules():
    model = nn.Module()
    model.nvfp4 = QATLinear(32, 8, mode=QATMode.W4A16, group_size=16)
    model.mxfp8 = MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="torch")

    model.nvfp4._weight_blockwise_scale = torch.ones(1)
    model.nvfp4._weight_global_scale = torch.ones(1)
    model.nvfp4._cached_weight_amax = torch.ones(1)
    model.mxfp8._last_weight_scale = torch.ones((8, 1), dtype=torch.uint8)
    model.mxfp8._last_input_scale = torch.ones((1, 1), dtype=torch.uint8)
    model.mxfp8._cached_weight_qdq = torch.ones((8, 32), dtype=torch.bfloat16)

    invalidate_all_scales(model)

    assert model.nvfp4._weight_blockwise_scale is None
    assert model.nvfp4._weight_global_scale is None
    assert model.nvfp4._cached_weight_amax is None
    assert model.mxfp8._last_weight_scale is None
    assert model.mxfp8._last_input_scale is None
    assert model.mxfp8._cached_weight_qdq is None
