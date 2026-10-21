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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.core import QATConfig, apply_qat, invalidate_all_scales, normalize_mxfp8_fallback_layers
from verl.utils.qat.linear import QATLinear, QATMode
from verl.utils.qat.mxfp8_experts import MXFP8QATExperts
from verl.utils.qat.mxfp8_linear import (
    MXFP8QATLinear,
    _dequantize_mxfp8,
    _round_mxfp8_scaled_abs,
    dequantize_mxfp4_tensor,
    flush_mxfp8_probe,
    linear_with_gradient_rotation,
    mxfp8_probe_step_context,
    normalize_mxfp8_fake_quant_targets,
    quantize_mxfp4_tensor,
    quantize_mxfp8_tensor,
    reset_mxfp8_probe,
)
from verl.utils.qat.mxfp8_rotation import (
    MXFP8RotationConfig,
    apply_mxfp8_block_rotation,
    normalize_mxfp8_rotation_targets,
)


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
            quant_backend="torch",
            rounding_mode="hash",
        ),
    )

    assert isinstance(model.proj, MXFP8QATLinear)
    assert model.proj.mode == QATMode(mode)
    assert model.proj.quant_backend == "torch"
    assert model.proj.rounding_mode == "hash"
    assert isinstance(model.lm_head, nn.Linear)
    assert any(
        "MXFP QAT applied to training model" in record.message
        and f"mode={mode}" in record.message
        and "quant_backend=torch" in record.message
        and "rounding_mode=hash" in record.message
        for record in caplog.records
    )


def test_apply_qat_mxfp8_fallback_layers_skip_complete_transformer_layers():
    model = _ProbeModel()

    assert normalize_mxfp8_fallback_layers("0-3,37-48") == ((0, 3), (37, 48))
    assert normalize_mxfp8_fallback_layers([[3, 5], [0, 1], [5, 6]]) == ((0, 1), (3, 6))

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="torch",
            fallback_layers="0",
        ),
    )

    assert isinstance(model.layers[0].q_proj, nn.Linear)
    assert isinstance(model.layers[1].q_proj, MXFP8QATLinear)


def test_mxfp8_block_rotation_preserves_linear_equivalence():
    cfg = MXFP8RotationConfig(enable=True, block_size=32, seed=11)
    x = torch.randn(3, 64, dtype=torch.float32)
    weight = torch.randn(5, 64, dtype=torch.float32)

    rotated_x = apply_mxfp8_block_rotation(x, cfg)
    rotated_w = apply_mxfp8_block_rotation(weight, cfg)

    baseline = F.linear(x, weight)
    rotated = F.linear(rotated_x, rotated_w)

    assert torch.allclose(rotated, baseline, atol=1e-5, rtol=1e-5)


def test_mxfp8_rotation_targets_normalize_and_default():
    assert normalize_mxfp8_rotation_targets(None) == ("fprop",)
    assert normalize_mxfp8_rotation_targets("Dgrad,Wgrad") == ("dgrad", "wgrad")
    assert normalize_mxfp8_rotation_targets('["Wgrad", "Fprop"]') == ("fprop", "wgrad")

    config = QATConfig(
        enable=True,
        mode="w8a8_mxfp8",
        group_size=32,
        quant_backend="torch",
        rotation_enable=True,
        rotation_targets=["Wgrad", "Dgrad"],
    )
    assert config.rotation_targets == ["dgrad", "wgrad"]


def test_mxfp8_fake_quant_targets_normalize_and_default():
    assert normalize_mxfp8_fake_quant_targets(None) == ("fprop",)
    assert normalize_mxfp8_fake_quant_targets("Dgrad,Wgrad") == ("dgrad", "wgrad")
    assert normalize_mxfp8_fake_quant_targets('["Wgrad", "Fprop"]') == ("fprop", "wgrad")

    config = QATConfig(
        enable=True,
        mode="w8a8_mxfp8",
        group_size=32,
        quant_backend="torch",
        fake_quant_targets=["Wgrad", "Dgrad"],
    )
    assert config.fake_quant_targets == ["dgrad", "wgrad"]


def test_linear_with_gradient_rotation_matches_standard_backward_with_padding():
    cfg = MXFP8RotationConfig(enable=True, block_size=32, seed=17, targets=("dgrad", "wgrad"))
    x = torch.randn(4, 37, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(23, 37, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(4, 23, dtype=torch.float32)

    rotated_out = linear_with_gradient_rotation(x, weight, None, cfg)
    rotated_out.backward(grad_output)
    rotated_x_grad = x.grad.detach().clone()
    rotated_weight_grad = weight.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    ref_out = F.linear(x_ref, weight_ref)
    ref_out.backward(grad_output)

    assert torch.allclose(rotated_out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotated_x_grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(rotated_weight_grad, weight_ref.grad, atol=1e-5, rtol=1e-5)


def test_dgrad_rotation_does_not_fake_quantize_when_fake_quant_target_excludes_dgrad(monkeypatch):
    cfg = MXFP8RotationConfig(enable=True, block_size=32, seed=17, targets=("dgrad",))
    x = torch.randn(4, 37, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(23, 37, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(4, 23, dtype=torch.float32)

    def fail_quantize(*args, **kwargs):
        raise AssertionError("Dgrad rotation should not call MXFP8 quantization when Dgrad is not a fake-quant target")

    monkeypatch.setattr("verl.utils.qat.mxfp8_linear.quantize_mxfp8_tensor", fail_quantize)
    rotated_out = linear_with_gradient_rotation(
        x,
        weight,
        None,
        cfg,
        fake_quant_targets=("fprop",),
        quant_backend="torch",
    )
    rotated_out.backward(grad_output)
    rotated_x_grad = x.grad.detach().clone()
    rotated_weight_grad = weight.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    ref_out = F.linear(x_ref, weight_ref)
    ref_out.backward(grad_output)

    assert torch.allclose(rotated_out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotated_x_grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(rotated_weight_grad, weight_ref.grad, atol=1e-5, rtol=1e-5)


def test_dgrad_fake_quant_does_not_require_rotation(monkeypatch):
    cfg = MXFP8RotationConfig(enable=False, block_size=32, seed=17, targets=("dgrad",))
    x = torch.randn(4, 37, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(23, 37, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(4, 23, dtype=torch.float32)
    quantized_shapes = []

    def fake_quantize(tensor, quant_backend="torch", rounding_mode="rint"):
        quantized_shapes.append(tuple(tensor.shape))
        scale = torch.full(
            (tensor.shape[0], tensor.shape[-1] // 32),
            127,
            dtype=torch.uint8,
            device=tensor.device,
        )
        return tensor, scale

    monkeypatch.setattr("verl.utils.qat.mxfp8_linear.quantize_mxfp8_tensor", fake_quantize)
    out = linear_with_gradient_rotation(
        x,
        weight,
        None,
        cfg,
        fake_quant_targets=("dgrad",),
        quant_backend="torch",
    )
    out.backward(grad_output)

    assert quantized_shapes == [(4, 32), (37, 32)]


def test_mxfp8_qat_linear_does_not_requantize_fprop_operands_in_backward(monkeypatch):
    linear = MXFP8QATLinear(
        32,
        8,
        bias=False,
        mode=QATMode.W8A8_MXFP8,
        quant_backend="torch",
        fake_quant_targets=["Fprop", "Dgrad", "Wgrad"],
        dtype=torch.float32,
    )
    x = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    quantized_shapes = []

    def fake_quantize(tensor, quant_backend="torch", rounding_mode="rint"):
        quantized_shapes.append(tuple(tensor.shape))
        scale = torch.full(
            (tensor.shape[0], tensor.shape[-1] // 32),
            127,
            dtype=torch.uint8,
            device=tensor.device,
        )
        return tensor, scale

    monkeypatch.setattr("verl.utils.qat.mxfp8_linear.quantize_mxfp8_tensor", fake_quantize)

    linear.weight_quantizer = fake_quantize
    linear.activation_quantizer = fake_quantize
    linear(x).sum().backward()

    assert quantized_shapes == [(8, 32), (4, 32), (4, 32), (8, 32)]


def test_mxfp8_qat_linear_uses_backward_reduction_axis_operands_when_enabled(monkeypatch):
    linear = MXFP8QATLinear(
        32,
        8,
        bias=False,
        mode=QATMode.W8A8_MXFP8,
        quant_backend="torch",
        fake_quant_targets=["Fprop", "Dgrad", "Wgrad"],
        backward_reduction_axis_fake_quant=True,
        dtype=torch.float32,
    )
    x = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    quantized_shapes = []

    def fake_quantize(tensor, quant_backend="torch", rounding_mode="rint"):
        quantized_shapes.append(tuple(tensor.shape))
        scale = torch.full(
            (tensor.shape[0], tensor.shape[-1] // 32),
            127,
            dtype=torch.uint8,
            device=tensor.device,
        )
        return tensor, scale

    monkeypatch.setattr("verl.utils.qat.mxfp8_linear.quantize_mxfp8_tensor", fake_quantize)

    linear.weight_quantizer = fake_quantize
    linear.activation_quantizer = fake_quantize
    linear(x).sum().backward()

    assert quantized_shapes == [(8, 32), (4, 32), (32, 32), (32, 32), (4, 32), (8, 32)]


def test_apply_qat_mxfp8_rotation_targets_propagate_to_wrappers():
    model = _TinyModel()

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="torch",
            rounding_mode="round",
            rotation_enable=True,
            rotation_targets=["Dgrad", "Wgrad"],
            fake_quant_targets=["Fprop", "Dgrad"],
            backward_reduction_axis_fake_quant=True,
        ),
    )

    assert isinstance(model.proj, MXFP8QATLinear)
    assert model.proj.mxfp8_rotation_config.targets == ("dgrad", "wgrad")
    assert model.proj.fake_quant_targets == ("fprop", "dgrad")
    assert model.proj.backward_reduction_axis_fake_quant is True


def test_mxfp8_qat_probe_mode_aggregates_by_step_layer_index_and_type(tmp_path):
    model = _ProbeModel()
    output_path = tmp_path / "mxfp8_probe.jsonl"

    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="torch",
            probe_quant_error=True,
            probe_quant_error_output_path=str(output_path),
        ),
    )

    assert isinstance(model.layers[0].q_proj, MXFP8QATLinear)
    assert model.layers[0].q_proj.probe_quant_error is True
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
        assert record["error_metric"] == "mae"


def test_mxfp8_qat_probe_mode_requires_output_path():
    with pytest.raises(ValueError, match="probe_quant_error_output_path"):
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="torch",
            probe_quant_error=True,
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
            quant_backend="torch",
            rounding_mode="round",
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
            quant_backend="torch",
            rounding_mode="round",
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
            quant_backend="torch",
            rounding_mode="round",
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
            quant_backend="torch",
            rounding_mode="round",
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


@pytest.mark.parametrize("mode", ["w8a16_mxfp8", "w8a8_mxfp8", "w4a8_mxfp4", "w4a4_mxfp4"])
@pytest.mark.parametrize("backward_reduction_axis_fake_quant", [False, True])
@pytest.mark.parametrize("rotation_targets", [(), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("fake_quant_targets", [("fprop", "dgrad", "wgrad"), ("dgrad", "wgrad")])
def test_mxfp_qat_expert_grouped_backward_matches_python_path(
    mode, backward_reduction_axis_fake_quant, rotation_targets, fake_quant_targets
):
    from types import SimpleNamespace

    torch.manual_seed(7)
    expert_count = 4
    hidden_size = 32
    intermediate_size = 32
    # Top-2 routing with uneven expert segments and an empty expert.
    expert_ids = torch.cat([torch.tensor([[0, 2]]).repeat(33, 1), torch.tensor([[0, 3]]).repeat(35, 1)])
    input_tokens = expert_ids.shape[0]
    # Keep the CPU GMM oracle independent of shape-dependent FP32 accumulation
    # near FP4 rounding midpoints. Real NPU kernels still run in the model dtype.
    source = torch.randn(input_tokens, hidden_size, dtype=torch.float64)
    expert_ids = expert_ids[torch.randperm(input_tokens)]
    routing_weights = torch.softmax(torch.randn(input_tokens, 2, dtype=source.dtype), dim=-1)
    packed_gate_up = torch.randn(expert_count, 2 * intermediate_size, hidden_size, dtype=source.dtype)
    packed_down = torch.randn(expert_count, hidden_size, intermediate_size, dtype=source.dtype)

    def make_experts():
        return MXFP8QATExperts(
            num_experts=expert_count,
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_size,
            act_fn=F.silu,
            gate_up_proj=packed_gate_up,
            down_proj=packed_down,
            mode=mode,
            quant_backend="torch",
            rounding_mode="rint",
            fake_quant_targets=fake_quant_targets,
            backward_reduction_axis_fake_quant=backward_reduction_axis_fake_quant,
            rotation_enable=bool(rotation_targets),
            rotation_targets=rotation_targets or ("fprop",),
        )

    def grouped_matmul(inputs, weights, *, group_list, split_item, group_type, **kwargs):
        groups = group_list.to(torch.int64).tolist()
        if split_item == 2:
            pieces = []
            offset = 0
            for count, weight in zip(groups, weights[0], strict=True):
                pieces.append(inputs[0][offset : offset + count] @ weight)
                offset += count
            return [torch.cat(pieces, dim=0)]

        assert split_item == 3 and group_type == 2
        pieces = []
        offset = 0
        for count in groups:
            pieces.append(inputs[0][:, offset : offset + count] @ weights[0][offset : offset + count])
            offset += count
        return [torch.stack(pieces)]

    def token_permute(hidden_states, selected_experts):
        expert_ids = selected_experts.reshape(-1).to(torch.int64)
        token_ids = torch.arange(hidden_states.shape[0]).repeat_interleave(selected_experts.shape[1])
        order = torch.argsort(expert_ids, stable=True)
        return hidden_states[token_ids[order]], order

    def token_unpermute(hidden_states, row_ids_map, *, probs):
        routed = hidden_states.new_zeros((row_ids_map.numel(), hidden_states.shape[-1]))
        routed = routed.index_copy(0, row_ids_map.to(torch.int64), hidden_states)
        routed = routed.view(probs.shape[0], probs.shape[1], -1)
        return (routed * probs.unsqueeze(-1)).sum(dim=1)

    torch_npu = SimpleNamespace(
        npu_grouped_matmul=grouped_matmul,
        npu_moe_token_permute=token_permute,
        npu_moe_token_unpermute=token_unpermute,
    )
    python_experts = make_experts()
    npu_experts = make_experts()
    python_input = source.clone().requires_grad_()
    npu_input = source.clone().requires_grad_()
    python_probs = routing_weights.clone().requires_grad_()
    npu_probs = routing_weights.clone().requires_grad_()

    expected = python_experts(python_input, expert_ids, python_probs)
    actual = npu_experts.forward_npu(npu_input, expert_ids, npu_probs, torch_npu)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(npu_input.grad, python_input.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(npu_probs.grad, python_probs.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(npu_experts.gate_up_proj.grad, python_experts.gate_up_proj.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(npu_experts.down_proj.grad, python_experts.down_proj.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("mode", ["w8a8_mxfp8", "w4a8_mxfp4", "w4a4_mxfp4"])
def test_mxfp_qat_python_npu_backend_skips_npu_ops(mode):
    from types import SimpleNamespace

    torch.manual_seed(19)
    model = _MoeModel()
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode=mode,
            group_size=32,
            quant_backend="torch",
            experts={"enable": True, "npu_backend": "python"},
            fake_quant_targets=["Fprop", "Dgrad", "Wgrad"],
            backward_reduction_axis_fake_quant=True,
            rotation_enable=True,
            rotation_targets=["Fprop", "Dgrad", "Wgrad"],
        ),
    )
    experts = model.layers[0].mlp.experts
    assert experts.npu_backend == "python"
    source = torch.randn(5, 32, dtype=torch.bfloat16)
    # Exercise top-2 routing and router-weight gradients, too.
    selected = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    probs = torch.rand(5, 2, dtype=torch.bfloat16)
    expected_input = source.clone().requires_grad_()
    expected_probs = probs.clone().requires_grad_()
    expected = experts(expected_input, selected, expected_probs)
    expected.float().sum().backward()
    weight_grads = [experts.gate_up_proj.grad.clone(), experts.down_proj.grad.clone()]
    experts.zero_grad(set_to_none=True)

    actual_input = source.clone().requires_grad_()
    actual_probs = probs.clone().requires_grad_()
    # An empty namespace fails if this backend touches any NPU GMM/permutation op.
    actual = experts.forward_npu(actual_input, selected, actual_probs, SimpleNamespace())
    actual.float().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(actual_probs.grad, expected_probs.grad)
    torch.testing.assert_close(experts.gate_up_proj.grad, weight_grads[0])
    torch.testing.assert_close(experts.down_proj.grad, weight_grads[1])


def test_mxfp8_qat_expert_probe_splits_gate_up_proj_into_gate_and_up(tmp_path):
    model = _MoeModel()
    output_path = tmp_path / "mxfp8_expert_probe.jsonl"
    apply_qat(
        model,
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="torch",
            rounding_mode="round",
            probe_quant_error=True,
            probe_quant_error_output_path=str(output_path),
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
        MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, quant_backend="rotate")


def test_mxfp8_qat_linear_rejects_unknown_rounding_mode():
    with pytest.raises(ValueError, match="Unsupported MXFP8 rounding mode"):
        MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, quant_backend="torch", rounding_mode="nearest")


def test_mxfp8_qat_config_rejects_stochastic_rounding_with_npu_backend():
    with pytest.raises(ValueError, match="quant_backend='torch'"):
        QATConfig(
            enable=True,
            mode="w8a8_mxfp8",
            group_size=32,
            quant_backend="npu",
            rounding_mode="random",
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
    linear = MXFP8QATLinear(32, 8, bias=True, mode=QATMode.W8A16_MXFP8, quant_backend="torch", dtype=torch.bfloat16)
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


def test_mxfp8_qat_linear_skips_fprop_fake_quant_when_target_excludes_fprop():
    linear = MXFP8QATLinear(
        32,
        8,
        bias=True,
        mode=QATMode.W8A8_MXFP8,
        quant_backend="torch",
        fake_quant_targets=["Dgrad"],
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-3.25, 3.25, steps=8 * 32, dtype=torch.bfloat16).view(8, 32))
        linear.bias.zero_()

    x = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.bfloat16).view(2, 32)
    out = linear(x)

    assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))
    assert linear._last_weight_scale is None
    assert linear._last_input_scale is None


def test_w8a8_mxfp8_qat_linear_fake_quantizes_activation():
    linear = MXFP8QATLinear(32, 8, bias=True, mode=QATMode.W8A8_MXFP8, quant_backend="torch", dtype=torch.bfloat16)
    w8a16_linear = MXFP8QATLinear(
        32, 8, bias=True, mode=QATMode.W8A16_MXFP8, quant_backend="torch", dtype=torch.bfloat16
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
    linear = MXFP8QATLinear(32, 8, bias=False, mode=QATMode.W8A8_MXFP8, quant_backend="torch", dtype=torch.bfloat16)
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
    model.mxfp8 = MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, quant_backend="torch")

    model.nvfp4._weight_blockwise_scale = torch.ones(1)
    model.nvfp4._weight_global_scale = torch.ones(1)
    model.nvfp4._cached_weight_amax = torch.ones(1)
    model.mxfp8._last_weight_scale = torch.ones((8, 1), dtype=torch.uint8)
    model.mxfp8._last_input_scale = torch.ones((1, 1), dtype=torch.uint8)

    invalidate_all_scales(model)

    assert model.nvfp4._weight_blockwise_scale is None
    assert model.nvfp4._weight_global_scale is None
    assert model.nvfp4._cached_weight_amax is None
    assert model.mxfp8._last_weight_scale is None
    assert model.mxfp8._last_input_scale is None


@pytest.mark.parametrize(
    "rounding,expected",
    [
        ("rint", [0, 1, 1, 2, 2, 4, 4, 6]),
        ("round", [0.5, 1, 1.5, 2, 3, 4, 6, 6]),
    ],
)
def test_mxfp4_e2m1_midpoints_and_saturation(rounding, expected):
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7])
    tensor = torch.cat([values, -values, values, -values]).reshape(1, 32)
    payload, scale = quantize_mxfp4_tensor(tensor, rounding_mode=rounding)
    expected = torch.tensor(expected, dtype=torch.float32)
    expected = torch.cat([expected, -expected, expected, -expected]).reshape_as(tensor)
    assert torch.equal(scale, torch.tensor([[127.0]]))
    assert torch.equal(payload, expected)
    assert torch.equal(dequantize_mxfp4_tensor(payload, scale, tensor.dtype), expected)


def test_mxfp4_zero_tiny_nonfinite_and_independent_scales():
    tensor = torch.cat([torch.zeros(32), torch.full((32,), 2.0**-140), torch.full((32,), 24.0)]).reshape(1, 96)
    payload, scale = quantize_mxfp4_tensor(tensor)
    assert torch.equal(scale, torch.tensor([[0.0, 0.0, 129.0]]))
    assert torch.equal(payload[:, :64], torch.zeros(1, 64))
    assert torch.equal(dequantize_mxfp4_tensor(payload, scale, tensor.dtype)[:, 64:], tensor[:, 64:])
    for value in [float("nan"), float("inf")]:
        tensor[0, 0] = value
        payload, scale = quantize_mxfp4_tensor(tensor)
        assert scale[0, 0] == 255
        assert torch.isnan(dequantize_mxfp4_tensor(payload, scale, tensor.dtype)[:, :32]).all()


def test_mxfp4_handles_empty_token_rows():
    tensor = torch.empty(0, 96)
    payload, scale = quantize_mxfp4_tensor(tensor)
    assert payload.shape == (0, 96) and scale.shape == (0, 3)
    assert dequantize_mxfp4_tensor(payload, scale, tensor.dtype).shape == tensor.shape


@pytest.mark.parametrize("rounding", ["random", "hash"])
def test_mxfp4_stochastic_rounding_is_reproducible_and_uses_adjacent_levels(rounding):
    tensor = torch.full((512, 32), 0.25)
    tensor[:, -1] = 6
    torch.manual_seed(17)
    payload, scale = quantize_mxfp4_tensor(tensor, rounding_mode=rounding)
    torch.manual_seed(17)
    again, again_scale = quantize_mxfp4_tensor(tensor, rounding_mode=rounding)
    assert torch.equal(payload, again)
    assert torch.equal(scale, again_scale)
    assert ((payload[:, :-1] == 0) | (payload[:, :-1] == 0.5)).all()
    if rounding == "random":
        assert abs(payload[:, :-1].mean().item() - 0.25) < 0.01


@pytest.mark.parametrize("shape", [(32,), (2, 31), (2, 3, 32)])
def test_mxfp4_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        quantize_mxfp4_tensor(torch.zeros(shape))


@pytest.mark.parametrize("mode", ["w4a8_mxfp4", "w4a4_mxfp4"])
@pytest.mark.parametrize("rotation", [False, True])
def test_mxfp4_dense_forward_uses_expected_formats_and_ste(mode, rotation):
    torch.manual_seed(71)
    linear = MXFP8QATLinear(32, 9, mode=mode, quant_backend="torch", rotation_enable=rotation)
    x = torch.randn(2, 3, 32, requires_grad=True)
    weight = linear._rotate_tensor(linear.weight)
    inputs = linear._rotate_tensor(x)
    wq, ws = quantize_mxfp4_tensor(weight)
    activation_quantizer = quantize_mxfp4_tensor if mode == "w4a4_mxfp4" else quantize_mxfp8_tensor
    aq, asc = activation_quantizer(inputs.reshape(-1, 32))
    expected = F.linear(
        _dequantize_mxfp8(aq, asc, x.dtype).reshape_as(x), dequantize_mxfp4_tensor(wq, ws, x.dtype), linear.bias
    )
    result = linear(x)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(linear.weight.grad).all()


@pytest.mark.parametrize("mode", ["w4a8_mxfp4", "w4a4_mxfp4"])
def test_mxfp4_backward_reduction_axis_preserves_weight_activation_formats(mode):
    linear = MXFP8QATLinear(
        32,
        9,
        mode=mode,
        quant_backend="torch",
        rotation_enable=True,
        rotation_targets=["Fprop", "Dgrad", "Wgrad"],
        fake_quant_targets=["Fprop", "Dgrad", "Wgrad"],
        backward_reduction_axis_fake_quant=True,
    )
    calls = []
    weight_quantizer, activation_quantizer = linear.weight_quantizer, linear.activation_quantizer

    def weights(tensor, **kwargs):
        calls.append(("weight", tuple(tensor.shape)))
        return weight_quantizer(tensor, **kwargs)

    def activations(tensor, **kwargs):
        calls.append(("activation", tuple(tensor.shape)))
        return activation_quantizer(tensor, **kwargs)

    linear.weight_quantizer, linear.activation_quantizer = weights, activations
    x = torch.randn(3, 32, requires_grad=True)
    linear(x).sum().backward()
    assert calls == [
        ("weight", (9, 32)),
        ("activation", (3, 32)),
        ("weight", (32, 32)),
        ("activation", (32, 32)),
        ("activation", (3, 32)),
        ("activation", (9, 32)),
    ]
    assert torch.isfinite(x.grad).all() and torch.isfinite(linear.weight.grad).all()


@pytest.mark.parametrize("mode", ["w4a8_mxfp4", "w4a4_mxfp4"])
def test_mxfp4_apply_qat_experts_fallback_and_probe(mode, tmp_path):
    model = _MoeModel()
    model.layers.append(_MoeLayer())
    output = tmp_path / "mxfp4_probe.jsonl"
    config = QATConfig(
        enable=True,
        mode=mode,
        group_size=32,
        quant_backend="torch",
        experts={"enable": True},
        fallback_layers="1",
        rotation_enable=True,
        probe_quant_error=True,
        probe_quant_error_output_path=str(output),
    )
    apply_qat(model, config)
    assert isinstance(model.layers[0].mlp.experts, MXFP8QATExperts)
    assert isinstance(model.layers[1].mlp.experts, _PackedExperts)
    expert = model.layers[0].mlp.experts
    x = torch.randn(3, 32, dtype=torch.bfloat16, requires_grad=True)
    with mxfp8_probe_step_context(1):
        result = expert(x, torch.tensor([[0], [1], [0]]), torch.ones(3, 1, dtype=x.dtype))
    result.float().sum().backward()
    assert torch.isfinite(expert.gate_up_proj.grad).all()
    flush_mxfp8_probe()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records and all(record["mode"] == mode for record in records)
    assert {record["error_type"] for record in records} == {"weight", "activation"}
    reset_mxfp8_probe()
