from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear
from verl.utils.qat.mxfp_quant import (
    MXFPMode,
    dequantize_mxfp4,
    dequantize_mxfp8,
    fake_quantize,
    normalize_mxfp4_scale_alg,
    normalize_mxfp_mode,
    quantize_mxfp4,
    quantize_mxfp8,
)
from verl.utils.qat.mxfp_rotation import RotationConfig, _matrix, rotate_blocks


@pytest.mark.parametrize("mode", ["mxfp_w8a8", "w8a8_mxfp8", "mxfp_w4a8", "mxfp_w4a4"])
def test_mode_names_normalize(mode):
    assert normalize_mxfp_mode(mode) in {MXFPMode.W8A8, MXFPMode.W4A8, MXFPMode.W4A4}


@pytest.mark.parametrize(
    ("scale_alg", "expected"),
    [
        ("standard", "standard"),
        ("ocp", "standard"),
        (0, "standard"),
        ("tfs", "tfs"),
        (1, "tfs"),
        ("cx", "cx"),
        (2, "cx"),
    ],
)
def test_mxfp4_scale_algorithm_names_normalize(scale_alg, expected):
    assert normalize_mxfp4_scale_alg(scale_alg) == expected


@pytest.mark.parametrize(
    ("scale_alg", "value", "expected_scale", "expected_decoded"),
    [
        ("standard", 7.0, 127.0, 6.0),
        ("tfs", 7.0, 128.0, 8.0),
        ("cx", 7.0, 127.0, 6.0),
        ("cx", 8.0, 128.0, 8.0),
        ("cx", 15.0, 129.0, 16.0),
    ],
)
def test_mxfp4_scale_algorithms_use_expected_block_scale(scale_alg, value, expected_scale, expected_decoded):
    x = torch.full((1, 32), value)
    payload, scale = quantize_mxfp4(x, scale_alg=scale_alg)
    assert scale.item() == expected_scale
    decoded = dequantize_mxfp4(payload, scale, x.dtype, scale_alg=scale_alg)
    assert torch.equal(decoded, torch.full_like(x, expected_decoded))


@pytest.mark.parametrize(
    "quantize,dequantize", [(quantize_mxfp8, dequantize_mxfp8), (quantize_mxfp4, dequantize_mxfp4)]
)
def test_mxfp_qdq_preserves_shape_and_is_finite(quantize, dequantize):
    x = torch.linspace(-2, 2, 96, dtype=torch.float32).reshape(3, 32)
    payload, scale = quantize(x)
    y = dequantize(payload, scale, x.dtype)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(y).item() > 0


def test_mxfp8_zero_block_stays_zero():
    x = torch.zeros((2, 64), dtype=torch.float32)
    payload, scale = quantize_mxfp8(x)
    assert torch.count_nonzero(dequantize_mxfp8(payload, scale, x.dtype)).item() == 0


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
def test_fake_quant_pads_and_restores_non_aligned_dimension(fmt):
    x = torch.randn((2, 37), dtype=torch.float32)
    y = fake_quantize(x, fmt)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mxfp4_mbs_matches_macroblock_qdq_with_inverse_correction():
    x = torch.linspace(-3.8, 4.0, 128, dtype=torch.float32).reshape(1, 128).to(torch.bfloat16)
    expected = (fake_quantize(x.to(torch.float32) * 1.5, "mxfp4") / 1.5).to(x.dtype)

    actual = fake_quantize(x, "mxfp4", mxfp4_mbs_enable=True)

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, fake_quantize(x, "mxfp4"))


def test_mxfp4_mbs_handles_partial_final_macroblock():
    x = torch.linspace(-2.0, 2.0, 160, dtype=torch.float32).reshape(2, 80)
    actual = fake_quantize(x, "mxfp4", mxfp4_mbs_enable=True)

    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


def test_rotation_is_seeded_and_orthogonal():
    config = RotationConfig(enable=True, targets=("fprop",), block_size=8, seed=17)
    matrix_a = _matrix(8, 17, torch.device("cpu"), torch.float32)
    matrix_b = _matrix(8, 17, torch.device("cpu"), torch.float32)
    assert torch.equal(matrix_a, matrix_b)
    assert torch.allclose(matrix_a @ matrix_a.T, torch.eye(8), atol=1e-6)
    x = torch.randn((3, 19))
    assert rotate_blocks(x, config).shape == x.shape


def test_linear_replacement_reuses_parameters_and_has_identity_without_transforms():
    linear = torch.nn.Linear(32, 64)
    config = MXFPQATConfig(mode="mxfp_w8a8", quant_backend="torch", fake_quant_targets=(), rotation_enable=False)
    qat = MXFPQATLinear.from_linear(linear, config)
    assert qat.weight is linear.weight
    assert qat.bias is linear.bias
    x = torch.randn((3, 32))
    assert torch.equal(qat(x), linear(x))


def test_mxfp4_mbs_config_rejects_unsupported_modes_and_scale_algorithms():
    config = MXFPQATConfig(mode="mxfp_w4a8", quant_backend="torch", mxfp4_mbs_enable=True)
    assert config.mxfp4_mbs_enable
    with pytest.raises(ValueError, match="requires an MXFP4 QAT mode"):
        MXFPQATConfig(mode="mxfp_w8a8", quant_backend="torch", mxfp4_mbs_enable=True)
    with pytest.raises(ValueError, match="mxfp4_scale_alg='standard'"):
        MXFPQATConfig(mode="mxfp_w4a8", quant_backend="torch", mxfp4_scale_alg="tfs", mxfp4_mbs_enable=True)


def test_mxfp4_mbs_linear_forward_and_backward_targets():
    linear = torch.nn.Linear(128, 128)
    config = MXFPQATConfig(
        mode="mxfp_w4a4",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        mxfp4_mbs_enable=True,
    )
    qat = MXFPQATLinear.from_linear(linear, config)
    x = torch.randn((4, 128), requires_grad=True)

    output = qat(x)
    output.square().mean().backward()

    assert output.shape == (4, 128)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert qat.weight.grad is not None and torch.isfinite(qat.weight.grad).all()


@pytest.mark.parametrize("targets", [("fprop",), ("dgrad",), ("wgrad",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("save_backward_axes", [False, True])
def test_linear_backward_targets_keep_gradient_shapes(targets, save_backward_axes):
    linear = torch.nn.Linear(32, 64)
    config = MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=targets,
        backward_reduction_axis_fake_quant=save_backward_axes,
    )
    qat = MXFPQATLinear.from_linear(linear, config)
    x = torch.randn((4, 32), requires_grad=True)
    qat(x).square().mean().backward()
    assert x.grad is not None and x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    assert qat.weight.grad is not None and qat.weight.grad.shape == qat.weight.shape
    assert torch.isfinite(qat.weight.grad).all()


def test_linear_fprop_rotation_returns_gradients_in_original_coordinates():
    linear = torch.nn.Linear(32, 64)
    config = MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=(),
        rotation_enable=True,
        rotation_targets=("fprop",),
        rotation_seed=23,
    )
    qat = MXFPQATLinear.from_linear(linear, config)
    x = torch.randn((4, 32), requires_grad=True)
    qat(x).sum().backward()
    assert x.grad.shape == x.shape
    assert qat.weight.grad.shape == linear.weight.shape


@pytest.mark.parametrize(
    "rotation_targets",
    [("fprop",), ("dgrad",), ("wgrad",), ("fprop", "dgrad", "wgrad")],
)
def test_linear_rotation_gradients_match_unrotated_linear(rotation_targets):
    torch.manual_seed(23)
    reference = torch.nn.Linear(32, 64)
    qat_source = torch.nn.Linear(32, 64)
    qat_source.load_state_dict(reference.state_dict())
    config = MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=(),
        rotation_enable=True,
        rotation_targets=rotation_targets,
        rotation_seed=29,
    )
    qat = MXFPQATLinear.from_linear(qat_source, config)
    x_reference = torch.randn((4, 32), requires_grad=True)
    x_qat = x_reference.detach().clone().requires_grad_()
    grad_output = torch.randn((4, 64))

    reference_output = reference(x_reference)
    qat_output = qat(x_qat)
    (reference_output * grad_output).sum().backward()
    (qat_output * grad_output).sum().backward()

    assert torch.allclose(qat_output, reference_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_qat.grad, x_reference.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(qat.weight.grad, reference.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(qat.bias.grad, reference.bias.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("mode", "mbs"), [("mxfp_w8a8", False), ("mxfp_w4a8", False), ("mxfp_w4a8", True), ("mxfp_w4a4", True)]
)
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("legacy_cache", [False, True])
def test_linear_backward_matches_independent_raw_operand_recipe(mode, mbs, rotation, legacy_cache):
    torch.manual_seed(47)
    config = MXFPQATConfig(
        mode=mode,
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_seed=13,
        mxfp4_mbs_enable=mbs,
        backward_reduction_axis_fake_quant=legacy_cache,
    )
    qat = MXFPQATLinear(32, 64, True, config)
    x = torch.randn(2, 17, 32, requires_grad=True)
    grad_output = torch.randn(2, 17, 64)

    def operand(tensor, fmt, *, pad=False):
        if rotation:
            tensor = rotate_blocks(tensor, config.rotation, pad_to_block=pad)
        return fake_quantize(tensor, fmt, mxfp4_mbs_enable=mbs and fmt == "mxfp4")

    with torch.no_grad():
        input_matrix = x.reshape(-1, 32)
        gradient_matrix = grad_output.reshape(-1, 64)
        expected_y = operand(input_matrix, config.activation_format) @ operand(qat.weight, config.weight_format).T
        expected_y += qat.bias
        expected_dx = operand(gradient_matrix, config.activation_format) @ operand(qat.weight.T, config.weight_format).T
        expected_dw = (
            operand(gradient_matrix.T, config.activation_format, pad=True)
            @ operand(input_matrix.T, config.activation_format, pad=True).T
        )

    output = qat(x)
    output.backward(grad_output)
    torch.testing.assert_close(output.reshape_as(expected_y), expected_y)
    torch.testing.assert_close(x.grad.reshape_as(expected_dx), expected_dx)
    torch.testing.assert_close(qat.weight.grad, expected_dw)
    torch.testing.assert_close(qat.bias.grad, gradient_matrix.sum(0))


@pytest.mark.parametrize(("need_input_grad", "need_weight_grad"), [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("legacy_cache", [False, True])
def test_linear_defers_backward_transforms_and_saves_only_required_raw_operands(
    monkeypatch, need_input_grad, need_weight_grad, legacy_cache
):
    from verl.utils.qat import mxfp_linear

    config = MXFPQATConfig(
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        backward_reduction_axis_fake_quant=legacy_cache,
        rotation_enable=True,
        rotation_targets=("fprop", "dgrad", "wgrad"),
    )
    qat = MXFPQATLinear(32, 64, True, config)
    qat.weight.requires_grad_(need_weight_grad)
    x = torch.randn(35, 32, requires_grad=need_input_grad)
    calls, saved = [], []
    original_transform = mxfp_linear._target_tensor

    def transform(tensor, config, target, fmt, **kwargs):
        calls.append((target, kwargs.get("probe_operand", "activation"), tensor.detach().clone()))
        return original_transform(tensor, config, target, fmt, **kwargs)

    def pack(tensor):
        saved.append(tensor)
        return tensor

    monkeypatch.setattr(mxfp_linear, "_target_tensor", transform)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = qat(x)
    assert [target for target, _, _ in calls] == ["fprop", "fprop"]
    expected_saved = ([x] if need_weight_grad else []) + ([qat.weight] if need_input_grad else [])
    assert len(saved) == len(expected_saved)
    assert all(actual is expected for actual, expected in zip(saved, expected_saved, strict=True))
    output.sum().backward()
    backward_operands = {(target, operand): tensor for target, operand, tensor in calls if target != "fprop"}
    if need_input_grad:
        torch.testing.assert_close(backward_operands["dgrad", "weight"], qat.weight.detach().T)
    if need_weight_grad:
        torch.testing.assert_close(backward_operands["wgrad", "activation"], x.detach().T)
    assert (("dgrad", "weight") in backward_operands) == need_input_grad
    assert (("wgrad", "activation") in backward_operands) == need_weight_grad


@pytest.mark.parametrize("reentrant", [False, True])
def test_linear_checkpoint_matches_deferred_backward_without_checkpoint(reentrant):
    torch.manual_seed(53)
    config = MXFPQATConfig(
        mode="mxfp_w4a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        rotation_enable=True,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        mxfp4_mbs_enable=True,
    )
    reference = MXFPQATLinear(32, 64, True, config)
    replayed = MXFPQATLinear(32, 64, True, replace(config, backward_reduction_axis_fake_quant=False))
    replayed.load_state_dict(reference.state_dict())
    x_reference = torch.randn(35, 32, requires_grad=True)
    x_replayed = x_reference.detach().clone().requires_grad_()
    grad_output = torch.randn(35, 64)
    expected = reference(x_reference)
    actual = checkpoint(replayed, x_replayed, use_reentrant=reentrant)
    expected.backward(grad_output)
    actual.backward(grad_output)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x_replayed.grad, x_reference.grad)
    torch.testing.assert_close(replayed.weight.grad, reference.weight.grad)
    torch.testing.assert_close(replayed.bias.grad, reference.bias.grad)


@pytest.mark.parametrize("targets", [("fprop",), ("fprop", "dgrad"), ("fprop", "wgrad")])
@pytest.mark.parametrize(("need_input_grad", "need_weight_grad"), [(True, True), (True, False), (False, True)])
def test_linear_disabled_backward_targets_preserve_fprop_ste(targets, need_input_grad, need_weight_grad):
    from verl.utils.qat.mxfp_linear import _target_tensor

    torch.manual_seed(67)
    config = MXFPQATConfig(
        mode="mxfp_w4a8",
        quant_backend="torch",
        fake_quant_targets=targets,
        rotation_enable=True,
        rotation_targets=("fprop",),
    )
    qat = MXFPQATLinear(32, 64, True, config)
    qat.weight.requires_grad_(need_weight_grad)
    x = torch.randn(35, 32, requires_grad=need_input_grad)
    reference_x = x.detach().clone().requires_grad_(need_input_grad)
    reference_weight = qat.weight.detach().clone().requires_grad_(need_weight_grad)
    reference_output = torch.nn.functional.linear(
        _target_tensor(reference_x, config, "fprop", config.activation_format),
        _target_tensor(reference_weight, config, "fprop", config.weight_format),
        qat.bias.detach(),
    )
    grad_output = torch.randn_like(reference_output)
    actual = qat(x)
    reference_output.backward(grad_output)
    actual.backward(grad_output)
    torch.testing.assert_close(actual, reference_output)
    if need_input_grad and "dgrad" not in targets:
        torch.testing.assert_close(x.grad, reference_x.grad)
    if need_weight_grad and "wgrad" not in targets:
        torch.testing.assert_close(qat.weight.grad, reference_weight.grad)
