import pytest
import torch
from torch import nn

from verl.utils.qat.mxfp_linear import MXFPQATConfig
from verl.utils.qat.mxfp_moe import (
    _transform_tokens,
    _transform_weight,
    attach_mxfp_expert_qat,
    get_mxfp_expert_qat_config,
    mxfp_grouped_matmul,
    prepare_mxfp_fprop_input,
)


class _FakeMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 4, bias=False)
        self.experts = nn.Module()


def _config():
    return MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        backward_reduction_axis_fake_quant=True,
        rotation_enable=True,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_block_size=32,
        rotation_seed=7,
    )


def _fake_grouped_matmul(x, weight, group_list, group_list_type, *, split_item=2, group_type=0):
    counts = group_list.tolist()
    if group_list_type == 0:
        counts = [count - (counts[index - 1] if index else 0) for index, count in enumerate(counts)]
    offset = 0
    if split_item == 2:
        outputs = []
        for expert, count in enumerate(counts):
            outputs.append(x[offset : offset + count] @ weight[expert])
            offset += count
        return torch.cat(outputs, dim=0)
    outputs = []
    for count in counts:
        outputs.append(x[:, offset : offset + count] @ weight[offset : offset + count])
        offset += count
    return torch.stack(outputs)


def test_attach_only_marks_moe_blocks():
    model = nn.Sequential(_FakeMoeBlock(), nn.Linear(32, 32))
    config = _config()
    assert attach_mxfp_expert_qat(model, config) == 1
    assert get_mxfp_expert_qat_config(model[0]) is config
    assert get_mxfp_expert_qat_config(model[1]) is None


def test_grouped_weight_transforms_use_reduction_axis():
    config = _config()
    weight = torch.randn(4, 32, 64)
    transformed = _transform_weight(weight, config, "fprop", config.weight_format, reduction_dim=1)
    assert transformed.shape == weight.shape
    dgrad_weight = _transform_weight(weight, config, "dgrad", config.weight_format, reduction_dim=2)
    assert dgrad_weight.shape == weight.shape
    assert torch.isfinite(transformed).all()
    assert torch.isfinite(dgrad_weight).all()


def test_wgrad_transform_quantizes_token_reduction_axis():
    config = _config()
    x = torch.randn(65, 32)
    transformed = _transform_tokens(x, config, "wgrad", config.activation_format)
    assert transformed.shape == x.shape
    assert torch.isfinite(transformed).all()


def test_gate_up_can_share_detached_fprop_input():
    config = _config()
    x = torch.randn(65, 32, requires_grad=True)
    x_fprop = prepare_mxfp_fprop_input(x, config)
    assert x_fprop.shape == x.shape
    assert not x_fprop.requires_grad


@pytest.mark.parametrize(
    ("need_input_grad", "need_weight_grad"),
    [(True, True), (True, False), (False, True)],
)
def test_grouped_matmul_recomputes_backward_operands_and_saves_only_inputs(
    monkeypatch, need_input_grad, need_weight_grad
):
    wgrad_input_strides = []

    def fake_grouped_matmul(x, weight, group_list, group_list_type, *, split_item=2, group_type=0):
        if split_item == 3:
            wgrad_input_strides.append(x.stride())
        return _fake_grouped_matmul(
            x, weight, group_list, group_list_type, split_item=split_item, group_type=group_type
        )

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", fake_grouped_matmul)
    config = _config()
    x = torch.randn(64, 32, requires_grad=need_input_grad)
    weight = torch.randn(1, 32, 32, requires_grad=need_weight_grad)
    group_list = torch.tensor([64])
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = mxfp_grouped_matmul(x, weight, group_list, config, name="test").sum()

    expected_shapes = []
    if need_weight_grad:
        expected_shapes.append(tuple(x.shape))
    if need_input_grad:
        expected_shapes.append(tuple(weight.shape))
    assert saved_shapes == expected_shapes
    output.backward()
    assert (x.grad is not None) == need_input_grad
    assert (weight.grad is not None) == need_weight_grad
    if x.grad is not None:
        assert torch.isfinite(x.grad).all()
    if weight.grad is not None:
        assert torch.isfinite(weight.grad).all()
    if need_weight_grad:
        assert wgrad_input_strides == [(1, 32)]


@pytest.mark.parametrize(
    "rotation_targets",
    [("fprop",), ("dgrad",), ("wgrad",), ("fprop", "dgrad", "wgrad")],
)
def test_grouped_matmul_rotation_gradients_match_unrotated_matmul(monkeypatch, rotation_targets):
    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", _fake_grouped_matmul)
    config = MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=(),
        rotation_enable=True,
        rotation_targets=rotation_targets,
        rotation_block_size=32,
        rotation_seed=29,
    )
    torch.manual_seed(31)
    x_reference = torch.randn(65, 32, requires_grad=True)
    x_qat = x_reference.detach().clone().requires_grad_()
    weight_reference = torch.randn(3, 32, 64, requires_grad=True)
    weight_qat = weight_reference.detach().clone().requires_grad_()
    grad_output = torch.randn(65, 64)
    group_list = torch.tensor([7, 30, 28])

    reference_output = torch.cat(
        [
            x_reference[:7] @ weight_reference[0],
            x_reference[7:37] @ weight_reference[1],
            x_reference[37:] @ weight_reference[2],
        ],
        dim=0,
    )
    qat_output = mxfp_grouped_matmul(
        x_qat,
        weight_qat,
        group_list,
        config,
        name="rotation-test",
        fprop_x=prepare_mxfp_fprop_input(x_qat, config),
    )
    (reference_output * grad_output).sum().backward()
    (qat_output * grad_output).sum().backward()

    assert torch.allclose(qat_output, reference_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_qat.grad, x_reference.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(weight_qat.grad, weight_reference.grad, atol=1e-5, rtol=1e-5)
