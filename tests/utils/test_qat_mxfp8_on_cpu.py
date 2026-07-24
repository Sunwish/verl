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

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

from verl.utils.qat.core import QATConfig, apply_qat, invalidate_all_scales
from verl.utils.qat.linear import QATLinear, QATMode
from verl.utils.qat.mxfp8_linear import MXFP8QATLinear


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 16, bias=False)
        self.lm_head = nn.Linear(32, 16, bias=False)


@pytest.mark.parametrize("mode", ["w8a16_mxfp8", "w8a8_mxfp8"])
def test_apply_qat_mxfp8_replaces_eligible_linear_layers(mode):
    model = _TinyModel()

    apply_qat(model, QATConfig(enable=True, mode=mode, group_size=32, mxfp8_quant_backend="torch"))

    assert isinstance(model.proj, MXFP8QATLinear)
    assert model.proj.mode == QATMode(mode)
    assert model.proj.mxfp8_quant_backend == "torch"
    assert isinstance(model.lm_head, nn.Linear)


def test_apply_qat_w4a16_still_uses_nvfp4_qat_linear():
    model = _TinyModel()

    apply_qat(model, QATConfig(enable=True, mode="w4a16", group_size=16))

    assert isinstance(model.proj, QATLinear)


def test_apply_qat_mxfp8_requires_group_size_32():
    model = _TinyModel()

    with pytest.raises(ValueError, match="group_size=32"):
        apply_qat(model, QATConfig(enable=True, mode="w8a8_mxfp8", group_size=16))


def test_mxfp8_qat_linear_rejects_unknown_quant_backend():
    with pytest.raises(ValueError, match="Unsupported MXFP8 quant backend"):
        MXFP8QATLinear(32, 8, mode=QATMode.W8A8_MXFP8, mxfp8_quant_backend="rotate")


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

    invalidate_all_scales(model)

    assert model.nvfp4._weight_blockwise_scale is None
    assert model.nvfp4._weight_global_scale is None
    assert model.nvfp4._cached_weight_amax is None
    assert model.mxfp8._last_weight_scale is None
    assert model.mxfp8._last_input_scale is None
