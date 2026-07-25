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

import sys
import types
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("vllm")

from verl.utils.vllm.vllm_fp8_utils import (
    MXFP8_QUANT_BACKEND_ENV,
    MXFP8_ROUNDING_MODE_ENV,
    is_mxfp8_vllm_ascend,
    quant_weights,
)


class _FakeAscendModelSlimConfig:
    def __init__(self, quant_description):
        self.quant_description = quant_description


def _install_fake_vllm_ascend(monkeypatch):
    root = types.ModuleType("vllm_ascend")
    quantization = types.ModuleType("vllm_ascend.quantization")
    modelslim = types.ModuleType("vllm_ascend.quantization.modelslim_config")
    modelslim.AscendModelSlimConfig = _FakeAscendModelSlimConfig
    monkeypatch.setitem(sys.modules, "vllm_ascend", root)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quantization)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.modelslim_config", modelslim)


def test_is_mxfp8_vllm_ascend_detects_top_level_quant_method(monkeypatch):
    _install_fake_vllm_ascend(monkeypatch)
    config = _FakeAscendModelSlimConfig({"quant_method": "ascend"})
    assert is_mxfp8_vllm_ascend(config) is True


def test_is_mxfp8_vllm_ascend_detects_per_parameter_entries(monkeypatch):
    _install_fake_vllm_ascend(monkeypatch)
    config = _FakeAscendModelSlimConfig({"layer.weight": "W8A8_MXFP8"})
    assert is_mxfp8_vllm_ascend(config) is True


def test_quant_weights_mxfp8_emits_scale_suffix(monkeypatch):
    _install_fake_vllm_ascend(monkeypatch)
    quant_config = _FakeAscendModelSlimConfig({"quant_method": "ascend"})

    fake_torch_npu = types.SimpleNamespace(
        float8_e4m3fn=torch.float8_e4m3fn,
        npu_dynamic_mx_quant=lambda tensor, axis, dst_type: (
            tensor.to(torch.float8_e4m3fn),
            torch.randint(1, 10, (tensor.shape[0], tensor.shape[1] // 32, 1), dtype=torch.uint8),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    model = object()
    weights = [("layer.weight", torch.randn(4, 32, dtype=torch.bfloat16))]

    with patch("verl.utils.vllm.vllm_fp8_utils.is_fp8_weight", return_value=True), patch(
        "torch.distributed.get_rank", return_value=0
    ):
        outputs = list(quant_weights(weights, model, quant_config, dtype=torch.bfloat16))

    assert outputs[0][0] == "layer.weight"
    assert outputs[1][0] == "layer.weight_scale"
    assert outputs[1][1].shape == (4,)
    assert outputs[1][1].dtype == torch.uint8


def test_quant_weights_mxfp8_torch_backend_emits_scale_suffix(monkeypatch):
    _install_fake_vllm_ascend(monkeypatch)
    monkeypatch.setenv(MXFP8_QUANT_BACKEND_ENV, "torch")
    monkeypatch.setenv(MXFP8_ROUNDING_MODE_ENV, "hash")
    quant_config = _FakeAscendModelSlimConfig({"quant_method": "ascend"})

    model = object()
    weights = [("layer.weight", torch.randn(4, 32, dtype=torch.bfloat16))]

    with patch("verl.utils.vllm.vllm_fp8_utils.is_fp8_weight", return_value=True), patch(
        "torch.distributed.get_rank", return_value=0
    ):
        outputs = list(quant_weights(weights, model, quant_config, dtype=torch.bfloat16))

    assert outputs[0][0] == "layer.weight"
    assert outputs[1][0] == "layer.weight_scale"
    assert outputs[1][1].shape == (4,)
    assert outputs[1][1].dtype == torch.uint8
