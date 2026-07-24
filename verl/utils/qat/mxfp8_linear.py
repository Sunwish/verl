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

"""MXFP8 QAT linear layer for FSDP training.

This layer applies MXFP8 quantize+dequantize during forward while keeping the
actual matmul in high precision. W8A16 quantizes weights only; W8A8 quantizes
both weights and activations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.linear import QATMode

__all__ = ["MXFP8QATLinear", "quantize_mxfp8_tensor"]

_MXFP8_BLOCK_SIZE = 32


def _dequantize_mxfp8(weight_q: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    weight_fp32 = weight_q.to(torch.float32)
    num_blocks = weight_q.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = weight_fp32.reshape(*weight_q.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    scale_shape = blocked.shape[:-1]
    descale = torch.exp2(weight_scale.to(torch.float32).reshape(scale_shape) - 127.0)
    dequant = blocked * descale.unsqueeze(-1)
    return dequant.reshape_as(weight_q).to(dtype)


def _check_mxfp8_last_dim(tensor: torch.Tensor):
    if tensor.shape[-1] % _MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires the last dimension to be divisible by {_MXFP8_BLOCK_SIZE}, "
            f"got shape={tuple(tensor.shape)}"
        )


def _quantize_mxfp8_torch(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _check_mxfp8_last_dim(tensor)
    tensor_fp32 = tensor.to(torch.float32)
    num_blocks = tensor.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = tensor_fp32.reshape(*tensor.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    amax = blocked.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254)
    tensor_scale = scale_biased.to(torch.uint8)
    descale = torch.exp2(scale_biased - 127.0)
    quant = (blocked / descale.unsqueeze(-1)).reshape_as(tensor_fp32).to(torch.float8_e4m3fn)
    return quant, tensor_scale


def quantize_mxfp8_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor along its last dimension using MXFP8 blocks."""
    _check_mxfp8_last_dim(tensor)
    if tensor.device.type == "npu":
        from verl.utils.vllm.vllm_fp8_utils import quantize_mxfp8_weight_ascend

        return quantize_mxfp8_weight_ascend(tensor, tensor.dtype)

    return _quantize_mxfp8_torch(tensor)


class MXFP8QATLinear(nn.Linear):
    """QAT linear layer that injects MXFP8 quantization error."""

    supports_qat_fusion = False
    _is_verl_qat_linear = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W8A16_MXFP8,
        group_size: int = _MXFP8_BLOCK_SIZE,
        activation_observer: str = "static_minmax",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        if isinstance(mode, str):
            mode = QATMode(mode.lower())
        if mode not in {QATMode.W8A16_MXFP8, QATMode.W8A8_MXFP8}:
            raise ValueError(f"MXFP8QATLinear only supports w8a16_mxfp8/w8a8_mxfp8 modes, got: {mode}")
        if group_size != _MXFP8_BLOCK_SIZE:
            raise ValueError(f"MXFP8 QAT requires group_size={_MXFP8_BLOCK_SIZE}, got: {group_size}")
        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer
        self.fake_quant_enabled = True
        self._last_weight_scale: Optional[torch.Tensor] = None
        self._last_input_scale: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W8A16_MXFP8,
        group_size: int = _MXFP8_BLOCK_SIZE,
        activation_observer: str = "static_minmax",
    ) -> "MXFP8QATLinear":
        has_bias = linear.bias is not None
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def invalidate_quant_state(self):
        self._last_weight_scale = None
        self._last_input_scale = None

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        weight_q, weight_scale = quantize_mxfp8_tensor(weight)
        self._last_weight_scale = weight_scale.detach()
        weight_fq = _dequantize_mxfp8(weight_q, weight_scale, weight.dtype)
        return weight + (weight_fq - weight).detach()

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = quantize_mxfp8_tensor(x)
        self._last_input_scale = x_scale.detach()
        x_fq = _dequantize_mxfp8(x_q, x_scale, x.dtype)
        return x + (x_fq - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)
        x_fq = self._fake_quantize_activation(x) if self.mode == QATMode.W8A8_MXFP8 else x
        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}"
        )
