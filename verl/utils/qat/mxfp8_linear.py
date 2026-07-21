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

This layer applies MXFP8 quantize+dequantize to weights during forward while
keeping the actual matmul in high precision.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.linear import QATMode

__all__ = ["MXFP8QATLinear"]

_MXFP8_BLOCK_SIZE = 32


def _dequantize_mxfp8(weight_q: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    weight_fp32 = weight_q.to(torch.float32)
    num_blocks = weight_q.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = weight_fp32.view(*weight_q.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    descale = torch.exp2(weight_scale.to(torch.float32) - 127.0)
    dequant = blocked * descale.unsqueeze(-1)
    return dequant.view_as(weight_q).to(dtype)


def _quantize_mxfp8_torch(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.to(torch.float32)
    num_blocks = weight.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = weight_fp32.view(*weight.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    amax = blocked.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254)
    weight_scale = scale_biased.to(torch.uint8)
    descale = torch.exp2(scale_biased - 127.0)
    quant = (blocked / descale.unsqueeze(-1)).reshape_as(weight_fp32).to(torch.float8_e4m3fn)
    return quant, weight_scale


def quantize_mxfp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.device.type == "npu":
        from verl.utils.vllm.vllm_fp8_utils import quantize_mxfp8_weight_ascend

        return quantize_mxfp8_weight_ascend(weight, weight.dtype)

    return _quantize_mxfp8_torch(weight)


def fake_quantize_mxfp8_weight(weight: torch.Tensor) -> torch.Tensor:
    weight_q, weight_scale = quantize_mxfp8_weight(weight)
    return _dequantize_mxfp8(weight_q, weight_scale, weight.dtype)


class MXFP8QATLinear(nn.Linear):
    """QAT linear layer that injects MXFP8 weight quantization error."""

    supports_qat_fusion = False
    _is_verl_qat_linear = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.MXFP8,
        group_size: int = _MXFP8_BLOCK_SIZE,
        activation_observer: str = "static_minmax",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        if mode != QATMode.MXFP8:
            raise ValueError(f"MXFP8QATLinear only supports mxfp8 mode, got: {mode}")
        if group_size != _MXFP8_BLOCK_SIZE:
            raise ValueError(f"MXFP8 QAT requires group_size={_MXFP8_BLOCK_SIZE}, got: {group_size}")
        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer
        self.fake_quant_enabled = True
        self._last_weight_scale: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.MXFP8,
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

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        weight_q, weight_scale = quantize_mxfp8_weight(weight)
        self._last_weight_scale = weight_scale.detach()
        weight_fq = _dequantize_mxfp8(weight_q, weight_scale, weight.dtype)
        return weight + (weight_fq - weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)
        return F.linear(x, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}"
        )
