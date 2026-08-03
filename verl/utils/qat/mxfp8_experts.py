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

"""MXFP8 QAT wrapper for packed MoE expert modules."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.linear import QATMode
from verl.utils.qat.mxfp8_linear import (
    _dequantize_mxfp8,
    _infer_mxfp8_layer_index,
    _infer_mxfp8_layer_type,
    _record_mxfp8_quant_error,
    normalize_mxfp8_quant_backend,
    normalize_mxfp8_rounding_mode,
    quantize_mxfp8_tensor,
)
from verl.utils.qat.mxfp8_rotation import (
    MXFP8RotationConfig,
    apply_mxfp8_block_rotation,
    normalize_mxfp8_rotation_kind,
    validate_mxfp8_rotation_config,
)

__all__ = ["MXFP8QATExperts", "MXFP8QATSparseMoeBlock"]


def _clone_parameter_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "meta":
        return tensor
    return tensor.detach().clone()


def _infer_packed_expert_dimensions(module: nn.Module) -> tuple[int, int, int]:
    num_experts = getattr(module, "num_experts", None)
    hidden_dim = getattr(module, "hidden_dim", getattr(module, "hidden_size", None))
    intermediate_dim = getattr(module, "intermediate_dim", getattr(module, "expert_dim", None))

    gate_up_proj = getattr(module, "gate_up_proj", None)
    down_proj = getattr(module, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        raise ValueError(f"Packed expert module {module.__class__.__name__} is missing gate_up_proj/down_proj")

    if num_experts is None:
        num_experts = gate_up_proj.shape[0]
    if hidden_dim is None:
        hidden_dim = gate_up_proj.shape[-1] if gate_up_proj.shape[-1] == down_proj.shape[-2] else gate_up_proj.shape[-2]
    if intermediate_dim is None:
        if gate_up_proj.shape[-2] % 2 == 0:
            intermediate_dim = gate_up_proj.shape[-2] // 2
        elif down_proj.shape[-1] % 2 == 0:
            intermediate_dim = down_proj.shape[-1]
        else:
            raise ValueError(
                "Unable to infer packed expert intermediate dimension from shapes "
                f"{tuple(gate_up_proj.shape)} / {tuple(down_proj.shape)}"
            )

    return int(num_experts), int(hidden_dim), int(intermediate_dim)


class MXFP8QATExperts(nn.Module):
    """MXFP8 QAT wrapper for packed expert modules such as Qwen MoE experts."""

    supports_qat_fusion = False
    _is_verl_qat_linear = True

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Any,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        mode: QATMode = QATMode.W8A16_MXFP8,
        group_size: int = 32,
        activation_observer: str = "static_minmax",
        mxfp8_quant_backend: str = "npu",
        mxfp8_rounding_mode: str = "rint",
        mxfp8_probe_quant_error: bool = False,
        mxfp8_rotation_enable: bool = False,
        mxfp8_rotation_kind: str = "block_hadamard_sign",
        mxfp8_rotation_block_size: int = 32,
        mxfp8_rotation_seed: int = 0,
        layer_name: Optional[str] = None,
        layer_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        quantize_gate_up_proj: bool = True,
        quantize_down_proj: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if isinstance(mode, str):
            mode = QATMode(mode.lower())
        if mode not in {QATMode.W8A16_MXFP8, QATMode.W8A8_MXFP8}:
            raise ValueError(f"MXFP8QATExperts only supports w8a16_mxfp8/w8a8_mxfp8 modes, got: {mode}")
        if group_size != 32:
            raise ValueError(f"MXFP8 QAT requires group_size=32, got: {group_size}")

        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.expert_dim = self.intermediate_dim
        self.act_fn = act_fn
        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer
        self.mxfp8_quant_backend = normalize_mxfp8_quant_backend(mxfp8_quant_backend)
        self.mxfp8_rounding_mode = normalize_mxfp8_rounding_mode(mxfp8_rounding_mode)
        if self.mxfp8_quant_backend == "npu" and self.mxfp8_rounding_mode in {"random", "hash"}:
            raise ValueError("MXFP8 stochastic rounding modes require mxfp8_quant_backend='torch'")
        self.mxfp8_probe_quant_error = mxfp8_probe_quant_error
        self.mxfp8_rotation_config = MXFP8RotationConfig(
            enable=mxfp8_rotation_enable,
            kind=normalize_mxfp8_rotation_kind(mxfp8_rotation_kind),
            block_size=mxfp8_rotation_block_size,
            seed=mxfp8_rotation_seed,
        )
        validate_mxfp8_rotation_config(self.mxfp8_rotation_config, group_size=self.group_size)
        self._mxfp8_layer_name = layer_name
        self._mxfp8_layer_type = layer_type if layer_type is not None else _infer_mxfp8_layer_type(layer_name)
        self._mxfp8_layer_index = layer_index if layer_index is not None else _infer_mxfp8_layer_index(layer_name)
        self._quantize_gate_up_proj = quantize_gate_up_proj
        self._quantize_down_proj = quantize_down_proj
        self.fake_quant_enabled = True
        self._last_gate_up_scale: Optional[torch.Tensor] = None
        self._last_down_scale: Optional[torch.Tensor] = None
        self._last_gate_up_input_scale: Optional[torch.Tensor] = None
        self._last_down_input_scale: Optional[torch.Tensor] = None

        self.gate_up_proj = nn.Parameter(_clone_parameter_tensor(gate_up_proj), requires_grad=True)
        self.down_proj = nn.Parameter(_clone_parameter_tensor(down_proj), requires_grad=True)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        mode: QATMode = QATMode.W8A16_MXFP8,
        group_size: int = 32,
        activation_observer: str = "static_minmax",
        mxfp8_quant_backend: str = "npu",
        mxfp8_rounding_mode: str = "rint",
        mxfp8_probe_quant_error: bool = False,
        mxfp8_rotation_enable: bool = False,
        mxfp8_rotation_kind: str = "block_hadamard_sign",
        mxfp8_rotation_block_size: int = 32,
        mxfp8_rotation_seed: int = 0,
        layer_name: Optional[str] = None,
        layer_type: Optional[str] = None,
        layer_index: Optional[int] = None,
        quantize_gate_up_proj: bool = True,
        quantize_down_proj: bool = True,
    ) -> "MXFP8QATExperts":
        num_experts, hidden_dim, intermediate_dim = _infer_packed_expert_dimensions(module)
        return cls(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=getattr(module, "act_fn"),
            gate_up_proj=getattr(module, "gate_up_proj"),
            down_proj=getattr(module, "down_proj"),
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            mxfp8_quant_backend=mxfp8_quant_backend,
            mxfp8_rounding_mode=mxfp8_rounding_mode,
            mxfp8_probe_quant_error=mxfp8_probe_quant_error,
            mxfp8_rotation_enable=mxfp8_rotation_enable,
            mxfp8_rotation_kind=mxfp8_rotation_kind,
            mxfp8_rotation_block_size=mxfp8_rotation_block_size,
            mxfp8_rotation_seed=mxfp8_rotation_seed,
            layer_name=layer_name,
            layer_type=layer_type,
            layer_index=layer_index,
            quantize_gate_up_proj=quantize_gate_up_proj,
            quantize_down_proj=quantize_down_proj,
            device=getattr(module.gate_up_proj, "device", None),
            dtype=getattr(module.gate_up_proj, "dtype", None),
        )

    def invalidate_quant_state(self):
        self._last_gate_up_scale = None
        self._last_down_scale = None
        self._last_gate_up_input_scale = None
        self._last_down_input_scale = None

    def _record_quant_error(
        self,
        error_type: str,
        sublayer_type: str | tuple[str, ...],
        original: torch.Tensor,
        quantized: torch.Tensor,
    ):
        if not self.mxfp8_probe_quant_error:
            return

        sublayer_types = (sublayer_type,) if isinstance(sublayer_type, str) else sublayer_type
        diff = (quantized.to(torch.float32) - original.detach().to(torch.float32)).abs()
        error_sum = diff.sum().item()
        element_count = diff.numel()
        for layer_type in sublayer_types:
            _record_mxfp8_quant_error(
                layer_type=layer_type,
                layer_index=self._mxfp8_layer_index,
                error_type=error_type,
                error_sum=error_sum,
                element_count=element_count,
                qat_mode=self.mode.value,
                quant_backend=self.mxfp8_quant_backend,
                rounding_mode=self.mxfp8_rounding_mode,
            )

    def _canonicalize_weight(self, weight: torch.Tensor, *, kind: str) -> torch.Tensor:
        if kind == "gate_up_proj":
            standard_shape = (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            transposed_shape = (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
        elif kind == "down_proj":
            standard_shape = (self.num_experts, self.hidden_dim, self.intermediate_dim)
            transposed_shape = (self.num_experts, self.intermediate_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unsupported packed expert kind: {kind}")

        if tuple(weight.shape) == standard_shape:
            return weight
        if tuple(weight.shape) == transposed_shape:
            return weight.transpose(-1, -2).contiguous()
        raise ValueError(
            f"Packed expert weight shape {tuple(weight.shape)} does not match expected layouts for {kind}: "
            f"{standard_shape} or {transposed_shape}"
        )

    def _rotate_if_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.mxfp8_rotation_config.enable:
            return tensor
        return apply_mxfp8_block_rotation(tensor, self.mxfp8_rotation_config)

    def _fake_quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        sublayer_type: str,
        scale_attr: str,
        split_sublayer_types: Optional[tuple[str, ...]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            weight_q, weight_scale = quantize_mxfp8_tensor(
                weight.reshape(-1, weight.shape[-1]),
                quant_backend=self.mxfp8_quant_backend,
                rounding_mode=self.mxfp8_rounding_mode,
            )
            setattr(self, scale_attr, weight_scale.detach())
            weight_fq = _dequantize_mxfp8(weight_q, weight_scale, weight.dtype).reshape(weight.shape)
            if self.mxfp8_probe_quant_error:
                if split_sublayer_types is None:
                    self._record_quant_error("weight", sublayer_type, weight, weight_fq)
                else:
                    original_chunks = weight.chunk(len(split_sublayer_types), dim=1)
                    quantized_chunks = weight_fq.chunk(len(split_sublayer_types), dim=1)
                    for layer_type, original_chunk, quantized_chunk in zip(
                        split_sublayer_types, original_chunks, quantized_chunks, strict=True
                    ):
                        self._record_quant_error("weight", layer_type, original_chunk, quantized_chunk)
                return weight
        return weight + (weight_fq - weight).detach()

    def _fake_quantize_activation(
        self,
        x: torch.Tensor,
        *,
        sublayer_type: str | tuple[str, ...],
        scale_attr: str,
    ) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            x_q, x_scale = quantize_mxfp8_tensor(
                x_2d,
                quant_backend=self.mxfp8_quant_backend,
                rounding_mode=self.mxfp8_rounding_mode,
            )
            setattr(self, scale_attr, x_scale.detach().reshape(*original_shape[:-1], x_scale.shape[-1]))
            x_fq = _dequantize_mxfp8(x_q, x_scale, x.dtype).reshape(original_shape)
            if self.mxfp8_probe_quant_error:
                self._record_quant_error("activation", sublayer_type, x, x_fq)
                return x
        return x + (x_fq - x).detach()

    def _prepare_gate_up_weight(self, apply_fake_quant: bool) -> torch.Tensor:
        weight = self._canonicalize_weight(self.gate_up_proj, kind="gate_up_proj")
        if not apply_fake_quant or not self._quantize_gate_up_proj:
            return weight
        weight = self._rotate_if_enabled(weight)
        return self._fake_quantize_weight(
            weight,
            sublayer_type="gate_up_proj",
            scale_attr="_last_gate_up_scale",
            split_sublayer_types=("gate_proj", "up_proj"),
        )

    def _prepare_down_weight(self, apply_fake_quant: bool) -> torch.Tensor:
        weight = self._canonicalize_weight(self.down_proj, kind="down_proj")
        if not apply_fake_quant or not self._quantize_down_proj:
            return weight
        weight = self._rotate_if_enabled(weight)
        return self._fake_quantize_weight(
            weight,
            sublayer_type="down_proj",
            scale_attr="_last_down_scale",
        )

    def _prepare_gate_up_input(self, x: torch.Tensor, apply_fake_quant: bool) -> torch.Tensor:
        if not apply_fake_quant or not self._quantize_gate_up_proj:
            return x
        x = self._rotate_if_enabled(x)
        if self.mode == QATMode.W8A8_MXFP8:
            x = self._fake_quantize_activation(
                x,
                sublayer_type=("gate_proj", "up_proj"),
                scale_attr="_last_gate_up_input_scale",
            )
        return x

    def _prepare_down_input(self, x: torch.Tensor, apply_fake_quant: bool) -> torch.Tensor:
        if not apply_fake_quant or not self._quantize_down_proj:
            return x
        x = self._rotate_if_enabled(x)
        if self.mode == QATMode.W8A8_MXFP8:
            x = self._fake_quantize_activation(
                x,
                sublayer_type="down_proj",
                scale_attr="_last_down_input_scale",
            )
        return x

    def prepare_npu_sparse_block_gate_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._prepare_gate_up_input(hidden_states, self.fake_quant_enabled)

    def prepare_npu_sparse_block_down_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._prepare_down_input(hidden_states, self.fake_quant_enabled)

    def get_npu_sparse_block_weights(self, input_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_up_weight = self._prepare_gate_up_weight(self.fake_quant_enabled)
        gate_proj_weight, up_proj_weight = gate_up_weight.chunk(2, dim=1)
        down_weight = self._prepare_down_weight(self.fake_quant_enabled)
        return (
            up_proj_weight.transpose(1, 2).to(input_dtype),
            gate_proj_weight.transpose(1, 2).to(input_dtype),
            down_weight.transpose(1, 2).to(input_dtype),
        )

    def _prepare_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        routing_arg1: torch.Tensor,
        routing_arg2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        original_shape = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(-1, original_shape[-1])

        if (
            routing_arg1.dtype.is_floating_point
            and routing_arg1.dim() == 2
            and routing_arg1.shape[0] == hidden_states_2d.shape[0]
            and routing_arg1.shape[-1] == self.num_experts
        ):
            routing_weights = routing_arg1
            router_indices = routing_arg2
            batch_idx = torch.arange(router_indices.shape[0], device=routing_weights.device).unsqueeze(1)
            top_k_index = router_indices
            top_k_weights = routing_weights[batch_idx.expand_as(router_indices), router_indices]
        else:
            top_k_index = routing_arg1
            top_k_weights = routing_arg2

        return hidden_states_2d, top_k_index, top_k_weights, original_shape

    def _forward_tokens(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        apply_fake_quant: bool,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_weight = self._prepare_gate_up_weight(apply_fake_quant)
        down_weight = self._prepare_down_weight(apply_fake_quant)

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_state = self._prepare_gate_up_input(current_state, apply_fake_quant)
            gate, up = F.linear(current_state, gate_up_weight[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self._prepare_down_input(current_hidden_states, apply_fake_quant)
            current_hidden_states = F.linear(current_hidden_states, down_weight[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_arg1: torch.Tensor,
        routing_arg2: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_2d, top_k_index, top_k_weights, original_shape = self._prepare_forward_inputs(
            hidden_states,
            routing_arg1,
            routing_arg2,
        )
        output = self._forward_tokens(
            hidden_states_2d,
            top_k_index,
            top_k_weights,
            apply_fake_quant=self.fake_quant_enabled,
        )
        return output.reshape(original_shape)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}, "
            f"mode={self.mode.value}, group_size={self.group_size}, mxfp8_quant_backend={self.mxfp8_quant_backend}, "
            f"mxfp8_rounding_mode={self.mxfp8_rounding_mode}, mxfp8_rotation_enable={self.mxfp8_rotation_config.enable}, "
            f"mxfp8_rotation_kind={self.mxfp8_rotation_config.kind}, mxfp8_rotation_block_size={self.mxfp8_rotation_config.block_size}, "
            f"mxfp8_rotation_seed={self.mxfp8_rotation_config.seed}, mxfp8_probe_quant_error={self.mxfp8_probe_quant_error}, "
            f"fake_quant_enabled={self.fake_quant_enabled}"
        )


MXFP8QATSparseMoeBlock = MXFP8QATExperts
