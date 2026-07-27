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

import atexit
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.linear import QATMode
from verl.utils.qat.mxfp8_rotation import (
    MXFP8RotationConfig,
    apply_mxfp8_block_rotation,
    normalize_mxfp8_rotation_kind,
    validate_mxfp8_rotation_config,
)

__all__ = [
    "MXFP8QATLinear",
    "configure_mxfp8_probe",
    "mxfp8_probe_step_context",
    "normalize_mxfp8_quant_backend",
    "normalize_mxfp8_rounding_mode",
    "quantize_mxfp8_tensor",
    "reset_mxfp8_probe",
    "set_mxfp8_probe_step",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MXFP8_BLOCK_SIZE = 32
_MXFP8_EMAX = 8
_MXFP8_SCALE_EMAX = 127
_MXFP8_MIN_PRIVATE_EXP = -6
_MXFP8_MANTISSA_SCALE = 8.0
_MXFP8_QUANT_BACKENDS = {"npu", "torch"}
_MXFP8_ROUNDING_MODES = {"round", "random", "hash"}
_MXFP8_HASH_MULTIPLIER = 1664525
_MXFP8_HASH_INCREMENT = 1013904223
_MXFP8_HASH_MODULUS = 2**32
_MXFP8_HASH_RANDOM_SHIFT = 2**8
_MXFP8_HASH_RANDOM_LEVELS = 2**24
_MXFP8_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


class _MXFP8ProbeRecorder:
    def __init__(self):
        self.enabled = False
        self.output_path: Optional[str] = None
        self.rank0_only = True
        self.current_step: Optional[int] = None
        self._fp = None
        self._lock = threading.Lock()
        self._atexit_registered = False

    def configure(self, enabled: bool, output_path: Optional[str], rank0_only: bool = True):
        with self._lock:
            self._close_locked()
            self.enabled = enabled
            self.output_path = output_path if enabled else None
            self.rank0_only = rank0_only
            self.current_step = None

    def set_step(self, step: Optional[int]):
        self.current_step = step

    def _is_rank0(self) -> bool:
        if not self.rank0_only:
            return True
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _ensure_open_locked(self):
        if self._fp is not None or self.output_path is None:
            return
        output_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(output_dir, exist_ok=True)
        self._fp = open(self.output_path, "a", encoding="utf-8")
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def _close_locked(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def close(self):
        with self._lock:
            self._close_locked()

    def reset(self):
        with self._lock:
            self._close_locked()
            self.enabled = False
            self.output_path = None
            self.rank0_only = True
            self.current_step = None

    def record(self, record: dict):
        if not self.enabled or not self._is_rank0():
            return

        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        logger.warning("[MXFP8 probe] %s", payload)

        if self.output_path is None:
            return

        with self._lock:
            self._ensure_open_locked()
            if self._fp is not None:
                self._fp.write(payload + "\n")
                self._fp.flush()


_MXFP8_PROBE_RECORDER = _MXFP8ProbeRecorder()


def configure_mxfp8_probe(enabled: bool, output_path: Optional[str], rank0_only: bool = True):
    _MXFP8_PROBE_RECORDER.configure(enabled=enabled, output_path=output_path, rank0_only=rank0_only)


def reset_mxfp8_probe():
    _MXFP8_PROBE_RECORDER.reset()


def set_mxfp8_probe_step(step: Optional[int]):
    _MXFP8_PROBE_RECORDER.set_step(step)


@contextmanager
def mxfp8_probe_step_context(step: Optional[int]):
    previous_step = _MXFP8_PROBE_RECORDER.current_step
    _MXFP8_PROBE_RECORDER.set_step(step)
    try:
        yield
    finally:
        _MXFP8_PROBE_RECORDER.set_step(previous_step)


def _infer_mxfp8_layer_type(layer_name: Optional[str]) -> Optional[str]:
    if layer_name is None:
        return None
    return layer_name.rsplit(".", 1)[-1]


def _infer_mxfp8_layer_index(layer_name: Optional[str]) -> Optional[int]:
    if layer_name is None:
        return None
    match = _MXFP8_LAYER_IDX_RE.search(layer_name)
    return int(match.group(1)) if match else None


def _record_mxfp8_quant_error(
    *,
    layer_name: Optional[str],
    layer_type: Optional[str],
    layer_index: Optional[int],
    error_type: str,
    error_value: float,
    qat_mode: str,
    quant_backend: str,
    rounding_mode: str,
):
    step = _MXFP8_PROBE_RECORDER.current_step
    _MXFP8_PROBE_RECORDER.record(
        {
            "error_metric": "mae",
            "error_type": error_type,
            "error_value": float(error_value),
            "layer_index": layer_index,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "mode": qat_mode,
            "quant_backend": quant_backend,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "rounding_mode": rounding_mode,
            "step": int(step) if step is not None else None,
        }
    )


def normalize_mxfp8_quant_backend(backend: str) -> str:
    backend = backend.lower()
    if backend not in _MXFP8_QUANT_BACKENDS:
        raise ValueError(
            f"Unsupported MXFP8 quant backend: {backend}. Supported backends: {sorted(_MXFP8_QUANT_BACKENDS)}"
        )
    return backend


def normalize_mxfp8_rounding_mode(rounding_mode: str) -> str:
    rounding_mode = rounding_mode.lower()
    if rounding_mode not in _MXFP8_ROUNDING_MODES:
        raise ValueError(
            f"Unsupported MXFP8 rounding mode: {rounding_mode}. "
            f"Supported modes: {sorted(_MXFP8_ROUNDING_MODES)}"
        )
    return rounding_mode


def _dequantize_mxfp8(weight_q: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    weight_fp32 = weight_q.to(torch.float32)
    num_blocks = weight_q.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = weight_fp32.reshape(*weight_q.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    scale_shape = blocked.shape[:-1]
    descale = torch.exp2(weight_scale.to(torch.float32).reshape(scale_shape) - 127.0)
    dequant = blocked * descale.unsqueeze(-1)
    return dequant.reshape_as(weight_q).to(dtype)


def _check_mxfp8_2d_tensor(tensor: torch.Tensor):
    if tensor.dim() != 2:
        raise ValueError(f"MXFP8 quantization only supports 2D tensors, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] % _MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires the last dimension to be divisible by {_MXFP8_BLOCK_SIZE}, "
            f"got shape={tuple(tensor.shape)}"
        )


def _round_mxfp8_scaled_abs(abs_scaled: torch.Tensor, rounding_mode: str) -> torch.Tensor:
    rounding_mode = normalize_mxfp8_rounding_mode(rounding_mode)
    if rounding_mode == "round":
        return torch.floor(abs_scaled + 0.5)

    floor_val = torch.floor(abs_scaled)
    frac = abs_scaled - floor_val
    if rounding_mode == "random":
        rand = torch.rand_like(frac)
    elif rounding_mode == "hash":
        int_bits = abs_scaled.contiguous().view(torch.int32).detach().to(torch.int64)
        uint_bits = torch.remainder(int_bits, _MXFP8_HASH_MODULUS)
        hashed = torch.remainder(
            uint_bits * _MXFP8_HASH_MULTIPLIER + _MXFP8_HASH_INCREMENT,
            _MXFP8_HASH_MODULUS,
        )
        rand_bits = torch.div(hashed, _MXFP8_HASH_RANDOM_SHIFT, rounding_mode="floor")
        rand = rand_bits.to(torch.float32) * (1.0 / float(_MXFP8_HASH_RANDOM_LEVELS))
    else:
        raise ValueError(f"Unsupported MXFP8 rounding mode: {rounding_mode}")

    return floor_val + (rand < frac).to(floor_val.dtype)


def _quantize_mxfp8_torch(tensor: torch.Tensor, rounding_mode: str = "round") -> tuple[torch.Tensor, torch.Tensor]:
    _check_mxfp8_2d_tensor(tensor)
    rounding_mode = normalize_mxfp8_rounding_mode(rounding_mode)
    tensor_fp32 = tensor.to(torch.float32)
    original_shape = tensor_fp32.shape
    max_norm = torch.finfo(torch.float8_e4m3fn).max
    num_blocks = tensor.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = tensor_fp32.reshape(tensor.shape[0], num_blocks, _MXFP8_BLOCK_SIZE)

    amax = blocked.abs().amax(dim=-1)
    amax_safe = torch.where(amax == 0, torch.full_like(amax, torch.finfo(torch.float32).tiny), amax)
    shared_exp = torch.floor(torch.log2(amax_safe)) - _MXFP8_EMAX
    shared_exp = torch.where(shared_exp > _MXFP8_SCALE_EMAX, torch.full_like(shared_exp, float("nan")), shared_exp)

    scale_factor = torch.pow(2.0, shared_exp.unsqueeze(-1))
    normalized = blocked / scale_factor
    abs_norm = normalized.abs()
    private_exp = torch.floor(torch.log2(abs_norm + (abs_norm == 0).float()))
    private_exp = private_exp.clamp(min=_MXFP8_MIN_PRIVATE_EXP)

    private_scale = torch.pow(2.0, private_exp)
    scaled = normalized / private_scale * _MXFP8_MANTISSA_SCALE
    quantized = torch.sign(scaled) * _round_mxfp8_scaled_abs(torch.abs(scaled), rounding_mode)
    quantized = quantized / _MXFP8_MANTISSA_SCALE * private_scale
    quantized = torch.clamp(quantized, min=-max_norm, max=max_norm)
    quantized = torch.where(torch.isinf(normalized), normalized, quantized)
    quantized = torch.where(torch.isnan(normalized), normalized, quantized)

    quant = quantized.reshape(original_shape).to(torch.float8_e4m3fn)
    shared_exp_fixed = torch.nan_to_num(shared_exp, nan=-127.0)
    tensor_scale = torch.clamp(shared_exp_fixed + 127.0, 0, 255).round().to(torch.uint8)
    return quant, tensor_scale


def _quantize_mxfp8_npu(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    tensor_q, tensor_scale = torch_npu.npu_dynamic_mx_quant(
        tensor,
        axis=-1,
        dst_type=torch_npu.float8_e4m3fn,
    )
    tensor_scale = tensor_scale.flatten(-2, -1)
    return tensor_q, tensor_scale.squeeze(-1)


def quantize_mxfp8_tensor(
    tensor: torch.Tensor, quant_backend: str = "torch", rounding_mode: str = "round"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor along its last dimension using MXFP8 blocks."""
    _check_mxfp8_2d_tensor(tensor)
    quant_backend = normalize_mxfp8_quant_backend(quant_backend)
    rounding_mode = normalize_mxfp8_rounding_mode(rounding_mode)
    if quant_backend == "npu":
        if rounding_mode != "round":
            raise ValueError("MXFP8 stochastic rounding modes require mxfp8_quant_backend='torch'")
        return _quantize_mxfp8_npu(tensor)

    return _quantize_mxfp8_torch(tensor, rounding_mode=rounding_mode)


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
        mxfp8_quant_backend: str = "npu",
        mxfp8_rounding_mode: str = "round",
        mxfp8_probe_quant_error: bool = False,
        mxfp8_rotation_enable: bool = False,
        mxfp8_rotation_kind: str = "block_hadamard_sign",
        mxfp8_rotation_block_size: int = _MXFP8_BLOCK_SIZE,
        mxfp8_rotation_seed: int = 0,
        layer_name: Optional[str] = None,
        layer_type: Optional[str] = None,
        layer_index: Optional[int] = None,
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
        self.mxfp8_quant_backend = normalize_mxfp8_quant_backend(mxfp8_quant_backend)
        self.mxfp8_rounding_mode = normalize_mxfp8_rounding_mode(mxfp8_rounding_mode)
        if self.mxfp8_quant_backend == "npu" and self.mxfp8_rounding_mode != "round":
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
        mxfp8_quant_backend: str = "npu",
        mxfp8_rounding_mode: str = "round",
        mxfp8_probe_quant_error: bool = False,
        mxfp8_rotation_enable: bool = False,
        mxfp8_rotation_kind: str = "block_hadamard_sign",
        mxfp8_rotation_block_size: int = _MXFP8_BLOCK_SIZE,
        mxfp8_rotation_seed: int = 0,
        layer_name: Optional[str] = None,
        layer_type: Optional[str] = None,
        layer_index: Optional[int] = None,
    ) -> "MXFP8QATLinear":
        has_bias = linear.bias is not None
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
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

    def _record_quant_error(self, error_type: str, original: torch.Tensor, quantized: torch.Tensor):
        if not self.mxfp8_probe_quant_error:
            return

        diff = (quantized.to(torch.float32) - original.detach().to(torch.float32)).abs()
        _record_mxfp8_quant_error(
            layer_name=self._mxfp8_layer_name,
            layer_type=self._mxfp8_layer_type,
            layer_index=self._mxfp8_layer_index,
            error_type=error_type,
            error_value=diff.mean().item(),
            qat_mode=self.mode.value,
            quant_backend=self.mxfp8_quant_backend,
            rounding_mode=self.mxfp8_rounding_mode,
        )

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight_q, weight_scale = quantize_mxfp8_tensor(
                weight, quant_backend=self.mxfp8_quant_backend, rounding_mode=self.mxfp8_rounding_mode
            )
            self._last_weight_scale = weight_scale.detach()
            weight_fq = _dequantize_mxfp8(weight_q, weight_scale, weight.dtype)
            if self.mxfp8_probe_quant_error:
                self._record_quant_error("weight", weight, weight_fq)
                return weight
        return weight + (weight_fq - weight).detach()

    def _rotate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.mxfp8_rotation_config.enable:
            return tensor
        return apply_mxfp8_block_rotation(tensor, self.mxfp8_rotation_config)

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            x_q, x_scale = quantize_mxfp8_tensor(
                x_2d, quant_backend=self.mxfp8_quant_backend, rounding_mode=self.mxfp8_rounding_mode
            )
            self._last_input_scale = x_scale.detach().reshape(*original_shape[:-1], x_scale.shape[-1])
            x_fq = _dequantize_mxfp8(x_q, x_scale, x.dtype).reshape(original_shape)
            if self.mxfp8_probe_quant_error:
                self._record_quant_error("activation", x, x_fq)
                return x
        return x + (x_fq - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        rotated_weight = self._rotate_tensor(self.weight)
        rotated_x = self._rotate_tensor(x)
        weight_fq = self._fake_quantize_weight(rotated_weight)
        x_fq = self._fake_quantize_activation(rotated_x) if self.mode == QATMode.W8A8_MXFP8 else rotated_x
        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, mxfp8_quant_backend={self.mxfp8_quant_backend}, "
            f"mxfp8_rounding_mode={self.mxfp8_rounding_mode}, "
            f"mxfp8_rotation_enable={self.mxfp8_rotation_config.enable}, "
            f"mxfp8_rotation_kind={self.mxfp8_rotation_config.kind}, "
            f"mxfp8_rotation_block_size={self.mxfp8_rotation_config.block_size}, "
            f"mxfp8_rotation_seed={self.mxfp8_rotation_config.seed}, "
            f"mxfp8_probe_quant_error={self.mxfp8_probe_quant_error}, "
            f"fake_quant_enabled={self.fake_quant_enabled}"
        )
