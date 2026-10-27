# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MXFP8/MXFP4 tensor quantize-dequantize primitives used by QAT."""

from __future__ import annotations

from enum import Enum

import torch

from verl.utils.qat.mxfp_debug import warn_once

MXFP_GROUP_SIZE = 32
MXFP8_EMAX = 8
MXFP8_SCALE_EMAX = 127
MXFP8_MIN_PRIVATE_EXP = -6
MXFP8_MANTISSA_SCALE = 8.0
MXFP4_EMAX = 2
MXFP4_MIN_PRIVATE_EXP = 0
MXFP4_MANTISSA_SCALE = 2.0
MXFP4_MAX_NORM = 6.0
MXFP4_CX_MAX_NORM = 7.25


class MXFPMode(str, Enum):
    W8A8 = "mxfp_w8a8"
    W4A8 = "mxfp_w4a8"
    W4A4 = "mxfp_w4a4"

    @staticmethod
    def values() -> tuple[str, ...]:
        return tuple(item.value for item in MXFPMode)


_MODE_ALIASES = {
    "mxfp_w8a8": MXFPMode.W8A8,
    "w8a8_mxfp8": MXFPMode.W8A8,
    "mxfp_w4a8": MXFPMode.W4A8,
    "w4a8_mxfp4": MXFPMode.W4A8,
    "mxfp_w4a4": MXFPMode.W4A4,
    "w4a4_mxfp4": MXFPMode.W4A4,
}


def normalize_mxfp_mode(mode: str | MXFPMode) -> MXFPMode:
    if isinstance(mode, MXFPMode):
        return mode
    try:
        return _MODE_ALIASES[str(mode).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MXFP QAT mode {mode!r}; expected one of {MXFPMode.values()}") from exc


def normalize_quant_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"torch", "npu"}:
        raise ValueError(f"Unsupported MXFP quant backend {backend!r}; expected 'torch' or 'npu'")
    return backend


def normalize_rounding_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"rint", "round", "random", "hash"}:
        raise ValueError(f"Unsupported MXFP rounding mode {mode!r}")
    return mode


_MXFP4_SCALE_ALIASES = {
    "standard": "standard",
    "ocp": "standard",
    "0": "standard",
    "tfs": "tfs",
    "1": "tfs",
    "cx": "cx",
    "2": "cx",
}


def normalize_mxfp4_scale_alg(scale_alg: str) -> str:
    """Normalize the MXFP4 shared-scale algorithm name.

    ``standard`` preserves the existing OCP-style scale. ``tfs`` and ``cx``
    are the two optimized scale choices used by the MXFP4 experiments.
    Numeric aliases match the corresponding hardware ``scale_alg`` values.
    """
    key = str(scale_alg).strip().lower()
    try:
        return _MXFP4_SCALE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported MXFP4 scale algorithm {scale_alg!r}; expected standard, tfs, or cx") from exc


def _mxfp4_shared_scale(amax: torch.Tensor, scale_alg: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the shared exponent and real scale factor for each 32-value block."""
    scale_alg = normalize_mxfp4_scale_alg(scale_alg)
    safe_amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    if scale_alg == "standard":
        shared_exp = torch.floor(torch.log2(safe_amax)) - MXFP4_EMAX
    elif scale_alg == "tfs":
        shared_exp = torch.ceil(torch.log2(safe_amax / MXFP4_MAX_NORM))
    else:  # CX uses a 7.25 effective maximum and therefore a non-power-of-two scale.
        shared_exp = torch.ceil(torch.log2(safe_amax))
    shared_exp = shared_exp.clamp(min=-MXFP8_SCALE_EMAX)
    shared_exp = torch.where(amax == 0, torch.full_like(shared_exp, -MXFP8_SCALE_EMAX), shared_exp)
    shared_exp = torch.where(shared_exp > MXFP8_SCALE_EMAX, torch.full_like(shared_exp, float("nan")), shared_exp)
    scale_factor = torch.pow(2.0, shared_exp)
    if scale_alg == "cx":
        scale_factor = scale_factor / MXFP4_CX_MAX_NORM
    return shared_exp, scale_factor


def _check_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"MXFP quantization expects a 2D tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] % MXFP_GROUP_SIZE:
        raise ValueError(
            f"MXFP quantization requires the last dimension divisible by {MXFP_GROUP_SIZE}, "
            f"got shape={tuple(tensor.shape)}"
        )


def _round_abs(values: torch.Tensor, mode: str) -> torch.Tensor:
    mode = normalize_rounding_mode(mode)
    if mode == "rint":
        return torch.round(values)
    if mode == "round":
        return torch.floor(values + 0.5)
    floor = torch.floor(values)
    fraction = values - floor
    if mode == "random":
        random = torch.rand_like(fraction)
    else:
        bits = values.contiguous().view(torch.int32).detach().to(torch.int64)
        hashed = torch.remainder(torch.remainder(bits, 2**32) * 1664525 + 1013904223, 2**32)
        random = torch.div(hashed, 2**8, rounding_mode="floor").to(torch.float32) * (1.0 / 2**24)
    return floor + (random < fraction).to(floor.dtype)


def _quantize_mxfp8_torch(tensor: torch.Tensor, rounding_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch MXFP8 quantizer copied from the established verl implementation."""
    _check_tensor(tensor)
    values = tensor.to(torch.float32)
    blocks = values.reshape(values.shape[0], -1, MXFP_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1)
    safe_amax = torch.where(amax == 0, torch.full_like(amax, torch.finfo(torch.float32).tiny), amax)
    shared_exp = torch.floor(torch.log2(safe_amax)) - MXFP8_EMAX
    shared_exp = torch.where(shared_exp > MXFP8_SCALE_EMAX, torch.full_like(shared_exp, float("nan")), shared_exp)
    normalized = blocks / torch.pow(2.0, shared_exp.unsqueeze(-1))
    private_exp = torch.floor(torch.log2(normalized.abs() + (normalized == 0).float())).clamp(
        min=MXFP8_MIN_PRIVATE_EXP
    )
    private_scale = torch.pow(2.0, private_exp)
    scaled = normalized / private_scale * MXFP8_MANTISSA_SCALE
    quantized = torch.sign(scaled) * _round_abs(scaled.abs(), rounding_mode)
    quantized = (quantized / MXFP8_MANTISSA_SCALE * private_scale).clamp(
        -torch.finfo(torch.float8_e4m3fn).max, torch.finfo(torch.float8_e4m3fn).max
    )
    quantized = torch.where(torch.isinf(normalized), normalized, quantized)
    quantized = torch.where(torch.isnan(normalized), normalized, quantized)
    payload = quantized.reshape_as(values).to(torch.float8_e4m3fn)
    scale = torch.clamp(torch.nan_to_num(shared_exp, nan=-127.0) + 127.0, 0, 255).round().to(torch.uint8)
    return payload, scale


def _quantize_mxfp8_torch_a2a3(tensor: torch.Tensor, rounding_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 fake-payload path for Ascend devices without float8 device casts."""
    _check_tensor(tensor)
    values = tensor.to(torch.float32)
    blocks = values.reshape(values.shape[0], -1, MXFP_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1)
    safe_amax = torch.where(amax == 0, torch.full_like(amax, torch.finfo(torch.float32).tiny), amax)
    shared_exp = torch.floor(torch.log2(safe_amax)) - MXFP8_EMAX
    shared_exp = torch.where(shared_exp > MXFP8_SCALE_EMAX, torch.full_like(shared_exp, float("nan")), shared_exp)
    normalized = blocks / torch.pow(2.0, shared_exp.unsqueeze(-1))
    private_exp = torch.floor(torch.log2(normalized.abs() + (normalized == 0).float())).clamp(
        min=MXFP8_MIN_PRIVATE_EXP
    )
    private_scale = torch.pow(2.0, private_exp)
    scaled = normalized / private_scale * MXFP8_MANTISSA_SCALE
    quantized = torch.sign(scaled) * _round_abs(scaled.abs(), rounding_mode)
    quantized = (quantized / MXFP8_MANTISSA_SCALE * private_scale).clamp(
        -torch.finfo(torch.float8_e4m3fn).max, torch.finfo(torch.float8_e4m3fn).max
    )
    quantized = torch.where(torch.isinf(normalized), normalized, quantized)
    quantized = torch.where(torch.isnan(normalized), normalized, quantized)
    payload = quantized.reshape_as(values)
    scale = torch.clamp(torch.nan_to_num(shared_exp, nan=-127.0) + 127.0, 0, 255).round()
    return payload, scale


def _quantize_mxfp8_npu(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    payload, scale = torch_npu.npu_dynamic_mx_quant(tensor, axis=-1, dst_type=torch_npu.float8_e4m3fn)
    return payload, scale.flatten(-2, -1).squeeze(-1)


def quantize_mxfp8(
    tensor: torch.Tensor, *, quant_backend: str = "torch", rounding_mode: str = "rint"
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_tensor(tensor)
    backend = normalize_quant_backend(quant_backend)
    rounding_mode = normalize_rounding_mode(rounding_mode)
    if backend == "npu":
        if rounding_mode in {"random", "hash"}:
            raise ValueError("MXFP8 stochastic rounding requires quant_backend='torch'")
        return _quantize_mxfp8_npu(tensor)
    if tensor.device.type == "npu":
        try:
            import torch_npu

            if int(torch_npu.npu.get_soc_version()) != 260:
                return _quantize_mxfp8_torch_a2a3(tensor, rounding_mode)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return _quantize_mxfp8_torch_a2a3(tensor, rounding_mode)
    return _quantize_mxfp8_torch(tensor, rounding_mode)


def quantize_mxfp4(
    tensor: torch.Tensor,
    *,
    quant_backend: str = "torch",
    rounding_mode: str = "rint",
    scale_alg: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_tensor(tensor)
    if normalize_quant_backend(quant_backend) != "torch":
        raise ValueError("MXFP4 QAT requires quant_backend='torch'")
    blocks = tensor.to(torch.float32).reshape(tensor.shape[0], -1, MXFP_GROUP_SIZE)
    amax = blocks.abs().amax(dim=-1)
    shared_exp, scale_factor = _mxfp4_shared_scale(amax, scale_alg)
    normalized = blocks / scale_factor.unsqueeze(-1)
    private_exp = torch.floor(torch.log2(normalized.abs() + (normalized == 0).float())).clamp(
        min=MXFP4_MIN_PRIVATE_EXP
    )
    private_scale = torch.pow(2.0, private_exp)
    scaled = normalized / private_scale * MXFP4_MANTISSA_SCALE
    quantized = torch.sign(scaled) * _round_abs(scaled.abs(), rounding_mode)
    quantized = (quantized / MXFP4_MANTISSA_SCALE * private_scale).clamp(-MXFP4_MAX_NORM, MXFP4_MAX_NORM)
    scale = torch.where(torch.isnan(shared_exp), torch.full_like(shared_exp, 255.0), shared_exp + MXFP8_SCALE_EMAX)
    return quantized.reshape_as(tensor), scale


def dequantize_mxfp8(payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    values = payload.to(torch.float32)
    blocks = values.reshape(*values.shape[:-1], -1, MXFP_GROUP_SIZE)
    scale_shape = blocks.shape[:-1]
    descale = torch.exp2(scale.to(torch.float32).reshape(scale_shape) - 127.0)
    return (blocks * descale.unsqueeze(-1)).reshape_as(payload).to(dtype)


def dequantize_mxfp4(
    payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, *, scale_alg: str = "standard"
) -> torch.Tensor:
    scale_alg = normalize_mxfp4_scale_alg(scale_alg)
    scale = scale.to(torch.float32)
    scale = torch.where(scale == 255, torch.full_like(scale, float("nan")), scale)
    result = dequantize_mxfp8(payload, scale, torch.float32)
    if scale_alg == "cx":
        result = result / MXFP4_CX_MAX_NORM
    return result.to(dtype)


def fake_quantize(
    tensor: torch.Tensor,
    fmt: str,
    *,
    quant_backend: str = "torch",
    rounding_mode: str = "rint",
    mxfp4_scale_alg: str = "standard",
) -> torch.Tensor:
    """Return the QDQ value; callers provide the STE boundary."""
    warn_once(
        ("qdq", fmt, quant_backend, rounding_mode, mxfp4_scale_alg),
        "MXFP QDQ executed: format=%s backend=%s rounding=%s mxfp4_scale_alg=%s",
        fmt,
        quant_backend,
        rounding_mode,
        mxfp4_scale_alg,
    )
    original_shape = tensor.shape
    padded = (-tensor.shape[-1]) % MXFP_GROUP_SIZE
    if padded:
        tensor = torch.nn.functional.pad(tensor, (0, padded))
    matrix = tensor.reshape(-1, tensor.shape[-1])
    if fmt == "mxfp4":
        payload, scale = quantize_mxfp4(
            matrix,
            quant_backend=quant_backend,
            rounding_mode=rounding_mode,
            scale_alg=mxfp4_scale_alg,
        )
        result = dequantize_mxfp4(payload, scale, tensor.dtype, scale_alg=mxfp4_scale_alg)
    elif fmt == "mxfp8":
        payload, scale = quantize_mxfp8(matrix, quant_backend=quant_backend, rounding_mode=rounding_mode)
        result = dequantize_mxfp8(payload, scale, tensor.dtype)
    else:
        raise ValueError(f"Unsupported MXFP format {fmt!r}")
    result = result.reshape_as(tensor)
    if padded:
        result = result[..., :-padded]
    return result.reshape(original_shape)
