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

"""Block-wise MXFP8 rotation helpers.

The first implementation uses one global 32x32 orthogonal block, built from a
normalized Hadamard matrix with deterministic random sign flips. The same block
is repeated along the hidden dimension, matching the MXFP8 group size.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

MXFP8_ROTATION_ENABLE_KEY = "mxfp8_rotation_enable"
MXFP8_ROTATION_KIND_KEY = "mxfp8_rotation_kind"
MXFP8_ROTATION_BLOCK_SIZE_KEY = "mxfp8_rotation_block_size"
MXFP8_ROTATION_SEED_KEY = "mxfp8_rotation_seed"
MXFP8_ROTATION_TARGETS_KEY = "mxfp8_rotation_targets"

MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN = "block_hadamard_sign"
MXFP8_ROTATION_TARGET_FPROP = "fprop"
MXFP8_ROTATION_TARGET_DGRAD = "dgrad"
MXFP8_ROTATION_TARGET_WGRAD = "wgrad"
MXFP8_ROTATION_TARGET_ORDER = (
    MXFP8_ROTATION_TARGET_FPROP,
    MXFP8_ROTATION_TARGET_DGRAD,
    MXFP8_ROTATION_TARGET_WGRAD,
)
_MXFP8_ROTATION_TARGET_ALIASES = {
    "fprop": MXFP8_ROTATION_TARGET_FPROP,
    "forward": MXFP8_ROTATION_TARGET_FPROP,
    "dgrad": MXFP8_ROTATION_TARGET_DGRAD,
    "wgrad": MXFP8_ROTATION_TARGET_WGRAD,
}
_DEFAULT_ROTATION_TARGETS = (MXFP8_ROTATION_TARGET_FPROP,)
_MXFP8_ROTATION_KIND_ALIASES = {
    MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN: MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
    "hadamard_sign": MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
    "block_hadamard": MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
}
_DEFAULT_BLOCK_SIZE = 32
_DEFAULT_SEED = 0
_MAX_TORCH_SEED = 2**63 - 1
_ROTATION_MATRIX_CACHE: dict[tuple[str, int, int, str, str], torch.Tensor] = {}
_ROTATION_LOGGED: set[tuple[str, int, int, str, str]] = set()


@dataclass(frozen=True)
class MXFP8RotationConfig:
    enable: bool = False
    kind: str = MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN
    block_size: int = _DEFAULT_BLOCK_SIZE
    seed: int = _DEFAULT_SEED
    targets: tuple[str, ...] = _DEFAULT_ROTATION_TARGETS


def normalize_mxfp8_rotation_kind(kind: str) -> str:
    normalized = str(kind).lower()
    if normalized not in _MXFP8_ROTATION_KIND_ALIASES:
        raise ValueError(
            f"Unsupported MXFP8 rotation kind: {kind}. "
            f"Supported kinds: {sorted(_MXFP8_ROTATION_KIND_ALIASES)}"
        )
    return _MXFP8_ROTATION_KIND_ALIASES[normalized]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_mxfp8_rotation_targets(targets: Any) -> tuple[str, ...]:
    if targets is None:
        return _DEFAULT_ROTATION_TARGETS
    if isinstance(targets, str):
        raw_targets = targets.strip()
        if raw_targets.startswith("["):
            try:
                targets = json.loads(raw_targets)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid MXFP8 rotation targets JSON: {targets}") from exc
        else:
            targets = [item.strip() for item in raw_targets.split(",") if item.strip()]
    if not isinstance(targets, (list, tuple, set)):
        if isinstance(targets, Iterable) and not isinstance(targets, Mapping):
            targets = list(targets)
        else:
            raise TypeError(
                f"MXFP8 rotation targets must be a list or comma-separated string, "
                f"got {type(targets).__name__}"
            )

    normalized = set()
    for target in targets:
        normalized_target = _MXFP8_ROTATION_TARGET_ALIASES.get(str(target).strip().lower())
        if normalized_target is None:
            raise ValueError(
                f"Unsupported MXFP8 rotation target: {target}. "
                f"Supported targets: {list(MXFP8_ROTATION_TARGET_ORDER)}"
            )
        normalized.add(normalized_target)
    return tuple(target for target in MXFP8_ROTATION_TARGET_ORDER if target in normalized)


def get_mxfp8_rotation_config(config: Mapping[str, Any] | None) -> MXFP8RotationConfig:
    if config is None:
        return MXFP8RotationConfig()

    enable = _coerce_bool(config.get(MXFP8_ROTATION_ENABLE_KEY, False))
    kind = normalize_mxfp8_rotation_kind(
        config.get(MXFP8_ROTATION_KIND_KEY, MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN)
    )
    return MXFP8RotationConfig(
        enable=enable,
        kind=kind,
        block_size=int(config.get(MXFP8_ROTATION_BLOCK_SIZE_KEY, _DEFAULT_BLOCK_SIZE)),
        seed=int(config.get(MXFP8_ROTATION_SEED_KEY, _DEFAULT_SEED)),
        targets=normalize_mxfp8_rotation_targets(config.get(MXFP8_ROTATION_TARGETS_KEY)),
    )


def mxfp8_rotation_config_to_dict(config: MXFP8RotationConfig) -> dict[str, Any]:
    return {
        MXFP8_ROTATION_ENABLE_KEY: bool(config.enable),
        MXFP8_ROTATION_KIND_KEY: normalize_mxfp8_rotation_kind(config.kind),
        MXFP8_ROTATION_BLOCK_SIZE_KEY: int(config.block_size),
        MXFP8_ROTATION_SEED_KEY: int(config.seed),
        MXFP8_ROTATION_TARGETS_KEY: list(normalize_mxfp8_rotation_targets(config.targets)),
    }


def validate_mxfp8_rotation_config(config: MXFP8RotationConfig, group_size: int = _DEFAULT_BLOCK_SIZE):
    if not config.enable:
        return
    normalize_mxfp8_rotation_kind(config.kind)
    if config.block_size != group_size:
        raise ValueError(
            f"MXFP8 rotation block_size must match group_size={group_size}, got: {config.block_size}"
        )
    if config.block_size <= 0 or config.block_size & (config.block_size - 1):
        raise ValueError(f"MXFP8 Hadamard rotation requires a power-of-two block_size, got: {config.block_size}")
    targets = normalize_mxfp8_rotation_targets(config.targets)
    if not targets:
        raise ValueError("MXFP8 rotation requires at least one target when rotation is enabled")


def is_mxfp8_rotation_target(config: MXFP8RotationConfig | Mapping[str, Any], target: str) -> bool:
    if isinstance(config, Mapping):
        config = get_mxfp8_rotation_config(config)
    normalized_target = _MXFP8_ROTATION_TARGET_ALIASES.get(str(target).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported MXFP8 rotation target: {target}. "
            f"Supported targets: {list(MXFP8_ROTATION_TARGET_ORDER)}"
        )
    return bool(config.enable and normalized_target in normalize_mxfp8_rotation_targets(config.targets))


def _normalize_seed(seed: int) -> int:
    return int(seed) % _MAX_TORCH_SEED


def _build_hadamard_matrix(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got: {size}")

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix * (1.0 / math.sqrt(size))


def _build_block_rotation_matrix(config: MXFP8RotationConfig) -> torch.Tensor:
    if config.kind != MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN:
        raise ValueError(f"Unsupported MXFP8 rotation kind: {config.kind}")

    matrix = _build_hadamard_matrix(config.block_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_normalize_seed(config.seed))
    signs = torch.randint(0, 2, (config.block_size,), generator=generator, dtype=torch.int8)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    return matrix * signs


def _get_block_rotation_matrix(
    config: MXFP8RotationConfig, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    cache_key = (
        normalize_mxfp8_rotation_kind(config.kind),
        int(config.block_size),
        _normalize_seed(config.seed),
        str(device),
        str(dtype),
    )
    cached = _ROTATION_MATRIX_CACHE.get(cache_key)
    if cached is None:
        cached = _build_block_rotation_matrix(config).to(device=device, dtype=dtype)
        _ROTATION_MATRIX_CACHE[cache_key] = cached
    return cached


def _rotation_work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def apply_mxfp8_block_rotation(
    tensor: torch.Tensor,
    config: MXFP8RotationConfig | Mapping[str, Any],
    *,
    transpose: bool = False,
    pad_to_block: bool = False,
) -> torch.Tensor:
    """Apply repeated block rotation on the tensor's last dimension.

    For row-vector activations and ``nn.Linear`` weights shaped
    ``[out_features, in_features]``, the default is right multiplication by
    ``Q``. Setting ``transpose=True`` applies ``Q.T`` and is kept for future
    inverse-rotation experiments.
    """

    if isinstance(config, Mapping):
        config = get_mxfp8_rotation_config(config)
    if not config.enable:
        return tensor

    validate_mxfp8_rotation_config(config)
    original_shape = tensor.shape
    last_dim = tensor.shape[-1]
    padding = (-last_dim) % config.block_size
    if padding and not pad_to_block:
        raise ValueError(
            f"MXFP8 block rotation requires last_dim divisible by block_size={config.block_size}, "
            f"got shape={tuple(tensor.shape)}"
        )
    if padding:
        tensor = torch.nn.functional.pad(tensor, (0, padding))
        last_dim = tensor.shape[-1]

    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dtype is not None
    )
    if tensor.dtype in fp8_dtypes:
        raise ValueError("MXFP8 block rotation must run before FP8 quantization")

    original_dtype = tensor.dtype
    work_dtype = _rotation_work_dtype(original_dtype)
    matrix = _get_block_rotation_matrix(config, device=tensor.device, dtype=work_dtype)
    if transpose:
        matrix = matrix.t()
    log_key = (
        normalize_mxfp8_rotation_kind(config.kind),
        int(config.block_size),
        _normalize_seed(config.seed),
        str(tensor.device),
        str(work_dtype),
    )
    if log_key not in _ROTATION_LOGGED:
        _ROTATION_LOGGED.add(log_key)
        logger.warning(
            "MXFP8 rotation matrix applied: kind=%s, block_size=%s, seed=%s, transpose=%s, "
            "device=%s, work_dtype=%s, tensor_shape=%s",
            config.kind,
            config.block_size,
            config.seed,
            transpose,
            tensor.device,
            work_dtype,
            tuple(tensor.shape),
        )

    tensor_2d = tensor.reshape(-1, last_dim).to(work_dtype)
    tensor_blocked = tensor_2d.reshape(tensor_2d.shape[0], last_dim // config.block_size, config.block_size)
    rotated = torch.matmul(tensor_blocked, matrix)
    rotated = rotated.reshape(tensor_2d.shape).to(original_dtype).reshape(tensor.shape)
    if padding:
        rotated = rotated[..., : original_shape[-1]]
    return rotated.reshape(original_shape)


def is_mxfp8_rotation_enabled(config: MXFP8RotationConfig | Mapping[str, Any] | None) -> bool:
    if config is None:
        return False
    if isinstance(config, Mapping):
        config = get_mxfp8_rotation_config(config)
    return bool(config.enable)
