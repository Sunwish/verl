# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Deterministic, block-local orthogonal transforms for MXFP QAT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from verl.utils.qat.mxfp_debug import warn_once

ROTATION_TARGETS = ("fprop", "wgrad", "dgrad")
_TARGET_ALIASES = {
    "fprop": "fprop",
    "forward": "fprop",
    "wgrad": "wgrad",
    "dgrad": "dgrad",
}
_MATRIX_CACHE: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}


def normalize_targets(targets: Any) -> tuple[str, ...]:
    if targets is None:
        return ("fprop",)
    if isinstance(targets, str):
        targets = [part.strip() for part in targets.split(",") if part.strip()]
    if not isinstance(targets, (list, tuple, set)):
        raise TypeError("MXFP targets must be a list or comma-separated string")
    normalized = {_TARGET_ALIASES.get(str(target).lower()) for target in targets}
    if None in normalized:
        raise ValueError(f"Unsupported MXFP target in {targets!r}; expected {ROTATION_TARGETS}")
    return tuple(target for target in ROTATION_TARGETS if target in normalized)


def _check_block_size(block_size: int) -> None:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"MXFP rotation block_size must be a positive power of two, got {block_size}")


@dataclass(frozen=True)
class RotationConfig:
    enable: bool = False
    targets: tuple[str, ...] = ("fprop",)
    block_size: int = 32
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "targets", normalize_targets(self.targets))
        if self.enable:
            _check_block_size(self.block_size)
            if not self.targets:
                raise ValueError("MXFP rotation requires at least one target")


def _matrix(block_size: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (block_size, int(seed), str(device), dtype)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    hadamard = torch.ones((1, 1), dtype=torch.float32)
    while hadamard.shape[0] < block_size:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1), torch.cat((hadamard, -hadamard), dim=1)), dim=0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    signs = torch.randint(0, 2, (block_size,), generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
    cached = (hadamard * signs / math.sqrt(block_size)).to(device=device, dtype=dtype)
    _MATRIX_CACHE[key] = cached
    return cached


def rotate_blocks(
    tensor: torch.Tensor, config: RotationConfig, *, transpose: bool = False, pad_to_block: bool = False
) -> torch.Tensor:
    """Rotate the last dimension in fixed-size blocks without forming a block diagonal matrix.

    ``pad_to_block`` keeps the zero-padded tail in the result for reductions whose
    transformed operands must share complete orthogonal blocks.
    """
    if not config.enable:
        return tensor
    block_size = config.block_size
    padding = (-tensor.shape[-1]) % block_size
    if padding:
        tensor = F.pad(tensor, (0, padding))
    work_dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    matrix = _matrix(block_size, config.seed, tensor.device, work_dtype)
    warn_once(
        ("rotation", block_size, config.seed, str(tensor.device), work_dtype),
        "MXFP block rotation executed: block_size=%s seed=%s device=%s dtype=%s transpose=%s",
        block_size,
        config.seed,
        tensor.device,
        work_dtype,
        transpose,
    )
    if transpose:
        matrix = matrix.transpose(-1, -2)
    blocks = tensor.to(work_dtype).reshape(*tensor.shape[:-1], -1, block_size)
    result = torch.matmul(blocks, matrix).reshape_as(tensor).to(tensor.dtype)
    return result if pad_to_block or not padding else result[..., :-padding]
