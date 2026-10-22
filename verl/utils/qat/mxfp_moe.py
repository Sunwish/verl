# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""MXFP QAT for packed MoE grouped matmul.

Qwen3's Ascend path bypasses expert submodules and feeds packed parameters
directly to ``npu_grouped_matmul``.  This module therefore applies QDQ at that
operator boundary instead of replacing expert ``Linear`` modules.
"""

from __future__ import annotations

import torch

from verl.utils.qat.mxfp_debug import warn_once
from verl.utils.qat.mxfp_linear import MXFPQATConfig
from verl.utils.qat.mxfp_quant import fake_quantize
from verl.utils.qat.mxfp_rotation import rotate_blocks


def _enabled(config: MXFPQATConfig, target: str) -> bool:
    return target in config.fake_quant_targets or (config.rotation.enable and target in config.rotation.targets)


def _transform_feature(tensor: torch.Tensor, config: MXFPQATConfig, target: str, fmt: str) -> torch.Tensor:
    """Transform the last feature dimension using the target's QDQ/rotation."""
    result = tensor
    if config.rotation.enable and target in config.rotation.targets:
        result = rotate_blocks(result, config.rotation)
    if target in config.fake_quant_targets:
        quantized = fake_quantize(
            result,
            fmt,
            quant_backend=config.quant_backend,
            rounding_mode=config.rounding_mode,
        )
        result = result + (quantized - result).detach()
    return result


def _transform_weight(
    weight: torch.Tensor, config: MXFPQATConfig, target: str, fmt: str, reduction_dim: int
) -> torch.Tensor:
    """Apply a feature transform along an arbitrary grouped-matmul axis."""
    moved = weight.movedim(reduction_dim, -1)
    moved = _transform_feature(moved, config, target, fmt)
    return moved.movedim(-1, reduction_dim).contiguous()


def _transform_tokens(tensor: torch.Tensor, config: MXFPQATConfig, target: str, fmt: str) -> torch.Tensor:
    """Transform the token reduction axis used by grouped-matmul wgrad."""
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D token matrix, got {tuple(tensor.shape)}")
    return _transform_feature(tensor.transpose(0, 1), config, target, fmt).transpose(0, 1).contiguous()


def _inverse_input_rotation(tensor: torch.Tensor, config: MXFPQATConfig, feature_dim: int) -> torch.Tensor:
    if not (config.rotation.enable and "fprop" in config.rotation.targets):
        return tensor
    moved = tensor.movedim(feature_dim, -1)
    moved = rotate_blocks(moved, config.rotation, transpose=True)
    return moved.movedim(-1, feature_dim).contiguous()


def prepare_mxfp_fprop_input(x: torch.Tensor, config: MXFPQATConfig) -> torch.Tensor:
    """Prepare a detached activation so gate/up can share fprop QDQ."""
    with torch.no_grad():
        return _transform_feature(x, config, "fprop", config.activation_format).detach()


def _call_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    *,
    split_item: int = 2,
    group_type: int = 0,
):
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        bias=None,
        group_list=group_list,
        split_item=split_item,
        group_type=group_type,
        group_list_type=group_list_type,
    )[0]


class _MXFPGroupedMatmul(torch.autograd.Function):
    """Grouped matmul whose three gradient paths use independent operands."""

    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type, config, name, fprop_x, fprop_weight):
        ctx.config = config
        ctx.group_list = group_list
        ctx.group_list_type = int(group_list_type)
        ctx.name = name
        need_input_grad, need_weight_grad = ctx.needs_input_grad[:2]

        x_fprop = fprop_x if fprop_x is not None else _transform_feature(x, config, "fprop", config.activation_format)
        weight_fprop = (
            fprop_weight
            if fprop_weight is not None
            else _transform_weight(weight, config, "fprop", config.weight_format, reduction_dim=1)
        )
        # MoE expert tensors are large. Retain only operands needed by a disabled
        # backward transform; enabled dgrad/wgrad operands are recomputed when
        # their gradient runs. save_for_backward lets checkpoint hooks release
        # retained tensors between forward and backward.
        ctx.save_for_backward(
            x if need_weight_grad else None,
            weight if need_input_grad else None,
            x_fprop.detach() if need_weight_grad and not _enabled(config, "wgrad") else None,
            weight_fprop.detach() if need_input_grad and not _enabled(config, "dgrad") else None,
        )
        warn_once(
            (
                "moe-fprop",
                name,
                config.mode,
                config.quant_backend,
                config.fake_quant_targets,
                config.rotation.targets,
            ),
            "moe %s fprop executed: fake_quant=%s rotation=%s reduction_axis=feature",
            name,
            "fprop" in config.fake_quant_targets,
            config.rotation.enable and "fprop" in config.rotation.targets,
        )
        return _call_grouped_matmul(x_fprop, weight_fprop, group_list, ctx.group_list_type)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, x_fprop, weight_fprop = ctx.saved_tensors
        config = ctx.config
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type
        need_input_grad, need_weight_grad = ctx.needs_input_grad[:2]

        if need_input_grad and _enabled(config, "dgrad"):
            warn_once(
                (
                    "moe-dgrad",
                    ctx.name,
                    config.mode,
                    config.quant_backend,
                    config.fake_quant_targets,
                    config.rotation.targets,
                ),
                "moe %s dgrad executed: fake_quant=%s rotation=%s reduction_axis=output operand=recomputed",
                ctx.name,
                "dgrad" in config.fake_quant_targets,
                config.rotation.enable and "dgrad" in config.rotation.targets,
            )
            grad_output_dgrad = _transform_feature(
                grad_output, config, "dgrad", config.activation_format
            ).contiguous()
            weight_dgrad = _transform_weight(weight, config, "dgrad", config.weight_format, reduction_dim=2)
        elif need_input_grad:
            grad_output_dgrad = grad_output
            weight_dgrad = weight_fprop

        if need_input_grad:
            dx = _call_grouped_matmul(
                grad_output_dgrad,
                weight_dgrad.transpose(1, 2).contiguous(),
                group_list,
                group_list_type,
            )
            dx = _inverse_input_rotation(dx, config, feature_dim=-1)
        else:
            dx = None

        if need_weight_grad and _enabled(config, "wgrad"):
            warn_once(
                (
                    "moe-wgrad",
                    ctx.name,
                    config.mode,
                    config.quant_backend,
                    config.fake_quant_targets,
                    config.rotation.targets,
                ),
                "moe %s wgrad executed: fake_quant=%s rotation=%s reduction_axis=token operand=recomputed",
                ctx.name,
                "wgrad" in config.fake_quant_targets,
                config.rotation.enable and "wgrad" in config.rotation.targets,
            )
            x_wgrad = _transform_tokens(x, config, "wgrad", config.activation_format)
            grad_output_wgrad = _transform_tokens(grad_output, config, "wgrad", config.activation_format)
        elif need_weight_grad:
            x_wgrad = x_fprop
            grad_output_wgrad = grad_output

        if need_weight_grad:
            dw = _call_grouped_matmul(
                x_wgrad.transpose(0, 1),
                grad_output_wgrad.contiguous(),
                group_list,
                group_list_type,
                split_item=3,
                group_type=2,
            )
            dw = _inverse_input_rotation(dw, config, feature_dim=1)
        else:
            dw = None
        return dx, dw, None, None, None, None, None, None


def mxfp_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    config: MXFPQATConfig,
    *,
    name: str,
    group_list_type: int = 1,
    fprop_x: torch.Tensor | None = None,
    fprop_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Ascend grouped matmul with MXFP QAT enabled."""
    return _MXFPGroupedMatmul.apply(x, weight, group_list, group_list_type, config, name, fprop_x, fprop_weight)


def _is_moe_block(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__
    if class_name in {
        "Qwen3MoeSparseMoeBlock",
        "Qwen3NextSparseMoeBlock",
        "Qwen3_5MoeSparseMoeBlock",
    }:
        return True
    return hasattr(module, "experts") and hasattr(module, "gate") and not isinstance(module, torch.nn.Linear)


def attach_mxfp_expert_qat(model: torch.nn.Module, config: MXFPQATConfig) -> int:
    """Attach one immutable QAT config to every packed MoE block."""
    count = 0
    for module in model.modules():
        if _is_moe_block(module):
            module._mxfp_expert_qat_config = config
            count += 1
    return count


def get_mxfp_expert_qat_config(module: torch.nn.Module) -> MXFPQATConfig | None:
    return getattr(module, "_mxfp_expert_qat_config", None)
