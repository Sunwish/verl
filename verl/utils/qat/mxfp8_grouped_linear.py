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

"""MXFP QAT backward for packed expert GMMs, using the dense operand transforms."""

import logging
from dataclasses import dataclass

import torch

from verl.utils.qat.mxfp8_linear import _transform_gradient_operand
from verl.utils.qat.mxfp8_rotation import MXFP8RotationConfig, is_mxfp8_rotation_target

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupedLinearConfig:
    """The same gradient operand settings used by MXFP8QATLinear."""

    rotation_config: MXFP8RotationConfig
    fake_quant_targets: tuple[str, ...] = ()
    quant_backend: str | None = None
    rounding_mode: str = "rint"
    quantize_fn: object = None
    weight_quantize_fn: object = None
    dequantize_fn: object = None
    skip_dgrad_weight_fake_quant: bool = False
    skip_wgrad_input_fake_quant: bool = False

    def enabled(self, target):
        return target in self.fake_quant_targets or is_mxfp8_rotation_target(self.rotation_config, target)

    def transform(self, tensor, target, *, weight=False, skip_fake_quant=False):
        return _transform_gradient_operand(
            tensor,
            self.rotation_config,
            target,
            self.fake_quant_targets,
            self.quant_backend,
            self.rounding_mode,
            skip_fake_quant=skip_fake_quant,
            quantize_fn=self.weight_quantize_fn if weight else self.quantize_fn,
            dequantize_fn=self.dequantize_fn,
        )


def _pad_expert_rows(input, grad_output, group_list, block_size=32):
    """Align each expert's token segment, without copying counts to the CPU."""
    counts = group_list.to(torch.int64)
    padded_counts = torch.div(counts + block_size - 1, block_size, rounding_mode="floor") * block_size
    positions = torch.arange(input.shape[0], device=input.device)
    expert_ids = torch.searchsorted(counts.cumsum(0), positions, right=True)
    padding = padded_counts - counts
    padding_before = padding.cumsum(0) - padding
    destinations = positions + padding_before.index_select(0, expert_ids)

    # The exact padded size is device data. Allocate its static upper bound and
    # include its zero-only tail in the last group; all expert boundaries stay
    # aligned without adding a dummy expert to GMM.
    capacity = ((input.shape[0] + (block_size - 1) * counts.numel() + block_size - 1) // block_size) * block_size
    padded_input = input.new_zeros((capacity, input.shape[1])).index_copy_(0, destinations, input)
    padded_grad = grad_output.new_zeros((capacity, grad_output.shape[1])).index_copy_(0, destinations, grad_output)
    padded_counts[-1] += capacity - padded_counts.sum()
    return padded_input, padded_grad, padded_counts.to(group_list.dtype)


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group_list, dgrad_weight_source, wgrad_input_source, config, grouped_matmul):
        # Canonical weights are [experts, output, input], just like nn.Linear.
        ctx.save_for_backward(input, weight, group_list, dgrad_weight_source, wgrad_input_source)
        ctx.config = config
        ctx.grouped_matmul = grouped_matmul
        return grouped_matmul(
            [input],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, group_list, dgrad_weight_source, wgrad_input_source = ctx.saved_tensors
        config = ctx.config
        grouped_matmul = ctx.grouped_matmul

        dgrad_output = config.transform(grad_output, "dgrad")
        dgrad_weight = config.transform(
            (weight if dgrad_weight_source is None else dgrad_weight_source).transpose(1, 2),
            "dgrad",
            weight=True,
            skip_fake_quant=config.skip_dgrad_weight_fake_quant,
        ).transpose(1, 2)
        grad_input = grouped_matmul(
            [dgrad_output],
            [dgrad_weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        if config.enabled("wgrad"):
            # Wgrad reduces over tokens, independently for each expert. A block
            # must never mix the last tokens of one expert with the next one.
            source = input if wgrad_input_source is None else wgrad_input_source
            padded_input, padded_grad, wgrad_groups = _pad_expert_rows(source, grad_output, group_list)
            wgrad_input_t = config.transform(
                padded_input.transpose(0, 1),
                "wgrad",
                skip_fake_quant=config.skip_wgrad_input_fake_quant,
            )
            wgrad_output = config.transform(padded_grad.transpose(0, 1), "wgrad").transpose(0, 1)
        else:
            wgrad_input_t = input.transpose(0, 1)
            wgrad_output = grad_output
            wgrad_groups = group_list
        grad_weight = grouped_matmul(
            [wgrad_input_t],
            [wgrad_output],
            bias=None,
            group_list=wgrad_groups,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0][: weight.shape[0]].transpose(1, 2)

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
            and not getattr(_GroupedLinear, "_backward_logged", False)
        ):
            _GroupedLinear._backward_logged = True
            logger.warning(
                "MXFP QAT grouped backward executed: fake_quant_targets=%s, rotation_enable=%s, "
                "rotation_targets=%s, wgrad_per_expert_padding=%s",
                config.fake_quant_targets,
                config.rotation_config.enable,
                config.rotation_config.targets,
                config.enabled("wgrad"),
            )
        return grad_input, grad_weight, None, None, None, None, None


def grouped_linear_with_gradient_rotation(
    input, weight, group_list, config, grouped_matmul, *, dgrad_weight_source=None, wgrad_input_source=None
):
    """Run a packed linear GMM with MXFP transforms on its backward operands."""
    return _GroupedLinear.apply(
        input, weight, group_list, dgrad_weight_source, wgrad_input_source, config, grouped_matmul
    )
