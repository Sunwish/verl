# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
QAT (Quantization-Aware Training) module for verl.

Supports NVFP4 (W4A4 and W4A16) and MXFP8 (W8A16 and W8A8) quantization modes for FSDP training.

Module Structure:
- core.py: QATConfig, apply_qat, enable_qat_fuse (training setup)
- linear.py: NVFP4 QATLinear layer with Triton kernels for fake quantization
- mxfp8_linear.py: MXFP8 fake-quant linear layer for high-precision matmul training
- quantizer.py: QATQuantizer for true quantization + scale computation utilities
- vllm_patch.py: Patches for vLLM dynamic weight loading

Usage:
    from verl.utils.qat import apply_qat, QATConfig

    config = QATConfig(enable=True, mode="w4a16")
    model = apply_qat(model, config)  # Before FSDP wrapping
"""

from verl.utils.qat.core import (
    QATConfig,
    QATExpertsConfig,
    apply_qat,
    enable_qat_fuse,
    format_mxfp8_fallback_layers,
    get_effective_ignore_patterns,
    get_mxfp8_fallback_ignore_patterns,
    invalidate_all_scales,
    load_quantization_config,
    normalize_mxfp8_fallback_layers,
)
from verl.utils.qat.mxfp8_linear import (
    configure_mxfp8_probe,
    mxfp8_probe_step_context,
    normalize_mxfp8_fake_quant_targets,
    reset_mxfp8_probe,
    set_mxfp8_probe_step,
)
from verl.utils.qat.vllm_patch import (
    apply_qat_patches,
    manual_process_weights_after_loading,
    prepare_qat_for_load_weights,
)

__all__ = [
    # Core
    "QATConfig",
    "QATExpertsConfig",
    "apply_qat",
    "load_quantization_config",
    "normalize_mxfp8_fallback_layers",
    "format_mxfp8_fallback_layers",
    "get_mxfp8_fallback_ignore_patterns",
    "get_effective_ignore_patterns",
    "enable_qat_fuse",
    "invalidate_all_scales",
    "configure_mxfp8_probe",
    "mxfp8_probe_step_context",
    "normalize_mxfp8_fake_quant_targets",
    "reset_mxfp8_probe",
    "set_mxfp8_probe_step",
    # vLLM Patch
    "apply_qat_patches",
    "manual_process_weights_after_loading",
    "prepare_qat_for_load_weights",
]
