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

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from verl.base_config import BaseConfig
from verl.utils.qat.mxfp8_rotation import (
    MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
    MXFP8RotationConfig,
    normalize_mxfp8_rotation_kind,
    validate_mxfp8_rotation_config,
)

logger = logging.getLogger(__name__)

_MXFP8_MODES = {"w8a16_mxfp8", "w8a8_mxfp8"}
_MXFP8_ROUNDING_MODES = {"rint", "round", "random", "hash"}
_MXFP8_STOCHASTIC_ROUNDING_MODES = {"random", "hash"}
_MXFP8_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")
_EXPERT_LINEAR_RE = re.compile(r".*(?:experts\.[^.]+|shared_expert|shared_experts)\.(gate_proj|up_proj|down_proj)$")
_INTERNAL_ROUTER_IGNORE_PATTERNS = [
    r"re:.*mlp\.gate(?:\.|$)",
    r"re:.*router(?:\.|$)",
    r"re:.*shared_expert_gate(?:\.|$)",
]
_INTERNAL_EXPERT_DISABLE_IGNORE_PATTERNS = [
    r"re:.*mlp\.experts(?:\.|$)",
    r"re:.*experts\.[^.]+\.(gate_proj|up_proj|down_proj)(?:\.weight)?$",
    r"re:.*shared_expert(?:\.|$)",
    r"re:.*shared_expert\.(gate_proj|up_proj|down_proj)(?:\.weight)?$",
    r"re:.*shared_experts(?:\.|$)",
]


@dataclass
class QATExpertsConfig(BaseConfig):
    """Configuration for expert-layer QAT within a broader QAT setup."""

    enable: bool = False


def _coerce_qat_experts_config(value: Any) -> QATExpertsConfig:
    if value is None:
        return QATExpertsConfig()
    if isinstance(value, QATExpertsConfig):
        return value
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return QATExpertsConfig(**value)
    raise TypeError(f"qat.experts must be a dict, DictConfig, or QATExpertsConfig; got {type(value).__name__}")


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    experts: QATExpertsConfig = field(default_factory=QATExpertsConfig)
    mxfp8_quant_backend: str = "npu"
    mxfp8_rounding_mode: str = "rint"
    mxfp8_probe_quant_error: bool = False
    mxfp8_probe_quant_error_output_path: Optional[str] = None
    mxfp8_rotation_enable: bool = False
    mxfp8_rotation_kind: str = MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN
    mxfp8_rotation_block_size: int = 32
    mxfp8_rotation_seed: int = 0
    quantization_config_path: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "experts", _coerce_qat_experts_config(self.experts))
        mxfp8_rounding_mode = self.mxfp8_rounding_mode.lower()
        if mxfp8_rounding_mode not in _MXFP8_ROUNDING_MODES:
            raise ValueError(
                f"Unsupported MXFP8 rounding mode: {self.mxfp8_rounding_mode}. "
                f"Supported modes: {sorted(_MXFP8_ROUNDING_MODES)}"
            )
        if (
            self.enable
            and self.mode.lower() in _MXFP8_MODES
            and mxfp8_rounding_mode in _MXFP8_STOCHASTIC_ROUNDING_MODES
            and self.mxfp8_quant_backend.lower() != "torch"
        ):
            raise ValueError("MXFP8 stochastic rounding modes require mxfp8_quant_backend='torch'")
        if self.mxfp8_probe_quant_error:
            if not self.enable:
                raise ValueError("mxfp8_probe_quant_error requires QAT enable=True")
            if self.mode.lower() not in _MXFP8_MODES:
                raise ValueError("mxfp8_probe_quant_error only supports w8a16_mxfp8/w8a8_mxfp8 modes")
            if not self.mxfp8_probe_quant_error_output_path:
                raise ValueError("mxfp8_probe_quant_error_output_path is required when mxfp8_probe_quant_error=True")
        rotation_config = MXFP8RotationConfig(
            enable=self.mxfp8_rotation_enable,
            kind=normalize_mxfp8_rotation_kind(self.mxfp8_rotation_kind),
            block_size=self.mxfp8_rotation_block_size,
            seed=self.mxfp8_rotation_seed,
        )
        if rotation_config.enable:
            if self.mode.lower() not in _MXFP8_MODES:
                raise ValueError("mxfp8_rotation_enable only supports w8a16_mxfp8/w8a8_mxfp8 modes")
            validate_mxfp8_rotation_config(rotation_config, group_size=self.group_size)


def _matches_ignore_pattern(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return pattern in name


def _is_ignored(name: str, ignore_patterns: list[str]) -> bool:
    return any(_matches_ignore_pattern(name, pattern) for pattern in ignore_patterns)


def get_effective_ignore_patterns(qat_config: QATConfig) -> list[str]:
    """Return the ignore list actually enforced for training and rollout handoff."""
    effective_ignore = list(qat_config.ignore_patterns or [])
    for pattern in _INTERNAL_ROUTER_IGNORE_PATTERNS:
        if pattern not in effective_ignore:
            effective_ignore.append(pattern)
    if not qat_config.experts.enable:
        for pattern in _INTERNAL_EXPERT_DISABLE_IGNORE_PATTERNS:
            if pattern not in effective_ignore:
                effective_ignore.append(pattern)
    return effective_ignore


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    effective_ignore = get_effective_ignore_patterns(qat_config)
    if effective_ignore:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = effective_ignore
        if original_ignore != effective_ignore:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {effective_ignore}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _is_packed_expert_block(name: str, module: nn.Module) -> bool:
    if not name.endswith(".experts"):
        return False
    gate_up_proj = getattr(module, "gate_up_proj", None)
    down_proj = getattr(module, "down_proj", None)
    return bool(
        gate_up_proj is not None
        and down_proj is not None
        and getattr(gate_up_proj, "dim", lambda: 0)() == 3
        and getattr(down_proj, "dim", lambda: 0)() == 3
    )


def _packed_expert_weight_supports_group_size(
    weight,
    *,
    input_dim: Optional[int],
    output_dim: Optional[int],
    group_size: int,
) -> bool:
    if weight is None or weight.dim() != 3 or input_dim is None or output_dim is None:
        return False
    if input_dim % group_size != 0:
        return False
    shape = tuple(weight.shape)
    if shape[-1] == input_dim and shape[-2] == output_dim:
        return True
    if shape[-2] == input_dim and shape[-1] == output_dim:
        return True
    return False


def _get_packed_expert_quant_plan(
    name: str,
    module: nn.Module,
    config: QATConfig,
    effective_ignore_patterns: list[str],
) -> Optional[dict[str, bool]]:
    if not config.experts.enable or not _is_packed_expert_block(name, module):
        return None

    hidden_dim = getattr(module, "hidden_dim", getattr(module, "hidden_size", None))
    intermediate_dim = getattr(module, "intermediate_dim", getattr(module, "expert_dim", None))
    if hidden_dim is None or intermediate_dim is None:
        logger.warning("Skipping %s: packed expert module missing hidden_dim/intermediate_dim metadata", name)
        return None

    gate_up_name = f"{name}.gate_up_proj"
    down_name = f"{name}.down_proj"

    quantize_gate_up_proj = not _is_ignored(gate_up_name, effective_ignore_patterns)
    quantize_down_proj = not _is_ignored(down_name, effective_ignore_patterns)

    gate_up_proj = getattr(module, "gate_up_proj", None)
    down_proj = getattr(module, "down_proj", None)

    if quantize_gate_up_proj and not _packed_expert_weight_supports_group_size(
        gate_up_proj,
        input_dim=hidden_dim,
        output_dim=2 * intermediate_dim,
        group_size=config.group_size,
    ):
        logger.warning(
            "Skipping packed expert sublayer %s: unsupported gate_up_proj layout/shape=%s for group_size=%s",
            gate_up_name,
            tuple(gate_up_proj.shape),
            config.group_size,
        )
        quantize_gate_up_proj = False

    if quantize_down_proj and not _packed_expert_weight_supports_group_size(
        down_proj,
        input_dim=intermediate_dim,
        output_dim=hidden_dim,
        group_size=config.group_size,
    ):
        logger.warning(
            "Skipping packed expert sublayer %s: unsupported down_proj layout/shape=%s for group_size=%s",
            down_name,
            tuple(down_proj.shape),
            config.group_size,
        )
        quantize_down_proj = False

    if not (quantize_gate_up_proj or quantize_down_proj):
        logger.warning("_get_packed_expert_quant_plan return None. quantize_gate_up_proj=%s, quantize_down_proj=%s", quantize_gate_up_proj, quantize_down_proj)
        return None

    return {
        "quantize_gate_up_proj": quantize_gate_up_proj,
        "quantize_down_proj": quantize_down_proj,
    }


def _should_quantize_linear(
    name: str,
    module: nn.Module,
    config: QATConfig,
    effective_ignore_patterns: list[str],
) -> bool:
    """Check if a linear module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    if _EXPERT_LINEAR_RE.match(name) and not config.experts.enable:
        return False

    if _is_ignored(name, effective_ignore_patterns):
        logger.debug("Ignoring %s due to ignore_patterns", name)
        return False

    if module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def _is_verl_qat_module(module: nn.Module) -> bool:
    return bool(getattr(module, "_is_verl_qat_linear", False))


def _get_qat_linear_cls(mode: str):
    from verl.utils.qat.linear import QATLinear, QATMode
    from verl.utils.qat.mxfp8_linear import MXFP8QATLinear

    qat_mode = QATMode(mode.lower())
    if qat_mode in {QATMode.W8A16_MXFP8, QATMode.W8A8_MXFP8}:
        return qat_mode, MXFP8QATLinear
    return qat_mode, QATLinear


def _infer_mxfp8_layer_metadata(name: str) -> tuple[Optional[str], Optional[int]]:
    layer_type = name.rsplit(".", 1)[-1]
    match = _MXFP8_LAYER_IDX_RE.search(name)
    layer_index = int(match.group(1)) if match else None
    return layer_type, layer_index


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model by replacing quantizable modules with mode-specific QAT wrappers."""
    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    mode, qat_linear_cls = _get_qat_linear_cls(config.mode)
    if mode.value in _MXFP8_MODES and config.group_size != 32:
        raise ValueError(f"MXFP8 QAT requires group_size=32, got: {config.group_size}")
    logger.info(
        f"Applying QAT with mode={mode.value}, group_size={config.group_size}, "
        f"mxfp8_rounding_mode={config.mxfp8_rounding_mode}, experts_enable={config.experts.enable}"
    )
    if mode.value in _MXFP8_MODES:
        from verl.utils.qat.mxfp8_linear import configure_mxfp8_probe

        logger.warning(
            "MXFP8 QAT requested on training model: mode=%s, group_size=%s, quant_backend=%s, "
            "rounding_mode=%s, probe_quant_error=%s, rotation_enable=%s, experts_enable=%s",
            mode.value,
            config.group_size,
            config.mxfp8_quant_backend,
            config.mxfp8_rounding_mode,
            config.mxfp8_probe_quant_error,
            config.mxfp8_rotation_enable,
            config.experts.enable,
        )
        configure_mxfp8_probe(
            enabled=config.mxfp8_probe_quant_error,
            output_path=config.mxfp8_probe_quant_error_output_path,
            rank0_only=True,
        )
        if config.mxfp8_probe_quant_error:
            logger.warning(
                "MXFP8 quant error probe enabled; writing JSONL records to %s",
                config.mxfp8_probe_quant_error_output_path,
            )
        if config.mxfp8_rotation_enable:
            logger.warning(
                "MXFP8 block rotation enabled for QAT: kind=%s, block_size=%s, seed=%s",
                config.mxfp8_rotation_kind,
                config.mxfp8_rotation_block_size,
                config.mxfp8_rotation_seed,
            )

    effective_ignore_patterns = get_effective_ignore_patterns(config)
    linear_modules_to_replace = []
    packed_expert_modules_to_replace = []
    for name, module in model.named_modules():
        packed_expert_plan = _get_packed_expert_quant_plan(name, module, config, effective_ignore_patterns)
        if packed_expert_plan is not None:
            packed_expert_modules_to_replace.append((name, module, packed_expert_plan))
            continue
        if _should_quantize_linear(name, module, config, effective_ignore_patterns):
            linear_modules_to_replace.append((name, module))

    logger.info(
        "Found %s linear layers and %s packed expert blocks to convert to QAT",
        len(linear_modules_to_replace),
        len(packed_expert_modules_to_replace),
    )

    converted_count = 0
    for name, module in linear_modules_to_replace:
        if _is_verl_qat_module(module):
            continue

        from_linear_kwargs = {
            "mode": mode,
            "group_size": config.group_size,
            "activation_observer": config.activation_observer,
        }
        if mode.value in _MXFP8_MODES:
            layer_type, layer_index = _infer_mxfp8_layer_metadata(name)
            from_linear_kwargs["mxfp8_quant_backend"] = config.mxfp8_quant_backend
            from_linear_kwargs["mxfp8_rounding_mode"] = config.mxfp8_rounding_mode
            from_linear_kwargs["mxfp8_probe_quant_error"] = config.mxfp8_probe_quant_error
            from_linear_kwargs["mxfp8_rotation_enable"] = config.mxfp8_rotation_enable
            from_linear_kwargs["mxfp8_rotation_kind"] = config.mxfp8_rotation_kind
            from_linear_kwargs["mxfp8_rotation_block_size"] = config.mxfp8_rotation_block_size
            from_linear_kwargs["mxfp8_rotation_seed"] = config.mxfp8_rotation_seed
            from_linear_kwargs["layer_name"] = name
            from_linear_kwargs["layer_type"] = layer_type
            from_linear_kwargs["layer_index"] = layer_index

        fake_quant_module = qat_linear_cls.from_linear(module, **from_linear_kwargs)

        _set_module(model, name, fake_quant_module)
        logger.warning("apply_qat _set_module: %s", name)
        converted_count += 1

    if mode.value in _MXFP8_MODES:
        from verl.utils.qat.mxfp8_experts import MXFP8QATSparseMoeBlock

        for name, module, packed_expert_plan in packed_expert_modules_to_replace:
            if _is_verl_qat_module(module):
                continue

            _, layer_index = _infer_mxfp8_layer_metadata(name)
            fake_quant_module = MXFP8QATSparseMoeBlock.from_module(
                module,
                mode=mode,
                group_size=config.group_size,
                activation_observer=config.activation_observer,
                mxfp8_quant_backend=config.mxfp8_quant_backend,
                mxfp8_rounding_mode=config.mxfp8_rounding_mode,
                mxfp8_probe_quant_error=config.mxfp8_probe_quant_error,
                mxfp8_rotation_enable=config.mxfp8_rotation_enable,
                mxfp8_rotation_kind=config.mxfp8_rotation_kind,
                mxfp8_rotation_block_size=config.mxfp8_rotation_block_size,
                mxfp8_rotation_seed=config.mxfp8_rotation_seed,
                layer_name=name,
                layer_index=layer_index,
                quantize_gate_up_proj=packed_expert_plan["quantize_gate_up_proj"],
                quantize_down_proj=packed_expert_plan["quantize_down_proj"],
            )
            _set_module(model, name, fake_quant_module)
            logger.warning("apply_qat _set_module: %s", name)
            converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")
    if mode.value in _MXFP8_MODES:
        logger.warning(
            "MXFP8 QAT applied to training model: mode=%s, converted_layers=%s, quant_backend=%s, "
            "rounding_mode=%s, weight_fake_quant=True, activation_fake_quant=%s, probe_quant_error=%s, "
            "rotation_enable=%s, experts_enable=%s",
            mode.value,
            converted_count,
            config.mxfp8_quant_backend,
            config.mxfp8_rounding_mode,
            mode.value == "w8a8_mxfp8",
            config.mxfp8_probe_quant_error,
            config.mxfp8_rotation_enable,
            config.experts.enable,
        )

    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    qat_modules = {
        name: m
        for name, m in model.named_modules()
        if _is_verl_qat_module(m) and bool(getattr(m, "supports_qat_fusion", False))
    }

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached quantization state after optimizer.step()."""
    count = 0
    for module in model.modules():
        if _is_verl_qat_module(module):
            if hasattr(module, "invalidate_quant_state"):
                module.invalidate_quant_state()
            else:
                module._weight_blockwise_scale = None
                module._weight_global_scale = None
                module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QAT layers")
