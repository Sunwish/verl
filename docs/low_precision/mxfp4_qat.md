# MXFP4 QAT for RL on Ascend A2/A3

MXFP4 experiments reuse high-precision training and rollout matmuls with
quantize/dequantize operands; native FP4 storage/kernels are not required.

| QAT mode | Weight | Activation |
| --- | --- | --- |
| `w8a16_mxfp8` | E4M3 | BF16/FP16 |
| `w8a8_mxfp8` | E4M3 | E4M3 |
| `w4a8_mxfp4` | E2M1 | E4M3 |
| `w4a4_mxfp4` | E2M1 | E2M1 |

All MXFP modes use 32-element groups and E8M0 shared scales. MXFP4 requires
`quant_backend: torch`; this torch implementation runs on NPU tensors on the
server. Native `npu` MXFP4 quantization is outside this fake-quant path.

## Reuse Plan and Implementation

1. Extend the QAT enum and select weight/activation quantizers within the existing
   dense and packed-expert wrappers. Reuse STE, scale invalidation, layer matching,
   expert selection, and JSONL probes.
2. Implement E2M1 quantize/dequantize in the current torch MXFP8 style. Follow
   [ops-nn DynamicMxQuant golden.py](https://gitcode.com/cann/ops-nn/blob/8b566a55cd41ee887aa4b82c3fc560c7010ef2c8/quant/dynamic_mx_quant/tests/assets/golden.py):
   `shared_exp = max(floor(log2(amax)) - 2, -127)`, private exponent minimum 0,
   one mantissa bit, saturation at 6. Zero blocks use E8M0 code 0; non-finite
   block scales use code 255. Store unpacked E2M1 values and biased exponent
   codes in FP32 for immediate fake dequantization on A2/A3.
3. Pass quantizers into the existing Dgrad/Wgrad transforms. W4A8 uses FP4 for
   Dgrad weights and FP8 for gradient/activation operands. Wgrad uses the
   activation format. Preserve optional reduction-axis preparation/padding,
   independent fake-quant targets, and independent rotation targets.
4. Remove `mxfp8_` from config fields across dataclasses, Hydra YAML, FSDP,
   rollout inheritance, docs, and tests. Historical Python helper/class/module
   names remain to reuse existing integration hooks.
5. Extend vllm-ascend's unquantized linear/MoE path with the same formula and
   format selection. Reuse weight-load cache refresh, rotation, and effective
   ignore/fallback patterns. Read fake-quant targets to support backward-only
   experiments without quantizing rollout Fprop.

## Configuration

Add these overrides to the existing RL launch command:

```bash
actor_rollout_ref.actor.fsdp_config.qat.enable=True \
actor_rollout_ref.actor.fsdp_config.qat.mode=w4a8_mxfp4 \
actor_rollout_ref.actor.fsdp_config.qat.group_size=32 \
actor_rollout_ref.actor.fsdp_config.qat.quant_backend=torch \
actor_rollout_ref.actor.fsdp_config.qat.rounding_mode=rint \
actor_rollout_ref.actor.fsdp_config.qat.rotation_enable=True \
actor_rollout_ref.actor.fsdp_config.qat.rotation_targets='[Fprop]' \
actor_rollout_ref.actor.fsdp_config.qat.fake_quant_targets='[Fprop]' \
actor_rollout_ref.actor.fsdp_config.qat.experts.enable=True \
actor_rollout_ref.actor.fsdp_config.qat.experts.npu_backend=grouped_matmul \
actor_rollout_ref.actor.fsdp_config.qat.fallback_layers='[0,1]' \
actor_rollout_ref.rollout.qat_fake_quant=True \
actor_rollout_ref.rollout.quantization=null
```

Change only `qat.mode=w4a4_mxfp4` to quantize activations to FP4 too. Rollout
inherits actor mode, rotation, rounding, experts, ignore patterns, and fallback
layers. Rollout fake quant can also run with actor QAT disabled.

On NPU, packed QAT experts default to `experts.npu_backend=grouped_matmul`.
Set `experts.npu_backend=python` to use the per-expert Python QAT loop instead.
Both paths support W4A8/W4A4 Fprop/Dgrad/Wgrad fake quantization and block
rotation; MXFP4 requires `quant_backend=torch` on either execution path. See
[NPU packed experts](mxfp8_qat.md#npu-packed-experts) for scope and examples.

The renamed keys apply to MXFP8 too: `quant_backend`, `rounding_mode`,
`fake_quant_targets`, `backward_reduction_axis_fake_quant`, `probe_quant_error`,
`probe_quant_error_output_path`, `rotation_enable`, `rotation_kind`,
`rotation_block_size`, `rotation_seed`, and `rotation_targets`. Remove the old
`mxfp8_` prefix from launch overrides. Rollout's `mxfp8_group_size` is now
`group_size`. Old config keys are intentionally not aliases.

The existing `VLLM_ASCEND_QAT_MXFP8_*` transport environment names remain between
the repositories and apply to both formats. Hardware MXFP8 rollout remains
supported. MXFP4 experiments use `qat_fake_quant=True` for low-precision rollout;
this does not export packed FP4 checkpoints or true FP4 GEMMs.

`rounding_mode` supports `rint` (nearest even), `round` (ties away from zero),
`random`, and deterministic `hash`. As in dense MXFP8 and rollout, forward
activation quantization uses `rint`; configured rounding controls weights and
backward operands. New MXFP4 MoE modes follow the same activation rule.

For backward experiments use `fake_quant_targets='[Fprop,Dgrad,Wgrad]'` and
`backward_reduction_axis_fake_quant=True`. Rotation can target the same set or
a subset. With the reduction-axis option disabled, historical reuse of already
fake-quantized forward operands is preserved.

For probes set `probe_quant_error=True` and an absolute
`probe_quant_error_output_path`. As in MXFP8, probes measure forward operand error
and retain the original forward operands. JSONL records include `mode`.
Probe mode is a diagnostic run; backward targets remain separately controlled.

## Validation

The development machine has no torch, torch_npu, vLLM, or NPU. Offline checks
cover syntax/lint, YAML/config handoff, and numerical formulas using a NumPy
adapter plus an independent E2M1 codebook oracle. They do not validate torch
device behavior, autograd, graph capture, or RL reward accuracy.

Run in server environments with the repositories' normal dependencies:

```bash
# In verl.
python -m pytest -q tests/utils/test_qat_mxfp8_on_cpu.py \
  tests/workers/config/test_engine_config_on_cpu.py \
  tests/utils/test_vllm_fp8_utils_on_cpu.py \
  tests/workers/rollout/test_rollout_bootstrap_model_path_on_cpu.py

# In vllm-ascend.
python -m pytest -q tests/ut/quantization/test_mxfp8_fake_quant.py
```

On A2/A3 first run one actor/rollout step for each mode, then repeat with
rotation, fallback layers, experts, stochastic rounding, and all backward
targets. Check finite actor gradients, the mode in rollout logs, and weight
cache refresh after actor synchronization. Compare reward with W8A8 using the
same data, seeds, and sampling settings. NPU training, reward accuracy, and
performance measurements have not been run on this development machine.
