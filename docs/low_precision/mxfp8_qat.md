# MXFP8 QAT (Quantization-Aware Training) in verl

Last updated: 07/27/2026

verl supports MXFP8 QAT for the Ascend low-precision RL stack. Training keeps matmul in high precision, but inserts
MXFP8 fake quantization on weights and, in W8A8 mode, activations. Rollout then loads a separate quantized artifact
through the vLLM / vLLM-Ascend MXFP8 path, which reduces the train/rollout precision gap and gives us a clean base for
later experiments such as rotation matrices and stochastic rounding.

| Training Backend | Training Precision | Rollout Precision | vLLM Quant Method |
|---|---|---|---|
| **FSDP / FSDP2** | BF16 + MXFP8 fake quantization | MXFP8 W8A16 or W8A8 | `ascend` |

> [!TIP]
> For ready-to-run scripts, environment setup, and experiment logs, see the QAT recipe. This page focuses on the
> verl-side configuration and runtime behavior.

---

## Current MXFP8 Stack

This stack was built in stages:

- rollout bootstrap from a separate quantized artifact path
- FSDP MXFP8 fake quantization in training
- W8A8 activation fake quantization
- torch quantizer aligned with rollout's shared-exponent flow
- configurable `npu` / `torch` quantization backend
- MXFP8 quantization error probes
- configurable rounding mode for the torch path

> [!NOTE]
> This page documents the FSDP / FSDP2 MXFP8 path. The Megatron QAT flow in verl still follows the NVFP4-oriented
> stack.

---

## Key Configuration

Configured under `actor_rollout_ref.actor.fsdp_config.qat`:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      qat:
        enable: true
        mode: "w8a8_mxfp8"
        group_size: 32
        ignore_patterns:
          - "lm_head"
          - "embed_tokens"
          - "re:.*mlp.gate$"
        quantization_config_path: "recipe/qat/config/mxfp8_w8a8_ascend.json"
        mxfp8_quant_backend: "torch"
        mxfp8_rounding_mode: "hash"
        mxfp8_probe_quant_error: false

actor_rollout_ref:
  rollout:
    bootstrap_model_path: /path/to/mxfp8-rollout-artifact
```

| Parameter | Description | Default / Notes |
|---|---|---|
| `fsdp_config.qat.enable` | Enable QAT | `False` |
| `fsdp_config.qat.mode` | MXFP8 mode | Use `w8a16_mxfp8` or `w8a8_mxfp8` |
| `fsdp_config.qat.group_size` | MXFP8 block size | Must be `32` |
| `fsdp_config.qat.ignore_patterns` | Layers to skip | `["lm_head", "embed_tokens", "re:.*mlp.gate$"]` |
| `fsdp_config.qat.quantization_config_path` | vLLM quantization config JSON | Required when QAT is enabled |
| `fsdp_config.qat.mxfp8_quant_backend` | Quantization backend | `npu` or `torch` |
| `fsdp_config.qat.mxfp8_rounding_mode` | Rounding mode | `rint`, `round`, `random`, or `hash` |
| `fsdp_config.qat.mxfp8_probe_quant_error` | Log aggregated per-layer quantization error | `False` |
| `fsdp_config.qat.mxfp8_probe_quant_error_output_path` | JSONL output path for probes | Required if probes are enabled |

---

## Backend And Rounding

The backend controls how fake quantization is computed:

| Backend | Behavior | When to use |
|---|---|---|
| `npu` | Uses `torch_npu.npu_dynamic_mx_quant` | Fast native Ascend path |
| `torch` | Uses verl's torch quantizer | Experiments, CPU validation, and custom rounding |

The rounding mode only affects the torch backend:

| Mode | Behavior | Notes |
|---|---|---|
| `rint` | Deterministic round-to-nearest-even | Default behavior, matches rollout-side low-precision quantization |
| `round` | Deterministic round-half-up | Legacy torch behavior |
| `random` | Independent stochastic rounding | Best for noise studies |
| `hash` | Deterministic pseudo-random rounding from tensor bits | Good when you want repeatable stochastic behavior |

`random` and `hash` require `mxfp8_quant_backend: "torch"`.

---

## Rollout Bootstrapping

MXFP8 rollout often starts from a separate artifact path. Keep the training model path where it belongs, and point the
rollout bootstrap path at the quantized artifact instead:

```yaml
actor_rollout_ref:
  model:
    path: /path/to/training/bf16/model
  rollout:
    bootstrap_model_path: /path/to/quantized/rollout/model
```

The rollout path reads the same MXFP8 config, exports `VERL_MXFP8_QUANT_BACKEND` and `VERL_MXFP8_ROUNDING_MODE`, and
quantizes high-precision weights before loading them into vLLM.

For current ModelSlim / Ascend exports, the rollout path accepts a top-level `quant_method: ascend` as well as
per-parameter descriptors that contain `MXFP8`.

---

## Probe Logging

Enable `mxfp8_probe_quant_error` when you want to inspect quantization error by step, layer index, layer type, and
error type. The probe aggregates all matching layer events and writes one final MAE record for each group. Records with
`step: null` are skipped.

- layer type
- layer index
- error type
- aggregated error value
- quantization backend
- rounding mode
- rank
- training step

The output format is JSONL, one rank-0 record per `(step, error_type, layer_index, layer_type)` group.

---

## Implementation Notes

- MXFP8 block size is `32`.
- The torch quantizer only accepts 2D tensors. Activations are flattened before quantization and reshaped back after
  dequantization.
- The torch quantizer now follows the same shared-exponent / private-exponent formula used by rollout.
- `w8a16_mxfp8` keeps activation precision high and fake-quantizes weights only.
- `w8a8_mxfp8` fake-quantizes both weights and activations.

---

## Validation

The current stack was exercised on Qwen3-30B-A3B in the FSDP / vLLM-Ascend flow.

Good starting points:

- baseline parity checks: `mxfp8_quant_backend: "npu"` with `mxfp8_rounding_mode: "rint"`
- repeatable stochastic studies: `mxfp8_quant_backend: "torch"` with `mxfp8_rounding_mode: "hash"`
- exploratory noise injection: `mxfp8_quant_backend: "torch"` with `mxfp8_rounding_mode: "random"`
