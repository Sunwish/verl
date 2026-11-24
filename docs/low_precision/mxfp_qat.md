# MXFP QAT forward and backward operands

The MXFP Dense and packed MoE paths simulate each selected GEMM independently.
They use the FSDP-Turbo MXFP8 idea of deriving each orientation from the same
pre-QDQ source, but defer backward operand preparation until backward needs it.
The GEMMs still run in the model's high-precision dtype after QDQ.

## Independent reduction axes

For a Dense layer with `X[M, K]`, `W[N, K]` and output gradient `G[M, N]`,
the unrotated recipe is:

| GEMM | Operands | Quantization reduction axis |
| --- | --- | --- |
| Fprop | `Q_K(X) @ Q_K(W).T` | Input feature `K` |
| Dgrad | `Q_N(G) @ Q_N(W)` | Output feature `N` |
| Wgrad | `Q_M(G).T @ Q_M(X)` | Flattened token dimension `M` |

Here `Q` includes the configured format, scaling algorithm, optional MXFP4
macro-block scaling (MBS), quantization, dequantization, and inverse MBS correction.
When a target enables rotation, both operands first receive the paired block
rotation along that GEMM's reduction dimension. Fprop rotation is not inherited
by the separately enabled Dgrad or Wgrad transforms.

Dense backward saves the original pre-QDQ activation and/or weight only when
their corresponding gradients need them. Backward then applies its own axis
layout, rotation and QDQ. Forward does not prepare a second quantization axis.
An unselected backward target preserves the existing Fprop STE derivative,
including the inverse Fprop rotation needed to return original coordinates.

Preparing a backward operand early or just before its GEMM is numerically
equivalent when the source tensor version, grouping, format, scales and rounding
are identical. Stochastic rounding additionally requires matching random draws;
changing execution order alone does not guarantee bitwise equality. This
implementation uses deferred preparation for both Dense and MoE.

## MoE memory and checkpointing

Packed MoE keeps its existing custom-autograd recomputation path. It does not
cache both quantized weight orientations or transformed activation buffers
when backward transforms are enabled. Required raw operands are retained through
`save_for_backward`, so activation-checkpoint saved-tensor hooks remain effective.
Dgrad temporary buffers are released before allocating Wgrad's padded operands.
Gate/up can continue sharing their detached Fprop activation.

Wgrad pads each expert's token segment independently to the quantization block
size before applying token-axis transforms. Blocks never mix different experts.
The group list is adjusted for those temporary rows; original routing and output
shapes are unchanged.

## Quantization error collection

Each record measures the local QDQ error for one
`step/layer/module/target/operand` key:

```text
relative_l2 = norm(QDQ(target_operand) - target_operand) / norm(target_operand)
```

`target_operand` is derived from the original pre-QDQ source using that target's
layout and rotation. For MXFP4 MBS, comparison is against the pre-MBS value,
after QDQ has undone its correction. The metric does not compare to a previous
Fprop QDQ value, an unrotated tensor, or the final GEMM output. Existing padding
masks exclude synthetic entries from the reported metrics.

Dgrad/Wgrad observations are collected when backward actually prepares those
operands. The existing per-step key deduplication prevents checkpoint replay or
shared expert inputs from adding duplicate records.
The active step is shared by the rank-local probe across threads from
`begin_quant_error_probe_step` until `finish_quant_error_probe_step`, covering
autograd device workers and checkpoint recomputation. A thread-local
`ContextVar` would lose Dgrad/Wgrad observations when backward runs on another
thread. Finishing the step closes sampling for all threads; first-step-only
sampling still leaves subsequent low-precision computation enabled.
`probe_quant_error_use_high_precision=false` keeps QDQ in the compute path;
`true` records QDQ but feeds the corresponding pre-QDQ value to the GEMM.
The output remains JSON with the existing record keys and metrics.

## Configuration compatibility and validation

No launcher argument or default changes are needed. `fake_quant_targets` and
`rotation_targets` still select Fprop, Dgrad and Wgrad independently. The existing
`backward_reduction_axis_fake_quant` field remains accepted for compatibility;
both values now use on-demand independent backward operands rather than
switching to a forward-time cache.

The existing W4A8/MXFP4 recipe and MBS behavior remain in use. This follows
FSDP-Turbo's independent-operand design; it does not reproduce Turbo's special
W4A8 MXFP4-to-block-MXFP8 weight conversion or claim hardware GEMM bitwise parity.

Focused CPU regression command on a configured server:

```bash
python -m pytest tests/utils/test_mxfp_quant_cpu.py \
  tests/utils/test_mxfp_moe_cpu.py tests/utils/test_mxfp_probe_cpu.py -q
```

Coverage includes independent raw sources and numerical gradients, both values
of the compatibility field, partial token blocks, W8A8/W4A8/W4A4, MXFP4 MBS,
rotation, checkpoint replay, probe metrics, frozen operands, and MoE saved
tensor/temporary-buffer lifetimes. Ascend/FSDP distributed execution, peak NPU
memory, and training stability still require server validation.
