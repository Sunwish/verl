# MXFP QDQ error analysis

[`scripts/mxfp_qdq_error.py`](../../scripts/mxfp_qdq_error.py) imports the
quantize and dequantize functions from the live `verl` checkout, or from a
specified `vllm-ascend` checkout. It does not copy their quantization
arithmetic. The report records the source file locations and git commits so
the measured implementation can be identified later.

## Forward path and dtypes

Each random weight and activation tensor is converted to BF16 before entering
the QDQ functions. The dequantized tensors are BF16, and both reference and
QDQ matrix multiplications run with BF16 inputs and produce BF16 outputs. The
script promotes those already-produced outputs to FP32 only when accumulating
error statistics. Thus FP32 is used for metric calculation, not as the input,
dequantized output, or forward-matmul dtype. The quantizers may use their own
internal working dtype or packed-value carrier representation; those are part
of the imported implementation.

For each mode, the measured output comparison is:

```text
reference = BF16_activation @ BF16_weight.T
qdq       = BF16_QDQ(activation) @ BF16_QDQ(weight).T
```

## Supported experiment settings

| Option | Default | Accepted values and meaning |
| --- | --- | --- |
| `--implementation` | `verl` | `verl`, `vllm-ascend`, or `both`. The latter compares both checkouts in one run. `vllm-ascend` needs its normal runtime dependencies. |
| `--vllm-ascend-root` | sibling `vllm-ascend` checkout | Path to that checkout; only needed for `vllm-ascend` or `both`. |
| `--modes` | `w8a8_mxfp8,w4a8_mxfp4,w4a4_mxfp4` | Comma-separated subset of `bf16`, `w8a16_mxfp8`, `w8a8_mxfp8`, `w4a8_mxfp4`, `w4a4_mxfp4`. `bf16` is the unquantized control; `w8a16` quantizes weights only; the other modes quantize the indicated weight and activation formats. |
| `--distributions` | `normal,uniform,heavy_tail,outlier` | Comma-separated subset of `normal`, `uniform`, `heavy_tail`, `outlier`. `normal` samples N(0,1); `uniform` samples U[-1,1]; `heavy_tail` multiplies normal samples by a log-normal scale mixture and normalizes to unit RMS; `outlier` multiplies each normal element independently by a random factor from 8 to 16 with the configured probability. |
| `--trials` | `32` | Positive integer. Number of independent weight/activation pairs generated per distribution. Increase it to reduce sensitivity to a particular random sample. |
| `--tokens-per-trial` | `128` | Positive integer. Number of activation rows (tokens) in each trial. Larger values increase the sampled activation/output count and matrix-multiply work without changing the weight matrix for that trial. |
| `--in-features` | `1024` | Positive integer divisible by 32. Input/hidden dimension and the MXFP block-quantization axis length. |
| `--out-features` | `1024` | Positive integer. Number of output rows in the weight matrix. |
| `--weight-rounding` | `rint` | `rint`, `round`, `random`, or `hash`. Controls weight quantization rounding. `random` and `hash` select stochastic rounding behavior supported by the live quantizer. |
| `--outlier-probability` | `0.01` | Number from 0 to 1. Per-element probability of applying the 8–16 multiplier in the `outlier` distribution; ignored by other distributions. |
| `--max-percentile-samples` | `100000` | Non-negative integer cap on retained error samples per implementation/distribution/mode/role (`weight`, `activation`, or `output`) used to estimate percentiles. Set to `0` to disable percentile sampling; percentile fields are then `null`. Other aggregate metrics are still computed over all values. |
| `--seed` | `20261021` | Integer seed for reproducible source tensors and percentile sampling. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `npu`, or another PyTorch device string. `auto` prefers NPU, then CUDA, then CPU. |
| `--output` | `mxfp_qdq_results/mxfp_qdq_error.json` | Main JSON report path. A per-trial JSONL file with the same stem is written beside it. |
| `--quiet` | off | Suppress periodic progress messages. |

All comma-separated values must be non-empty and drawn from the listed choices.
Feature dimensions, trials, and tokens per trial must be positive; the input
feature count must also be divisible by 32.

With the defaults, each mode/distribution/backend contributes
`32 × 128 × 1024 = 4,194,304` output values to aggregate metrics. The retained
percentile sample is capped separately. Increase trials and token count for
more random coverage; increase feature dimensions to test longer MX blocks and
larger GEMMs. Runtime and memory use grow with these dimensions, and `both`
also runs each case through two implementations.

## Features included and excluded

The standalone probe models random tensors passed through the imported tensor
QDQ functions and a linear matrix multiplication. It does not implement
rotation, TIS, or layer fallback configuration. Those features are therefore
unsupported by this probe, rather than supported switches that are merely
disabled by default. In particular, it cannot measure their impact on QDQ or
the resulting forward error.

Activation rounding is fixed to `rint`, matching the current MXFP QAT linear
and expert forward path; stochastic activation rounding is not configurable in
this probe. Weight rounding is configurable with `--weight-rounding`, including
`random` and `hash` modes where supported by the live quantizer. The script
does not run a training loop, emulate the full model, or model hardware kernel
accumulation behavior.

## Running the probe

From the `verl` checkout, a quick CPU-sized run is:

```bash
python scripts/mxfp_qdq_error.py \
  --modes w8a8_mxfp8,w4a8_mxfp4,w4a4_mxfp4 \
  --distributions normal,heavy_tail,outlier \
  --trials 4 --tokens-per-trial 16 --in-features 256 --out-features 128 \
  --output results/mxfp_qdq.json
```

For a server run, use the default 1024 dimensions or increase dimensions,
trials, and tokens as resources allow. `--device auto` selects an available
NPU, CUDA device, or CPU; pass `--device npu` to require NPU execution.

To compare the current `vllm-ascend` implementation too:

```bash
python scripts/mxfp_qdq_error.py \
  --implementation both \
  --vllm-ascend-root /path/to/vllm-ascend \
  --output results/mxfp_qdq_both.json
```

The `verl` implementation is the default and does not require importing
`vllm-ascend`. The latter backend requires the dependencies needed to import
its quantization module in the environment running the script.

## Visualizing results

Copy the JSON report (and optionally its JSONL companion) to a workstation,
then build a standalone HTML report and optional CSV summary:

```bash
python scripts/plot_mxfp_qdq_error.py \
  --input results/mxfp_qdq.json \
  --output results/mxfp_qdq.html \
  --csv-output results/mxfp_qdq_summary.csv
```

Open the HTML file directly in a browser. It has no network or plotting-package
dependency and provides aggregate comparisons, per-trial error curves, and a
table with MAE, RMSE, normalized RMSE, percentiles, cosine similarity, zero
fraction, and saturation fraction. The JSON report also includes input
configuration, source locations, and per-trial weight, activation, and output
metrics.
