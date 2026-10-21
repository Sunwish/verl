#!/usr/bin/env python3
"""Measure MXFP8/MXFP4 quantize-dequantize error with the live repository code.

The script deliberately imports the quantizers from the current checkout instead
of reimplementing their arithmetic. It generates paired BF16 weights and
activations, applies the same QDQ choices used by the linear QAT forward path,
and compares a BF16 reference matmul with the QDQ BF16 matmul. Error metrics
are accumulated in FP32 after both forward outputs have been produced.

Examples (run from the verl checkout)::

    python scripts/mxfp_qdq_error.py --output results/mxfp_qdq.json
    python scripts/mxfp_qdq_error.py --modes w8a8_mxfp8,w4a8_mxfp4,w4a4_mxfp4 \
        --distributions normal,heavy_tail,outlier --trials 64 \
        --tokens-per-trial 256 --in-features 2048 --out-features 2048 \
        --output results/mxfp_qdq.json

To compare the two checked-out implementations, pass the vllm-ascend root::

    python scripts/mxfp_qdq_error.py --implementation both \
        --vllm-ascend-root D:/Source/github/vllm-ascend \
        --output results/mxfp_qdq_both.json

Activation rounding is intentionally fixed to ``rint``.  That is what the
current MXFP QAT linear and expert forward paths use; ``--weight-rounding``
controls the weight side only.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLLM_ASCEND_ROOT = REPO_ROOT.parent / "vllm-ascend"
BLOCK_SIZE = 32
DATA_DTYPE = torch.bfloat16
ACTIVATION_ROUNDING_MODE = "rint"
MODES = {
    "bf16": (None, None),
    "w8a16_mxfp8": ("mxfp8", None),
    "w8a8_mxfp8": ("mxfp8", "mxfp8"),
    "w4a8_mxfp4": ("mxfp4", "mxfp8"),
    "w4a4_mxfp4": ("mxfp4", "mxfp4"),
}
DISTRIBUTIONS = ("normal", "uniform", "heavy_tail", "outlier")


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_location(function: Callable[..., Any]) -> dict[str, Any]:
    try:
        path = Path(inspect.getsourcefile(function) or "").resolve()
        _, first_line = inspect.getsourcelines(function)
        return {"file": str(path), "first_line": first_line, "name": function.__name__}
    except (OSError, TypeError):
        return {"file": None, "first_line": None, "name": getattr(function, "__name__", None)}


@dataclass(frozen=True)
class QuantBackend:
    """Small adapter around one checkout's quantize/dequantize functions."""

    name: str
    module_name: str
    source_root: Path
    quantize_mxfp8: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    dequantize_mxfp8: Callable[..., torch.Tensor]
    quantize_mxfp4: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    dequantize_mxfp4: Callable[..., torch.Tensor]

    def source_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module_name,
            "root": str(self.source_root),
            "git_commit": _git_commit(self.source_root),
            "functions": {
                "quantize_mxfp8": _source_location(self.quantize_mxfp8),
                "dequantize_mxfp8": _source_location(self.dequantize_mxfp8),
                "quantize_mxfp4": _source_location(self.quantize_mxfp4),
                "dequantize_mxfp4": _source_location(self.dequantize_mxfp4),
            },
        }

    def qdq(
        self,
        tensor: torch.Tensor,
        format_name: str | None,
        *,
        rounding_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run exactly one current-checkout quantize/dequantize path."""

        if format_name is None:
            return tensor, None, None
        if format_name == "mxfp8":
            quantizer_kwargs: dict[str, Any] = {"rounding_mode": rounding_mode}
            # verl exposes quant_backend on the public helper. vllm-ascend's
            # torch helper is already the backend-specific private function.
            if "quant_backend" in inspect.signature(self.quantize_mxfp8).parameters:
                quantizer_kwargs["quant_backend"] = "torch"
            quantized, scale = self.quantize_mxfp8(tensor, **quantizer_kwargs)
            dequantized = self.dequantize_mxfp8(quantized, scale, tensor.dtype)
        elif format_name == "mxfp4":
            quantized, scale = self.quantize_mxfp4(
                tensor, quant_backend="torch", rounding_mode=rounding_mode
            )
            dequantized = self.dequantize_mxfp4(quantized, scale, tensor.dtype)
        else:
            raise ValueError(f"Unsupported format: {format_name}")
        return dequantized, quantized, scale


def _load_backend(name: str, vllm_ascend_root: Path | None) -> QuantBackend:
    if name == "verl":
        module_name = "verl.utils.qat.mxfp8_linear"
        source_root = REPO_ROOT
        # Running ``python scripts/...`` puts the scripts directory at sys.path[0].
        # Prefer this checkout so the report always measures the source tree
        # containing this script rather than an unrelated installed package.
        sys.path.insert(0, str(source_root))
        module = importlib.import_module(module_name)
        quantize_mxfp8 = module.quantize_mxfp8_tensor
        dequantize_mxfp8 = module._dequantize_mxfp8
    elif name == "vllm-ascend":
        source_root = (vllm_ascend_root or DEFAULT_VLLM_ASCEND_ROOT).resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"vllm-ascend checkout does not exist: {source_root}")
        sys.path.insert(0, str(source_root))
        module_name = "vllm_ascend.quantization.mxfp8_fake_quant"
        module = importlib.import_module(module_name)
        quantize_mxfp8 = module._quantize_mxfp8_torch
        dequantize_mxfp8 = module._dequantize_mxfp8
    else:
        raise ValueError(f"Unsupported implementation: {name}")

    return QuantBackend(
        name=name,
        module_name=module_name,
        source_root=source_root,
        quantize_mxfp8=quantize_mxfp8,
        dequantize_mxfp8=dequantize_mxfp8,
        quantize_mxfp4=module.quantize_mxfp4_tensor,
        dequantize_mxfp4=module.dequantize_mxfp4_tensor,
    )


class MetricAccumulator:
    """Streaming error statistics with a bounded sample for percentiles."""

    def __init__(self, max_samples: int, seed: int):
        self.count = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.rel_abs_sum = 0.0
        self.rel_sq_sum = 0.0
        self.ref_sq_sum = 0.0
        self.approx_sq_sum = 0.0
        self.dot_sum = 0.0
        self.max_abs = 0.0
        self.zero_count = 0
        self.saturated_count = 0
        self.scale_nan_count = 0
        self.scale_count = 0
        self.payload_count = 0
        self.max_samples = max(0, max_samples)
        self._sample_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._abs_samples: list[torch.Tensor] = []
        self._rel_samples: list[torch.Tensor] = []
        self._sample_count = 0

    def update(
        self,
        reference: torch.Tensor,
        approximation: torch.Tensor,
        *,
        quantized: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        payload_max: float | None = None,
    ) -> None:
        reference_cpu = reference.detach().to(torch.float32).reshape(-1).cpu()
        approximation_cpu = approximation.detach().to(torch.float32).reshape(-1).cpu()
        error = approximation_cpu - reference_cpu
        abs_error = error.abs()
        relative_error = abs_error / reference_cpu.abs().clamp_min(1.0e-8)
        n = int(reference_cpu.numel())
        self.count += n
        self.abs_sum += float(abs_error.sum())
        self.sq_sum += float(error.square().sum())
        self.rel_abs_sum += float(relative_error.sum())
        self.rel_sq_sum += float(relative_error.square().sum())
        self.ref_sq_sum += float(reference_cpu.square().sum())
        self.approx_sq_sum += float(approximation_cpu.square().sum())
        self.dot_sum += float(torch.dot(reference_cpu, approximation_cpu))
        self.max_abs = max(self.max_abs, float(abs_error.max()) if n else 0.0)

        if quantized is not None:
            quantized_cpu = quantized.detach().to(torch.float32).reshape(-1).cpu()
            self.payload_count += int(quantized_cpu.numel())
            self.zero_count += int((quantized_cpu == 0).sum())
            if payload_max is not None:
                self.saturated_count += int((quantized_cpu.abs() >= payload_max - 1.0e-5).sum())
        if scale is not None:
            scale_cpu = scale.detach().reshape(-1).cpu()
            self.scale_count += int(scale_cpu.numel())
            self.scale_nan_count += int((scale_cpu == 255).sum())

        if self.max_samples <= 0 or n <= 0:
            return
        # Keep a small random sample from every update, then downsample the
        # merged sample. This avoids making percentile estimates depend only
        # on the first few trials.
        sample_size = min(4_096, self.max_samples, n)
        indices = torch.randint(
            low=0,
            high=n,
            size=(sample_size,),
            generator=self._sample_generator,
        )
        self._abs_samples.append(abs_error[indices])
        self._rel_samples.append(relative_error[indices])
        self._sample_count += sample_size
        if self._sample_count > self.max_samples:
            abs_sample = torch.cat(self._abs_samples)
            rel_sample = torch.cat(self._rel_samples)
            keep = torch.randperm(abs_sample.numel(), generator=self._sample_generator)[: self.max_samples]
            self._abs_samples = [abs_sample[keep]]
            self._rel_samples = [rel_sample[keep]]
            self._sample_count = self.max_samples

    @staticmethod
    def _percentiles(values: Iterable[torch.Tensor]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None, "p999": None}
        sample = torch.cat(list(values))
        result = torch.quantile(sample, torch.tensor([0.5, 0.95, 0.99, 0.999]))
        return {
            "p50": float(result[0]),
            "p95": float(result[1]),
            "p99": float(result[2]),
            "p999": float(result[3]),
        }

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot finalize an empty metric")
        count = float(self.count)
        ref_rms = math.sqrt(self.ref_sq_sum / count)
        rmse = math.sqrt(self.sq_sum / count)
        approx_rms = math.sqrt(self.approx_sq_sum / count)
        cosine = self.dot_sum / max(math.sqrt(self.ref_sq_sum * self.approx_sq_sum), 1.0e-12)
        return {
            "count": self.count,
            "mae": self.abs_sum / count,
            "rmse": rmse,
            "normalized_rmse": rmse / max(ref_rms, 1.0e-12),
            "relative_mae": self.rel_abs_sum / count,
            "relative_rmse": math.sqrt(self.rel_sq_sum / count),
            "reference_rms": ref_rms,
            "approximation_rms": approx_rms,
            "cosine_similarity": cosine,
            "max_abs": self.max_abs,
            "absolute_percentiles": self._percentiles(self._abs_samples),
            "relative_percentiles": self._percentiles(self._rel_samples),
            "quantized_zero_fraction": self.zero_count / self.payload_count if self.payload_count else None,
            "quantized_saturation_fraction": (
                self.saturated_count / self.payload_count if self.payload_count else None
            ),
            "scale_nan_fraction": self.scale_nan_count / self.scale_count if self.scale_count else None,
            "sample_count": sum(int(sample.numel()) for sample in self._abs_samples),
        }


def _summarize_pair(reference: torch.Tensor, approximation: torch.Tensor) -> dict[str, float | int]:
    accumulator = MetricAccumulator(max_samples=0, seed=0)
    accumulator.update(reference, approximation)
    result = accumulator.finalize()
    return {
        key: value
        for key, value in result.items()
        if key
        in {
            "count",
            "mae",
            "rmse",
            "normalized_rmse",
            "relative_mae",
            "relative_rmse",
            "max_abs",
            "cosine_similarity",
        }
    }


def _make_random_tensor(
    shape: tuple[int, ...], distribution: str, generator: torch.Generator, *, outlier_probability: float
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(shape, generator=generator, dtype=torch.float32)
    if distribution == "uniform":
        return torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0

    normal = torch.randn(shape, generator=generator, dtype=torch.float32)
    if distribution == "heavy_tail":
        # A symmetric log-normal scale mixture gives reproducible heavy tails
        # without allocating a separate Student-t degrees-of-freedom tensor.
        tensor = normal * torch.exp(0.75 * torch.randn(shape, generator=generator, dtype=torch.float32))
        return tensor / tensor.square().mean().sqrt().clamp_min(1.0e-8)
    if distribution == "outlier":
        mask = torch.rand(shape, generator=generator, dtype=torch.float32) < outlier_probability
        multiplier = 8.0 + 8.0 * torch.rand(shape, generator=generator, dtype=torch.float32)
        return torch.where(mask, normal * multiplier, normal)
    raise ValueError(f"Unsupported distribution: {distribution}")


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    npu = getattr(torch, "npu", None)
    if npu is not None and getattr(npu, "is_available", lambda: False)():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_csv(value: str, allowed: Iterable[str], argument: str) -> list[str]:
    choices = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(choices) - set(allowed))
    if invalid or not choices:
        raise ValueError(f"Invalid {argument}: {invalid or value}; allowed={sorted(allowed)}")
    return choices


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.in_features <= 0 or args.out_features <= 0:
        raise ValueError("--in-features and --out-features must be positive")
    if args.in_features % BLOCK_SIZE:
        raise ValueError(f"--in-features must be divisible by {BLOCK_SIZE}")
    if args.trials <= 0 or args.tokens_per_trial <= 0:
        raise ValueError("--trials and --tokens-per-trial must be positive")
    if args.max_percentile_samples < 0:
        raise ValueError("--max-percentile-samples must be non-negative")
    if not 0.0 <= args.outlier_probability <= 1.0:
        raise ValueError("--outlier-probability must be between 0 and 1")

    modes = _parse_csv(args.modes, MODES, "--modes")
    distributions = _parse_csv(args.distributions, DISTRIBUTIONS, "--distributions")
    if args.implementation == "verl":
        implementation_names = ["verl"]
    elif args.implementation == "vllm-ascend":
        implementation_names = ["vllm-ascend"]
    else:
        implementation_names = ["verl", "vllm-ascend"]
    backends = [_load_backend(name, args.vllm_ascend_root) for name in implementation_names]
    device = _resolve_device(args.device)
    tensor_dtype = DATA_DTYPE

    accumulator_map: dict[tuple[str, str, str, str], MetricAccumulator] = {}
    trial_records: list[dict[str, Any]] = []
    payload_max = {"mxfp4": 6.0, "mxfp8": 448.0}

    def get_accumulator(backend: str, distribution: str, mode: str, role: str) -> MetricAccumulator:
        key = (backend, distribution, mode, role)
        if key not in accumulator_map:
            seed = args.seed + sum(ord(char) for char in "|".join(key))
            accumulator_map[key] = MetricAccumulator(args.max_percentile_samples, seed)
        return accumulator_map[key]

    total_jobs = len(backends) * len(distributions) * args.trials
    completed_jobs = 0
    for distribution_index, distribution in enumerate(distributions):
        for trial in range(args.trials):
            # One paired source tensor per role makes mode comparisons fair.
            trial_seed = args.seed + distribution_index * 1_000_003 + trial * 9_973
            generator = torch.Generator(device="cpu").manual_seed(trial_seed)
            weight_source = _make_random_tensor(
                (args.out_features, args.in_features),
                distribution,
                generator,
                outlier_probability=args.outlier_probability,
            ).to(dtype=tensor_dtype)
            activation_source = _make_random_tensor(
                (args.tokens_per_trial, args.in_features),
                distribution,
                generator,
                outlier_probability=args.outlier_probability,
            ).to(dtype=tensor_dtype)
            weight = weight_source.to(device=device)
            activation = activation_source.to(device=device)
            # Match the live BF16 QAT forward path: QDQ tensors and matmul
            # outputs remain BF16. MetricAccumulator promotes only the
            # already-produced outputs to FP32 for stable error statistics.
            reference_output = activation @ weight.transpose(-1, -2)
            if reference_output.dtype != DATA_DTYPE:
                raise TypeError(f"BF16 reference matmul produced {reference_output.dtype}")

            for backend in backends:
                for mode in modes:
                    weight_format, activation_format = MODES[mode]
                    quantized_weight, weight_payload, weight_scale = backend.qdq(
                        weight, weight_format, rounding_mode=args.weight_rounding
                    )
                    quantized_activation, activation_payload, activation_scale = backend.qdq(
                        activation, activation_format, rounding_mode=ACTIVATION_ROUNDING_MODE
                    )
                    for label, tensor in (
                        ("weight source", weight),
                        ("activation source", activation),
                        ("dequantized weight", quantized_weight),
                        ("dequantized activation", quantized_activation),
                    ):
                        if tensor.dtype != DATA_DTYPE:
                            raise TypeError(f"Expected {label} to be BF16, got {tensor.dtype}")
                    if mode == "bf16":
                        quantized_weight = weight
                        quantized_activation = activation

                    weight_accumulator = get_accumulator(backend.name, distribution, mode, "weight")
                    weight_accumulator.update(
                        weight,
                        quantized_weight,
                        quantized=weight_payload,
                        scale=weight_scale,
                        payload_max=payload_max.get(weight_format or ""),
                    )
                    activation_accumulator = get_accumulator(backend.name, distribution, mode, "activation")
                    activation_accumulator.update(
                        activation,
                        quantized_activation,
                        quantized=activation_payload,
                        scale=activation_scale,
                        payload_max=payload_max.get(activation_format or ""),
                    )

                    quantized_output = quantized_activation @ quantized_weight.transpose(-1, -2)
                    if quantized_output.dtype != DATA_DTYPE:
                        raise TypeError(f"BF16 QDQ matmul produced {quantized_output.dtype}")
                    output_accumulator = get_accumulator(backend.name, distribution, mode, "output")
                    output_accumulator.update(reference_output, quantized_output)

                    trial_records.append(
                        {
                            "implementation": backend.name,
                            "distribution": distribution,
                            "mode": mode,
                            "trial": trial,
                            "weight": _summarize_pair(weight, quantized_weight),
                            "activation": _summarize_pair(activation, quantized_activation),
                            "output": _summarize_pair(reference_output, quantized_output),
                        }
                    )
                completed_jobs += 1
                if not args.quiet and (completed_jobs == total_jobs or completed_jobs % max(1, total_jobs // 20) == 0):
                    print(f"completed {completed_jobs}/{total_jobs} distribution-trials")

            del weight, activation, reference_output

    summary = []
    for (implementation, distribution, mode, role), accumulator in sorted(accumulator_map.items()):
        summary.append(
            {
                "implementation": implementation,
                "distribution": distribution,
                "mode": mode,
                "role": role,
                **accumulator.finalize(),
            }
        )

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "verl_root": str(REPO_ROOT),
            "verl_commit": _git_commit(REPO_ROOT),
            "backends": [backend.source_metadata() for backend in backends],
            "note": (
                "Quantization and dequantization functions are imported from the live checkout; "
                "no arithmetic is duplicated here."
            ),
        },
        "config": {
            "implementation": args.implementation,
            "device": str(device),
            "input_dtype": "bfloat16",
            "qdq_output_dtype": "bfloat16",
            "matmul_input_dtype": "bfloat16",
            "matmul_output_dtype": "bfloat16",
            "metric_accumulator_dtype": "float32",
            "modes": modes,
            "distributions": distributions,
            "trials": args.trials,
            "tokens_per_trial": args.tokens_per_trial,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "weight_rounding": args.weight_rounding,
            "activation_rounding": ACTIVATION_ROUNDING_MODE,
            "outlier_probability": args.outlier_probability,
            "max_percentile_samples": args.max_percentile_samples,
            "seed": args.seed,
            "block_size": BLOCK_SIZE,
            "forward_path": (
                "BF16 reference matmul and BF16 QDQ matmul; error statistics promote the "
                "result tensors to FP32 only after forward computation."
            ),
        },
        "summary": summary,
        "trials": trial_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("mxfp_qdq_results/mxfp_qdq_error.json"))
    parser.add_argument("--implementation", choices=("verl", "vllm-ascend", "both"), default="verl")
    parser.add_argument("--vllm-ascend-root", type=Path, default=None)
    parser.add_argument("--modes", default="w8a8_mxfp8,w4a8_mxfp4,w4a4_mxfp4")
    parser.add_argument("--distributions", default=",".join(DISTRIBUTIONS))
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--tokens-per-trial", type=int, default=128)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--weight-rounding", choices=("rint", "round", "random", "hash"), default="rint")
    parser.add_argument("--outlier-probability", type=float, default=0.01)
    parser.add_argument("--max-percentile-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, npu, or another torch device string")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(output_path)
    trials_path = output_path.with_name(output_path.stem + ".trials.jsonl")
    with trials_path.open("w", encoding="utf-8") as stream:
        for record in result["trials"]:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"wrote {output_path}")
    print(f"wrote {trials_path}")


if __name__ == "__main__":
    main()
