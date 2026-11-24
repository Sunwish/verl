import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.qat import mxfp_probe, mxfp_quant
from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear
from verl.utils.qat.mxfp_moe import mxfp_grouped_matmul
from verl.utils.qat.mxfp_probe import (
    begin_quant_error_probe_step,
    finish_quant_error_probe_step,
    quantize_with_probe,
)
from verl.utils.qat.mxfp_rotation import rotate_blocks


def _config(**kwargs):
    defaults = {
        "fake_quant_targets": ("fprop",),
        "quant_backend": "torch",
        "rounding_mode": "rint",
        "probe_quant_error": True,
        "probe_quant_error_use_high_precision": True,
        "layer_name": "model.layers.3.self_attn.q_proj",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_probe_deduplicates_key_uses_high_precision_and_writes_json_only(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    calls = 0

    def fake_quantize(tensor, fmt, **kwargs):
        nonlocal calls
        calls += 1
        return torch.round(tensor * 2) / 2

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 5)
    source = torch.arange(64, dtype=torch.float32).reshape(2, 32) / 7

    first = quantize_with_probe(source, "mxfp8", _config(), target="fprop", operand="activation")
    second = quantize_with_probe(source + 1, "mxfp8", _config(), target="fprop", operand="activation")
    assert torch.equal(first, source)
    assert torch.equal(second, source + 1)
    assert calls == 1

    finish_quant_error_probe_step(step)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert (record["step"], record["layer"], record["module"]) == (5, 3, "q")
    assert record["target"] == "fprop"
    assert record["operand"] == "activation"
    assert record["count"] == source.numel()
    assert record["mae"] > 0
    quantized = torch.round(source * 2) / 2
    expected_relative_abs = torch.where(source != 0, (quantized - source).abs() / source.abs(), 0).mean()
    assert record["relative_abs"] == pytest.approx(expected_relative_abs.item())
    assert not output.with_suffix(".html").exists()


def test_probe_low_precision_path_returns_qdq_and_keeps_one_record(tmp_path, monkeypatch):
    output = tmp_path / "probe"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    calls = 0

    def fake_quantize(tensor, fmt, **kwargs):
        nonlocal calls
        calls += 1
        return tensor + 0.25

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 6)
    source = torch.ones((1, 32), dtype=torch.float32)
    config = _config(probe_quant_error_use_high_precision=False)

    first = quantize_with_probe(source, "mxfp8", config, target="fprop", operand="weight")
    second = quantize_with_probe(source * 2, "mxfp8", config, target="fprop", operand="weight")
    assert torch.equal(first, source + 0.25)
    assert torch.equal(second, source * 2 + 0.25)
    assert calls == 2  # QDQ remains in the matmul path when explicitly requested.

    finish_quant_error_probe_step(step)
    payload = json.loads((tmp_path / "probe" / "mxfp_quant_error.json").read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    assert not (tmp_path / "probe" / "mxfp_quant_error.html").exists()


def test_probe_first_step_only_keeps_qdq_enabled_after_sampling(tmp_path, monkeypatch):
    output = tmp_path / "probe"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    quantize_calls = 0

    def fake_quantize(tensor, fmt, **kwargs):
        nonlocal quantize_calls
        quantize_calls += 1
        return tensor + 0.25

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    config = _config(probe_quant_error_use_high_precision=False)
    source = torch.ones((1, 32))

    first_step = begin_quant_error_probe_step(output, 1, first_step_only=True)
    assert first_step == 1
    first = quantize_with_probe(source, "mxfp8", config, target="fprop", operand="activation")
    finish_quant_error_probe_step(first_step)

    second_step = begin_quant_error_probe_step(output, 2, first_step_only=True)
    assert second_step is None
    second = quantize_with_probe(source * 2, "mxfp8", config, target="fprop", operand="activation")

    assert torch.equal(first, source + 0.25)
    assert torch.equal(second, source * 2 + 0.25)
    assert quantize_calls == 2
    payload = json.loads((tmp_path / "probe" / "mxfp_quant_error.json").read_text(encoding="utf-8"))
    assert {record["step"] for record in payload["records"]} == {1}


@pytest.mark.parametrize(("high_precision", "expected_quantize_calls"), ((True, 1), (False, 2)))
def test_probe_is_stable_during_checkpoint_recompute(tmp_path, monkeypatch, high_precision, expected_quantize_calls):
    output = tmp_path / "probe"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    quantize_calls = []

    def fake_quantize(tensor, fmt, **kwargs):
        quantize_calls.append((torch.is_grad_enabled(), tensor.requires_grad))
        return torch.round(tensor * 2) / 2

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 7)
    source = torch.randn((2, 32), requires_grad=True)

    def run(value):
        quantized = quantize_with_probe(
            value,
            "mxfp8",
            _config(probe_quant_error_use_high_precision=high_precision),
            target="fprop",
            operand="activation",
        )
        return quantized.sin()

    result = checkpoint(run, source, use_reentrant=False)
    assert result.shape == source.shape
    assert result.dtype == source.dtype

    # Simulate checkpoint recomputation after the probe ContextVar is no longer active.
    finish_quant_error_probe_step(step)
    result.sum().backward()

    assert quantize_calls == [(False, False)] * expected_quantize_calls
    qdq_input = source.detach() if high_precision else torch.round(source.detach() * 2) / 2
    torch.testing.assert_close(result.detach(), qdq_input.sin())
    torch.testing.assert_close(source.grad, qdq_input.cos())


def test_qat_without_probe_preserves_fake_quant_grad_context(monkeypatch):
    # Keep a live probe object around to verify the config switch alone keeps
    # the normal QAT path unchanged.
    monkeypatch.setattr(mxfp_probe, "_PROBE", SimpleNamespace(active=True))
    quantize_calls = []

    def fake_quantize(tensor, fmt, **kwargs):
        quantize_calls.append((torch.is_grad_enabled(), tensor.requires_grad))
        return tensor + 0.25

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    source = torch.ones((1, 32), requires_grad=True)

    result = quantize_with_probe(
        source,
        "mxfp8",
        _config(probe_quant_error=False),
        target="fprop",
        operand="activation",
    )

    torch.testing.assert_close(result, source + 0.25)
    assert quantize_calls == [(True, True)]
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("high_precision", [False, True])
def test_dense_probe_measures_independent_backward_qdq(tmp_path, monkeypatch, rotation, high_precision):
    torch.manual_seed(59)
    output_path = tmp_path / "independent.json"
    probe = mxfp_probe.MXFPQuantErrorProbe(output_path)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    config = MXFPQATConfig(
        mode="mxfp_w4a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        mxfp4_mbs_enable=True,
        layer_name="model.layers.3.self_attn.q_proj",
        probe_quant_error=True,
        probe_quant_error_use_high_precision=high_precision,
    )
    qat = MXFPQATLinear(32, 64, False, config)
    x = torch.randn(64, 32, requires_grad=True)
    grad_output = torch.randn(64, 64)
    step = begin_quant_error_probe_step(output_path, 8)
    qat(x).backward(grad_output)
    finish_quant_error_probe_step(step)
    records = {
        (record["target"], record["operand"]): record
        for record in json.loads(output_path.read_text(encoding="utf-8"))["records"]
    }
    sources = {
        ("fprop", "activation"): x.detach(),
        ("fprop", "weight"): qat.weight.detach(),
        ("dgrad", "weight"): qat.weight.detach().T,
        ("dgrad", "grad_output"): grad_output,
        ("wgrad", "activation"): x.detach().T,
        ("wgrad", "grad_output"): grad_output.T,
    }
    assert records.keys() == sources.keys()
    for key, source in sources.items():
        if rotation:
            source = rotate_blocks(source, config.rotation)
        fmt = config.weight_format if key[1] == "weight" else config.activation_format
        quantized = mxfp_quant.fake_quantize(source, fmt, mxfp4_mbs_enable=fmt == "mxfp4")
        expected_l2 = ((quantized - source).square().sum() / source.square().sum()).sqrt().item()
        assert records[key]["relative_l2"] == pytest.approx(expected_l2, rel=1e-5)
        assert records[key]["count"] == source.numel()


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("high_precision", [False, True])
def test_moe_probe_records_recomputed_backward_operands(tmp_path, monkeypatch, reentrant, high_precision):
    def grouped_matmul(x, weight, groups, group_list_type, *, split_item=2, group_type=0):
        counts = groups.tolist()
        if split_item == 2:
            return torch.cat([part @ matrix for part, matrix in zip(x.split(counts), weight, strict=True)])
        return torch.stack(
            [left @ right for left, right in zip(x.split(counts, dim=1), weight.split(counts), strict=True)]
        )

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", grouped_matmul)
    output_path = tmp_path / "moe.json"
    probe = mxfp_probe.MXFPQuantErrorProbe(output_path)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    config = MXFPQATConfig(
        mode="mxfp_w4a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        rotation_enable=True,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        mxfp4_mbs_enable=True,
        layer_name="model.layers.3.mlp",
        probe_quant_error=True,
        probe_quant_error_use_high_precision=high_precision,
    )
    x = torch.randn(35, 32, requires_grad=True)
    weight = torch.randn(2, 32, 64, requires_grad=True)
    groups = torch.tensor([7, 28])

    def run(value, matrices):
        return mxfp_grouped_matmul(value, matrices, groups, config, name="gate_proj")

    step = begin_quant_error_probe_step(output_path, 9)
    output = checkpoint(run, x, weight, use_reentrant=reentrant)
    assert {key[3] for key in probe._pending} == {"fprop"}
    output.sum().backward()
    finish_quant_error_probe_step(step)
    records = json.loads(output_path.read_text(encoding="utf-8"))["records"]
    assert {(record["target"], record["operand"]) for record in records} == {
        ("fprop", "activation"),
        ("fprop", "weight"),
        ("dgrad", "grad_output"),
        ("dgrad", "weight"),
        ("wgrad", "activation"),
        ("wgrad", "grad_output"),
    }
    assert len(records) == 6
    for record in records:
        expected_count = (
            weight.numel()
            if record["operand"] == "weight"
            else (x.numel() if record["operand"] == "activation" else output.numel())
        )
        assert record["count"] == expected_count
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()
