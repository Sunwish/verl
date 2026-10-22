import json
from types import SimpleNamespace

import torch

from verl.utils.qat import mxfp_probe, mxfp_quant
from verl.utils.qat.mxfp_probe import (
    begin_quant_error_probe_step,
    finish_quant_error_probe_step,
    quantize_with_probe,
)


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


def test_probe_deduplicates_key_uses_high_precision_and_writes_html(tmp_path, monkeypatch):
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
    html = output.with_suffix(".html").read_text(encoding="utf-8")
    assert "MXFP QDQ Error Probe" in html
    assert "q" in html


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
    assert (tmp_path / "probe" / "mxfp_quant_error.html").exists()
