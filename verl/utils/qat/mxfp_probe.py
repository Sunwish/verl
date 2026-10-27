# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Streaming MXFP QDQ error collection for FSDP QAT experiments."""

# HTML/CSS/JavaScript is embedded in a single self-contained template below.
# ruff: noqa: E501

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)
_STATS_CHUNK_SIZE = 1 << 20

_ACTIVE_STEP: contextvars.ContextVar[int | None] = contextvars.ContextVar("mxfp_probe_step", default=None)
_MODULE_ALIASES = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down_projection",
}
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PROBE: MXFPQuantErrorProbe | None = None
_PROBE_LOCK = threading.Lock()


def _output_paths(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    output = Path(path).expanduser()
    if output.suffix.lower() in {".json", ".html"}:
        base = output.with_suffix("")
    else:
        base = output / "mxfp_quant_error"
    return base.with_suffix(".json"), base.with_suffix(".html")


def _module_label(module_name: str) -> str:
    suffix = module_name.rsplit(".", 1)[-1]
    return _MODULE_ALIASES.get(suffix, suffix)


def _record_key(record: dict[str, Any]) -> tuple[int, str, str, str, str]:
    return record["step"], record["layer_name"], record["module"], record["target"], record["operand"]


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, str, str, str]:
    return record["step"], record["layer"], record["module"], record["target"], record["operand"]


class MXFPQuantErrorProbe:
    """Keep one compact error summary per step/layer/module/target/operand."""

    def __init__(self, output_path: str | os.PathLike[str]):
        self.json_path, self.html_path = _output_paths(output_path)
        self._pending: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
        self._history: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
        self._claimed: set[tuple[int, str, str, str, str]] = set()
        self._fallback_step = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return _ACTIVE_STEP.get() is not None

    def begin_step(self, step: int | None) -> int:
        if step is None:
            self._fallback_step += 1
            step = self._fallback_step
        step = int(step)
        _ACTIVE_STEP.set(step)
        return step

    def claim(self, layer_name: str, module_name: str, target: str, operand: str) -> bool:
        """Reserve a key before QDQ so simultaneous expert paths quantify it once."""
        step = _ACTIVE_STEP.get()
        if step is None:
            return False
        full_name = f"{layer_name}.{module_name}" if module_name else layer_name
        key = (step, layer_name, _module_label(module_name or full_name), target, operand)
        with self._lock:
            if key in self._claimed or key in self._pending or key in self._history:
                return False
            self._claimed.add(key)
            return True

    def record_many(
        self,
        layer_name: str,
        module_names: list[str],
        target: str,
        operand: str,
        reference: torch.Tensor,
        quantized: torch.Tensor,
        *,
        valid_shape: tuple[int, ...] | None = None,
        valid_indices: torch.Tensor | None = None,
        indices_dim: int = 0,
    ) -> bool:
        step = _ACTIVE_STEP.get()
        if step is None:
            return False

        if valid_shape is not None and tuple(reference.shape) != valid_shape:
            slices = tuple(slice(0, size) for size in valid_shape)
            reference = reference[slices]
            quantized = quantized[slices]
        if valid_indices is not None:
            reference = reference.index_select(indices_dim, valid_indices)
            quantized = quantized.index_select(indices_dim, valid_indices)

        reference_flat = reference.detach().reshape(-1)
        quantized_flat = quantized.detach().reshape(-1)
        device = reference.device
        abs_sum = torch.zeros((), device=device, dtype=torch.float32)
        sq_sum = torch.zeros_like(abs_sum)
        max_abs = torch.zeros_like(abs_sum)
        reference_sq_sum = torch.zeros_like(abs_sum)
        # Bound temporary memory: model weights can be very large, and probe collection
        # must not allocate a second full-size error tensor on the NPU.
        for start in range(0, reference_flat.numel(), _STATS_CHUNK_SIZE):
            stop = min(start + _STATS_CHUNK_SIZE, reference_flat.numel())
            reference_chunk = reference_flat[start:stop].to(torch.float32)
            error_chunk = quantized_flat[start:stop].to(torch.float32) - reference_chunk
            abs_chunk = error_chunk.abs()
            abs_sum += abs_chunk.sum()
            sq_sum += error_chunk.square().sum()
            max_abs = torch.maximum(max_abs, abs_chunk.amax())
            reference_sq_sum += reference_chunk.square().sum()
        # Transfer all metrics in one synchronization. Keep reductions in float32 for
        # accelerator compatibility, then persist only Python scalars.
        stats = (
            torch.stack(
                (
                    abs_sum,
                    sq_sum,
                    max_abs,
                    reference_sq_sum,
                )
            )
            .cpu()
            .tolist()
        )
        layer_match = _LAYER_PATTERN.search(layer_name)
        common_record = {
            "step": step,
            "layer": int(layer_match.group(1)) if layer_match else -1,
            "layer_name": layer_name or "<unknown>",
            "target": target,
            "operand": operand,
            "_count": reference_flat.numel(),
            "_abs_sum": float(stats[0]),
            "_sq_sum": float(stats[1]),
            "_max_abs": float(stats[2]),
            "_reference_sq_sum": float(stats[3]),
            "rank_count": 1,
        }
        records = []
        for module_name in module_names:
            full_name = f"{layer_name}.{module_name}" if module_name else layer_name
            module = _module_label(module_name or full_name)
            records.append(((step, layer_name, module, target, operand), dict(common_record, module=module)))
        with self._lock:
            # Concurrent calls for one logical key retain only the first observation.
            for key, record in records:
                self._pending.setdefault(key, record)
        return True

    def finish_step(self, step: int) -> None:
        step = int(step)
        with self._lock:
            local_records = [record for key, record in self._pending.items() if key[0] == step]

        gathered = [local_records]
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local_records)
        except (RuntimeError, ValueError) as exc:
            logger.warning("MXFP QDQ probe rank aggregation failed; writing local metrics: %s", exc)
            gathered = [local_records]

        try:
            import torch.distributed as dist

            is_writer = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        except (RuntimeError, ValueError):
            is_writer = True

        if is_writer:
            merged: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
            for rank_records in gathered:
                for record in rank_records or []:
                    key = _record_key(record)
                    current = merged.get(key)
                    if current is None:
                        merged[key] = dict(record)
                    else:
                        current["_count"] += record["_count"]
                        current["_abs_sum"] += record["_abs_sum"]
                        current["_sq_sum"] += record["_sq_sum"]
                        current["_max_abs"] = max(current["_max_abs"], record["_max_abs"])
                        current["_reference_sq_sum"] += record["_reference_sq_sum"]
                        current["rank_count"] += record.get("rank_count", 1)
            for key, record in merged.items():
                self._history.setdefault(key, record)
            self._write_reports()

        with self._lock:
            self._pending = {key: record for key, record in self._pending.items() if key[0] != step}
            self._claimed = {key for key in self._claimed if key[0] != step}
        _ACTIVE_STEP.set(None)

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        count = record["_count"]
        sq_sum = record["_sq_sum"]
        ref_sq_sum = record["_reference_sq_sum"]
        return {
            "step": record["step"],
            "layer": record["layer"],
            "layer_name": record["layer_name"],
            "module": record["module"],
            "target": record["target"],
            "operand": record["operand"],
            "count": count,
            "mae": record["_abs_sum"] / count if count else 0.0,
            "rmse": (sq_sum / count) ** 0.5 if count else 0.0,
            "max_abs": record["_max_abs"],
            "relative_l2": (sq_sum / ref_sq_sum) ** 0.5 if ref_sq_sum else 0.0,
            "rank_count": record.get("rank_count", 1),
        }

    def _write_reports(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        records = [self._public_record(record) for record in self._history.values()]
        records.sort(key=_record_sort_key)
        payload = {"schema_version": 1, "records": records}
        self._atomic_write(self.json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        self._atomic_write(self.html_path, _build_html(records))
        logger.info("MXFP QDQ probe wrote %d unique records to %s and %s", len(records), self.json_path, self.html_path)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _build_html(records: list[dict[str, Any]]) -> str:
    safe_data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MXFP QDQ Error Probe</title><style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#18212b;background:#f5f7fa}}h1{{margin-bottom:6px}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}label{{display:grid;gap:4px}}select{{min-width:150px;padding:7px}}
.summary{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:white;border:1px solid #dce2e8;border-radius:8px;padding:12px 16px}}
canvas{{background:white;border:1px solid #dce2e8;border-radius:8px;width:100%;height:280px}}
table{{width:100%;border-collapse:collapse;background:white;margin-top:16px}}th,td{{padding:8px;border-bottom:1px solid #e5e9ee;text-align:right}}
th{{position:sticky;top:0;background:#eef2f6}}th:nth-child(-n+5),td:nth-child(-n+5){{text-align:left}}
.table-wrap{{max-height:55vh;overflow:auto;background:white}}.muted{{color:#627080}}
</style></head><body><h1>MXFP QDQ Error Probe</h1>
<p class="muted">One observation per step, layer, module, target and operand. Values are aggregated across distributed ranks.</p>
<div class="summary" id="summary"></div><div class="controls">
<label>Step<select id="step"></select></label><label>Layer<select id="layer"></select></label>
<label>Module<select id="module"></select></label><label>Target<select id="target"></select></label>
</div><canvas id="chart" width="1400" height="280"></canvas>
<div class="table-wrap"><table><thead><tr><th>Step</th><th>Layer</th><th>Module</th><th>Target</th><th>Operand</th><th>Count</th><th>MAE</th><th>RMSE</th><th>Max abs</th><th>Relative L2</th><th>Ranks</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>const allRecords={safe_data};
const fields=['step','layer','module','target'];const selects=Object.fromEntries(fields.map(k=>[k,document.getElementById(k)]));
function options(key){{const values=[...new Set(allRecords.map(r=>String(r[key])))].sort((a,b)=>Number(a)-Number(b)||a.localeCompare(b));return ['all',...values]}}
for(const key of fields){{selects[key].innerHTML=options(key).map(v=>`<option value="${{v}}">${{v==='all'?'All':key==='layer'&&v!=='all'?'Layer '+v:v}}</option>`).join('');selects[key].onchange=render}}
if(allRecords.length)selects.step.value=String(Math.max(...allRecords.map(r=>r.step)));
function filtered(){{return allRecords.filter(r=>fields.every(k=>selects[k].value==='all'||String(r[k])===selects[k].value))}}
function render(){{const rows=filtered();document.getElementById('rows').innerHTML=rows.map(r=>`<tr><td>${{r.step}}</td><td>${{r.layer_name}}</td><td>${{r.module}}</td><td>${{r.target}}</td><td>${{r.operand}}</td><td>${{r.count.toLocaleString()}}</td><td>${{r.mae.toExponential(3)}}</td><td>${{r.rmse.toExponential(3)}}</td><td>${{r.max_abs.toExponential(3)}}</td><td>${{r.relative_l2.toExponential(3)}}</td><td>${{r.rank_count}}</td></tr>`).join('');
document.getElementById('summary').innerHTML=[['Records',rows.length],['Steps',new Set(rows.map(r=>r.step)).size],['Layers',new Set(rows.map(r=>r.layer_name)).size],['Modules',new Set(rows.map(r=>r.module)).size]].map(([k,v])=>`<div class="card"><b>${{k}}</b><br>${{v}}</div>`).join('');draw(rows)}}
function draw(rows){{const c=document.getElementById('chart'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!rows.length)return;const vals=rows.map(r=>r.rmse),max=Math.max(...vals,1e-30),left=76,top=22,bottom=42,w=(c.width-left-20)/rows.length;ctx.fillStyle='#263746';ctx.font='13px system-ui';ctx.fillText('RMSE (log scale)',12,18);rows.forEach((r,i)=>{{const h=(Math.log10(Math.max(r.rmse,1e-30))-Math.log10(max/1000))/(3)* (c.height-top-bottom);const bar=Math.max(2,Math.min(c.height-top-bottom,h));const x=left+i*w;ctx.fillStyle=['#3478c5','#e48636','#3b9c74','#a45ca8'][i%4];ctx.fillRect(x+2,c.height-bottom-bar,Math.max(3,w-4),bar);ctx.save();ctx.translate(x+w/2,c.height-bottom+8);ctx.rotate(-Math.PI/4);ctx.textAlign='right';ctx.fillStyle='#334';ctx.fillText(`${{r.step}} · L${{r.layer}} ${{r.module}}/${{r.target}} ${{r.operand}}`,0,0);ctx.restore()}})}}render();</script></body></html>"""


def configure_quant_error_probe(output_path: str | os.PathLike[str]) -> MXFPQuantErrorProbe:
    global _PROBE
    with _PROBE_LOCK:
        if _PROBE is None or _PROBE.json_path != _output_paths(output_path)[0]:
            _PROBE = MXFPQuantErrorProbe(output_path)
        return _PROBE


def begin_quant_error_probe_step(output_path: str | os.PathLike[str], step: int | None) -> int:
    return configure_quant_error_probe(output_path).begin_step(step)


def finish_quant_error_probe_step(step: int) -> None:
    if _PROBE is not None:
        _PROBE.finish_step(step)


def quantize_with_probe(
    tensor: torch.Tensor,
    fmt: str,
    config: Any,
    *,
    target: str,
    operand: str,
    module_name: str | None = None,
    probe_module_names: tuple[str, ...] | None = None,
    valid_shape: tuple[int, ...] | None = None,
    valid_indices: torch.Tensor | None = None,
    indices_dim: int = 0,
) -> torch.Tensor:
    """Quantize and record one unique QDQ observation when a probe step is active."""
    from verl.utils.qat.mxfp_quant import fake_quantize

    if target not in config.fake_quant_targets:
        return tensor

    probe_enabled = bool(getattr(config, "probe_quant_error", False)) and _PROBE is not None and _PROBE.active
    layer_name = getattr(config, "layer_name", None) or "<unknown>"
    module_name = module_name or layer_name.rsplit(".", 1)[-1]
    module_names = probe_module_names or (module_name,)
    record_names = (
        [name for name in dict.fromkeys(module_names) if _PROBE.claim(layer_name, name, target, operand)]
        if probe_enabled
        else []
    )
    high_precision = probe_enabled and bool(getattr(config, "probe_quant_error_use_high_precision", True))
    if high_precision and not record_names:
        return tensor

    quantized = fake_quantize(
        tensor,
        fmt,
        quant_backend=config.quant_backend,
        rounding_mode=config.rounding_mode,
        mxfp4_scale_alg=getattr(config, "mxfp4_scale_alg", "standard"),
    )
    if record_names:
        _PROBE.record_many(
            layer_name,
            record_names,
            target,
            operand,
            tensor,
            quantized,
            valid_shape=valid_shape,
            valid_indices=valid_indices,
            indices_dim=indices_dim,
        )
    if high_precision:
        return tensor
    return tensor + (quantized - tensor).detach()
