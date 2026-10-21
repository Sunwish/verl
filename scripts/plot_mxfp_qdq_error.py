#!/usr/bin/env python3
# ruff: noqa: E501
"""Build a dependency-free HTML report from ``mxfp_qdq_error.py`` output.

The generated report embeds the JSON data, so it can be opened directly from a
local filesystem after copying results off a server.  No web server, CDN, or
Python plotting package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MXFP QDQ error report</title>
<style>
:root { color-scheme: light dark; --bg: light-dark(#f7f8fa,#111418); --fg: light-dark(#1c232b,#eef2f6); --muted: light-dark(#5e6875,#aab5c2); --line: light-dark(#d6dce3,#36404c); --panel: light-dark(#fff,#191e24); --blue: #4c83d1; --orange: #e38b45; --green: #46a77a; --purple: #a06ac4; --red: #d45c65; --yellow: #c6a340; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 500; margin: 0 0 4px; }
h2 { font-size: 16px; font-weight: 500; margin: 24px 0 8px; }
.muted { color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 12px 18px; margin: 20px 0 12px; align-items: end; }
label { display: grid; gap: 4px; color: var(--muted); }
select { min-width: 170px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 5px; background: var(--panel); color: var(--fg); }
.chart { width: 100%; min-height: 320px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
svg { width: 100%; height: 320px; display: block; }
.axis { stroke: var(--line); stroke-width: 1; }
.grid { stroke: var(--line); stroke-width: 1; opacity: .55; }
.tick { fill: var(--muted); font-size: 11px; }
.axis-title { fill: var(--fg); font-size: 12px; }
.series-0 { fill: var(--blue); stroke: var(--blue); }
.series-1 { fill: var(--orange); stroke: var(--orange); }
.series-2 { fill: var(--green); stroke: var(--green); }
.series-3 { fill: var(--purple); stroke: var(--purple); }
.series-4 { fill: var(--red); stroke: var(--red); }
.series-5 { fill: var(--yellow); stroke: var(--yellow); }
.line { fill: none; stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 8px 0 0; color: var(--muted); }
.legend span { display: inline-flex; gap: 6px; align-items: center; }
.swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
pre { overflow-x: auto; padding: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--muted); font-size: 12px; }
.empty { color: var(--muted); padding: 32px 0; }
</style>
</head>
<body>
<main>
  <h1>MXFP QDQ error report</h1>
  <div id="subtitle" class="muted"></div>
  <div class="controls">
    <label>Implementation<select id="implementation"></select></label>
    <label>Distribution<select id="distribution"></select></label>
    <label>Role<select id="role"><option value="output">output</option><option value="weight">weight</option><option value="activation">activation</option></select></label>
    <label>Bar metric<select id="metric"><option value="normalized_rmse">normalized RMSE</option><option value="relative_mae">relative MAE</option><option value="relative_p95">relative error p95</option><option value="mae">MAE</option><option value="rmse">RMSE</option></select></label>
  </div>
  <h2 id="bar-heading">Aggregate comparison</h2>
  <div class="chart"><svg id="bar-chart" role="img" aria-label="Aggregate error comparison"></svg></div>
  <div id="bar-legend" class="legend"></div>
  <h2 id="line-heading">Per-trial normalized RMSE</h2>
  <div class="chart"><svg id="line-chart" role="img" aria-label="Per-trial normalized RMSE"></svg></div>
  <div id="line-legend" class="legend"></div>
  <h2>Aggregate statistics</h2>
  <div id="table" class="table-wrap"></div>
  <h2>Run metadata</h2>
  <pre id="metadata"></pre>
</main>
<script>
const REPORT = __REPORT_JSON__;
const COLORS = ['var(--blue)','var(--orange)','var(--green)','var(--purple)','var(--red)','var(--yellow)'];
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const unique = (values) => [...new Set(values)];
const fmt = (value) => value === null || value === undefined || !Number.isFinite(value) ? '—' : Number(value).toPrecision(5);
const metricValue = (record, metric) => metric === 'relative_p95' ? record.relative_percentiles.p95 : record[metric];
const options = (element, values) => { element.innerHTML = values.map(value => `<option value="${value}">${value}</option>`).join(''); };
const selectedSummary = () => REPORT.summary.filter(row => row.implementation === $('implementation').value && row.distribution === $('distribution').value && row.role === $('role').value);
const selectedTrials = () => REPORT.trials.filter(row => row.implementation === $('implementation').value && row.distribution === $('distribution').value);
function svgElement(tag, attrs = {}) { const node = document.createElementNS(NS, tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value); return node; }
function clearSvg(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function drawAxes(svg, width, height, margin, yMax, yTitle) {
  const x0 = margin.left, y0 = height - margin.bottom, x1 = width - margin.right, y1 = margin.top;
  svg.appendChild(svgElement('line', {x1:x0,y1:y0,x2:x1,y2:y0,class:'axis'}));
  svg.appendChild(svgElement('line', {x1:x0,y1:y0,x2:x0,y2:y1,class:'axis'}));
  for (let i = 0; i <= 4; i++) {
    const value = yMax * i / 4, y = y0 - (y0-y1) * i / 4;
    svg.appendChild(svgElement('line', {x1:x0,y1:y,x2:x1,y2:y,class:'grid'}));
    const tick = svgElement('text', {x:x0-8,y:y+4,'text-anchor':'end',class:'tick'}); tick.textContent = fmt(value); svg.appendChild(tick);
  }
  const title = svgElement('text', {x:14,y:(y1+y0)/2,transform:`rotate(-90 14 ${(y1+y0)/2})`,class:'axis-title','text-anchor':'middle','data-axis':'y'}); title.textContent = yTitle; svg.appendChild(title);
  return {x0,y0,x1,y1};
}
function drawBars() {
  const rows = selectedSummary(), svg = $('bar-chart'); clearSvg(svg);
  const width = Math.max(680, svg.clientWidth || 900), height = 320, margin = {left:72,right:20,top:18,bottom:55}; svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  if (!rows.length) return;
  const values = rows.map(row => metricValue(row, $('metric').value)); const yMax = Math.max(...values, 1e-12) * 1.12; const axes = drawAxes(svg,width,height,margin,yMax,$('metric').selectedOptions[0].textContent);
  const groupWidth = (axes.x1-axes.x0) / rows.length, barWidth = Math.min(80, groupWidth * .68);
  rows.forEach((row,index) => { const value = metricValue(row, $('metric').value), x = axes.x0 + groupWidth*index + (groupWidth-barWidth)/2, h = (axes.y0-axes.y1) * value/yMax, y = axes.y0-h; svg.appendChild(svgElement('rect',{x,y,width:barWidth,height:h,class:`series-${index%COLORS.length}`,rx:2})); const label=svgElement('text',{x:x+barWidth/2,y:axes.y0+22,'text-anchor':'middle',class:'tick'}); label.textContent=row.mode; svg.appendChild(label); const valueLabel=svgElement('text',{x:x+barWidth/2,y:Math.max(axes.y1+12,y-5),'text-anchor':'middle',class:'tick'}); valueLabel.textContent=fmt(value); svg.appendChild(valueLabel); });
  $('bar-heading').textContent = `Aggregate comparison · ${$('role').value} · ${$('metric').selectedOptions[0].textContent}`;
  $('bar-legend').innerHTML = rows.map((row,index)=>`<span><i class="swatch" style="background:${COLORS[index%COLORS.length]}"></i>${row.mode}</span>`).join('');
}
function drawLines() {
  const rows = selectedTrials(), svg = $('line-chart'); clearSvg(svg);
  const width = Math.max(680, svg.clientWidth || 900), height = 320, margin = {left:72,right:20,top:18,bottom:42}; svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const modes = unique(rows.map(row => row.mode)); if (!rows.length) return;
  const byMode = Object.fromEntries(modes.map(mode => [mode, rows.filter(row => row.mode === mode).sort((a,b)=>a.trial-b.trial)]));
  const role = $('role').value;
  const values = rows.map(row => row[role].normalized_rmse); const yMax = Math.max(...values,1e-12)*1.12; const axes = drawAxes(svg,width,height,margin,yMax,`${role} normalized RMSE`);
  const maxTrial = Math.max(...rows.map(row=>row.trial),1), xScale = trial => axes.x0 + (axes.x1-axes.x0) * (maxTrial ? trial/maxTrial : 0), yScale = value => axes.y0 - (axes.y0-axes.y1) * value/yMax;
  for (let i=0;i<=4;i++) { const trial=Math.round(maxTrial*i/4), x=xScale(trial); const tick=svgElement('text',{x,y:axes.y0+20,'text-anchor':'middle',class:'tick'}); tick.textContent=trial; svg.appendChild(tick); }
  modes.forEach((mode,index)=>{ const points=byMode[mode].map(row=>`${xScale(row.trial)},${yScale(row[role].normalized_rmse)}`).join(' '); svg.appendChild(svgElement('polyline',{points,class:`line series-${index%COLORS.length}`})); });
  $('line-heading').textContent = `Per-trial normalized RMSE · ${role}`;
  $('line-legend').innerHTML=modes.map((mode,index)=>`<span><i class="swatch" style="background:${COLORS[index%COLORS.length]}"></i>${mode}</span>`).join('');
}
function drawTable() {
  const rows = selectedSummary(); if (!rows.length) { $('table').innerHTML='<div class="empty">No matching data.</div>'; return; }
  $('table').innerHTML=`<table><thead><tr><th>mode</th><th>MAE</th><th>RMSE</th><th>normalized RMSE</th><th>relative MAE</th><th>relative p95</th><th>cosine</th><th>zero fraction</th><th>saturation</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${row.mode}</td><td>${fmt(row.mae)}</td><td>${fmt(row.rmse)}</td><td>${fmt(row.normalized_rmse)}</td><td>${fmt(row.relative_mae)}</td><td>${fmt(row.relative_percentiles.p95)}</td><td>${fmt(row.cosine_similarity)}</td><td>${fmt(row.quantized_zero_fraction)}</td><td>${fmt(row.quantized_saturation_fraction)}</td></tr>`).join('')}</tbody></table>`;
}
function render() { drawBars(); drawLines(); drawTable(); }
options($('implementation'), unique(REPORT.summary.map(row=>row.implementation))); options($('distribution'), unique(REPORT.summary.map(row=>row.distribution)));
$('subtitle').textContent=`${REPORT.config.trials} trials · ${REPORT.config.tokens_per_trial} tokens/trial · ${REPORT.config.in_features}→${REPORT.config.out_features} · input ${REPORT.config.input_dtype} · matmul output ${REPORT.config.matmul_output_dtype} · seed ${REPORT.config.seed}`;
$('metadata').textContent=JSON.stringify({created_utc:REPORT.created_utc,source:REPORT.source,config:REPORT.config},null,2);
['implementation','distribution','role','metric'].forEach(id=>$(id).addEventListener('change',render)); window.addEventListener('resize',render); render();
</script>
</body>
</html>
"""


def _write_csv(data: dict[str, Any], path: Path) -> None:
    fields = [
        "implementation",
        "distribution",
        "mode",
        "role",
        "count",
        "mae",
        "rmse",
        "normalized_rmse",
        "relative_mae",
        "relative_rmse",
        "relative_p50",
        "relative_p95",
        "relative_p99",
        "relative_p999",
        "cosine_similarity",
        "max_abs",
        "quantized_zero_fraction",
        "quantized_saturation_fraction",
        "scale_nan_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data.get("summary", []):
            output = dict(row)
            percentiles = output.pop("relative_percentiles", {})
            for key in ("p50", "p95", "p99", "p999"):
                output[f"relative_{key}"] = percentiles.get(key)
            writer.writerow({field: output.get(field) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON output from mxfp_qdq_error.py")
    parser.add_argument("--output", type=Path, default=None, help="HTML report path; defaults beside input")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional aggregate CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = (args.output or input_path.with_suffix(".html")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    report = REPORT_TEMPLATE.replace("__REPORT_JSON__", embedded)
    output_path.write_text(report, encoding="utf-8")
    csv_path = args.csv_output
    if csv_path is not None:
        _write_csv(data, csv_path.resolve())
    print(f"wrote {output_path}")
    if csv_path is not None:
        print(f"wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
