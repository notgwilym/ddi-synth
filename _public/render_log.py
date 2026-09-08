"""Render experiments.json as one self-contained HTML file.

The page is a chronological ledger, not a dashboard. It exists to answer "what happened,
when, and what did it score", so the date axis is the spine and the notes field is given
as much room as the numbers. Everything is embedded; the file opens with no server and
no network.
"""
import argparse
import json
from pathlib import Path

from collect_runs import build

CSS = """
:root {
  --paper: #fbfaf7;
  --rule: #ddd9d0;
  --ink: #22201c;
  --quiet: #7b756a;
  --real: #2f6f6a;
  --llm: #8a5a1f;
  --synth: #6a4a8f;
  --ceiling: #b03a2e;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.num, .id { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
            font-variant-numeric: tabular-nums; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 28px 120px; }

header h1 { font-size: 26px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }
header p { margin: 0; color: var(--quiet); max-width: 62ch; }

.chart { margin: 34px 0 10px; border-top: 1px solid var(--rule);
         border-bottom: 1px solid var(--rule); padding: 22px 0 8px; }
.chart svg { width: 100%; height: 300px; display: block; }
.legend { display: flex; gap: 20px; flex-wrap: wrap; color: var(--quiet);
          font-size: 13px; padding-top: 8px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
            margin-right: 6px; vertical-align: 1px; }

.controls { display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
            margin: 26px 0 8px; }
.controls input[type=search] {
  flex: 1 1 300px; min-width: 220px; padding: 9px 12px; font: inherit;
  background: #fff; border: 1px solid var(--rule); border-radius: 3px; color: inherit;
}
.controls label { color: var(--quiet); font-size: 14px; cursor: pointer;
                  display: inline-flex; align-items: center; gap: 6px; }
.count { color: var(--quiet); font-size: 14px; }

.day { display: grid; grid-template-columns: 108px 1fr; gap: 20px;
       border-top: 1px solid var(--rule); padding: 18px 0 6px; }
.day > h2 { position: sticky; top: 12px; align-self: start; margin: 0;
            font-size: 14px; font-weight: 600; }
.day > h2 span { display: block; color: var(--quiet); font-weight: 400; font-size: 13px; }

.exp { padding: 9px 0 11px; border-bottom: 1px dotted var(--rule); }
.exp:last-child { border-bottom: 0; }
.exp > summary { cursor: pointer; list-style: none; display: grid;
                 grid-template-columns: 1fr 74px 62px 62px 46px; gap: 12px;
                 align-items: baseline; }
.exp > summary::-webkit-details-marker { display: none; }
.exp[open] > summary .name { font-weight: 600; }
.name { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.name b { font-weight: 600; }
.name .q { color: var(--quiet); font-size: 13px; }
.f1 { text-align: right; font-size: 15px; }
.f1 .sd { color: var(--quiet); font-size: 12px; }
.pr { text-align: right; color: var(--quiet); font-size: 13px; }
.seeds { text-align: right; color: var(--quiet); font-size: 12px; }
.note { color: var(--quiet); font-size: 13.5px; margin: 3px 0 0;
        grid-column: 1 / -1; max-width: 78ch; }
.tag { font-size: 11px; padding: 1px 5px; border-radius: 2px; border: 1px solid; }
.t-real { color: var(--real); border-color: var(--real); }
.t-llm { color: var(--llm); border-color: var(--llm); }
.t-synth { color: var(--synth); border-color: var(--synth); }
.t-dirty { color: var(--ceiling); border-color: var(--ceiling); }
.t-one { color: var(--quiet); border-color: var(--rule); }

.detail { margin: 12px 0 4px 0; padding: 14px 16px; background: #fff;
          border: 1px solid var(--rule); border-radius: 3px; font-size: 13px; }
.detail h3 { margin: 0 0 8px; font-size: 12px; font-weight: 600; color: var(--quiet); }
.detail + .detail { margin-top: 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
        gap: 4px 20px; }
.kv { display: flex; justify-content: space-between; gap: 12px;
      border-bottom: 1px dotted var(--rule); padding: 2px 0; }
.kv span:first-child { color: var(--quiet); }
.ids { color: var(--quiet); font-size: 11.5px; word-break: break-all; line-height: 1.7; }
.empty { padding: 40px 0; color: var(--quiet); }
@media (max-width: 720px) {
  .day { grid-template-columns: 1fr; gap: 6px; }
  .day > h2 { position: static; }
  .exp > summary { grid-template-columns: 1fr 70px; }
  .pr, .seeds { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = r"""
const DATA = __DATA__;
const CEIL = 0.790;

function kind(e) {
  const n = (e.name + ' ' + (e.provenance ? e.provenance.provenance || '' : '')).toLowerCase();
  if (n.includes('llm-label') || n.includes('label-')) return 'llm';
  if (n.startsWith('human') || n.startsWith('xr-human')) return 'real';
  if (n.includes('synth') || n.startsWith('v1') || n.includes('union')) return 'synth';
  return 'real';
}
const COLOR = { real: '#2f6f6a', llm: '#8a5a1f', synth: '#6a4a8f' };
const f1 = e => (e.metrics.micro_f1_pos || {}).mean;
const fmt = (v, d = 3) => v == null ? '\u2014' : v.toFixed(d);

function chart(exps) {
  const W = 1120, H = 300, L = 40, R = 12, T = 14, B = 34;
  const ts = exps.map(e => Date.parse(e.first_run));
  const t0 = Math.min(...ts), t1 = Math.max(...ts);
  const x = t => L + (t1 === t0 ? 0.5 : (t - t0) / (t1 - t0)) * (W - L - R);
  const y = v => T + (1 - v) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Micro-F1 of every experiment over time">`;
  for (let v = 0; v <= 1.0001; v += 0.2) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#ddd9d0"/>`;
    s += `<text x="4" y="${y(v) + 4}" fill="#7b756a" font-size="11" font-family="monospace">${v.toFixed(1)}</text>`;
  }
  s += `<line x1="${L}" x2="${W - R}" y1="${y(CEIL)}" y2="${y(CEIL)}" stroke="#b03a2e" stroke-dasharray="4 3"/>`;
  s += `<text x="${W - R}" y="${y(CEIL) - 6}" text-anchor="end" fill="#b03a2e" font-size="11">human ceiling 0.790</text>`;
  const days = new Map();
  exps.forEach(e => days.set(e.date, Date.parse(e.first_run)));
  [...days].forEach(([d, t]) => {
    s += `<text x="${x(t)}" y="${H - 12}" text-anchor="middle" fill="#7b756a" font-size="10.5">${d.slice(5)}</text>`;
  });
  exps.forEach(e => {
    const v = f1(e); if (v == null) return;
    const sd = (e.metrics.micro_f1_pos || {}).sd;
    const c = COLOR[kind(e)];
    if (sd) s += `<line x1="${x(Date.parse(e.first_run))}" x2="${x(Date.parse(e.first_run))}" y1="${y(v - sd)}" y2="${y(v + sd)}" stroke="${c}" stroke-opacity="0.45"/>`;
    s += `<circle cx="${x(Date.parse(e.first_run))}" cy="${y(v)}" r="3.4" fill="${c}" fill-opacity="0.8"><title>${e.date}  ${e.name}  F1 ${fmt(v)} (n=${e.n_seeds})</title></circle>`;
  });
  return s + '</svg>';
}

function detail(e) {
  const cfg = Object.entries(e.config).map(([k, v]) =>
    `<div class="kv"><span>${k}</span><span class="num">${v === null ? 'null' : v}</span></div>`).join('');
  const per = Object.entries(e.per_label).map(([k, v]) =>
    `<div class="kv"><span>${k}</span><span class="num">${fmt(v.f1)} <span style="color:#7b756a">n=${Math.round(v.support)}</span></span></div>`).join('');
  const p = e.provenance;
  const prov = p ? Object.entries(p).filter(([k, v]) => v != null && k !== 'label_distribution')
    .map(([k, v]) => `<div class="kv"><span>${k}</span><span class="num">${typeof v === 'object' ? JSON.stringify(v) : v}</span></div>`).join('') : '';
  const ld = p && p.label_distribution ? Object.entries(p.label_distribution)
    .map(([k, v]) => `<div class="kv"><span>${k}</span><span class="num">${v}</span></div>`).join('') : '';
  return `
    <div class="detail"><h3>Configuration</h3><div class="grid">${cfg}</div></div>
    ${per ? `<div class="detail"><h3>F1 by relation type</h3><div class="grid">${per}</div></div>` : ''}
    ${prov ? `<div class="detail"><h3>Training data</h3><div class="grid">${prov}</div>
      ${ld ? `<h3 style="margin-top:12px">Label distribution</h3><div class="grid">${ld}</div>` : ''}</div>` : ''}
    <div class="detail"><h3>Provenance</h3>
      <div class="kv"><span>git</span><span class="num">${e.git_shas.join(', ')}${e.git_dirty ? '  uncommitted changes' : ''}</span></div>
      <div class="kv"><span>seeds</span><span class="num">${e.seeds.join(', ')}</span></div>
      <div class="kv"><span>first run</span><span class="num">${e.first_run}</span></div>
      <div class="kv"><span>last run</span><span class="num">${e.last_run}</span></div>
      <p class="ids" style="margin:8px 0 0">${e.run_ids.join('  ')}</p>
    </div>`;
}

function row(e) {
  const m = e.metrics, v = f1(e), sd = (m.micro_f1_pos || {}).sd;
  const k = kind(e);
  const tags = [`<span class="tag t-${k}">${k === 'real' ? 'human text + labels' : k === 'llm' ? 'LLM labels' : 'synthetic'}</span>`];
  if (e.n_seeds === 1) tags.push('<span class="tag t-one">1 seed</span>');
  if (e.git_dirty) tags.push('<span class="tag t-dirty">uncommitted</span>');
  return `<details class="exp"><summary>
      <span class="name"><b>${e.name}</b>
        ${e.qualifiers.length ? `<span class="q">${e.qualifiers.join('  ')}</span>` : ''}
        ${tags.join(' ')}</span>
      <span class="f1 num">${fmt(v)}${sd != null ? ` <span class="sd">±${sd.toFixed(3)}</span>` : ''}</span>
      <span class="pr num">P ${fmt((m.micro_p_pos || {}).mean)}</span>
      <span class="pr num">R ${fmt((m.micro_r_pos || {}).mean)}</span>
      <span class="seeds num">${e.n_seeds}\u00d7</span>
      ${e.notes.filter(Boolean).length ? `<p class="note">${e.notes.filter(Boolean).join(' \u2014 ')}</p>` : ''}
    </summary>${detail(e)}</details>`;
}

function render() {
  const q = document.getElementById('q').value.toLowerCase().trim();
  const multi = document.getElementById('multi').checked;
  const arch = document.getElementById('arch').checked;
  const exps = DATA.experiments.filter(e => {
    if (multi && e.n_seeds < 2) return false;
    if (!arch && e.archived) return false;
    if (!q) return true;
    return (e.name + ' ' + e.qualifiers.join(' ') + ' ' + e.notes.join(' ') + ' ' +
            (e.train_id || '') + ' ' + JSON.stringify(e.provenance || {})).toLowerCase().includes(q);
  });

  document.getElementById('chart').innerHTML = exps.length ? chart(exps) : '';
  document.getElementById('count').textContent =
    `${exps.length} experiments, ${exps.reduce((a, e) => a + e.n_seeds, 0)} training runs`;

  const days = new Map();
  exps.forEach(e => { if (!days.has(e.date)) days.set(e.date, []); days.get(e.date).push(e); });
  const out = [...days].map(([d, es]) => {
    const wd = new Date(d + 'T12:00:00').toLocaleDateString('en-GB', { weekday: 'long' });
    return `<section class="day"><h2>${d}<span>${wd}</span><span>${es.length} exp</span></h2>
            <div>${es.map(row).join('')}</div></section>`;
  }).join('');
  document.getElementById('log').innerHTML = out ||
    '<p class="empty">Nothing matches that search.</p>';
}

['q', 'multi', 'arch'].forEach(id =>
  document.getElementById(id).addEventListener('input', render));
render();
"""

HTML = """<!doctype html>
<html lang="en-GB"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DDI experiment log</title>
<style>__CSS__</style>
</head><body>
<div class="wrap">
  <header>
    <h1>DDI synthetic data: every training run</h1>
    <p>__SUB__</p>
  </header>

  <div class="chart"><div id="chart"></div>
    <div class="legend">
      <span><i style="background:#2f6f6a"></i>human text and labels</span>
      <span><i style="background:#8a5a1f"></i>LLM labels on real text</span>
      <span><i style="background:#6a4a8f"></i>synthetic text</span>
      <span>vertical bars are one standard deviation across seeds</span>
    </div>
  </div>

  <div class="controls">
    <input type="search" id="q" placeholder="Search names, notes, dataset ids">
    <label><input type="checkbox" id="multi"> Two or more seeds only</label>
    <label><input type="checkbox" id="arch" checked> Include archived</label>
    <span class="count" id="count"></span>
  </div>

  <div id="log"></div>
</div>
<script>__JS__</script>
</body></html>
"""


def render(data, out_path):
    lean = {"experiments": data["experiments"]}      # manifests already inlined
    n_exp = len(data["experiments"])
    dates = [e["date"] for e in data["experiments"]]
    dirty = sum(1 for e in data["experiments"] if e["git_dirty"])
    sub = (f"{data['n_runs']} logged training runs, grouped into {n_exp} experiments by "
           f"configuration, spanning {min(dates)} to {max(dates)}. "
           f"{dirty} experiments were logged against a tree with uncommitted changes and "
           f"cannot be reproduced from a commit alone. Nothing is filtered out: "
           f"single-seed probes and arms that went nowhere are kept and marked.")

    payload = json.dumps(lean, separators=(",", ":")).replace("<", "\\u003c")
    js = JS.replace("__DATA__", payload)
    html = (HTML.replace("__CSS__", CSS).replace("__SUB__", sub).replace("__JS__", js))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--manifests", default="datasets/manifests")
    ap.add_argument("--out", default="reports/experiment_log.html")
    a = ap.parse_args()
    data = build(a.runs, a.manifests)
    p = render(data, a.out)
    print(f"{data['n_runs']} runs, {len(data['experiments'])} experiments -> {p} "
          f"({Path(p).stat().st_size / 1e6:.2f} MB)")
