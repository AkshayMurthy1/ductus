"""Self-contained HTML eval report. One file per run; diff two runs by opening both.

No CDN, no build step -- inline SVG generated in Python so the report renders on a GPU box with
no network. Palette is the validated two-series default (blue #2a78d6 / orange #eb6834; dark
steps #3987e5 / #d95926), which clears all-pairs CVD and contrast gates in both modes.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

SURFACE_LIGHT = "#fcfcfb"
CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-1: #fcfcfb; --surface-2: #f4f3f0; --border: #dedcd6;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7c7a74;
  --series-1: #2a78d6; --series-2: #eb6834;
  --good: #008300; --bad: #e34948; --warn: #eda100;
  --grid: #e7e5e0;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1: #1a1a19; --surface-2: #232322; --border: #3a3a37;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #93918a;
    --series-1: #3987e5; --series-2: #d95926;
    --good: #4caf50; --bad: #e66767; --warn: #c98500;
    --grid: #2e2e2c;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1: #1a1a19; --surface-2: #232322; --border: #3a3a37;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #93918a;
  --series-1: #3987e5; --series-2: #d95926;
  --good: #4caf50; --bad: #e66767; --warn: #c98500; --grid: #2e2e2c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-1); }
.viz-root {
  background: var(--surface-1); color: var(--text-primary); min-height: 100vh;
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  padding: 40px 28px 72px; max-width: 1080px; margin: 0 auto;
}
h1 { font-size: 25px; font-weight: 620; letter-spacing: -0.015em; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 44px 0 6px; letter-spacing: -0.005em; }
.sub { color: var(--text-secondary); font-size: 13.5px; margin: 0 0 6px; }
.muted { color: var(--text-muted); font-size: 12.5px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
         gap: 12px; margin: 22px 0 0; }
.tile { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
        background: var(--surface-2); }
.tile .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em;
               color: var(--text-muted); font-weight: 600; }
.tile .value { font-size: 27px; font-weight: 640; letter-spacing: -0.02em; margin-top: 5px;
               font-variant-numeric: tabular-nums; }
.tile .note { font-size: 12px; color: var(--text-secondary); margin-top: 3px; }
.pill { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 600;
        border-radius: 999px; padding: 2px 9px; border: 1px solid currentColor; }
.pass { color: var(--good); } .fail { color: var(--bad); } .unk { color: var(--warn); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 10px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: .05em; }
td { font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 16px; align-items: center; margin: 4px 0 10px;
          font-size: 12.5px; color: var(--text-secondary); }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block;
          margin-right: 6px; vertical-align: -1px; }
.samples { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }
.card { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
        background: var(--surface-2); font-size: 13.5px; white-space: pre-wrap; }
.card .who { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em;
             font-weight: 600; margin-bottom: 8px; }
pre.cfg { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
          padding: 14px 16px; font-size: 12px; overflow-x: auto; color: var(--text-secondary); }
svg text { font: 11.5px ui-sans-serif, -apple-system, sans-serif; }
.bar:hover rect { opacity: .82; }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--text-secondary);
          font-size: 13px; }
.toggle { position: fixed; top: 14px; right: 16px; font-size: 12px; padding: 5px 11px;
          border-radius: 999px; border: 1px solid var(--border); background: var(--surface-2);
          color: var(--text-secondary); cursor: pointer; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _verdict_pill(text: str | None) -> str:
    if not text:
        return '<span class="pill unk">not run</span>'
    cls = "pass" if text.upper().startswith("PASS") else (
        "fail" if text.upper().startswith("FAIL") else "unk"
    )
    return f'<span class="pill {cls}">{_esc(text)}</span>'


# --------------------------------------------------------------------------- charts
def hbar_chart(
    rows: list[tuple[str, float]],
    *,
    width: int = 660,
    row_h: int = 26,
    label_w: int = 210,
    vmax: float | None = None,
    unit: str = "",
) -> str:
    """Horizontal bars, single series -> one hue, no legend, every bar directly labeled.

    4px rounded data-end anchored to the baseline; recessive gridlines; hover tooltip via
    <title> so it works with no JS.
    """
    if not rows:
        return '<p class="muted">no data</p>'
    vmax = vmax or max(max(v for _, v in rows), 1e-9)
    plot_w = width - label_w - 56
    height = row_h * len(rows) + 26
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="stylometry gaps">'
    ]
    # gridlines
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = label_w + plot_w * frac
        parts.append(
            f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{row_h * len(rows) + 8}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{row_h * len(rows) + 22}" fill="var(--text-muted)" '
            f'text-anchor="middle">{vmax * frac:.2f}</text>'
        )
    for i, (name, val) in enumerate(rows):
        y = 14 + i * row_h
        w = max(2.0, plot_w * (val / vmax))
        parts.append(
            f'<g class="bar"><title>{_esc(name)}: {val:.4f}{unit}</title>'
            f'<text x="{label_w - 10}" y="{y + 11}" text-anchor="end" '
            f'fill="var(--text-secondary)">{_esc(name)}</text>'
            f'<rect x="{label_w}" y="{y + 2}" width="{w:.1f}" height="14" rx="4" '
            f'fill="var(--series-1)"/>'
            f'<text x="{label_w + w + 7:.1f}" y="{y + 13}" fill="var(--text-primary)">'
            f"{val:.3f}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    width: int = 660,
    height: int = 220,
    ylabel: str = "share",
) -> str:
    """Two series max. Legend is emitted by the caller; 2px surface gap between adjacent bars."""
    if not categories or not series:
        return '<p class="muted">no data</p>'
    pad_l, pad_b, pad_t = 44, 34, 12
    plot_w = width - pad_l - 14
    plot_h = height - pad_b - pad_t
    vmax = max((max(v) for _, v in series), default=1.0) or 1.0
    vmax = vmax * 1.12
    group_w = plot_w / len(categories)
    bar_w = max(3.0, (group_w - 8) / len(series) - 2)  # 2px surface gap

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="{_esc(ylabel)} by bucket">'
    ]
    for frac in (0, 0.5, 1.0):
        y = pad_t + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="var(--text-muted)">{vmax * frac:.2f}</text>'
        )
    for gi, cat in enumerate(categories):
        gx = pad_l + gi * group_w + 4
        for si, (sname, vals) in enumerate(series):
            v = vals[gi] if gi < len(vals) else 0.0
            h = plot_h * (v / vmax)
            x = gx + si * (bar_w + 2)
            y = pad_t + plot_h - h
            color = "var(--series-1)" if si == 0 else "var(--series-2)"
            parts.append(
                f'<g class="bar"><title>{_esc(sname)} · {_esc(cat)}: {v:.3f}</title>'
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(1.0, h):.1f}" '
                f'rx="4" fill="{color}"/></g>'
            )
        parts.append(
            f'<text x="{gx + group_w / 2 - 4:.1f}" y="{height - 12}" text-anchor="middle" '
            f'fill="var(--text-muted)">{_esc(cat)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def legend(series: list[str]) -> str:
    """A legend is always present for >= 2 series -- identity is never color-alone."""
    colors = ["var(--series-1)", "var(--series-2)"]
    items = "".join(
        f'<span><span class="swatch" style="background:{colors[i % 2]}"></span>{_esc(s)}</span>'
        for i, s in enumerate(series)
    )
    return f'<div class="legend">{items}</div>'


SL_BUCKETS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-60", "60+"]


# --------------------------------------------------------------------------- report
def render_report(result: dict[str, Any]) -> str:
    run = result.get("run_name", "run")
    av = result.get("av") or {}
    sty = result.get("stylometry") or {}
    flu = result.get("fluency") or {}
    leak = result.get("leakage") or {}
    gaps = result.get("biggest_gaps") or []
    baseline = result.get("baseline_comparison") or {}

    attribution = av.get("attribution_rate")
    tiles = [
        (
            "AV attribution rate",
            f"{attribution:.0%}" if isinstance(attribution, (int, float)) else "—",
            f"verifier AUC {_fmt(av.get('verifier_auc'), 3)} · threshold "
            f"{_fmt(av.get('threshold'), 3)}",
        ),
        (
            "Stylometry distance",
            _fmt(sty.get("overall"), 4),
            (
                f"95% CI [{_fmt(sty.get('ci_lo'), 3)}, {_fmt(sty.get('ci_hi'), 3)}] · "
                "lower is better"
            ),
        ),
        (
            "Fluency",
            _fmt(flu.get("relative_regression"), 3) if flu.get("relative_regression") is not None
            else "—",
            f"ppl on {_fmt(flu.get('ppl_adapter_on'), 2)} vs off "
            f"{_fmt(flu.get('ppl_adapter_off'), 2)}",
        ),
        (
            "Content leakage",
            _fmt((leak.get("verbatim") or {}).get("rate"), 3),
            f"longest verbatim run "
            f"{(leak.get('verbatim') or {}).get('longest_verbatim_run_tokens', '—')} tokens",
        ),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{_esc(a)}</div>'
        f'<div class="value">{_esc(b)}</div><div class="note">{_esc(c)}</div></div>'
        for a, b, c in tiles
    )

    verdicts = (
        f'<p class="sub">fluency {_verdict_pill(flu.get("verdict"))} &nbsp; '
        f'leakage {_verdict_pill(leak.get("verdict"))} &nbsp; '
        f'verifier {_verdict_pill(av.get("verifier_verdict"))}</p>'
    )

    # --- gap chart
    gap_rows = [(n.split(":", 1)[-1].replace("_", " "), float(v)) for n, v in gaps]
    gap_html = hbar_chart(gap_rows, vmax=max([v for _, v in gap_rows] + [0.3]))

    # --- sentence-length distribution: real vs generated
    sl = result.get("sent_len_hist") or {}
    sl_html, sl_legend = "", ""
    if sl.get("real") and sl.get("generated"):
        sl_legend = legend(["author (real)", "model"])
        sl_html = grouped_bar_chart(
            SL_BUCKETS,
            [("author (real)", sl["real"]), ("model", sl["generated"])],
            ylabel="share of sentences",
        )

    # --- baseline comparison
    cmp_html = ""
    if baseline:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_fmt(v.get('baseline'), 4)}</td>"
            f"<td>{_fmt(v.get('this_run'), 4)}</td><td>{_fmt(v.get('delta'), 4)}</td></tr>"
            for k, v in baseline.items()
        )
        cmp_html = (
            "<h2>Versus the Phase-0 prompting baseline</h2>"
            '<p class="sub">The number that decides whether tuning earned its keep.</p>'
            "<table><thead><tr><th>metric</th><th>baseline</th><th>this run</th>"
            f"<th>delta</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    # --- scalar table (the required table view)
    scal = (sty.get("scalars_raw") or {})
    scal_rows = "".join(
        f"<tr><td>{_esc(k.replace('_', ' '))}</td><td>{_fmt(v[0], 3)}</td>"
        f"<td>{_fmt(v[1], 3)}</td>"
        f"<td>{_fmt(sty.get('scalars_normalized', {}).get(k), 3)}</td></tr>"
        for k, v in scal.items()
    )
    dist_rows = "".join(
        f"<tr><td>{_esc(k.replace('_', ' '))}</td><td colspan='2' "
        f"class='muted'>Jensen-Shannon</td><td>{_fmt(v, 3)}</td></tr>"
        for k, v in (sty.get("distributions") or {}).items()
    )

    # --- samples
    samples = result.get("samples") or []
    sample_html = "".join(
        f'<div class="samples">'
        f'<div class="card"><div class="who" style="color:var(--series-1)">author (real)</div>'
        f"{_esc(s.get('real', ''))}</div>"
        f'<div class="card"><div class="who" style="color:var(--series-2)">model</div>'
        f"{_esc(s.get('generated', ''))}</div></div>"
        for s in samples[:6]
    )

    smoke = result.get("smoke") or []
    smoke_html = "".join(
        f"<details><summary>{_esc(s['prompt'])}</summary>"
        f'<div class="card">{_esc(s["completion"])}</div></details>'
        for s in smoke
    )

    cfg = json.dumps(result.get("config") or {}, indent=2)[:12000]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ductus — {_esc(run)}</title><style>{CSS}</style></head>
<body><div class="viz-root">
<button class="toggle" onclick="document.documentElement.dataset.theme =
  document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'">theme</button>

<h1>{_esc(run)}</h1>
<p class="sub">{_esc(result.get('generated_at', ''))} · {_esc(result.get('n_generations', 0))}
generations on the <strong>{_esc(result.get('split', 'blind'))}</strong> split ·
model {_esc(result.get('model', '—'))}</p>
{verdicts}
<div class="tiles">{tiles_html}</div>

{cmp_html}

<h2>Where the voice is still off</h2>
<p class="sub">Largest normalized gaps between the model's output and the author's real text.
This is the to-do list: the top item is what the next ablation should target.</p>
{gap_html}

<h2>Sentence-length rhythm</h2>
<p class="sub">Distribution shape, not just the mean. Matching the mean while missing the
spread is the classic signature of generic AI cadence.</p>
{sl_legend}{sl_html}

<h2>Stylometry detail</h2>
<table><thead><tr><th>feature</th><th>author</th><th>model</th><th>normalized gap</th></tr>
</thead><tbody>{scal_rows}{dist_rows}</tbody></table>

<h2>Side by side</h2>
<p class="sub">Read these. The human blind test is the real bar; the numbers above are proxies
for it.</p>
{sample_html or '<p class="muted">no paired samples provided</p>'}

<h2>Fluency smoke test</h2>
<p class="sub">General questions, greedy decoding. Looping, dropped syntax, or answering a
factual question in the author's voice all mean the adapter is bleeding into capability.</p>
{smoke_html or '<p class="muted">not run</p>'}

<h2>Run configuration</h2>
<pre class="cfg">{_esc(cfg)}</pre>
</div></body></html>
"""


def write_report(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result), encoding="utf-8")
    (path.parent / (path.stem + ".json")).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return path
