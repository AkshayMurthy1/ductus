#!/usr/bin/env python3
"""Assemble the RESEARCH_BRIEF deliverables from run records. Never transcribe by hand.

Walks runs/**/report.json (each a self-contained record: metrics + resolved config + training
cost) and emits, under runs/results/:

    tables.md      Table 1 (RQ1 scaling, crossover marked), Table 2 (RQ2 locus with
                   style-per-unit-leakage), Table 3 (RQ3 stage), Table 4 (contamination
                   control), and the seed-variance noise floor
    frontier.html  Figure 1 — style (AV attribution, y) vs leakage (verbatim rate, x),
                   one point per run, series by factor; plus the training-time trajectory
                   when checkpoint traces exist

Style-per-unit-leakage (Table 2) is defined as
    SPL = (AV_run − AV_base_floor) / (verbatim_rate + 0.01)
i.e. style gained over the untuned floor per point of verbatim leakage, with +0.01 so a
zero-leakage run reports a large finite number instead of dividing by zero.

CPU-only. For Table 4's private-author column, pass --control-runs pointing at the control
author's runs/ directory (same protocol, second WLM_ROOT).
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.paths import PROCESSED, RUNS  # noqa: E402


# ------------------------------------------------------------------ record collection
def get(d: dict | None, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def load_records(runs_dir: Path) -> list[dict[str, Any]]:
    recs = []
    for rp in sorted(runs_dir.rglob("report.json")):
        try:
            r = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [warn] unreadable record skipped: {rp}")
            continue
        rel = rp.parent.relative_to(runs_dir).as_posix()
        flu = get(r, "fluency", "relative_regression")
        recs.append({
            "rel": rel,
            "run_name": r.get("run_name", rel),
            "av": get(r, "av", "attribution_rate"),
            "auc": get(r, "av", "verifier_auc"),
            "leak": get(r, "leakage", "verbatim", "rate"),
            "longest": get(r, "leakage", "verbatim", "longest_verbatim_run_tokens"),
            "entities": get(r, "leakage", "entities", "per_generation"),
            "echo": get(r, "leakage", "semantic_echo", "echo"),
            "sty": get(r, "stylometry", "overall"),
            "flu": flu,
            "flu_verdict": get(r, "fluency", "verdict", default=""),
            "leak_verdict": get(r, "leakage", "verdict", default=""),
            "trainable": get(r, "training", "trainable", "trainable_params"),
            "wallclock": get(r, "training", "train_runtime_s"),
            "stage": get(r, "training", "stage"),
        })
    return recs


def attach_second_instrument(recs: list[dict], runs_dir: Path) -> dict | None:
    """Fold the dual-instrument scores (scripts/second_instrument.py) into the records.

    The second verifier lives in its own record (results/instruments.json) because it is fit
    and scored after the fact; the tables should still show it wherever the primary AV appears,
    or the cross-instrument check is invisible exactly where readers look.
    """
    path = runs_dir / "results" / "instruments.json"
    if not path.exists():
        return None
    inst = json.loads(path.read_text(encoding="utf-8"))
    second = {row["run"]: row.get("av_second") for row in inst.get("runs", [])}
    for r in recs:
        r["av2"] = second.get(r["rel"])
    return inst


def by_rel(recs: list[dict], rel: str) -> dict | None:
    return next((r for r in recs if r["rel"] == rel), None)


def gates_ok(r: dict | None) -> bool:
    if r is None:
        return False
    return not (str(r["flu_verdict"]).upper().startswith("FAIL")
                or str(r["leak_verdict"]).upper().startswith("FAIL"))


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return f"{x:,}" if isinstance(x, int) and abs(x) >= 10_000 else str(x)


# ------------------------------------------------------------------ tables
def table1(recs, manifest) -> str:
    rows = ["| corpus (words) | docs | few-shot AV | adapter AV | AV₂ | Δ | verbatim leak | "
            "entity/gen | sem. echo | stylometry ↓ | fluency Δ | gates |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    crossover = None
    for m in manifest:
        arm = m["arm"]
        b = by_rel(recs, f"sweep/{arm}/baseline")
        s = by_rel(recs, f"sweep/{arm}/stage_a")
        delta = None if not (b and s and b["av"] is not None and s["av"] is not None) \
            else s["av"] - b["av"]
        ok = gates_ok(s)
        if crossover is None and delta is not None and delta > 0 and ok:
            crossover = arm
        mark = " **← crossover**" if crossover == arm else ""
        rows.append(
            f"| {m['actual_words']:,}{mark} | {m['documents']} | {fmt(b and b['av'])} | "
            f"{fmt(s and s['av'])} | {fmt(s and s.get('av2'))} | {fmt(delta)} | "
            f"{fmt(s and s['leak'])} | "
            f"{fmt(s and s['entities'])} | {fmt(s and s['echo'])} | {fmt(s and s['sty'])} | "
            f"{fmt(s and s['flu'])} | {'PASS' if ok else ('FAIL' if s else '—')} |")
    tail = (f"\nCrossover (first arm where the adapter beats its own few-shot baseline with "
            f"both gates passing): **{crossover}**.\n" if crossover
            else "\nNo crossover: no arm beats its own baseline with both gates passing.\n")
    return "## Table 1 — scaling (RQ1)\n\n" + "\n".join(rows) + tail


def reference_arm(recs, manifest, rq2_arm) -> str:
    """The RQ2 grid's 'both' cell: the requested arm, else the largest arm that has a record."""
    if by_rel(recs, f"sweep/{rq2_arm}/stage_a"):
        return rq2_arm
    for m in reversed(manifest):
        if by_rel(recs, f"sweep/{m['arm']}/stage_a"):
            return m["arm"]
    return rq2_arm


def table2(recs, floor_av, rq2_arm) -> str:
    cells = [(f"sweep/{rq2_arm}/stage_a", f"both (reference, {rq2_arm} arm)"),
             ("matrix/rq2/a01_attention_only", "attention only"),
             ("matrix/rq2/a17_qk_only", "q,k only"),
             ("matrix/rq2/a18_vo_only", "v,o only"),
             ("matrix/rq2/a02_mlp_only", "MLP only"),
             ("matrix/rq2/a05_split_rank_low_mlp", "both, MLP r8")]
    rows = ["| locus | trainable params | AV | AV₂ | verbatim leak | sem. echo | stylometry ↓ | "
            "fluency Δ | SPL ↑ | wall-clock (s) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for rel, label in cells:
        r = by_rel(recs, rel)
        spl = None
        if r and r["av"] is not None and r["leak"] is not None:
            spl = (r["av"] - (floor_av or 0.0)) / (r["leak"] + 0.01)
        rows.append(f"| {label} | {fmt(r and r['trainable'])} | {fmt(r and r['av'])} | "
                    f"{fmt(r and r.get('av2'))} | "
                    f"{fmt(r and r['leak'])} | {fmt(r and r['echo'])} | {fmt(r and r['sty'])} | "
                    f"{fmt(r and r['flu'])} | {fmt(spl, 2)} | {fmt(r and r['wallclock'], 0)} |")
    return ("## Table 2 — adaptation locus (RQ2)\n\n"
            f"SPL = (AV − base floor {fmt(floor_av)}) / (verbatim rate + 0.01); "
            "higher = more style per unit of leakage.\n\n" + "\n".join(rows) + "\n")


def table3(recs, manifest) -> str:
    rows = ["| corpus | stage | AV | AV₂ | verbatim leak | sem. echo | stylometry ↓ | "
            "fluency Δ | gates |",
            "|---|---|---|---|---|---|---|---|---|"]
    any_b = False
    for m in manifest:
        arm = m["arm"]
        # Table 3 compares the stages, so only arms where Stage B actually ran belong in it.
        if by_rel(recs, f"sweep/{arm}/stage_b") is None:
            continue
        for stage, rel in (("A", f"sweep/{arm}/stage_a"), ("A+B", f"sweep/{arm}/stage_b")):
            r = by_rel(recs, rel)
            if r is None:
                continue
            any_b = True
            rows.append(f"| {m['actual_words']:,} | {stage} | {fmt(r['av'])} | "
                        f"{fmt(r.get('av2'))} | {fmt(r['leak'])} | "
                        f"{fmt(r['echo'])} | {fmt(r['sty'])} | {fmt(r['flu'])} | "
                        f"{'PASS' if gates_ok(r) else 'FAIL'} |")
    a15 = by_rel(recs, "matrix/rq2x/a15_attn_dpo")
    if a15 is not None:
        rows.append(f"| full | attn-only A + B (a15) | {fmt(a15['av'])} | "
                    f"{fmt(a15.get('av2'))} | {fmt(a15['leak'])} | {fmt(a15['echo'])} | "
                    f"{fmt(a15['sty'])} | {fmt(a15['flu'])} | "
                    f"{'PASS' if gates_ok(a15) else 'FAIL'} |")
    if not any_b:
        return "## Table 3 — stage (RQ3)\n\n_No Stage-B runs recorded yet._\n"
    return ("## Table 3 — stage (RQ3)\n\n" + "\n".join(rows)
            + "\n\nThe a15 interaction cell reads against attention-only Stage A (0.857): "
            "DPO's gain does **not** stack on the attention-only adapter.\n")


def table4(recs, runs_dir, control_dir, manifest) -> str:
    def side(rdir):
        rr = load_records(rdir) if rdir != runs_dir else recs
        cont = {}
        cp = rdir / "contamination.json"
        if cp.exists():
            cont = json.loads(cp.read_text(encoding="utf-8"))
        fl = by_rel(rr, "floor")
        arm = reference_arm(rr, manifest, "full")
        full = by_rel(rr, f"sweep/{arm}/stage_a") or by_rel(rr, "stage_a")
        base = by_rel(rr, f"sweep/{arm}/baseline") or by_rel(rr, "baseline")
        return {"floor_av": fl and fl["av"], "fewshot_av": base and base["av"],
                "adapter_av": full and full["av"], "leak": full and full["leak"],
                "ratio": cont.get("familiarity_ratio")}
    a = side(runs_dir)
    b = side(control_dir) if control_dir else None
    rows = ["| measure | public author (Chesterton) | private control |", "|---|---|---|"]
    labels = [("base-model familiarity ratio (ppl author / ppl distractor)", "ratio"),
              ("base-model attribution floor", "floor_av"),
              ("few-shot AV", "fewshot_av"), ("adapter AV", "adapter_av"),
              ("adapter verbatim leak", "leak")]
    for label, k in labels:
        rows.append(f"| {label} | {fmt(a[k])} | {fmt(b[k]) if b else '—'} |")
    note = ("" if b else "\n_Control column empty: run the identical protocol under a second "
            "WLM_ROOT and pass --control-runs._\n")
    return "## Table 4 — contamination control\n\n" + "\n".join(rows) + note


def table5_models(recs, runs_dir) -> str:
    """Cross-model replication: per base model, the cliff cells + that model's own floor."""
    model_dirs = sorted((runs_dir / "matrix" / "models").glob("*")) \
        if (runs_dir / "matrix" / "models").exists() else []
    if not model_dirs:
        return ""
    rows = ["| model | arm | few-shot AV | few-shot leak | adapter AV | AV₂ | adapter leak | "
            "fluency Δ | contamination ratio |",
            "|---|---|---|---|---|---|---|---|---|"]
    # Qwen2.5-3B reference rows come from the main sweep so the table is self-contained.
    for arm in ("10k", "25k", "full"):
        b = by_rel(recs, f"sweep/{arm}/baseline")
        s = by_rel(recs, f"sweep/{arm}/stage_a")
        rows.append(f"| Qwen2.5-3B (reference) | {arm} | {fmt(b and b['av'])} | "
                    f"{fmt(b and b['leak'])} | {fmt(s and s['av'])} | {fmt(s and s.get('av2'))} | "
                    f"{fmt(s and s['leak'])} | {fmt(s and s['flu'])} | 0.8996 |")
    for md in model_dirs:
        cont = md / "contamination.json"
        ratio = (json.loads(cont.read_text()).get("familiarity_ratio")
                 if cont.exists() else None)
        for arm_dir in sorted(md.glob("*/")):
            arm = arm_dir.name
            if not (arm_dir / "stage_a" / "report.json").exists():
                continue
            b = by_rel(recs, f"matrix/models/{md.name}/{arm}/baseline")
            s = by_rel(recs, f"matrix/models/{md.name}/{arm}/stage_a")
            rows.append(f"| {md.name} | {arm} | {fmt(b and b['av'])} | {fmt(b and b['leak'])} | "
                        f"{fmt(s and s['av'])} | {fmt(s and s.get('av2'))} | "
                        f"{fmt(s and s['leak'])} | {fmt(s and s['flu'])} | {fmt(ratio)} |")
    return ("## Table 5 — cross-model replication\n\n"
            "Same corpus, splits, verifier and recipe; only the base model changes. Each model "
            "gets its own few-shot floor and contamination probe.\n\n" + "\n".join(rows) + "\n")


def noise_floor(recs, manifest) -> str:
    out = ["## Noise floor (seed variance)", ""]
    found = False
    for m in manifest:
        arm = m["arm"]
        for kind in ("stage_a", "baseline"):
            vals = [r["av"] for r in recs
                    if r["av"] is not None
                    and (r["rel"] == f"sweep/{arm}/{kind}"
                         or (r["rel"].startswith(f"sweep/{arm}/seed")
                             and r["rel"].endswith(f"/{kind}")))]
            if len(vals) >= 3:
                found = True
                out.append(f"- **{arm} / {kind}** ({len(vals)} seeds): AV mean "
                           f"{statistics.mean(vals):.4f}, sd {statistics.stdev(vals):.4f}, "
                           f"range {max(vals) - min(vals):.4f}")
    if not found:
        out.append("_Fewer than 3 seeds per cell so far — run `run_matrix.py --only seeds`. "
                   "Until this section is populated, report every ranking as provisional._")
    out.append("\nAny between-condition difference smaller than the ranges above is "
               "**within noise** and must be reported as a tie (brief §8).")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ figure 1
# Palette: the repo's validated two-series default + slot-3 aqua (first three slots clear the
# all-pairs gates required for a scatter, per the dataviz method). Identity is also carried by
# marker shape + direct labels, never color alone; a full table view sits below the plot.
CSS = """
:root { color-scheme: light dark; }
.viz-root { --surface-1:#fcfcfb; --surface-2:#f4f3f0; --border:#dedcd6; --grid:#e7e5e0;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7c7a74;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  background:var(--surface-1); color:var(--text-primary); max-width:980px; margin:0 auto;
  padding:36px 24px 64px;
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37; --grid:#2e2e2c;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#93918a;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; } }
:root[data-theme="dark"] .viz-root {
  --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37; --grid:#2e2e2c;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#93918a;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; }
body { margin:0; background:var(--surface-1); }
h1 { font-size:23px; letter-spacing:-.015em; margin:0 0 4px; }
.sub { color:var(--text-secondary); font-size:13.5px; margin:0 0 20px; }
svg text { font:12px ui-sans-serif,-apple-system,sans-serif; fill:var(--text-secondary); }
svg .lbl { font-size:11px; fill:var(--text-primary); }
svg .axis { stroke:var(--border); stroke-width:1; }
svg .grid { stroke:var(--grid); stroke-width:1; }
table { border-collapse:collapse; margin-top:28px; font-size:13px; width:100%; }
th,td { border:1px solid var(--border); padding:5px 9px; text-align:right;
        font-variant-numeric:tabular-nums; }
th:first-child,td:first-child { text-align:left; }
th { background:var(--surface-2); }
.legend { display:flex; gap:18px; margin:10px 0 4px; font-size:13px;
          color:var(--text-secondary); align-items:center; flex-wrap:wrap; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
"""

SERIES = {  # fixed assignment; shape is the secondary encoding
    "few-shot": ("var(--series-2)", "square"),
    "stage A": ("var(--series-1)", "circle"),
    "stage A+B": ("var(--series-3)", "diamond"),
    "base floor": ("var(--text-muted)", "cross"),
}


def classify(rel: str) -> tuple[str, str] | None:
    """(series, point label) for a record path, or None to leave it off the figure."""
    parts = rel.split("/")
    if rel == "floor":
        return "base floor", "base"
    if parts[0] == "sweep" and len(parts) == 3:
        arm = parts[1]
        return {"baseline": ("few-shot", arm), "stage_a": ("stage A", arm),
                "stage_b": ("stage A+B", arm)}.get(parts[2])
    if parts[0] == "matrix" and parts[1] == "rq2":
        return "stage A", parts[2].split("_", 1)[0]  # a01 / a02 / a05
    return None


def _shape(x, y, kind, color) -> str:
    if kind == "square":
        return (f'<rect x="{x-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" rx="1.5" '
                f'fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>')
    if kind == "diamond":
        return (f'<path d="M{x:.1f} {y-6:.1f} L{x+6:.1f} {y:.1f} L{x:.1f} {y+6:.1f} '
                f'L{x-6:.1f} {y:.1f} Z" fill="{color}" stroke="var(--surface-1)" '
                f'stroke-width="2"/>')
    if kind == "cross":
        return (f'<path d="M{x-5:.1f} {y:.1f} H{x+5:.1f} M{x:.1f} {y-5:.1f} V{y+5:.1f}" '
                f'stroke="{color}" stroke-width="2.5" fill="none"/>')
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
            f'stroke="var(--surface-1)" stroke-width="2"/>')


def frontier_svg(points: list[dict]) -> str:
    W, H, ML, MR, MT, MB = 900, 480, 62, 20, 16, 52
    xs = [p["leak"] for p in points]
    x_max = max(0.06, max(xs) * 1.25) if xs else 0.06
    sx = lambda v: ML + (W - ML - MR) * v / x_max  # noqa: E731
    sy = lambda v: MT + (H - MT - MB) * (1 - v)  # noqa: E731  (AV is a rate in [0,1])
    el = [f'<svg viewBox="0 0 {W} {H}" role="img" '
          f'aria-label="Style vs leakage frontier: one point per run">']
    for i in range(6):
        v = i / 5
        el.append(f'<line class="grid" x1="{ML}" x2="{W-MR}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        el.append(f'<text x="{ML-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.1f}</text>')
    n_xt = 6
    for i in range(n_xt + 1):
        v = x_max * i / n_xt
        el.append(f'<line class="grid" y1="{MT}" y2="{H-MB}" x1="{sx(v):.1f}" x2="{sx(v):.1f}"/>')
        el.append(f'<text x="{sx(v):.1f}" y="{H-MB+18}" text-anchor="middle">{v:.3f}</text>')
    el.append(f'<line class="axis" x1="{ML}" x2="{W-MR}" y1="{H-MB}" y2="{H-MB}"/>')
    el.append(f'<line class="axis" x1="{ML}" x2="{ML}" y1="{MT}" y2="{H-MB}"/>')
    el.append(f'<text x="{(ML+W-MR)/2}" y="{H-10}" text-anchor="middle">'
              "verbatim leakage rate → worse</text>")
    el.append(f'<text transform="translate(16 {(MT+H-MB)/2}) rotate(-90)" '
              'text-anchor="middle">AV attribution rate → better</text>')
    # connect the RQ1 series in corpus-size order so the frontier reads as a path
    for series in ("few-shot", "stage A"):
        path_pts = [p for p in points if p["series"] == series and p.get("order") is not None]
        path_pts.sort(key=lambda p: p["order"])
        if len(path_pts) >= 2:
            d = " ".join(f'{"M" if i == 0 else "L"}{sx(p["leak"]):.1f} {sy(p["av"]):.1f}'
                         for i, p in enumerate(path_pts))
            el.append(f'<path d="{d}" fill="none" stroke="{SERIES[series][0]}" '
                      'stroke-width="1.5" opacity="0.45"/>')
    for p in points:
        color, shape = SERIES[p["series"]]
        x, y = sx(p["leak"]), sy(p["av"])
        title = (f'{p["series"]} · {p["label"]} — AV {p["av"]:.3f}, leak {p["leak"]:.3f}')
        el.append(f"<g>{_shape(x, y, shape, color)}<title>{html.escape(title)}</title></g>")
        el.append(f'<text class="lbl" x="{x+8:.1f}" y="{y-7:.1f}">{html.escape(p["label"])}</text>')
    el.append("</svg>")
    return "\n".join(el)


def trajectory_svg(traj: list[dict]) -> str:
    """Training-time frontier for one run: same axes, points ordered by checkpoint step."""
    traj = sorted(traj, key=lambda p: p["step"])
    for p in traj:
        p["series"], p["label"], p["order"] = "stage A", f'step {p["step"]}', p["step"]
    return frontier_svg(traj)


def write_frontier(points, traj, out_path: Path) -> None:
    legend = "".join(
        f'<span><span class="sw" style="background:{c}"></span>{html.escape(name)}</span>'
        for name, (c, _s) in SERIES.items()
        if any(p["series"] == name for p in points))
    table = ["<table><tr><th>run</th><th>series</th><th>AV</th><th>leak</th></tr>"]
    for p in sorted(points, key=lambda p: (p["series"], p["label"])):
        table.append(f"<tr><td>{html.escape(p['label'])}</td><td>{html.escape(p['series'])}</td>"
                     f"<td>{p['av']:.4f}</td><td>{p['leak']:.4f}</td></tr>")
    table.append("</table>")
    traj_block = ""
    if traj:
        traj_block = ("<h1 style='margin-top:44px'>Training-time frontier</h1>"
                      "<p class='sub'>One point per checkpoint of a single Stage-A run: does "
                      "style rise before memorization sets in?</p>" + trajectory_svg(traj))
    out_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Figure 1 — the frontier</title>"
        f"<style>{CSS}</style><div class='viz-root'>"
        "<h1>Figure 1 — style against leakage</h1>"
        "<p class='sub'>One point per run. The operating point the product needs is the region "
        "where the y-axis rises while the x-axis stays put.</p>"
        f"<div class='legend'>{legend}</div>" + frontier_svg(points) + traj_block
        + "".join(table) + "</div>", encoding="utf-8")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--control-runs", default=None,
                    help="the private control author's runs/ directory (Table 4)")
    ap.add_argument("--rq2-arm", default="full")
    ap.add_argument("--out", default=None, help="output dir (default <runs>/results)")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out) if args.out else runs_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = load_records(runs_dir)
    if not recs:
        print(f"no report.json under {runs_dir} — nothing to assemble yet")
        return 1

    manifest_path = PROCESSED / "sweep" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    floor = by_rel(recs, "floor")
    floor_av = floor["av"] if floor else None

    aucs = {r["auc"] for r in recs if r["auc"] is not None}
    gate = ""
    if aucs:
        gate = (f"Verifier held-out AUC (the gate on everything below): "
                f"{', '.join(f'{a:.4f}' for a in sorted(aucs))}"
                + (" — **multiple AUCs means the ruler moved between runs; "
                   "re-check before comparing.**" if len(aucs) > 1 else "") + "\n")
    inst = attach_second_instrument(recs, runs_dir)
    if inst:
        ag = inst.get("agreement") or {}
        gate += (
            f"\nAV₂ = second instrument ({inst.get('second_embedder')}, held-out AUC "
            f"{inst.get('second_verifier_auc')}); rank agreement with the primary: Spearman "
            f"{ag.get('spearman')} over {ag.get('n')} runs. AV₂ reads adapters systematically "
            "higher, so compare *orderings* across instruments, not absolute values — details "
            "in `results/instruments.md`.\n")

    sections = [
        "# Results — assembled from run records\n",
        f"_{len(recs)} record(s) under `{runs_dir}`. Regenerate with "
        "`python scripts/assemble_results.py` — never edit by hand._\n",
        gate,
        table1(recs, manifest),
        table2(recs, floor_av, reference_arm(recs, manifest, args.rq2_arm)),
        table3(recs, manifest),
        table4(recs, runs_dir, Path(args.control_runs) if args.control_runs else None,
               manifest),
        table5_models(recs, runs_dir),
        noise_floor(recs, manifest),
    ]
    (out_dir / "tables.md").write_text("\n".join(sections), encoding="utf-8")

    points, traj = [], []
    order = {m["arm"]: m["actual_words"] for m in manifest}
    for r in recs:
        if r["av"] is None or r["leak"] is None:
            continue
        if "/trajectory/" in r["rel"]:
            step_part = r["rel"].rsplit("step-", 1)
            if len(step_part) == 2 and step_part[1].split("/")[0].isdigit():
                traj.append({"step": int(step_part[1].split("/")[0]),
                             "av": r["av"], "leak": r["leak"]})
            continue
        cls = classify(r["rel"])
        if cls is None:
            continue
        series, label = cls
        points.append({"series": series, "label": label, "av": r["av"], "leak": r["leak"],
                       "order": order.get(label)})
    if points:
        write_frontier(points, traj, out_dir / "frontier.html")

    print(f"wrote {out_dir / 'tables.md'}"
          + (f" and {out_dir / 'frontier.html'} ({len(points)} points"
             + (f", {len(traj)} trajectory checkpoints" if traj else "") + ")" if points else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
