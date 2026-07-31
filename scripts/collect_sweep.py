#!/usr/bin/env python3
"""Collect the corpus-size sweep into one table: where does the adapter beat the prompt?

Reads every runs/sweep/<arm>/{baseline,stage_a}/report.json and prints AV attribution for both
arms side by side, plus the gates. Runs on CPU -- pull `runs/` back from the GPU box first.

The column that matters is `delta`: adapter attribution minus its own few-shot baseline at the
same corpus size. The crossover -- the first arm where delta is positive AND both gates pass --
is the answer to "how much writing does a user need before training is worth it".

An arm with a positive delta and a failing gate is not a win. At small corpus sizes the adapter
tends to memorise, which raises attribution and trips leakage at the same time; reporting the
attribution alone would turn that failure into a headline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def short(verdict):
    if not verdict:
        return "—"
    return "PASS" if str(verdict).upper().startswith("PASS") else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs/sweep"))
    ap.add_argument("--sweep", default=str(ROOT / "data/processed/sweep"))
    ap.add_argument("--out", default=None, help="also write the table here as markdown")
    args = ap.parse_args()

    runs, sweep = Path(args.runs), Path(args.sweep)
    manifest = load(sweep / "manifest.json") or []
    sizes = {m["arm"]: m for m in manifest}

    arms = sorted([d.name for d in runs.iterdir() if d.is_dir()],
                  key=lambda a: sizes.get(a, {}).get("actual_words", 1 << 60))
    if not arms:
        print(f"no arms under {runs} -- has the sweep run yet?")
        return 1

    head = (f"| {'arm':<6} | {'words':>9} | {'docs':>5} | {'few-shot AV':>11} | "
            f"{'adapter AV':>10} | {'delta':>8} | {'fluency':>7} | {'leakage':>7} | verdict |")
    rule = "|" + "|".join("-" * len(c) for c in head.split("|")[1:-1]) + "|"
    lines = [head, rule]

    for arm in arms:
        b = load(runs / arm / "baseline" / "report.json") or {}
        s = load(runs / arm / "stage_a" / "report.json") or {}
        b_av = get(b, "av", "attribution_rate")
        s_av = get(s, "av", "attribution_rate")
        delta = None if (b_av is None or s_av is None) else s_av - b_av
        flu = short(get(s, "fluency", "verdict"))
        leak = short(get(s, "leakage", "verdict"))
        gates_ok = flu != "FAIL" and leak != "FAIL"
        verdict = ("WIN" if (delta or 0) > 0 and gates_ok else
                   "gated" if (delta or 0) > 0 else "no win")
        m = sizes.get(arm, {})
        lines.append(
            f"| {arm:<6} | {m.get('actual_words', 0):>9,} | {m.get('documents', 0):>5} | "
            f"{fmt(b_av):>11} | {fmt(s_av):>10} | {fmt(delta):>8} | {flu:>7} | {leak:>7} | "
            f"{verdict} |")

    table = "\n".join(lines)
    print(table)

    wins = [a for a, line in zip(arms, lines[2:], strict=False) if line.rstrip().endswith("WIN |")]
    print()
    if wins:
        first = wins[0]
        w = sizes.get(first, {}).get("actual_words", 0)
        print(f"Crossover at the {first} arm (~{w:,} words): below this, few-shot prompting is "
              f"the better product.")
    else:
        print("No arm beats its own few-shot baseline with both gates passing. Either the corpus "
              "is too small at every size tested, or the recipe needs work before scale helps.")

    auc = get(load(runs / arms[0] / "stage_a" / "report.json") or {}, "av", "verifier_auc")
    if auc is not None:
        print(f"Verifier AUC (same ruler for every arm): {auc:.4f}")

    if args.out:
        Path(args.out).write_text(
            "# Corpus-size sweep\n\n" + table + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
