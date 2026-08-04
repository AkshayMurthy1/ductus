#!/usr/bin/env python3
"""Re-score every run under a second, independently-trained style instrument.

The threat this closes (STATUS §6.2): all style numbers read through one embedder + one head,
and the 50k arm's 0.825 > 0.714 anomaly has an ungenerous reading — "the adapter overfits the
verifier's decision boundary better than the true author does." A second instrument with a
different embedder (default: StyleDistance, arXiv 2410.12757, contrastively trained on
synthetic parallel examples — a genuinely different training signal from the primary's
same-author/same-topic Reddit contrastive) turns that from a gloss into a measurement:

  - agreement (Spearman over run rankings) means the orderings are instrument-independent;
  - per-run disagreements are the cells to hand-inspect (read the flagged runs' gen.jsonl
    next to real held-out text);
  - the real-text calibration row shows whether "adapter above real Chesterton" replicates
    under a ruler the adapter was never scored against during development.

Entirely additive and CPU-friendly: existing report.json files are read, never rewritten —
the primary numbers stay exactly what the committed records say. The second verifier is fit
ONCE per corpus (same discipline as the primary: it is a ruler, it must not move) into
runs/av2 and reused on every call.

    python scripts/second_instrument.py               # fit av2 if missing, re-score all runs
    python scripts/second_instrument.py --runs runs --embedder StyleDistance/styledistance
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.config import Config  # noqa: E402
from wlm.eval.agreement import instrument_agreement  # noqa: E402
from wlm.paths import PROCESSED, RUNS, read_jsonl  # noqa: E402

SKIP_MARKERS = ("checkpoint", "trajectory")  # per-checkpoint gens: --include-trajectory


def fit_second(av2_dir: Path, embedder: str, author_path: Path, distractor_dir: Path,
               cfg: Config) -> None:
    """Same fit protocol as `wlm eval fit-av` (dedupe, document groups, class balance) with
    only the embedder swapped — the comparison is between embedders, not between protocols."""
    from wlm.eval.av import AuthorshipVerifier, StyleEmbedder, load_distractor_texts

    rows = read_jsonl(author_path)
    by_text: dict[str, str] = {}
    for r in rows:
        by_text.setdefault(r["response"].strip(), r.get("doc_id", "unknown"))
    author = list(by_text.keys())
    groups = list(by_text.values())
    distractor = load_distractor_texts(
        distractor_dir, min_words=cfg.data.chunk_min_words,
        max_words=cfg.data.chunk_max_words)
    k = min(len(author), len(distractor))
    author, groups, distractor = author[:k], groups[:k], distractor[:k]

    v = AuthorshipVerifier(StyleEmbedder(embedder), head=cfg.eval.av_classifier)
    metrics = v.fit(author, distractor, seed=cfg.data.seed, author_groups=groups)
    v.save(av2_dir)
    print(f"[av2] fitted {embedder}: held-out AUC {metrics.auc:.4f} -> {av2_dir}")
    if metrics.auc < 0.75:
        print("[av2] WARNING second-instrument AUC < 0.75 — its rankings below are noise, "
              "and the agreement number will understate the real agreement.")


def collect_runs(runs_dir: Path, include_trajectory: bool) -> list[Path]:
    out = []
    for gen in sorted(runs_dir.rglob("gen.jsonl")):
        rel = gen.parent.relative_to(runs_dir).as_posix()
        if not include_trajectory and any(m in rel for m in SKIP_MARKERS):
            continue
        if (gen.parent / "report.json").exists():
            out.append(gen.parent)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--av2", default=str(RUNS / "av2"))
    ap.add_argument("--embedder", default=None,
                    help="default: eval.second_style_embedder from the config")
    ap.add_argument("--config", default=str(REPO / "configs/stage_a.yaml"))
    ap.add_argument("--author", default=str(PROCESSED / "train.jsonl"),
                    help="fit corpus for av2 (same as the primary: TRAIN only)")
    ap.add_argument("--distractor", default=str(REPO / "data/raw/distractor"))
    ap.add_argument("--blind", default=str(PROCESSED / "blind.jsonl"),
                    help="real held-out text for the calibration row")
    ap.add_argument("--include-trajectory", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (default <runs>/results)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    embedder = args.embedder or cfg.eval.second_style_embedder
    runs_dir, av2_dir = Path(args.runs), Path(args.av2)
    out_dir = Path(args.out) if args.out else runs_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    from wlm.eval.av import AuthorshipVerifier

    if not (av2_dir / "av_meta.json").exists():
        fit_second(av2_dir, embedder, Path(args.author), Path(args.distractor), cfg)
    v2 = AuthorshipVerifier.load(av2_dir)
    if v2.embedder.model_name == cfg.eval.style_embedder:
        print(f"[av2] WARNING {av2_dir} was fit with the PRIMARY embedder "
              f"({v2.embedder.model_name}) — this measures nothing. Delete it and re-run.")
        return 1

    rows = []
    for run_dir in collect_runs(runs_dir, args.include_trajectory):
        rel = run_dir.relative_to(runs_dir).as_posix()
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        primary = (report.get("av") or {}).get("attribution_rate")
        if primary is None:
            continue  # no primary AV -> nothing to agree with
        texts = [g["completion"] for g in read_jsonl(run_dir / "gen.jsonl")
                 if g.get("completion")]
        if not texts:
            continue
        second = v2.evaluate(texts)
        rows.append({"run": rel, "av_primary": primary,
                     "av_second": second["attribution_rate"],
                     "second_score_mean": second["score_mean"], "n": second["n"]})
        print(f"  {rel:<44} primary {primary:.4f}  second {second['attribution_rate']:.4f}")

    if not rows:
        print(f"no dual-scorable runs under {runs_dir}")
        return 1

    # Calibration row: real held-out author text under the second ruler. Read next to the
    # primary's real_reference_attribution_rate (0.714 on the dev corpus) — if the adapter sits
    # above real text under ONE ruler but not the other, that is the gaming signature.
    calibration = None
    blind_path = Path(args.blind)
    if blind_path.exists():
        real = [r["response"] for r in read_jsonl(blind_path) if r.get("response")]
        if real:
            calibration = v2.evaluate(real)
            print(f"  {'[real blind text]':<44} second {calibration['attribution_rate']:.4f}")

    agreement = instrument_agreement(rows)
    result = {
        "second_embedder": v2.embedder.model_name,
        "second_verifier_auc": v2.metrics.auc if v2.metrics else None,
        "real_blind_under_second": calibration,
        "agreement": agreement,
        "runs": rows,
    }
    (out_dir / "instruments.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = ["# Dual-instrument check\n",
          f"Second instrument: `{v2.embedder.model_name}` "
          f"(held-out AUC {v2.metrics.auc:.4f})\n" if v2.metrics else "",
          f"Rank agreement over {agreement.get('n')} runs: Spearman "
          f"**{agreement.get('spearman')}** (disagreement {agreement.get('rank_disagreement')}, "
          f"mean |Δ| {agreement.get('mean_abs_diff')})\n",
          "| run | primary AV | second AV | Δ |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["run"]):
        md.append(f"| {r['run']} | {r['av_primary']:.4f} | {r['av_second']:.4f} | "
                  f"{r['av_primary'] - r['av_second']:+.4f} |")
    if calibration:
        md.append(f"\nReal held-out author text under the second ruler: attribution "
                  f"**{calibration['attribution_rate']:.4f}** (mean score "
                  f"{calibration['score_mean']:.4f}). Compare adapter rows against this the "
                  "same way the primary is compared against 0.714.")
    if agreement.get("flagged_runs"):
        md.append("\nRuns the two instruments disagree on (hand-inspect these first):")
        for f in agreement["flagged_runs"]:
            md.append(f"- `{f['run']}`: primary {f['av_primary']:.4f} vs "
                      f"second {f['av_second']:.4f}")
    (out_dir / "instruments.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'instruments.json'} and {out_dir / 'instruments.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
