# EXPANSION — closing the venue-readiness gaps

The critique this answers (external doc, Aug 3 2026): the science is sound but the empirical
scope is one cell — one author, one model, one verifier — and three named confounds are still
open. This document is the execution plan for closing all of it, with the tooling already in
place on this branch. Target: everything runnable started **by end of August 2026**, with the
MPLR-FM workshop deadline (Aug 29, 4 pages, non-archival) as the first checkpoint.

Everything below follows the standing rules: one variable per run, the verifier never moves
within a corpus, tables come from `assemble_results.py`, and a run failing a gate is a failure
regardless of its AV score.

## Tier 0 — confound closers (all scaffolded; GPU-cheap; BLOCK the workshop submission)

| run | command | closes |
|---|---|---|
| seed 43 @ full | `python scripts/run_matrix.py --only seeds` (resumes) | 2-seeds-at-the-decisive-cell |
| a14 no-scrub | `scripts/make_a14_noscrub.py`, then Stage A + eval on it | "adapters don't emit entities" being true *by construction* |
| checkpoint trajectory | `scripts/05_checkpoint_trajectory.sh runs/trajectory` | rate claim with no training-time measurement |
| attention-only + DPO | `python scripts/run_matrix.py --only rq2x` | recommending a configuration that was never run |

`rq2x` is new on this branch: DPO on top of the a01 adapter, baseline-delta'd against a01, into
`runs/matrix/rq2x/a15_attn_dpo`.

## Tier 1 — triangulate the ruler

**Second instrument.** `python scripts/second_instrument.py` — fits a StyleDistance-based
verifier once (`runs/av2`, same protocol, different embedder), re-scores every existing
`gen.jsonl` (CPU, no retraining), and writes `runs/results/instruments.{md,json}`: per-run
primary-vs-second attribution, Spearman rank agreement, a real-text calibration row, and a
flagged list of disagreeing runs. Existing `report.json` records are never modified.

**Human panel.** `python scripts/make_panel_sheets.py --gen <run>/gen.jsonl --out
runs/human_panel` builds blinded anchored-comparison sheets (reference + candidate, 1-5
same-author rating; equal thirds real/adapter/distractor). Collect ≥10 raters, then
`python scripts/score_panel.py --ratings <filled csvs> --key runs/human_panel/key.json`.
Sample the 50k arm (the 0.825 > 0.714 anomaly) plus any runs `instruments.md` flags. The
decision table lives in `score_panel.py`'s docstring; `real_minus_distractor` is the power
check.

**Cliff shape (Mirage defense).** Two intermediate arms resolve the transition:
`python scripts/make_size_sweep.py --sizes 2000 5000 10000 15000 20000 25000 50000 100000`
(nested subsampling means existing arms are unchanged; then `run_matrix.py --only rq1` picks
up only the new ones). Present the continuous mean score with per-seed traces as primary
evidence, citing Schaeffer et al. 2023 explicitly.

## Tier 2 — generality (the acceptance-deciding tier)

**Extra authors.** One command scaffolds a complete author root outside the repo:

    python scripts/new_author.py ~/authors/<name> --register informal [--private]

It writes a `RUNBOOK.md` with the corpus checklist and the exact per-author command sequence
(phase-0, both verifier fits, matched sweep arms, `run_matrix`, per-root assembly). Planned
arms: **2 extra public authors (≥1 informal register) + the private control author** — the
control is the highest-priority cell because it kills the pretraining-recall objection.
Cross-author claims compare within-author deltas and cliff *shapes*, never raw AV numbers
(each corpus has its own ruler). `make_size_sweep.py` is now `WLM_ROOT`-aware (it previously
ignored it — fixed on this branch), so the documented `WLM_ROOT=...` protocol works end to end.

**Second base model.** `python scripts/run_matrix.py --only models` runs, per model config
(default `a16_llama3b`; add `a13_1p5b` for the within-family point), the few-shot baseline AND
Stage A at the cliff-region arms (`10k 25k full` by default) plus that model's own
contamination probe, into `runs/matrix/models/<config>/<arm>/`. The baseline moves with the
base model — an adapter must beat *its own model's* few-shot floor. The verifier is text-side
and shared, so cross-model AV numbers ARE comparable within a corpus.

Minimum defensible package for a full-paper venue: 3 authors (1 informal) × 2 model families
at 3-4 sizes, plus the complete Chesterton matrix.

## Tier 3 — mechanism sharpening (opportunistic, after Tiers 0-2 are queued)

    python scripts/run_matrix.py --only rq2 --rq2-configs a17_qk_only a18_vo_only

splits the attention cell into q,k (routing) vs v,o (the matrices Savine, arXiv 2507.21009,
finds memorize most). If v,o-only still shows zero verbatim and background-rate entities, the
attention result survives its strongest published counterexample. Run
`scripts/adapter_anatomy.py` on both, and on a14-vs-reference.

## Sequencing to Aug 31

1. **Week 1 (now):** finish Tier 0 on the SCC; run `second_instrument.py` over everything that
   exists (CPU, immediate); generate panel sheets and send to raters.
2. **Week 2:** write + submit the 4-page MPLR-FM abstract from Tier-0-complete results
   (deadline **Aug 29**); start the 15k/20k arms and `--only models` on the GPU box.
3. **Weeks 3-4:** scaffold both public extra authors (`new_author.py`), run their phase-0 +
   decisive arms; score the panel as ratings return. Private-author corpus lands whenever
   available — the runbook makes it a fill-in-the-directory task.
4. **September:** assemble, then target ARR Oct 12 / TMLR / COLM 2027 per the venue analysis.

## What this branch deliberately does NOT change

The reference recipe, the existing run records, the brief's default matrix (`--only` defaults
are untouched), and `report.json` schemas. Every addition is a new config, a new script, a new
optional driver section, or a fix that only affects new roots (`make_size_sweep.py` under
`WLM_ROOT`).
