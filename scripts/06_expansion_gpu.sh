#!/usr/bin/env bash
# The expansion GPU workload (docs/EXPANSION.md), in priority order. Every unit is resumable —
# a unit is done when its report.json exists — so this script can be re-run after any
# preemption with the same command and it continues where it stopped.
#
# Before running on the GPU box, rsync from the laptop (the CPU half is already done there):
#   rsync -av data/processed/ runs/av/ runs/av2/ GPUBOX:~/ductus/...
#   rsync -av ~/authors/twain/ GPUBOX:~/authors/twain/       # corpus + splits + verifiers
#
# Priority 4 (a16) needs gated-model access:
#   meta-llama/Llama-3.2-3B-Instruct requires an accepted license on huggingface.co plus
#   `huggingface-cli login` (or HF_TOKEN in the environment) on the GPU box. The section is
#   skipped with a warning if no token is visible; a13 (Qwen 1.5B, ungated) still runs.
set -euo pipefail
cd "$(dirname "$0")/.."
TWAIN_ROOT="${TWAIN_ROOT:-$HOME/authors/twain}"

# run_matrix exits non-zero when any unit fails — but a unit "failing" is often a leakage-gate
# veto, which is an expected research outcome (small-arm baselines parrot their exemplars),
# not a reason to abandon the remaining tiers. Record and continue; read matrix_state.json.
matrix() {
  python scripts/run_matrix.py "$@" || \
    echo "[06] run_matrix recorded failed unit(s) — continuing; gate vetoes are data points. See runs/matrix_state.json"
}

echo "== 1. Tier 0: remaining confound closers (Chesterton) =="
# a14 no-scrub diagnostic (already ran and committed at runs/matrix/a14_no_scrubbing; the
# guard keeps this re-runnable from scratch on a fresh box). DIAGNOSTIC ONLY — never ship it.
if [ ! -f runs/matrix/a14_no_scrubbing/report.json ]; then
  wlm train sft --config configs/ablations/a14_no_scrubbing.yaml \
      --train data/processed/a14_noscrub/train.jsonl \
      --val data/processed/a14_noscrub/val.jsonl \
      --out runs/matrix/a14_no_scrubbing
  wlm generate --config configs/ablations/a14_no_scrubbing.yaml \
      --adapter runs/matrix/a14_no_scrubbing \
      --out runs/matrix/a14_no_scrubbing/gen.jsonl
  wlm eval run --gen runs/matrix/a14_no_scrubbing/gen.jsonl \
      --train data/processed/a14_noscrub/train.jsonl --av runs/av \
      --out runs/matrix/a14_no_scrubbing/report.html --run-name a14_no_scrubbing \
      --baseline runs/sweep/full/stage_a/report.json
fi
python scripts/adapter_anatomy.py runs/matrix/a14_no_scrubbing \
    --json runs/results/anatomy_a14.json || true
matrix --only seeds rq2x    # third seed + the interaction cell
# Checkpoint-trajectory eval: interrupted after step-100 — the script skips finished steps.
if [ -d runs/trajectory ]; then
  scripts/05_checkpoint_trajectory.sh runs/trajectory
fi

echo "== 2. Cliff shape: the 15k/20k arms (only the new arms actually run) =="
matrix --only rq1

echo "== 3. Per-matrix locus split (Savine test): q,k vs v,o =="
matrix --only rq2 --rq2-configs a17_qk_only a18_vo_only

echo "== 4. Cross-model replication =="
if python -c "import huggingface_hub as h; h.whoami()" >/dev/null 2>&1; then
  matrix --only models --model-configs a16_llama3b a13_1p5b
else
  echo "  no HF login found — meta-llama/Llama-3.2-3B-Instruct is gated. Running the"
  echo "  ungated within-family point only; accept the Llama license + huggingface-cli login,"
  echo "  then re-run this script for a16."
  matrix --only models --model-configs a13_1p5b
fi

echo "== 5. Second author (Twain, informal register) =="
if [ -f "$TWAIN_ROOT/data/processed/blind.jsonl" ]; then
  WLM_ROOT="$TWAIN_ROOT" matrix --only floor rq1
  WLM_ROOT="$TWAIN_ROOT" python scripts/assemble_results.py \
      --runs "$TWAIN_ROOT/runs" --out "$TWAIN_ROOT/runs/results"
else
  echo "  $TWAIN_ROOT has no processed splits — finish its CPU phase-0 first (RUNBOOK.md)."
fi

echo "== 6. Assemble everything (Chesterton root) =="
python scripts/assemble_results.py
python scripts/second_instrument.py   # picks up the new runs; av2 is already fitted
echo "Done. Read runs/results/tables.md and runs/results/instruments.md."
