#!/usr/bin/env bash
# Training-time frontier [GPU]: evaluate EVERY checkpoint of one Stage-A run.
#
# The brief's primary question — "does a LoRA acquire an author's form faster than it memorizes
# their content?" — is a question about *dynamics*, but the run matrix only measures endpoints.
# One training run already saves a checkpoint every sft.save_steps steps; evaluating each one
# traces style (AV) and leakage per step and shows which rises first, at the cost of a
# generation+eval pass per checkpoint and zero extra training.
#
# Train the run with configs/trajectory.yaml (same recipe, save_total_limit high enough that
# checkpoints survive pruning):
#
#   wlm train sft --config configs/trajectory.yaml --out runs/trajectory
#   scripts/05_checkpoint_trajectory.sh runs/trajectory
#
# Results land in <run>/trajectory/step-<N>/report.json; scripts/assemble_results.py picks them
# up automatically and adds the training-time panel to Figure 1.
#
# Caveat recorded so nobody over-reads the output: the fluency verdict shown in each
# checkpoint's report comes from the run's fluency log (nearest logged step), not a fresh
# probe, and generations at every checkpoint share the config's sampling seed.
set -euo pipefail

RUN="${1:?usage: scripts/05_checkpoint_trajectory.sh <stage-a run dir> [config]}"
CFG="${2:-configs/trajectory.yaml}"
BLIND="${BLIND:-data/processed/blind.jsonl}"
TRAIN="${TRAIN:-data/processed/train.jsonl}"

shopt -s nullglob
CKPTS=("$RUN"/checkpoint-*)
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoint-* under $RUN — was it trained with save_total_limit high enough?"; exit 1; }

for ck in "${CKPTS[@]}"; do
  step=$(basename "$ck" | cut -d- -f2)
  out="$RUN/trajectory/step-$step"
  if [ -f "$out/report.json" ]; then
    echo "[trajectory] step $step already evaluated, skipping"
    continue
  fi
  mkdir -p "$out"
  echo "=== checkpoint step $step ==="
  wlm generate --config "$CFG" --adapter "$ck" --split-path "$BLIND" --out "$out/gen.jsonl"
  # `|| true`: a mid-training checkpoint failing a gate is a *data point* for the trajectory,
  # not a reason to stop tracing it.
  wlm eval run --gen "$out/gen.jsonl" --ref "$BLIND" --train "$TRAIN" --av runs/av \
               --out "$out/report.html" --run-name "trajectory-step-$step" \
               --fluency "$RUN" || true
done

echo
echo "Trace complete. Assemble the figure:  python scripts/assemble_results.py"
