#!/usr/bin/env bash
# Corpus-size sweep [GPU]: find where a Stage-A LoRA starts beating few-shot prompting.
#
# For each arm produced by scripts/make_size_sweep.py, this runs BOTH arms of the comparison
# against the same fixed blind split and the same fitted verifier:
#
#   few-shot baseline   exemplars drawn from that arm's train set, no training
#   Stage-A adapter     LoRA trained on that arm's train set
#
# The interesting output is not either number alone but the delta between them as corpus size
# grows -- the crossover is the point where training earns its cost.
#
# Prerequisites (see the bottom of this file if any fail):
#   data/processed/sweep/*/train.jsonl   from scripts/make_size_sweep.py
#   runs/av/                             fitted ONCE on the FULL train set
set -euo pipefail

CFG="${CFG:-configs/stage_a.yaml}"
SWEEP="${SWEEP:-data/processed/sweep}"
OUT="${OUT:-runs/sweep}"
BLIND="${BLIND:-data/processed/blind.jsonl}"

[ -d "$SWEEP" ]   || { echo "missing $SWEEP -- run: python scripts/make_size_sweep.py"; exit 1; }
[ -d "runs/av" ]  || { echo "missing runs/av -- run: wlm eval fit-av --author data/processed/train.jsonl --distractor data/raw/distractor --out runs/av"; exit 1; }
[ -f "$BLIND" ]   || { echo "missing $BLIND"; exit 1; }

ARMS=$(find "$SWEEP" -mindepth 1 -maxdepth 1 -type d | sort -t/ -k4 -V)
echo "arms: $(echo "$ARMS" | tr '\n' ' ')"

for arm_dir in $ARMS; do
  arm=$(basename "$arm_dir")
  train="$arm_dir/train.jsonl"
  val="$arm_dir/val.jsonl"
  base_out="$OUT/$arm/baseline"
  sft_out="$OUT/$arm/stage_a"
  mkdir -p "$base_out" "$sft_out"

  echo
  echo "======================= $arm ======================="

  # --- arm 1: few-shot prompting, no training. Exemplars come from THIS arm's train set.
  if [ ! -f "$base_out/report.json" ]; then
    wlm baseline --config "$CFG" --train "$train" --split-path "$BLIND" --out "$base_out"
    wlm eval run --gen "$base_out/gen.jsonl" --ref "$BLIND" --train "$train" \
                 --av runs/av --out "$base_out/report.html" --run-name "baseline-$arm"
  else
    echo "[$arm] baseline already done, skipping"
  fi

  # --- arm 2: Stage-A LoRA on the same subsample. Same recipe every arm: one variable only.
  if [ ! -f "$sft_out/report.json" ]; then
    wlm train sft --config "$CFG" --train "$train" --val "$val" --out "$sft_out"
    wlm generate  --config "$CFG" --adapter "$sft_out" --split-path "$BLIND" \
                  --out "$sft_out/gen.jsonl"
    wlm eval run  --gen "$sft_out/gen.jsonl" --ref "$BLIND" --train "$train" \
                  --av runs/av --out "$sft_out/report.html" --run-name "stage_a-$arm" \
                  --baseline "$base_out/report.json"
  else
    echo "[$arm] stage_a already done, skipping"
  fi
done

echo
python scripts/collect_sweep.py --out "$OUT/summary.md"
echo
echo "Read the table, not the loss curves. An arm only counts as a win if the AV attribution"
echo "beats its own baseline AND fluency and leakage both PASS -- a leakage failure at small"
echo "corpus sizes is the expected failure mode, not a surprise."
