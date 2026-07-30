#!/usr/bin/env bash
# Phase 0 (baseline) + Phase 1 (Stage A) on the GPU box.
set -euo pipefail
CFG="${CFG:-configs/stage_a.yaml}"
RUN="${RUN:-stage_a}"

# Phase-0 baseline: few-shot style prompting, exemplars from TRAIN only. This is the bar.
wlm baseline --config "$CFG" --out runs/baseline
wlm eval run --gen runs/baseline/gen.jsonl --av runs/av \
             --out runs/baseline/report.html --run-name baseline

# Phase 1: Stage-A LoRA SFT (attention + MLP).
wlm train sft --config "$CFG" --out "runs/$RUN"
wlm generate  --config "$CFG" --adapter "runs/$RUN" --split blind --out "runs/$RUN/gen.jsonl"
wlm eval run  --gen "runs/$RUN/gen.jsonl" --av runs/av \
              --out "runs/$RUN/report.html" --run-name "$RUN" \
              --baseline runs/baseline/report.json

echo
echo "Open runs/baseline/report.html and runs/$RUN/report.html side by side."
echo "The deliverable is: AV attribution up on INFORMAL text, fluency PASS, leakage PASS."
