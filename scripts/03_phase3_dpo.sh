#!/usr/bin/env bash
# Phase 3: Stage-B on-policy DPO. Plain sigmoid DPO only -- change one thing at a time.
set -euo pipefail
CFG="${CFG:-configs/stage_b.yaml}"
BASE_ADAPTER="${BASE_ADAPTER:-runs/stage_a}"

# Pairs first, and READ THE STATS before training. `wlm dpo-pairs` exits non-zero if fewer than
# ~40 pairs survive the length and degeneracy filters, because DPO on that few is noise.
# --val-frac holds a slice back so Stage B can run eval, which is what makes the fluency probe
# (and therefore the veto in `wlm eval run`) work for this stage.
wlm dpo-pairs --config "$CFG" --adapter "$BASE_ADAPTER" --split val \
              --out data/processed/dpo.jsonl --av runs/av --val-frac 0.15

wlm train dpo --config "$CFG" --adapter "$BASE_ADAPTER" --out runs/stage_b \
              --pairs data/processed/dpo.jsonl \
              --val-pairs data/processed/dpo_val.jsonl
wlm generate  --config "$CFG" --adapter runs/stage_b --split blind --out runs/stage_b/gen.jsonl
wlm eval run  --gen runs/stage_b/gen.jsonl --av runs/av \
              --out runs/stage_b/report.html --run-name stage_b \
              --baseline runs/baseline/report.json

echo
echo "Deliverable: gain on the INFORMAL cases and reduced generic-AI-voice, fluency preserved."
echo "For 'reduced generic voice', name the stylometry features that moved (hedge rate,"
echo "burstiness, sentence-opener distribution) rather than reporting a vibe."
