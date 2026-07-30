#!/usr/bin/env bash
# Phase 0, laptop half: corpus -> splits -> fitted authorship verifier.
# Nothing here needs a GPU. Stop and read the printed reports before moving to the GPU box.
set -euo pipefail
CFG="${CFG:-configs/stage_a.yaml}"

wlm ingest local --in data/raw/author --out data/interim/docs.jsonl
wlm chunk        --in data/interim/docs.jsonl   --out data/interim/chunks.jsonl   --config "$CFG"
wlm scrub        --in data/interim/chunks.jsonl --out data/interim/scrubbed.jsonl --config "$CFG" \
                 ${SCRUB_TERMS:+--terms "$SCRUB_TERMS"}
wlm backtranslate --in data/interim/scrubbed.jsonl --out data/interim/pairs.jsonl --config "$CFG" --audit
wlm split        --in data/interim/pairs.jsonl  --outdir data/processed --config "$CFG"
wlm eval fit-av  --author data/processed/train.jsonl --distractor data/raw/distractor \
                 --out runs/av --config "$CFG"

echo
echo "Phase 0 (CPU half) done. Before spending a GPU hour, confirm:"
echo "  1. chunk health: frac_starts_mid_thought < 0.35"
echo "  2. split summary: _doc_overlap is empty and both registers appear in val and blind"
echo "  3. verifier AUC >= 0.75 -- and if it is above 0.97, check that your distractors are"
echo "     register-matched rather than trivially different from your own writing"
echo
echo "Then: rsync -av data/processed/ runs/av/ GPUBOX:~/writelikeme/  and run scripts/01_phase1_gpu.sh"
