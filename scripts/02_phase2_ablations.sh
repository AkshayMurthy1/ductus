#!/usr/bin/env bash
# Phase 2: the ablation grid. Cheapest-first so a broken harness surfaces in minutes, not hours.
set -euo pipefail
ORDER="${ORDER:-a13 a03 a06 a07 a08 a01 a02 a05 a11 a10 a09 a04 a12 a14}"

for tag in $ORDER; do
  cfg=$(ls configs/ablations/${tag}_*.yaml)
  name=$(basename "$cfg" .yaml)
  echo "=== $name ==="
  if [ "$tag" = "a14" ]; then
    echo "a14 is a DIAGNOSTIC (scrubbing off). It requires rebuilding the dataset with"
    echo "  wlm scrub --no-entities ... && wlm backtranslate ... && wlm split ..."
    echo "into a separate directory. Do not ship this adapter. Skipping automatic run."
    continue
  fi
  wlm train sft --config "$cfg" --out "runs/$name"
  wlm generate  --config "$cfg" --adapter "runs/$name" --split blind --out "runs/$name/gen.jsonl"
  wlm eval run  --gen "runs/$name/gen.jsonl" --av runs/av \
                --out "runs/$name/report.html" --run-name "$name" \
                --baseline runs/baseline/report.json
done
echo "Now fill docs/ABLATIONS.md. Mark anything inside seed variance as 'within noise'."
