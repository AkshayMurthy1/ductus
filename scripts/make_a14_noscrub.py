#!/usr/bin/env python3
"""Build the a14 (no-scrub) diagnostic dataset WITHOUT re-spending API budget.

a14 answers the question the scrubbed runs cannot: would this recipe absorb the author's
entities and specific vocabulary if they were present in the targets? With scrubbing on, clean
entity numbers partly measure the scrubber, not the model; a14 separates "LoRA learns form
faster than content" (intrinsic) from "scrubbing prevents content" (preprocessing).

The trick that makes it cheap: backtranslated questions are keyed by chunk_id, and scrubbing
only altered the *response* text. So this script reuses the already-paid-for questions and
swaps each pair's response for the UNSCRUBBED chunk text, preserving everything else:

  - identical pair_ids, doc_ids, prompts, and question provenance
  - identical train/val membership, pair for pair (read from data/processed/), so the split
    inherits the canonical document-level integrity and the diff vs the reference run is
    exactly one variable: scrubbing
  - blind.jsonl is NOT rebuilt — the a14 arm generates and is scored on the canonical blind
    split with the canonical verifier, because the ruler must not move

Outputs data/processed/a14_noscrub/{train,val}.jsonl. Run it, rsync the directory to the GPU
box, then (single diagnostic run — do not ship this adapter):

  wlm train sft --config configs/ablations/a14_no_scrubbing.yaml \
      --train data/processed/a14_noscrub/train.jsonl \
      --val   data/processed/a14_noscrub/val.jsonl \
      --out   runs/matrix/a14_no_scrubbing
  wlm generate --config configs/ablations/a14_no_scrubbing.yaml \
      --adapter runs/matrix/a14_no_scrubbing --split-path data/processed/blind.jsonl \
      --out runs/matrix/a14_no_scrubbing/gen.jsonl
  wlm eval run --gen runs/matrix/a14_no_scrubbing/gen.jsonl \
      --train data/processed/a14_noscrub/train.jsonl \
      --av runs/av --out runs/matrix/a14_no_scrubbing/report.html \
      --run-name a14_no_scrubbing --baseline runs/sweep/full/stage_a/report.json

Note --train points at the UNSCRUBBED train file: the leakage suite must compare generations
against the text the model actually saw, or verbatim overlap would be measured against a
corpus the model was never trained on.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.paths import INTERIM, PROCESSED, read_jsonl, write_jsonl  # noqa: E402

OUT = PROCESSED / "a14_noscrub"


def main() -> int:
    chunks_path = INTERIM / "chunks.jsonl"
    for p in (chunks_path, PROCESSED / "train.jsonl", PROCESSED / "val.jsonl"):
        if not p.exists():
            print(f"missing {p} — this script needs the interim chunks and the canonical "
                  "splits produced by the phase-0 pipeline.")
            return 1

    unscrubbed = {c["chunk_id"]: c["text"] for c in read_jsonl(chunks_path)}
    stats = {"swapped": 0, "identical": 0, "missing_chunk": 0}

    def swap(rows):
        out = []
        for r in rows:
            raw = unscrubbed.get(r.get("chunk_id"))
            if raw is None:
                # A pair whose source chunk vanished would silently train on scrubbed text and
                # dilute the diagnostic — drop it loudly instead.
                stats["missing_chunk"] += 1
                continue
            stats["swapped" if raw != r["response"] else "identical"] += 1
            out.append({**r, "response": raw, "n_words": len(raw.split())})
        return out

    for name in ("train", "val"):
        rows = swap(read_jsonl(PROCESSED / f"{name}.jsonl"))
        n = write_jsonl(OUT / f"{name}.jsonl", rows)
        words = sum(r["n_words"] for r in {r["chunk_id"]: r for r in rows}.values())
        print(f"  {name:<6} {n:>5} pairs  {words:>8,} unique-passage words -> {OUT / name}.jsonl")

    print(f"\nstats: {stats}")
    if stats["missing_chunk"]:
        print(f"WARNING {stats['missing_chunk']} pair(s) dropped — their chunks are gone from "
              f"{chunks_path}. If this is more than a handful, the interim data is out of sync "
              "with the splits; rebuild phase 0 before trusting the diagnostic.")
    if not stats["swapped"]:
        print("ERROR nothing differed from the scrubbed splits — scrubbing appears to have "
              "been a no-op, so a14 would compare a run against itself.")
        return 1
    print("\nDIAGNOSTIC DATASET — train/val targets contain real entities. Do not ship any "
          "adapter trained on it; see the run commands in this script's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
