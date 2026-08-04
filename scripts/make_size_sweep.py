#!/usr/bin/env python3
"""Build nested, document-level subsamples of train.jsonl for a corpus-size sweep.

The question this sweep answers: **at what corpus size does a Stage-A LoRA start beating
few-shot prompting?** That number decides how much writing a real user must supply before the
adapter is worth training at all -- i.e. whether the product is an adapter or a prompt.

Three properties make the comparison honest, and all three are easy to lose:

1. **The eval never moves.** Every arm generates on the same `blind.jsonl` and is scored by the
   same verifier in `runs/av/`. Refitting the AV per arm would change the ruler between
   measurements. Fit it once, on the FULL train set, before running the sweep.
2. **Subsamples are nested.** The 5k set is a subset of the 25k set, which is a subset of the
   100k set. Differences between arms are then attributable to size rather than to which
   documents got drawn.
3. **Subsampling is by document, never by pair.** Two chunks of one document share phrasing;
   splitting at pair level would put near-identical text at two different corpus sizes and
   quietly inflate the small arms.

`val.jsonl` is deliberately NOT subsampled: early stopping and best-checkpoint restore should
behave identically across arms, so the only variable is training-set size.

    python scripts/make_size_sweep.py                       # 5k / 25k / 100k / full
    python scripts/make_size_sweep.py --sizes 2000 10000    # custom targets, in words
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# WLM_ROOT-aware, like every other entry point: the control-author and extra-author protocols
# (docs/RUN_MATRIX.md §1, docs/EXPANSION.md) run this script under a second root, and a
# repo-relative path here would silently sweep the wrong corpus.
from wlm.paths import PROCESSED  # noqa: E402

SWEEP = PROCESSED / "sweep"
# RESEARCH_BRIEF §3 (RQ1): ≈{2k, 5k, 10k, 25k, 50k, 100k} words.
DEFAULT_SIZES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES,
                    help="target training words per arm (a 'full' arm is always added)")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    train_path = PROCESSED / "train.jsonl"
    if not train_path.exists():
        print(f"missing {train_path} -- run `wlm split` first")
        return 1
    rows = [json.loads(ln) for ln in train_path.read_text(encoding="utf-8").splitlines() if ln]

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    # Count each passage ONCE. `questions_per_chunk` emits several pairs per chunk, so summing
    # pair word counts double-counts the text and would label a 2.7k-word arm as "5k" -- the
    # exact number this sweep exists to report.
    doc_words = {d: sum(len(t.split()) for t in {p["response"] for p in ps})
                 for d, ps in by_doc.items()}
    total = sum(doc_words.values())

    # One shuffle, then prefixes of it -- this is what makes the arms nested.
    order = sorted(by_doc)
    random.Random(args.seed).shuffle(order)

    targets = sorted({s for s in args.sizes if s < total}) + [total]
    SWEEP.mkdir(parents=True, exist_ok=True)
    manifest = []

    for target in targets:
        picked, words = [], 0
        for doc in order:
            if words >= target:
                break
            picked.append(doc)
            words += doc_words[doc]
        label = "full" if target == total else f"{target // 1000}k"
        arm = SWEEP / label
        arm.mkdir(parents=True, exist_ok=True)

        pairs = [p for d in picked for p in by_doc[d]]
        (arm / "train.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        # val and blind are shared, not subsampled: the ruler must not move between arms.
        for name in ("val", "blind"):
            src = PROCESSED / f"{name}.jsonl"
            if src.exists():
                shutil.copyfile(src, arm / f"{name}.jsonl")

        manifest.append({"arm": label, "target_words": target, "actual_words": words,
                         "documents": len(picked), "pairs": len(pairs)})
        print(f"  {label:<6} {words:>8,} words  {len(picked):>4} docs  {len(pairs):>5} pairs")

    # Nesting is the property the whole comparison rests on, so assert it rather than trust it.
    for a, b in zip(manifest, manifest[1:], strict=False):
        sa = {p["pair_id"] for p in
              (json.loads(x) for x in (SWEEP / a["arm"] / "train.jsonl").read_text().splitlines())}
        sb = {p["pair_id"] for p in
              (json.loads(x) for x in (SWEEP / b["arm"] / "train.jsonl").read_text().splitlines())}
        assert sa <= sb, f"{a['arm']} is not a subset of {b['arm']}"

    (SWEEP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} arms -> {SWEEP}")
    print("Fit the verifier ONCE on the full train set before sweeping:")
    print("  wlm eval fit-av --author data/processed/train.jsonl "
          "--distractor data/raw/distractor --out runs/av")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
