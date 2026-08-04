#!/usr/bin/env python3
"""Snapshot an expansion author's committable artifacts into the repo (transparency policy).

This is a research repository first: every detail a verification needs should live in the
repo when it lawfully can. For a PUBLIC-DOMAIN expansion author (e.g. Twain), that means
committing the same things the Chesterton dev corpus commits — the exact corpus a result was
produced from, the paid non-regenerable pairs.jsonl, and (as they appear) run records:

    data/authors/<name>/author/**       the corpus, verbatim from the author root
    data/authors/<name>/distractor/**   the verifier's negative class
    data/authors/<name>/pairs.jsonl     the API-generated pairs (cannot be re-made identically)
    runs/authors/<name>/**              run records — report.json etc. are already allow-listed

Two guards make this safe to leave lying around:

  1. A root whose RUNBOOK carries the PRIVATE AUTHOR marker is refused outright — private
     writing and research-use-only corpora (the BAC blogger) can never be snapshotted.
  2. The destination must already be allow-listed by an explicit `!data/authors/<name>/**`
     line in .gitignore. Committing a new author is therefore always a deliberate two-step
     (edit .gitignore, then snapshot), never a side effect of running this script.

    python scripts/snapshot_author.py ~/authors/twain twain
    python scripts/snapshot_author.py ~/authors/twain twain --runs   # also copy run records
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RECORD_NAMES = {"report.json", "report.html", "run_meta.json", "gen.jsonl",
                "contamination.json", "matrix_state.json"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="the author's WLM_ROOT tree")
    ap.add_argument("name", help="snapshot name -> data/authors/<name>/")
    ap.add_argument("--runs", action="store_true",
                    help="also snapshot run records into runs/authors/<name>/")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    runbook = root / "RUNBOOK.md"
    if not runbook.exists():
        print(f"{root} is not a scaffolded author root (no RUNBOOK.md)")
        return 1
    if "PRIVATE AUTHOR" in runbook.read_text(encoding="utf-8"):
        print(f"REFUSED: {root} is marked PRIVATE AUTHOR — its data never enters the repo. "
              "Only public-domain authors are snapshotted.")
        return 1
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    if f"!data/authors/{args.name}/**" not in gitignore:
        print(f"REFUSED: .gitignore has no `!data/authors/{args.name}/**` allow-list line.\n"
              "Committing a new author is a deliberate two-step: add that line (with a\n"
              "provenance comment) to .gitignore first, then re-run this snapshot.")
        return 1

    dest = REPO / "data" / "authors" / args.name
    copied = 0
    for src_rel, dst_rel in (("data/raw/author", "author"),
                             ("data/raw/distractor", "distractor")):
        src = root / src_rel
        if not src.exists():
            print(f"  [skip] {src} missing")
            continue
        target = dest / dst_rel
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
        copied += sum(1 for p in target.rglob("*") if p.is_file())
    pairs = root / "data" / "interim" / "pairs.jsonl"
    if pairs.exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pairs, dest / "pairs.jsonl")
        copied += 1
    else:
        print("  [note] no pairs.jsonl yet — re-run after backtranslation to snapshot it")

    if args.runs and (root / "runs").exists():
        rdest = REPO / "runs" / "authors" / args.name
        for p in (root / "runs").rglob("*"):
            if p.is_file() and (p.name in RECORD_NAMES or "results" in p.parts):
                out = rdest / p.relative_to(root / "runs")
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, out)
                copied += 1

    print(f"snapshotted {copied} file(s) -> {dest}"
          + (f" and runs/authors/{args.name}/" if args.runs else ""))
    print("Run `make check-data` before committing, as always.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
