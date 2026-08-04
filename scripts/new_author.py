#!/usr/bin/env python3
"""Scaffold a new author root for the generality expansion (docs/EXPANSION.md).

Every path in the pipeline honors WLM_ROOT, so an extra author is a directory tree, not a code
change. This script creates that tree, writes the corpus checklist and the exact command
sequence into it, and refuses to scaffold inside the repo — extra-author corpora (and
especially private ones) live outside version control, always.

    python scripts/new_author.py ~/authors/orwell --name "George Orwell" --register formal
    python scripts/new_author.py ~/authors/blogger01 --register informal --private

The tree it creates is exactly what scripts/00_phase0_cpu.sh expects under WLM_ROOT:

    <root>/data/raw/author/       the author's plain-text documents (one file per document)
    <root>/data/raw/distractor/   era/register-matched text by OTHER writers
    <root>/RUNBOOK.md             checklist + the full command sequence for this author
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RUNBOOK = """\
# {name} — author runbook

Scaffolded by scripts/new_author.py. Root: `{root}`
Register: **{register}**{private_line}

## Corpus checklist (fill before running anything)

- [ ] `data/raw/author/` holds plain-text files, one per document, author-typed prose only
      (strip quotations, boilerplate, and anything not written by the author).
- [ ] Total is >= 25k words if this author only needs the decisive cells, >= 100k+ if it
      should support the full size sweep. Nothing below ~10k words has ever produced signal.
- [ ] `data/raw/distractor/` holds 15-25 OTHER writers matched on era AND register — the
      verifier learns "this author vs plausible alternatives," and mismatched distractors
      give an inflated AUC that poisons every downstream number.
- [ ] Provenance recorded below (source, rights, date pulled).
- [ ] Private author? Then no file under this root is ever committed anywhere. The repo's
      `make check-data` blocks private paths, but the real guard is that this root is not a
      git repository at all.

## Command sequence (from the repo checkout, in order)

```bash
export WLM_ROOT={root}

# CPU half: ingest -> chunk -> scrub -> pairs -> split -> fit BOTH verifiers.
scripts/00_phase0_cpu.sh
python scripts/second_instrument.py --runs "$WLM_ROOT/runs"   # fits av2 for this corpus

# STOP: check the printed gates (AUC in [0.75, 0.97], zero doc overlap) before GPU money.

# Size arms. Match the dev author's cliff-region arms unless the corpus is smaller.
python scripts/make_size_sweep.py --sizes 10000 25000 50000

# GPU half: floor + per-arm baseline & Stage A (add rq3/seeds per docs/EXPANSION.md).
python scripts/run_matrix.py --only floor rq1

# Assemble THIS author's tables from its own records.
python scripts/assemble_results.py --runs "$WLM_ROOT/runs" --out "$WLM_ROOT/runs/results"
```

Per-corpus rules that do not bend (docs/RUN_MATRIX.md):

- The verifier is fit ONCE per corpus, on that corpus's full train set, and never refit
  per arm. Verifiers are never shared across authors — each author gets their own ruler,
  and cross-author claims compare *within-author deltas and shapes*, never raw AV numbers.
- Splits are by document. `wlm split` hard-fails on overlap; do not work around it.
- A run failing the fluency or leakage gate is a failure regardless of its AV score.

## Provenance

| file/source | origin | rights | pulled |
|---|---|---|---|
| | | | |
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="directory to scaffold (created if missing); NOT inside the repo")
    ap.add_argument("--name", default=None, help="author display name (default: directory name)")
    ap.add_argument("--register", default="formal", choices=["formal", "informal", "mixed"])
    ap.add_argument("--private", action="store_true",
                    help="real person's writing: adds the never-commit warnings")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if root == REPO or REPO in root.parents or root in (REPO / "data").parents:
        print(f"refusing to scaffold inside the repo ({REPO}) — author corpora live outside "
              "version control. Pick a path like ~/authors/<name>.")
        return 1
    if (root / "RUNBOOK.md").exists():
        print(f"{root} already has a RUNBOOK.md — not overwriting an existing author root.")
        return 1

    name = args.name or root.name
    for sub in ("data/raw/author", "data/raw/distractor", "data/interim",
                "data/processed", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    private_line = ("\nPrivacy: **PRIVATE AUTHOR — never commit anything under this root; "
                    "results are reported only as aggregates (Table 4 protocol).**"
                    if args.private else "")
    (root / "RUNBOOK.md").write_text(
        RUNBOOK.format(name=name, root=root, register=args.register,
                       private_line=private_line),
        encoding="utf-8")

    print(f"scaffolded {name} at {root}")
    print(f"  1. drop author text into   {root / 'data/raw/author'}")
    print(f"  2. drop distractors into   {root / 'data/raw/distractor'}")
    print(f"  3. follow                  {root / 'RUNBOOK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
