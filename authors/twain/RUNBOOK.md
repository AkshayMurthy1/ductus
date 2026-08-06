# Mark Twain — author runbook

Scaffolded by scripts/new_author.py. Root: `authors/twain` (in-repo — see authors/README.md)
Register: **informal**

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
export WLM_ROOT="$PWD/authors/twain"   # run from the repo root

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
