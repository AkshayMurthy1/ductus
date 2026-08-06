# `authors/` — replication-author experiment trees

The repository root is itself the primary author's experiment (Chesterton, the reproducible
dev corpus). Every additional author lives here as a **complete, self-contained tree with the
identical layout** — the standard replication structure:

```
authors/<name>/
  RUNBOOK.md              # provenance + the exact per-author command sequence
  data/raw/author/        # the corpus (committed — pins what every number was produced from)
  data/raw/distractor/    # the verifier's negative class (committed)
  data/interim/           # pairs.jsonl committed (paid API calls); the rest regenerates
  data/processed/         # splits + sweep — deterministic, never committed
  runs/                   # av/, av2/, per-arm run dirs; records commit, weights never
```

Everything in the pipeline is `WLM_ROOT`-driven, so running an author is the same protocol as
the root, pointed one directory down:

```bash
export WLM_ROOT="$PWD/authors/<name>"       # absolute path — relative roots break subprocesses
python scripts/new_author.py "$WLM_ROOT" --name "..." --register informal   # scaffold
python scripts/build_author_corpus.py <name> --root "$WLM_ROOT"             # pinned-ID corpus
# then the RUNBOOK's sequence: ingest -> pairs -> split -> fit both verifiers -> sweep -> GPU
```

Rules:

- **One `.gitignore` allow-list block per author, added deliberately.** A new author commits
  nothing until someone asserts, by editing `.gitignore`, that its corpus is redistributable.
  Research-use-only corpora (e.g. the Blog Authorship Corpus) stay untracked forever — usable
  locally, never committed.
- **Private authors do not belong in this repository at all.** This is a public research repo;
  applying the protocol to a private individual's writing is future product work and happens in
  a separate, private repository (docs/STATUS.md §5).
- **The ruler rule applies per author:** each tree fits its own verifiers once (`runs/av`,
  `runs/av2`) and every arm of that author is scored by them. Cross-author comparisons compare
  *shapes* (cliff, frontier), never raw AV values — different verifiers are different rulers.

Current authors: **twain** (Mark Twain, informal first-person American prose — see
`authors/twain/RUNBOOK.md` for provenance and gates).
