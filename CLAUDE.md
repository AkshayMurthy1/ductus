# CLAUDE.md — repo context for Claude Code

Read this before touching anything. It exists so you don't have to re-explain the project every
session.

## What this is

`ductus` learns one person's **writing form** (grammar, rhythm, flow, idiom) from their own
prose and emits a small reversible LoRA adapter on an open-weight 1–3B instruct model. It
deliberately does **not** learn their facts. Full design: `docs/PLAN.md`.

## The three invariants — never break these

1. **The user's real text is the ground truth in every training signal.** Stage-A targets are
   real passages. Stage-B `chosen` is a real passage; only `rejected` is generated. If you ever
   find yourself training on model output as the target, stop — that is the recursive-collapse
   failure mode this design exists to avoid.
2. **The question generator is a separate hosted API model, never the model being trained.**
   `src/wlm/backtranslate.py` talks to the Anthropic API. It must never be swapped for the
   trainee.
3. **Splits are by document, never by pair.** Two chunks from one document share topic and
   phrasing; splitting at chunk level leaks and inflates every metric. `wlm split` hard-fails on
   document overlap — keep that check.

## Two more rules that are easy to violate by accident

- **An HMM is never a training loss here.** See `docs/PLAN.md` §8. It is allowed as an eval
  metric or a decode-time rerank, nothing else. If a task asks for a "distillation loss toward a
  stylometric model," push back and cite §8.
- **LoRA goes on attention *and* MLP.** `target_modules: all-linear`. If topic leak appears, the
  fix is a lower MLP rank via `rank_pattern` (see `configs/ablations/a05_*.yaml`), not dropping
  the MLP.

## Where things live

| Path | Role |
|---|---|
| `src/wlm/ingest/` | Google Docs + local ingest, normalization (strip anything not author-typed) |
| `src/wlm/chunk.py` | documents → response-sized units, plus a health report |
| `src/wlm/scrub.py` | PII + topic-entity scrubbing to typed placeholders, with an audit log |
| `src/wlm/backtranslate.py` | (question → passage) pair construction via the API model |
| `src/wlm/dataset.py` | splits (by document), chat formatting, raw next-token mix |
| `src/wlm/eval/` | stylometry, authorship verifier, leakage (verbatim + entity + semantic echo), fluency, contamination probe, harness, HTML report |
| `src/wlm/train/` | Stage-A SFT, Stage-B DPO |
| `src/wlm/generate.py` | few-shot prompting baseline + adapter generation (same decode path, seeded) |
| `configs/` | `stage_a.yaml`, `stage_b.yaml`, `trajectory.yaml`, and one file per ablation |
| `scripts/run_matrix.py` | resumable driver for the whole RESEARCH_BRIEF matrix (floor/RQ1/RQ2/RQ3/seeds) |
| `scripts/assemble_results.py` | Tables 1–4 + Figure 1 (frontier) from run records — never hand-transcribe |
| `scripts/adapter_anatomy.py` | per-layer/module LoRA update-norm decomposition (RQ2's mechanistic companion) |
| `docs/BRIEF_AUDIT.md` / `docs/RUN_MATRIX.md` | brief-to-code audit; how to run and assemble everything |
| `docs/EXPANSION.md` | venue-readiness expansion: extra authors (`scripts/new_author.py`, `scripts/build_author_corpus.py`), second base model (`run_matrix --only models`), second verifier (`scripts/second_instrument.py`) |
| `docs/STATUS.md` | **the living record of results and open work — read this for where the research stands** |

## Four traps this codebase has already been bitten by

Undoing any of these looks like a simplification and is not one.

1. **The prompt format must be identical in training and at inference.** `generate()` always
   sends `cfg.gen.system_prompt`, so training rows carry it too. SFT rows are trl
   *conversational prompt-completion* (`{"prompt": [msgs], "completion": [msgs]}`) and DPO rows
   are conversational preference rows — plain strings would skip the chat template entirely and
   train on a format the model never sees.
2. **`gen.max_prompt_tokens` is not `model.max_seq_len`.** The latter is a training length chosen
   for GPU memory. Budgeting generation against it truncates the few-shot baseline's system
   prompt, which quietly hands Stage A a win it did not earn. `generate()` raises rather than
   truncates for this reason — do not "fix" that by re-enabling silent truncation.
3. **Stage B's reference policy is Stage A, not the base model.** With `ref_model=None` and a
   PEFT model, trl disables the adapter for reference logprobs, i.e. anchors KL to the *base*
   model. `dpo.reference_policy: stage_a` loads a second frozen adapter and passes
   `ref_adapter_name` so the anchor lands where the plan intends.
4. **The AV verifier must be fit on deduped, document-grouped text.** `build_pairs` emits several
   questions per chunk, so `train.jsonl` contains each passage multiple times; a row-level split
   puts identical text on both sides and the "held-out" AUC becomes memorization. The AUC is the
   guardrail on every other number, so a compromised AUC compromises everything.

## Conventions

- Every knob lives in `src/wlm/config.py` and is set from YAML. **Do not hardcode
  hyperparameters in training code** — an ablation must be a one-line config diff.
- Ablations inherit: `_extends: ../stage_a.yaml`, then override exactly one thing. One variable
  per run or the table means nothing.
- CPU-side code (ingest → split, eval harness) must import and run with no CUDA and no network.
  Keep `transformers`/`peft`/`trl` imports **inside functions**, never at module top level.
- Data files are JSONL. Read/write via `wlm.paths.read_jsonl` / `write_jsonl`.
- **Data commits follow provenance, not path** (policy in `.gitignore` + `data/README.md`):
  the public-domain fixture corpus, the paid-for `pairs.jsonl`, and run *records*
  (`report.json`, `run_meta.json`, `gen.jsonl`, `runs/results/`) are committed for
  verifiability; weights, checkpoints, and deterministic intermediates never are. Run
  `make check-data` before any commit touching `data/`, `runs/`, or `authors/` — and if
  `data/raw/author` ever holds a real person's writing, it must not ship regardless of the
  allow-list. **Replication authors live in-repo as full trees** under `authors/<name>/`
  (same layout and rules as the repo root; per-author `.gitignore` allow-list — see
  `authors/README.md`). Research-use-only corpora (the BAC blogger) stay untracked, and
  private-author work happens in a separate repository, never here (this repo is pure
  research as of 2026-08-06; see docs/STATUS.md §5).

## Testing

`pytest -q` must pass before any commit; it's CPU-only and takes under a second.
`make demo` runs the whole CPU pipeline on `tests/fixtures/author/` with offline template
questions — use it as the smoke test after touching the pipeline.

When you add a metric to the eval harness, add a test that it is **zero on identical inputs and
larger on genuinely different inputs**. A silently-broken metric is worse than a missing one
because it produces confident wrong conclusions.

## Research-brief invariants (added 2026-07-31)

- **`report.json` is the one record of a run.** The harness merges the resolved config,
  trainable-parameter counts and wall-clock into it; tables and figures are assembled from these
  records by `scripts/assemble_results.py`, never typed in.
- **Seeds are CLI-overridable** (`--seed` on `wlm train/generate/baseline`) and generation
  sampling is genuinely seeded (`gen.seed`). Variance cells are flags, not YAML edits.
- **The verifier is fitted once per corpus** and shared by every arm; if the assembler prints
  more than one AUC, the ruler moved and the comparison is invalid.
- **Entity leakage needs the pre-scrub docs** — `wlm eval run` reads them from
  `data/interim/docs.jsonl` by default; deleting that file silently halves the leakage axis.

## What "done" means for a phase

Never report a phase complete on the basis of a loss curve. Each phase has a deliverable defined
in `docs/PLAN.md` §5, and every one of them is a number from `wlm eval run` compared against the
previous phase's report. If the fluency probe fails or the leakage check fails, the run is not a
success regardless of the AV score.
