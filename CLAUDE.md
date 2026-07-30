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
| `src/wlm/eval/` | stylometry, authorship verifier, leakage, fluency, harness, HTML report |
| `src/wlm/train/` | Stage-A SFT, Stage-B DPO |
| `src/wlm/generate.py` | few-shot prompting baseline + adapter generation (same decode path) |
| `configs/` | `stage_a.yaml`, `stage_b.yaml`, and one file per ablation |

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
- Never commit anything under `data/` or `runs/`.

## Testing

`pytest -q` must pass before any commit; it's CPU-only and takes under a second.
`make demo` runs the whole CPU pipeline on `tests/fixtures/author/` with offline template
questions — use it as the smoke test after touching the pipeline.

When you add a metric to the eval harness, add a test that it is **zero on identical inputs and
larger on genuinely different inputs**. A silently-broken metric is worse than a missing one
because it produces confident wrong conclusions.

## What "done" means for a phase

Never report a phase complete on the basis of a loss curve. Each phase has a deliverable defined
in `docs/PLAN.md` §5, and every one of them is a number from `wlm eval run` compared against the
previous phase's report. If the fluency probe fails or the leakage check fails, the run is not a
success regardless of the AV score.
