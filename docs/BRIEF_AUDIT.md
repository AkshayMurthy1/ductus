# Repository audit against RESEARCH_BRIEF.md — 2026-07-31

The brief (§6) asks for each requirement to be mapped onto what the code actually provides:
(a) what already satisfies it, (b) what exists but is wrong, (c) what is missing. This is that
report, plus the fixes applied in the same change set. Anything marked **fixed** below was
changed on this date; the diff is the authoritative record.

## §6 checklist

| # | Requirement | Verdict | Detail |
|---|---|---|---|
| 1 | Subsample by word count, by document, deterministic | **(a) satisfied** | `scripts/make_size_sweep.py` — nested prefixes of one seeded shuffle, counted on deduped passage words. **Fixed:** defaults were {5k, 25k, 100k}; now the brief's {2k, 5k, 10k, 25k, 50k, 100k}. |
| 2 | Splits by document, hard-fail on overlap | **(a) satisfied** | `wlm split` exits non-zero on any doc overlap; near-duplicate documents are union-grouped first (`dataset.near_duplicate_groups`) so a recycled passage cannot straddle the boundary. |
| 3 | Verifier deduped, document-grouped, held-out AUC | **(a) satisfied** | `eval/av.py` dedupes both sides, groups the author side by doc id (`GroupShuffleSplit`), reports AUC and exits non-zero below 0.75. **Fixed:** added the brief's other tail — a warning when AUC > 0.97, which signals a trivial confound, not success. |
| 4 | Leakage = verbatim n-grams AND entity emission | **(b) existed but was wrong** | Both metrics were implemented, but `run_from_files` never passed the pre-scrub author text, so **entity emission never ran in any real report** — half the leakage axis was silently absent. **Fixed:** `wlm eval run --raw-docs` (default `data/interim/docs.jsonl`) feeds it. Also **added** a third, paraphrase-robust leakage dimension: `semantic_echo` (see docs/RUN_MATRIX.md). |
| 5 | Fluency probe, adapter attached vs detached | **(a) satisfied** | `eval/fluency.py` via `disable_adapter()`, run per checkpoint by callback in both stages; the report vetoes on it. |
| 6 | Few-shot baseline: shared decode path, refuses truncation | **(a) satisfied** | `generate()` is the single decode path; it raises on prompt-budget overflow rather than truncating. Exemplars come from train only and are deduped. |
| 7 | Every hyperparameter in config | **(a) satisfied** | One dataclass per knob group, YAML-driven, unknown keys rejected. **Fixed:** the one hardcoded knob found (`save_total_limit=3` in the SFT trainer) is now `sft.save_total_limit`. |
| 8 | Prompt format provably identical train/inference | **(a) satisfied** | Training rows carry `cfg.gen.system_prompt` because `generate()` always sends it; `tests/test_training_contracts.py` pins this, including for the raw-mix rows and DPO rows. |
| 9 | Seeding threaded end-to-end | **(b) existed but was wrong** | Subsampling, splits, and training were seeded; **generation sampling was not** — no `torch.manual_seed` existed anywhere, so "same run, different seed" was not actually controlled, and changing any seed required editing YAML (breaking one-variable-per-run hygiene). **Fixed:** `gen.seed` in config, seeded in `generate()` (sampler + few-shot exemplar draw), and a `--seed` override on `wlm train / generate / baseline` for variance cells. |
| 10 | Self-contained per-run record with resolved config | **(b) existed but was split** | `report.json` had metrics; trainable params + wall-clock lived only in `run_meta.json`; baseline runs recorded no config at all. **Fixed:** `wlm baseline` and no-adapter `wlm generate` now write `run_meta.json`, and `run_from_files` merges stage, trainable-parameter counts and wall-clock into `report.json` — one record per run, as required. |
| 11 | Unattended, resumable run-matrix driver | **(c) was missing** | `scripts/04_size_sweep_gpu.sh` resumed RQ1 only; RQ2/RQ3 scripts were not resumable, and nothing drove seeds, the base-model floor, or the contamination probe. **Fixed:** `scripts/run_matrix.py` — one idempotent driver for floor + RQ1 + RQ2 + RQ3 + variance seeds, unit-by-unit resume, failure log in `runs/matrix_state.json`. |
| 12 | Corpus artifacts and adapters out of VCS | **(a) satisfied** | `.gitignore` excludes `data/**` and `runs/**` with `.gitkeep`/README re-includes. |

## §5 validity requirements

- **Distractors era/genre/length-matched** — *(b) partially wrong.* Length matching is enforced
  (`load_distractor_texts` windows to the author chunk bounds). But the dev distractor set was
  missing the brief's named hard negatives. **Fixed:** added Beerbohm (PG 1956), Lynd (PG 13448),
  E.V. Lucas (PG 73174) and Gardiner (PG 10675), with Belloc documented as the primary hard
  negative. *Shaw's prefaces were deliberately skipped*: on Gutenberg they exist only inside
  play volumes, where the essay extractor would sweep in stage dialogue. Note kept in
  `build_dev_corpus.py`: a few holdover authors (Addison, Emerson, Thoreau, Twain) are **not**
  era-matched — if the verifier AUC looks too good, drop them first.
- **Contamination check (untuned base-model performance)** — *(c) was missing.* **Fixed** twice
  over: the driver's `floor` step records base-model attribution on the blind set, and the new
  `wlm eval contamination` records a perplexity familiarity ratio (author vs matched distractors
  under the untuned model). Both feed Table 4.
- **Control author** — *(c) was missing orchestration.* No new code needed: every path honors
  `WLM_ROOT`, so the control protocol is the identical driver run under a second root. Documented
  in docs/RUN_MATRIX.md; `assemble_results.py --control-runs` merges the two into Table 4.
- **Preference pairs length-matched** — *(a) satisfied* (`dpo.length_match_tolerance`, with
  regeneration retries before dropping).
- **Noise floor before ranking** — *(c) was missing tooling.* **Fixed:** driver `seeds` section
  (3 seeds × 2 cells by default) plus a noise-floor section in the assembled tables that
  explicitly instructs reporting within-noise differences as ties.
- **Log every imposed bound** — *(a) largely satisfied* (drop counts, retry counts, caps are
  printed and recorded); the new contamination probe records its sampling cap in its output.

## §7 deliverables

Tables were previously hand-assembled (only the RQ1 table had a collector). **Fixed:**
`scripts/assemble_results.py` builds Tables 1–4 and Figure 1 (the style-vs-leakage frontier,
with a training-time trajectory panel) purely from `report.json` records — never transcribed by
hand, per §4. `scripts/collect_sweep.py` remains as a quick RQ1-only view.

## Beyond the brief (novel additions)

See docs/RUN_MATRIX.md §"Novel measurements" for the rationale and usage of:

1. **Training-time frontier** (`scripts/05_checkpoint_trajectory.sh` + `configs/trajectory.yaml`)
   — evaluates every checkpoint of one run, turning the brief's primary question about *rates*
   ("faster than") into a measured curve rather than an endpoint comparison.
2. **Semantic echo** (`wlm.eval.leakage.semantic_echo`) — paraphrase-robust content leakage,
   zero-referenced against the author's own held-out text.
3. **Adapter anatomy** (`scripts/adapter_anatomy.py`) — per-layer/per-module update-norm
   decomposition of the trained LoRA, the mechanistic companion to RQ2.
4. **Contamination familiarity ratio** (`wlm eval contamination`) — quantifies how much of the
   public-author arm sits on pretraining recall.
