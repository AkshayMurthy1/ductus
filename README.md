# ductus

Style transfer without content transfer: `ductus` fine-tunes a small, deletable LoRA adapter
on a 1–3B open-weight model from a single author's prose, learning their writing form
(grammar, rhythm, idiom) while leaking none of their words. Every run scores **style
acquired** (attribution by a frozen authorship verifier) against **content leaked** (verbatim
n-grams, entity emission, paraphrase-level echo).

Full results and threats to validity: [`docs/STATUS.md`](docs/STATUS.md). Design:
[`docs/PLAN.md`](docs/PLAN.md).

## Research questions

- **RQ1 — corpus size:** How many words of an author's writing does an adapter need to acquire
  their voice, and does leakage grow with it?
- **RQ2 — adapter placement:** Which transformer modules (attention vs. MLP) carry the style?
- **RQ3 — training stage:** Does on-policy DPO over real-vs-generated pairs improve style, and
  at what memorization cost?

## Key findings

1. **Style acquisition is a phase transition, not a dial.** Attribution is ≈ 0 below ~10k
   training words, jumps to 0.52 between 10k and 15k, and saturates at 0.79–0.90 by 25–50k —
   above the 0.71 attribution rate of the author's own held-out prose. Step counts across arms
   suggest optimization steps, not words, are the binding variable.

2. **Fine-tuning leaks less than prompting.** Few-shot prompting copied training passages
   verbatim (up to 11.9% of generations) without acquiring the style. Adapters acquired the
   style with zero verbatim 12-grams across every run, corpus size, placement, and checkpoint.

   | method | style acquired (AV) | verbatim leakage |
   |---|---|---|
   | few-shot prompting | ≈ 0.00 | up to **11.9%** (5 of 9 arms fail the 5% gate) |
   | LoRA adapter | **0.77 – 0.90** at saturation | **0.0%**, everywhere |

   ![Style-vs-leakage frontier](runs/results/figures/fig1_frontier.png)

3. **The style lives in attention.** Attention-only adapters beat all-linear: higher
   attribution (0.857 vs 0.790), 4× fewer parameters, 3× less wall-clock, and improved general
   fluency. The q,k projections carry most of the style; v,o contribute the least. Training
   takes ~10 minutes on one GPU.

   ![Locus ablation](runs/results/figures/fig3_locus.png)

4. **It replicates.** The same cliff-then-plateau shape and zero verbatim leakage hold across
   three base models (Qwen-3B, Qwen-1.5B, Llama-3.2-3B) and a second author in a different
   register (Twain, 0.47 → 0.904). Two independently trained verifiers rank all runs the same
   way (Spearman 0.93). The cliff's location moves with the model; the shape doesn't.

   ![Phase transition replication](runs/results/figures/fig2_phase_transition.png)

5. **Caveat: entity hygiene depends on the pipeline.** Zero verbatim copying survives training
   on unscrubbed text, but unscrubbed training doubles the rate at which the adapter emits
   names from the corpus — the PII scrubbing stage is load-bearing.

DPO (RQ3) helps the all-linear adapter (+0.055) and mildly hurts the attention-only one; the
recommended recipe is attention-only SFT.

## Method

Documents from one author (dev corpus: G.K. Chesterton, public domain, ~256k words) are
chunked into response-sized passages and scrubbed of names/places to typed placeholders. A
separate hosted API model writes the question each passage answers (instruction
backtranslation), so every training target is real author text, never model output. Stage A
fine-tunes a LoRA adapter on the (question → passage) pairs; Stage B optionally adds on-policy
DPO with real text as `chosen`. Each run generates 252 blind-set responses scored by a frozen
per-corpus style verifier (document-grouped, deduped, held-out AUC 0.896), three leakage
channels, a fluency budget, and a pretraining-contamination probe. One run = one
`report.json`; all tables and figures are assembled from those records.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"

pytest -q                             # full CPU test suite, < 3 s
make demo                             # end-to-end pipeline on the bundled fixture author
python scripts/assemble_results.py    # rebuild every table + figure from committed run records
python scripts/make_figures.py        # rebuild the publication figures
```

Committed `runs/**/report.json` records are the ground truth; `assemble_results.py`
regenerates every table and figure byte-for-byte. Re-running the training matrix needs one
24 GB+ GPU (`pip install -e ".[gpu]"`); the full evidence base cost ~6 GPU-hours on a single
NVIDIA L40S. See [`docs/RUN_MATRIX.md`](docs/RUN_MATRIX.md) for drivers and run order.

## Design invariants

- Real author text is the target in every training signal; the question generator is a
  separate API model, never the trainee (prevents self-distillation collapse).
- One verifier per corpus, fit once, frozen across every arm.
- Splits are by document, never by chunk (sibling chunks leak topic and phrasing).
- A run passes only if all gates pass (leakage ≤ 5%, fluency within budget), regardless of
  style score.

## Layout

```
src/wlm/           ingest → chunk → scrub → backtranslate → train → generate → eval
configs/           one YAML per arm; every ablation is a one-line diff
scripts/           numbered phase drivers, resumable matrix runner, results assembler
runs/              committed per-run records (report.json, gen.jsonl) + assembled results
authors/twain/     the replication author — a complete parallel tree, same layout
docs/              PLAN (design) · STATUS (living results) · RUN_MATRIX (how to run it all)
```

## Privacy

The dev corpus is public-domain fixture text (`data/README.md`); real writing never ships
(`make check-data` guards every commit). Scrubbing writes an audit log. The only personal
artifact produced is the adapter itself — a few tens of MB, deletable.
