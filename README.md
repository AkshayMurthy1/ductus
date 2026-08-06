# ductus

Personal style tuning of an open-weight transformer. Learns **form** (grammar, rhythm, flow,
idiom) from one person's own writing and emits a small, reversible, deletable LoRA adapter.
Does **not** try to learn their facts.

## Results at a glance

The research question: **does a LoRA adapter acquire an author's *form* faster than it
memorizes their *content*?** We trace a style-vs-leakage frontier across corpus size, adapter
placement, training stage, and base model — dev author G.K. Chesterton (public domain) vs 25
era-matched essayists, scored by two independent style verifiers with pre-registered leakage
and fluency gates. Answer: **yes, decisively** — the frontier is L-shaped.

| # | Finding | The number |
|---|---|---|
| 1 | **Style is a phase transition, not a slope.** Nothing below ~10k training words; the voice switches on between 10k and 15k, saturates by ~25–50k at or above the rate real held-out Chesterton scores. In training time it's even sharper — fully formed by gradient step 25 — and the step counts across arms suggest optimization *steps*, not corpus words, set the threshold. | AV 0.004 → **0.524** between 10k and 15k words |
| 2 | **A double dissociation.** Few-shot prompting copies whole training passages verbatim but never sounds like the author; the trained adapter sounds like the author and copies *nothing* — zero verbatim 12-grams across all 20+ adapter runs, at every checkpoint. | prompting: ≤11.9% leak, AV ≈ 0 · adapter: **0% leak**, AV 0.79–0.89 |
| 3 | **The voice lives in attention — mostly in q,k.** Attention-only adapters beat all-linear with 4× fewer parameters, ⅓ the compute, and *better* fluency; the memorization-prone v,o matrices contribute the least style. | attention-only **0.857** vs full recipe 0.790 |
| 4 | **On-policy DPO sharpens without stealing — but only where it's needed.** It lifts the full-locus adapter (+0.055) with zero verbatim overlap *even against its own preference targets*, yet subtracts from the attention-only adapter. Recommended config: attention-only SFT, no second stage. | A+B **0.845**, 0.0% leak vs DPO targets |
| 5 | **It all replicates across base models; the cliff moves, the shape doesn't.** Qwen-1.5B transitions at the same corpus size and peaks highest; Llama-3.2-3B needs more data; every model shows zero adapter leakage and leaky few-shot baselines. | 3 model families, adapter leak **0.0%** at every cell |
| 6 | **Caveat we found ourselves:** entity hygiene is the *pipeline's* (scrubbing) — trained on unscrubbed text, the adapter does absorb entities (4.3/gen vs 2.0 background) while still copying zero verbatim text. Style-not-content is a property of pipeline + adapter together. | a14 diagnostic |

Every number regenerates from committed per-run records (`python scripts/assemble_results.py`
→ `runs/results/tables.md` + the frontier figure); claims are labeled against a measured
seed-noise floor. Full detail, confidence labels, and threats to validity: **`docs/STATUS.md`**.

Implements the plan in `docs/PLAN.md`:

- **Phase 0** — eval harness + few-shot prompting baseline (built *before* any tuning)
- **Phase 1** — Stage A: LoRA SFT on instruction-backtranslated pairs (attention **and** MLP)
- **Phase 2** — ablations on Stage A
- **Phase 3** — Stage B: on-policy DPO sharpening
- **Phase 4** — optional style-vector steering + stylometric HMM as an eval only
- **Phase 5** — one-command pipeline, per-user adapters, hot-swap serving

## Two-machine layout

This repo is built to be split across your laptop and a rented 24 GB GPU.

| Where | What runs | Install |
|---|---|---|
| Laptop (CPU) | ingest, chunk, scrub, backtranslation, eval harness, reports | `pip install -e ".[cpu]"` |
| GPU box (24 GB) | Stage-A SFT, Stage-B DPO, generation | `pip install -e ".[gpu]"` |

Everything crosses the boundary as files under `data/` and `runs/`, so `rsync` is the only
integration you need:

```bash
rsync -av data/processed/ gpu:~/ductus/data/processed/     # push pairs
rsync -av gpu:~/ductus/runs/ runs/                          # pull adapters + generations
```

## Quickstart (laptop, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"
cp .env.example .env          # add ANTHROPIC_API_KEY for backtranslation

# 0. Smoke-test the whole CPU pipeline on the bundled fixture author
make demo

# 0b. Optional: build the larger development fixture (public-domain, ~256k words).
#     See "Development corpus" below -- this is for exercising the pipeline, not a real run.
python scripts/build_dev_corpus.py

# 1. Real corpus: drop .txt/.md/.docx into data/raw/author/, or link Google Docs
wlm ingest local  --in data/raw/author --out data/interim/docs.jsonl
#   ... or:
wlm ingest gdocs --out data/interim/docs.jsonl --typed-only

# 2. Chunk -> scrub -> questions -> splits
wlm chunk       --in data/interim/docs.jsonl --out data/interim/chunks.jsonl
wlm scrub       --in data/interim/chunks.jsonl --out data/interim/scrubbed.jsonl
wlm backtranslate --in data/interim/scrubbed.jsonl --out data/interim/pairs.jsonl -n 2
wlm split       --in data/interim/pairs.jsonl --outdir data/processed

# 3. Fit the eval harness (needs a distractor corpus in data/raw/distractor/)
wlm eval fit-av --author data/processed/train.jsonl --distractor data/raw/distractor
```

## Then on the GPU box

```bash
pip install -e ".[gpu]"
wlm baseline  --config configs/stage_a.yaml --out runs/baseline      # few-shot prompting
wlm train sft --config configs/stage_a.yaml --out runs/stage_a
wlm generate  --adapter runs/stage_a --split blind --out runs/stage_a/gen.jsonl
wlm eval run  --gen runs/stage_a/gen.jsonl --out runs/stage_a/report.html
```

Compare `runs/baseline/report.html` against `runs/stage_a/report.html`. If Stage A wins on the
authorship verifier for *informal* text with fluency intact, proceed to Stage B:

```bash
wlm dpo-pairs --adapter runs/stage_a --split val --out data/processed/dpo.jsonl
wlm train dpo --config configs/stage_b.yaml --out runs/stage_b
```

## Running the full research matrix

The experiment defined in `RESEARCH_BRIEF.md` (corpus-size scaling, adaptation locus, DPO
stage, contamination control, seed variance) has a single resumable driver and a deliverables
assembler — see `docs/RUN_MATRIX.md`:

```bash
python scripts/make_size_sweep.py         # nested {2k..100k} corpus arms
python scripts/run_matrix.py              # everything, resumable; --only rq1 etc. for subsets
python scripts/assemble_results.py        # Tables 1-4 + Figure 1 from run records
```

## The rule that keeps this honest

Every training signal is anchored on the user's **real text**:

- Stage A target = a real passage the person wrote.
- Stage B `chosen` = a real passage the person wrote; only `rejected` is generated.
- The question generator is a **separate hosted API model**, never the model being trained.

That is what prevents model collapse in a single-person, scarce-data setup. See `docs/PLAN.md` §8
for why the parallel-HMM distillation loss was rejected as a *loss* and kept only as an eval.

## Layout

```
configs/          YAML training configs + ablation variants
data/raw/         your writing (gitignored) + distractor corpus
data/interim/     docs -> chunks -> scrubbed -> pairs
data/processed/   train / val / blind splits, dpo pairs
runs/             adapters, generations, eval reports (gitignored)
src/wlm/ingest/   Google Docs + local file ingest, normalization
src/wlm/eval/     stylometry, authorship verification, leakage, fluency, report
src/wlm/train/    Stage-A SFT, Stage-B DPO
docs/PROMPTS.md   phase-by-phase Claude Code prompts
docs/ABLATIONS.md ablation table to fill in (Phase 2 deliverable)
```

## Development corpus

`data/raw/author/` and `data/raw/distractor/` may contain a **development fixture** rather than
anyone's real writing. Check `data/raw/author/README.md` before assuming otherwise.

The fixture is public-domain Project Gutenberg text, rebuilt by `scripts/build_dev_corpus.py`:

| | contents |
|---|---|
| author | **G.K. Chesterton** — 148 documents, ~256k words, five essay collections |
| distractor | **21 other essayists** — 420 windows, Addison through Twain, Chesterton excluded |

It exists because the eval harness only means something at scale. On a 5k-word corpus the
verifier's AUC swung 0.26 depending on whether classes were balanced; on this fixture the two
fits agree to within 0.003 (AUC ≈ 0.92). Use it to validate changes to chunking, splitting,
scrubbing or the AV before pointing anything at real writing.

It is **not** a product, and it proves nothing about any real user. Delete it before a real run —
`python scripts/build_dev_corpus.py --clean`. Do not mix it with a real corpus: the adapter would
learn the average of two voices and the verifier would have no coherent positive class.

## Privacy posture

`data/` and `runs/` are gitignored. Scrubbing runs on every target and writes an audit log to
`data/interim/scrub_log.jsonl`. The only artifact worth shipping is the adapter (a few tens of
MB), and deleting it deletes the personalization.
