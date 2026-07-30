# writelikeme

Personal style tuning of an open-weight transformer. Learns **form** (grammar, rhythm, flow,
idiom) from one person's own writing and emits a small, reversible, deletable LoRA adapter.
Does **not** try to learn their facts.

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
rsync -av data/processed/ gpu:~/writelikeme/data/processed/     # push pairs
rsync -av gpu:~/writelikeme/runs/ runs/                          # pull adapters + generations
```

## Quickstart (laptop, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"
cp .env.example .env          # add ANTHROPIC_API_KEY for backtranslation

# 0. Smoke-test the whole CPU pipeline on the bundled fixture author
make demo

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

## Privacy posture

`data/` and `runs/` are gitignored. Scrubbing runs on every target and writes an audit log to
`data/interim/scrub_log.jsonl`. The only artifact worth shipping is the adapter (a few tens of
MB), and deleting it deletes the personalization.
