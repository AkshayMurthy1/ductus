# Running the RESEARCH_BRIEF matrix

How to execute every arm the brief requires, resume after failures, and assemble the
deliverables (Tables 1–4, Figure 1) without ever transcribing a number by hand. Read
`docs/BRIEF_AUDIT.md` for how each brief requirement maps onto the code.

## 0. One-time setup (CPU, then rsync to the GPU box)

```bash
scripts/00_phase0_cpu.sh                 # ingest -> chunk -> scrub -> pairs -> split -> fit AV
python scripts/make_size_sweep.py        # nested {2k,5k,10k,25k,50k,100k,full} arms (RQ1)
```

Two things must be true before spending GPU money, and both are checked for you:

- the verifier AUC is between ~0.75 and ~0.97 (outside that band, downstream numbers are either
  noise or a confound — the fit warns in both directions);
- the split summary shows zero document overlap.

The verifier is fitted **once, on the full train set**, and never refit per arm: it is the
ruler, and the ruler must not move between measurements.

## 1. The driver

```bash
python scripts/run_matrix.py                  # floor + rq1 + rq2 + rq3 + seeds
python scripts/run_matrix.py --only rq1       # any subset
python scripts/run_matrix.py --dry-run        # print the plan without running
```

| section | what it runs | brief item |
|---|---|---|
| `floor` | base model, no adapter, no exemplars, on blind + the perplexity contamination probe | §5 contamination check |
| `rq1` | per size arm: few-shot baseline + Stage-A adapter, same blind set, same verifier | §3 RQ1 |
| `rq2` | attention-only / MLP-only / low-MLP-rank at one fixed arm (`--rq2-arm`, default `full`; the "both" cell is that arm's rq1 run) | §3 RQ2 |
| `rq3` | Stage-B DPO on top of Stage A at `--rq3-arms` (default `10k full`) | §3 RQ3 |
| `seeds` | seeds {29, 43} at the smallest and largest arms (+ the main run's 17 = 3 seeds/cell) | §3 variance |

Every unit is done when its `report.json` exists, so re-running the same command after a crash,
OOM, or preemption resumes exactly where it stopped. Failures are recorded in
`runs/matrix_state.json` and don't block independent units; units that need a failed
prerequisite (rq3 on a failed stage_a) are skipped with a reason.

**Control author (Table 4).** Every path honors `WLM_ROOT`. Put the private author's corpus in
`<control-root>/data/raw/{author,distractor}`, then run the identical protocol:

```bash
WLM_ROOT=~/control-author scripts/00_phase0_cpu.sh
WLM_ROOT=~/control-author python scripts/make_size_sweep.py --sizes <matched size>
WLM_ROOT=~/control-author python scripts/run_matrix.py --only floor rq1
```

Nothing else changes — which is the point: identical protocol, different corpus. The
Chesterton-vs-private gap is what separates learning from pretraining recall.

## 2. Assembling the deliverables

```bash
python scripts/assemble_results.py \
    [--control-runs ~/control-author/runs]     # adds Table 4's second column
```

Writes `runs/results/tables.md` (Tables 1–4 + the seed-variance noise floor) and
`runs/results/frontier.html` (Figure 1: AV attribution vs verbatim leakage, one point per run,
series by factor, with a table view). Table 2 includes the brief's **style-per-unit-leakage**
ratio, defined as `SPL = (AV − base floor) / (verbatim rate + 0.01)`.

The assembler also prints every distinct verifier AUC it saw — if there is more than one, the
ruler moved between runs and the comparison is suspect.

## 3. Novel measurements (beyond the brief)

These were added because each one either sharpens the primary question or removes a way the
existing numbers could lie. Each is optional and none changes the training recipe.

### 3a. The training-time frontier — the primary question as a curve

The brief's headline question is about *rates* — "does a LoRA acquire form **faster** than it
memorizes content" — but endpoint runs only measure where training stopped. A single Stage-A
run already checkpoints every `sft.save_steps` steps; evaluating each checkpoint traces
style-vs-leakage across training time at zero extra training cost:

```bash
wlm train sft --config configs/trajectory.yaml --out runs/trajectory   # keeps all checkpoints
scripts/05_checkpoint_trajectory.sh runs/trajectory
python scripts/assemble_results.py        # picks the trace up as a second panel in Figure 1
```

If style attribution rises steps before verbatim/entity leakage does, the frontier's usable
regime is an *early-stopping band*, which is a much stronger (and more product-relevant) result
than a corpus-size crossover alone. If they rise together at every corpus size, the premise
fails — publishably. Caveats: each checkpoint's fluency verdict is read from the training-time
log at the nearest step, and all checkpoints share one sampling seed.

### 3b. Semantic echo — leakage that survives a paraphrase

Verbatim 12-grams miss content the model reproduces in different words. `semantic_echo`
(in `wlm.eval.leakage`, reported automatically by `wlm eval run`) embeds generations with a
*content* encoder (MiniLM — deliberately the opposite of the AV's topic-invariant style
embedder) and measures how much closer generations sit to the training passages than the
author's own held-out text does. ≈0 is healthy; large positive means paraphrase-level
memorization. It is advisory (not yet a gate) until calibrated across a few runs; the verbatim
gate still vetoes alone.

### 3c. Adapter anatomy — where the voice lives

```bash
python scripts/adapter_anatomy.py runs/sweep/full/stage_a --json runs/results/anatomy.json
```

Computes ‖ΔW‖_F for every adapted matrix (without materializing ΔW) and shows how update mass
distributes across depth and attention-vs-MLP. Read next to the per-feature stylometry gaps of
the RQ2 ablations: if attention-only runs close the rhythm gap while the reference run's update
mass concentrates in upper-layer MLPs as the idiom gap closes, that is mechanistic support for
the plan's "attention carries cadence, MLP carries lexicon" claim — a figure the ablation
deltas alone can't give you. Also worth running on a14 (no-scrub) vs the reference: if
scrubbing changes *where* update mass lands, that localizes the content-memorization circuit.

### 3d. Contamination familiarity ratio

```bash
wlm eval contamination --author data/processed/blind.jsonl --out runs/contamination.json
```

Untuned-model perplexity on held-out author text divided by perplexity on era-matched
distractor text. Distractors control for "the model knows Edwardian prose"; a ratio well below
1.0 means it knows *this author*, and the public arm's gains must be discounted against the
private control accordingly. Run by the driver's `floor` section; lands in Table 4.

## 4. Reporting rules (unchanged, restated)

- A run failing the fluency or leakage gate is a failure regardless of its AV score.
- No ranking before the noise floor section of `tables.md` is populated; differences inside it
  are ties.
- The crossover claim in Table 1 requires the winning arm to pass both gates.
