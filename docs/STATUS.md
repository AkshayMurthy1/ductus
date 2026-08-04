# STATUS — the style–content frontier: what we know now

**Updated 2026-08-03.** This is the living record of the project: motive, results with
confidence labels, what the data overturned, and what remains. It supersedes the *assumptions*
of `docs/PLAN.md` (kept as the original design) and answers the questions posed by
`RESEARCH_BRIEF.md`. Numbers regenerate from committed run records via
`python scripts/assemble_results.py` — nothing here is hand-transcribed.

---

## 1. Why this exists

`ductus` turns one person's writing into a small, reversible LoRA adapter that reproduces their
**form** — grammar, rhythm, flow, idiom — while deliberately not learning their **facts**. The
research question that decides whether that product can exist: **does a LoRA adapter acquire an
author's form faster than it memorizes their content, and what governs the gap?** Operationalized
as a frontier — style acquired (authorship-verifier attribution on held-out text) against content
leaked (verbatim n-grams, entity emission, semantic echo) — traced across corpus size (RQ1),
adaptation locus (RQ2), and training stage (RQ3). A usable product needs a regime where style
rises steeply while leakage stays flat; if they rise together, the premise fails.

**Setup:** Qwen2.5-3B-Instruct, DoRA LoRA r16, two-stage curriculum (SFT on
instruction-backtranslated pairs, then on-policy DPO with the author's real text as *chosen*).
Dev author: G. K. Chesterton (149 docs / 256k words, public domain, reproducible) against 25
era-matched essayists; Belloc is the primary hard negative. Verifier: topic-invariant style
embedder + logistic head, fit once (held-out AUC **0.896**, document-grouped, deduped), frozen
across every run. 252 blind-set generations per run; splits by document with near-duplicate
grouping; every run emits a self-contained `report.json`.

## 2. Results

Confidence labels: **established** (exceeds the measured seed noise floor, gates pass),
**within noise** (real data, no ordering claim), **provisional** (awaiting queued runs).

### R1 — Style acquisition is a phase transition, not a slope *(established)*

Attribution vs the frozen verifier, identical recipe per arm, nested corpora:

| arm (train words) | few-shot AV | adapter AV | adapter mean score | verbatim leak | gates |
|---|---|---|---|---|---|
| 2k (3,156) | 0.004 | 0.000 | 0.072 | 0.0% | PASS |
| 5k (5,909) | 0.000 | 0.000 | 0.102 | 0.0% | PASS |
| 10k (10,226) | 0.000 | 0.004 | 0.146 | 0.0% | PASS |
| **25k (27,729)** | 0.000 | **0.718** | 0.634 | 0.0% | PASS |
| 50k (50,020) | 0.000 | 0.825 | 0.693 | 0.0% | PASS |
| 100k (100,411) | 0.000 | 0.798 | 0.685 | 0.0% | PASS |
| full (178,708) | 0.004 | 0.790 | 0.683 | 0.0% | PASS |

The voice snaps in between ~10 and ~55 gradient steps (10k → 25k words): 0.004 → 0.718, then a
plateau at 0.79–0.83. Real held-out Chesterton attributes at **0.714**, so from 25k words on,
the adapter's output is attributed to Chesterton *at or above the rate of his own held-out
prose*. The continuous mean score shows the same cliff, ruling out a threshold artifact. The
product answer to "how much writing does a user need": **≈25k words**; below it, nothing works
(and few-shot prompting never works at any size — see R2).

### R2 — A double dissociation between prompting and tuning *(established)*

The two methods fail in opposite directions on the same corpus and ruler:

- **Few-shot prompting: content without style.** AV ≈ 0 at every corpus size — while up to
  **11.9%** of its generations contain verbatim 12-grams of training text, and every copied run
  hits the 80-token measurement cap (whole exemplar passages parroted, not phrases). Four of
  seven few-shot baselines fail the ≤5% leakage gate outright.
- **Adapters: style without content.** Across **all 13 adapter runs** (7 sizes, 4 loci, 2
  Stage-B): **zero verbatim 12-grams; longest copied run 0 tokens**. Semantic echo — the
  paraphrase-robust check, zero-referenced against the author's own held-out text — is *negative*
  everywhere (−0.03 to −0.10): generations sit topically farther from the training passages
  than real held-out Chesterton does. Entity emission (1.8–3.3/generation) matches the bare
  base model's background rate (2.0) and *falls* as corpus size grows.

The frontier (Figure 1 in `runs/results/dashboard.html`) is therefore **L-shaped**: the adapter
series climbs the style axis without leaving leakage = 0.

### R3 — The voice lives in attention *(established vs Stage A; ordering monotone)*

Locus ablations at the full corpus, one-variable diffs from the reference recipe:

| locus | trainable params | AV | fluency Δ | train time |
|---|---|---|---|---|
| **attention-only (q,k,v,o)** | **7.5M** | **0.857** | **−0.5%** (improved) | **611 s** |
| both, MLP rank 8 | 19.7M | 0.810 | +2.0% | 1,941 s |
| MLP-only (gate,up,down) | 23.4M | 0.794 | +2.5% | 1,534 s |
| both, r16 (reference) | 31.0M | 0.790 | +1.9% | 1,670 s |

Monotone: the less MLP, the better. Attention-only beats the reference by +0.067 (≈5× the
measured seed spread) with 4× fewer parameters, a third of the wall-clock, and *improved*
general-domain fluency — the MLP contributes fluency tax, not voice. This **reverses the
original plan's assumption** ("attention alone captures cadence but underfits word choice; you
need both"). And since no locus leaks anything, RQ2's premise — rank allocation as a *privacy*
dial — dissolves into an *efficiency* dial: there is no leakage to trade against.

### R4 — Stage B (on-policy DPO) moves the frontier and steals nothing *(established)*

| corpus | stage | AV | stylometry ↓ | fluency Δ | verbatim vs train | verbatim vs DPO targets |
|---|---|---|---|---|---|---|
| 10k | A | 0.004 | 0.072 | −0.4% | 0.0% | — |
| 10k | A+B | **0.167** | 0.083 | −0.2% | 0.0% | **0.0%** |
| full | A | 0.790 | 0.064 | +1.9% | 0.0% | — |
| full | A+B | **0.845** | **0.055** | +1.5% | 0.0% | **0.0%** |

DPO adds +0.055 at full (the study's best number) and conjures 0.167 from 0.004 at 10k — it
extracts signal below the SFT phase boundary. Fluency and stylometric distance both *improve*.
The last column is a check standard protocols lack: DPO pushes probability toward real passages
drawn from the **val** split, which routine train-set leakage checks never examine — a designed
blind spot we measured directly. Zero verbatim overlap with DPO's own chosen targets at both
sizes. Mechanism: the KL anchor to Stage A plus one epoch at 5e-6 sharpens the policy without
collapsing onto the targets.

### R5 — Cheapest-vs-best is a tie; the interaction cell is untested *(within noise / queued)*

Attention-only Stage A alone (0.857, 7.5M params, 611 s) vs full-locus Stage A+B (0.845, 31M
params + a DPO stage): Δ = 0.012 — exactly the measured seed spread. **Reported as a tie**, and
the honest headline is "the cheapest configuration matches the most expensive pipeline at ~20×
less compute." The obvious best-of-both cell — **attention-only + DPO** — was never in the
matrix (RQ2 and RQ3 each varied against the reference); it is queued, with the hypothesis it
lands 0.87–0.91 at 7.5M params.

### R6 — The scrubbing diagnostic: entity hygiene is the pipeline's, verbatim hygiene is the LoRA's *(single run; gates pass)*

a14 (identical recipe, **unscrubbed** targets, identical splits, full corpus): AV **0.8532**,
verbatim **0.0%** (longest run 0 tokens), semantic echo −0.035, fluency PASS — but entity
emission **4.27/generation**, roughly double the base model's 2.0 background rate and above
every scrubbed run (1.8–3.3). Reading: when entities are present in training targets, the
adapter *does* absorb and emit them — so the scrubbed runs' entity-flatness is substantially
**the scrubbing working** (an engineering property), while the zero-verbatim result survives
unscrubbed training and is a property of the recipe itself. The paper must state the entity
claim as "pipeline + adapter", not "adapter alone". (One seed; no measured noise floor for
entity emission yet.)

### R7 — A second instrument agrees with the ruler — including about the anomaly *(established)*

Every run re-scored under an independently-trained embedder (StyleDistance, fit on the same
train/distractor protocol, held-out AUC 0.966; `runs/results/instruments.md`): Spearman rank
agreement **0.897** over 29 runs, mean |Δ| 0.046. The run orderings behind R1–R5 are
instrument-independent. Sharper: the 50k adapter scores **0.952** under the second ruler
against real held-out Chesterton's **0.738** — the "adapter above the real author" anomaly
**replicates on an instrument the adapter was never developed against**, which argues against
verifier-gaming and for the hyper-typicality reading. The second ruler systematically rates
adapters *higher* than the primary (Δ up to −0.15); the human blind panel
(`runs/human_panel/`, sheets generated, ratings pending) is the remaining adjudicator.

### Supporting measurements

- **Noise floor** (brief §3): 2k adapter ±0.004 over 3 seeds; full adapter range 0.012 over 2
  seeds (third queued); baselines ±0.002–0.016. Claims above are labeled against these.
- **Contamination control:** base-model perplexity familiarity ratio **0.90** (author vs matched
  period prose) — mild familiarity; yet the untuned base attributes **0.000** and few-shot
  0.004. Whatever the base model remembers of Chesterton, it cannot *perform* him: the gains
  are trained, not recalled. Private-author control arm (Table 4's second column) not yet run.
- **Verifier sanity:** real-text attribution 0.714 constant across runs; injecting fake scrub
  placeholders into distractor text drops attribution to 0.0 (the verifier is not keying on
  placeholders); replacing placeholders with a neutral word raises real-text attribution to
  0.80, so 0.714 slightly *understates* the ceiling.
- **Curiosities to explain in the paper:** (a) 50k-arm attribution 0.825 > the 0.714 real-text
  rate — the adapter may write "hyper-typical" Chesterton, more modal than the man himself;
  (b) trained adapters emit `<PERSON>`-style scrub placeholders at roughly corpus rate — the
  privacy mechanism made visible; tested not to inflate AV; strip at serving time.

## 3. What the data overturned

1. **"You need attention *and* MLP"** (PLAN §3) → reversed; attention-only wins everything (R3).
2. **"Rank allocation is a privacy control"** (RQ2's product framing) → there is no leakage for
   it to control; it is a cost control.
3. **"Few-shot prompting is the safe baseline"** → it is the *leaky* method; the training-based
   method is the clean one. This inverts the intuitive privacy ordering and is the single most
   product-relevant sentence in the study.
4. **"Style should scale smoothly with data"** → it is a phase transition with a cliff at ~25k
   words (~55 gradient steps).

## 4. In flight (queued on the SCC, resumable chain)

| item | purpose | state |
|---|---|---|
| full-arm seed 43 | completes the 3-seed noise floor at the high-AV cell | **done** (report local, AV 0.790; commit with the next record batch) |
| a14 no-scrub diagnostic | separates "LoRA learns form over content" from "scrubbing prevents content" | **done** — see R6 |
| checkpoint trajectory | AV + leakage per checkpoint — the phase transition in *training time* | 1/6 checkpoints evaluated (step-100: AV 0.814, verbatim 0, entities 2.0); resume via `scripts/06_expansion_gpu.sh` |
| attention-only + DPO | the untested interaction cell; likely headline configuration (R5) | queued — `run_matrix.py --only rq2x` |
| 15k/20k arms, per-matrix a17/a18, second model (a13/a16), Twain author root | the expansion matrix (docs/EXPANSION.md) | CPU halves done; GPU workload scripted in `scripts/06_expansion_gpu.sh` |

## 5. Future work (beyond the current matrix)

- **Private control author** (Table 4, brief §3 — "not optional"): identical protocol under a
  second `WLM_ROOT`; the Chesterton-vs-private gap isolates pretraining recall. Records from
  that arm are never committed (see data policy below).
- **The informal register**: the dev corpus is all formal essays; the product's hard case is
  casual first-person prose. Requires a real user corpus; the pipeline stratifies by register
  and the per-register AV breakdown already exists.
- **Cross-model replication**: does the ~25k cliff move with base-model scale? (`a13_1p5b.yaml`
  exists; a Llama-3.2-3B config would test family-dependence.)
- **Serving path**: merge adapter for inference (DoRA unmerged costs ~10× generation latency —
  measured), strip/ban placeholder tokens at decode time, per-user adapter storage (Phase 5).
- **Verifier hardening**: second embedder **done** (R7); human blind panel sheets generated
  (`runs/human_panel/`), ratings pending — the last open piece of the single-ruler threat.

## 6. Threats to validity (say them before a reviewer does)

1. **One author, one register, one base model** so far. The phase-transition *location* (25k)
   is likely Chesterton-specific; the *shape* (cliff + L-frontier) is the claim.
2. **One verifier** → largely closed by R7: a second, independently-trained instrument ranks
   the runs the same way (Spearman 0.897) and replicates the above-real-text anomaly. Residual:
   both instruments are neural embedders; the human panel (sheets ready) is the non-neural
   check.
3. **Chesterton is in pretraining.** Familiarity ratio 0.90 bounds the concern and the floor is
   0.000, but the private-author control is the real answer and hasn't run.
4. **Two seeds at the decisive cell** (third queued). Every ordering claim is labeled against
   the current floor.
5. **The 80-token cap** on verbatim-run measurement means baseline copying is *understated*.

## 7. Reproduction and data policy

Everything a verification needs is in this repository:

- **Committed:** the public-domain corpus (`data/raw/author`, `data/raw/distractor` — provenance
  in `data/README.md`), the API-generated pairs (`data/interim/pairs.jsonl` — ~3k
  non-deterministic API calls, not regenerable), every run's `report.json`/`run_meta.json`/
  `gen.jsonl`, and assembled results (`runs/results/`). Guard: `make check-data` previews
  exactly what would ship and blocks private paths; adapters/checkpoints/weights never commit.
- **Regenerable:** chunks/scrub/splits/sweep are deterministic functions of raw + seed
  (`scripts/00_phase0_cpu.sh`, `make_size_sweep.py`); tables and figures from records
  (`assemble_results.py`); the corpus itself from Gutenberg (`build_dev_corpus.py`).
- **Pipeline docs:** `docs/RUN_MATRIX.md` (how to run everything), `docs/BRIEF_AUDIT.md`
  (brief-requirement → code mapping), `docs/GPU_SETUP.md`.

Full result set: `runs/results/tables.md` (Tables 1–4 + noise floor),
`runs/results/dashboard.html` (Figures), per-run records under `runs/`.
