# Research brief — the style–content frontier in personal style adapters

**Note to the agent auditing this repository:** this document states *what must be true*, not what
anything is named. The codebase has changed substantially since this brief's author last saw it —
do not assume any file path, module name, CLI flag, or config key mentioned in conversation still
exists. Map each requirement below onto whatever the code actually provides today, and report
(a) what already satisfies it, (b) what exists but is wrong, (c) what is missing.

---

## 1. Primary question

**Does a LoRA adapter acquire an author's form faster than it memorizes their content, and what
governs the gap between the two?**

Operationalized as a **frontier**: style acquired (authorship-verifier attribution on held-out
text) plotted against content memorized (verbatim + entity leakage), traced while varying one
factor at a time. A usable product needs a regime where style rises steeply and leakage stays
flat. If the two rise together, the premise fails — which is itself a publishable result.

## 2. Sub-questions

| | Question | Product relevance |
|---|---|---|
| **RQ1** | How does the frontier move with corpus size, and at what size does adapter training overtake few-shot prompting? | How much writing to require from a user; when to bother training at all |
| **RQ2** | Does adapting attention vs MLP change the style-per-unit-leakage exchange rate? | Whether rank allocation is a *privacy control* with a measurable dial |
| **RQ3** | Does on-policy preference sharpening move the frontier, or only slide along it? | Whether stage two earns its complexity |

## 3. Design

**Author (main):** Chesterton — public domain, large corpus, reproducible.
**Negative class:** era-, genre-, and length-matched contemporaries. **Belloc is the primary hard
negative** (same decade, same essay form, overlapping subject matter, different voice); add
Beerbohm, Lynd, Lucas, Shaw's prefaces, Stevenson.
**Control author:** one private, uncontaminated author with a modest corpus, identical protocol.

**The control is not optional.** Chesterton is in the base model's pretraining data, so part of any
apparent gain is recall, not learning. The Chesterton-vs-private gap is what isolates that, and it
is the first thing a skeptical reader will ask about.

**Run matrix**

- **RQ1** — subsample the author corpus to ≈{2k, 5k, 10k, 25k, 50k, 100k} words. Full protocol at
  each. Subsampling must be by *document*, deterministic, and seed-controlled.
- **RQ2** — at one fixed corpus size: attention-only, MLP-only, both, both-with-reduced-MLP-rank.
- **RQ3** — at 2–3 corpus sizes: with and without the second stage.
- **Variance** — ≥3 seeds at a minimum of two cells. Establish the noise floor *before* ranking
  anything. Report any difference inside it as "within noise."

**Held constant across all runs:** base model, decoding parameters, prompt format (training and
inference must match exactly), evaluation splits, blind set.

## 4. Measurements required per run

1. **Authorship attribution rate** on the blind split — primary style metric.
2. **Content leakage** — verbatim n-gram overlap rate against training text, longest verbatim run,
   proper-noun/entity emission rate.
3. **Stylometric distance** to real author text, with per-feature breakdown (sentence-length
   distribution, burstiness, function words, punctuation, POS n-grams). The breakdown is what makes
   a result *actionable*; the scalar alone is not.
4. **Fluency regression** — perplexity on fixed general-domain text, adapter attached vs detached.
5. **Verifier's own held-out AUC** — a gate, not a result. Below ~0.75 nothing else in the run is
   interpretable. Above ~0.97, suspect a trivial confound rather than celebrate.
6. Trainable parameter count and wall-clock, for the cost axis.

Every run must emit a **machine-readable record** (one JSON per run) containing all of the above
plus the full resolved config. Tables are assembled from those records, never transcribed by hand.

## 5. Validity requirements — silent invalidators

Each of these produces good-looking numbers that mean nothing.

- **Splits by document, never by chunk.** Sibling chunks share topic and phrasing. Must fail loudly
  on document overlap between train / val / blind.
- **Verifier fit on training-split author text only**, deduplicated (multiple prompts per passage
  produce duplicate rows), and split by document group.
- **Distractors matched on era, genre, and chunk length.** A length or era mismatch means the
  verifier learns length or era.
- **Few-shot baseline:** exemplars drawn from the train split only; identical decoding config to
  adapter runs; the prompt must **fail rather than silently truncate** if it exceeds the input
  budget. A truncated baseline hands the adapter an unearned win and corrupts every RQ1 crossover.
- **Preference pairs length-matched** between the real passage and the model's, or the second stage
  learns length instead of voice.
- **Contamination check:** record base-model (untuned) performance on the author, not just the
  few-shot baseline. That is the floor the whole Chesterton arm sits on.

## 6. Repository audit — verify each, report gaps

1. Can the author corpus be **subsampled to a target word count**, by document, deterministically?
   (Required for RQ1; the most likely thing to be missing.)
2. Do splits enforce document-level separation and **hard-fail** on overlap?
3. Is verifier fitting deduplicated and document-grouped, and does it report held-out AUC?
4. Does leakage measurement cover **both** verbatim n-grams and entity emission?
5. Is there a fluency probe comparing adapter attached vs detached on fixed text?
6. Does the few-shot baseline share the decoding path with adapter runs, and refuse to truncate?
7. Is **every** hyperparameter in config rather than hardcoded, so a condition is a one-line diff?
8. Is the training-time prompt format provably identical to the inference-time one? (A mismatch
   here is invisible and poisons everything.)
9. Is seeding threaded end-to-end — data subsampling, splits, training, sampling?
10. Does each run write a self-contained result record including its resolved config?
11. Is there a driver that can execute the run matrix unattended and resume after a failure?
12. Do corpus artifacts and trained adapters stay out of version control?

Report anything ambiguous rather than assuming; a wrong assumption here costs a whole experimental
arm.

## 7. Deliverables

- **Table 1 — scaling (RQ1).** Rows: corpus size. Columns: AV attribution, leakage rate,
  stylometric distance, fluency delta, few-shot baseline AV. Mark the crossover.
- **Table 2 — locus (RQ2).** Rows: adaptation target. Same columns, plus trainable parameters and a
  **style-per-unit-leakage** ratio.
- **Table 3 — stage (RQ3).** Rows: stage one vs stage one + two, at each corpus size tested.
- **Figure 1 — the frontier.** Style (y) against leakage (x), one point per run, series by factor.
  This single plot is the paper's argument and the product's operating-point decision.
- **Table 4 — contamination control.** Chesterton vs private author, matched corpus size.

## 8. Standing rules

- One variable per run. A run that changes two things produces an uninterpretable delta.
- Never report a phase or condition as successful on training loss. The deliverable is always the
  measured triple: style up, leakage flat, fluency intact.
- A run failing the fluency or leakage gate is a failure regardless of its attribution score.
- State the noise floor before any ranking. Ties are "within noise," not ordered.
- Log every bound the pipeline imposes — sampling caps, truncation, dropped rows, filtered pairs.
  Silent truncation reads as full coverage when it is not.
