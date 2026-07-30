# Implementation Plan — "Write Like Me": Personal Style Tuning of a Transformer

> The design document this repo implements. `CLAUDE.md` points here; keep them in sync. When a
> code decision contradicts this file, one of the two is wrong — resolve it explicitly rather
> than letting them drift.

## 0. Project goal

Build a system that takes one person's **data dump** (their own writing) and produces a
lightweight, reversible **style adapter** on top of an open-weight LLM, so the model generates
text that reproduces that person's *grammatical structure, rhythm, and flow* — not their facts.
Success = a blind reader (and an automated authorship verifier) can't easily tell the model's
output from the real person's writing, while general fluency is preserved.

**Design principle:** we are learning *form (style)*, not *content (topics/facts)*. Every decision
below serves that.

## 1. Approach — a two-stage curriculum

Style is distributed across attention **and** MLP layers, so we tune **LoRA/PEFT across both**
(not attention alone). The training signal comes in two stages, chosen after honestly reviewing
the alternatives (a parallel-HMM distillation loss was rejected — see §8):

- **Stage A — SFT via instruction backtranslation.** Turn the user's unpaired prose into
  supervised (question → user's answer) pairs, then LoRA-SFT on them. This establishes the
  behavior reliably and trains the model in its real usage mode. (This is the published
  "Humpback" method, applied to one person.)
- **Stage B — on-policy sharpening via DPO.** Using the Stage-A adapter, generate answers on the
  model's *own* distribution, treat the user's real matching passage as "chosen" and the model's
  generation as "rejected," and run DPO (optionally with an authorship-verifier reward). This
  corrects the residual "generic AI voice" that SFT alone can't fix, while the user's real text
  stays the anchor that prevents model collapse.

Everything keeps the user's genuine writing as ground truth in every step — that is what protects
a scarce-data, single-person setup from both overfitting and collapse.

## 2. Recommended stack (locked-in decisions)

- **Base model:** a **1–3B open-weight instruct model** — **Qwen2.5-3B-Instruct** as the primary
  pick (strong for its size, permissive), with **Qwen2.5-1.5B-Instruct** or
  **Llama-3.2-3B-Instruct** as faster-iteration alternatives. Open-weight is required (LoRA on
  chosen matrices, steering, and DPO all need direct weight access — a closed API can't do this).
- **Precision:** at 1–3B on a 24 GB GPU you can train in **bf16 with LoRA and skip
  quantization** — a 3B model's weights are ~6 GB, so it fits comfortably. Reserve **QLoRA
  (4-bit)** only if you later scale to bigger models or use a 16 GB card, since 4-bit
  quantization slightly costs quality and isn't needed at this size.
- **Compute:** rent a single **24 GB GPU** (RTX 4090 / A10 / L4) on RunPod, Lambda, Modal, or
  Vast. A 1–3B LoRA run is minutes to low hours; no hardware purchase needed.
- **Training framework:** Hugging Face `transformers` + `peft` + `trl`. Strongly consider
  **Unsloth** (≈2× faster, lower memory, drop-in for Qwen/Llama 1–3B) or **Axolotl** (clean
  config-file workflow) for ergonomics.
- **Question-generator (Stage A):** a hosted **API** instruct model. It is never trained, so a
  closed API is ideal and convenient here — keep it separate from the model being trained to
  avoid a self-generation feedback loop.
- **Serving:** `vllm` or `llama.cpp`; adapters hot-swap per user.
- **Eval:** an **authorship-verification** model (embedding-based style classifier / contrastive
  AV model) + classical stylometry (avg sentence length, burstiness, function-word distribution,
  punctuation histograms) + a human blind test.

## 3. Training configuration & customizability (LoRA + MLP)

**Adapt all linear layers — attention *and* MLP.** Best practice since the QLoRA paper is to put
LoRA on *every* linear projection, not just attention. For Qwen/Llama that means:

- Attention: `q_proj, k_proj, v_proj, o_proj` — shapes structure, agreement, and flow.
- MLP / SwiGLU: `gate_proj, up_proj, down_proj` — carries lexical choice, idiom, and phrasing.

In `peft` set `target_modules="all-linear"`. You need both groups: attention alone captures
cadence but underfits word-choice; the MLP is where a person's characteristic phrasings live.

**Mind the MLP's trade-off.** The MLP holds the most parameters, so adapting it adds the most
capacity — and therefore the most *memorization / topic-leak risk* on a tiny personal corpus. Two
controls: (1) keep entity scrubbing strong so the MLP learns function-word and idiom patterns
rather than topic nouns; (2) if you see topic leak in ablations, **lower the MLP rank relative to
attention** instead of dropping MLP entirely (e.g., attention `r=16`, MLP `r=8` via `peft`'s
`rank_pattern`/`alpha_pattern`).

**Rank / alpha.** Start uniform `r=16, alpha=32` (alpha ≈ 2×r). For a 1–3B model on one person's
data, `r=8–16` is plenty; higher ranks mostly buy overfitting. Tune rank as a first-class ablation
knob.

**Modern LoRA variants to enable:**

- **DoRA** — decomposes the update into magnitude + direction; consistently beats plain LoRA,
  especially at low rank and small data. `use_dora=True`. **Recommended default here.**
- **rsLoRA** — stabilizes training if you raise rank; cheap to turn on.
- **LoRA+** — higher learning rate for the B matrices than A; a small near-free quality gain.
- **NEFTune** — noise on input embeddings during SFT; helps *most* in exactly the small-data
  regime you're in. `trl`'s `neftune_noise_alpha` (try 5).

**Layer selection (optional ablation).** Middle-to-upper layers tend to carry more
stylistic/semantic abstraction; restricting LoRA to a subset (`layers_to_transform`) cuts
parameters and overfitting. Start with all layers, then try trimming and measure.

**Embeddings / lm_head (optional).** If the user has very distinctive tokens (emoji, slang,
unusual punctuation), a *small*-rank adapter on `embed_tokens` / `lm_head` can help — but it
overfits easily, so treat it as an ablation, not a default.

**Keep the style adapter modular.** Train it standalone, separate from any future task adapters,
so adapters compose, stay hot-swappable, and are deletable per user.

**Overfitting controls (critical at 1–3B + tiny data):** 1–3 epochs, early-stop on validation, LR
~1e-4–2e-4, LoRA dropout 0.05–0.1, small raw next-token mix, fluency probe every checkpoint.
Precision bf16, gradient checkpointing on, effective batch 16–32 via gradient accumulation.

## 4. Data pipeline (the hard part — invest here)

**Source:** link the user's **Google Docs** (Drive API, read-only) and pull documents they typed
themselves. Where possible use revision history to filter out pasted/imported text so the corpus
is genuinely first-party.

1. **Ingest & normalize** — pull docs, strip signatures/quoted replies/boilerplate/templates, keep
   only author-typed passages.
2. **Segment into "response" chunks** — coherent, response-sized units via document structure,
   discourse-marker heuristics, or semantic chunking. Chunk quality bounds target quality, so
   evaluate it.
3. **PII & entity scrubbing** — remove names, addresses, secrets, and topic-specific named
   entities from the *targets*. Both a privacy control and the main defense against learning
   content instead of style. Log what was removed.
4. **Backtranslate questions (Stage-A pairs)** — for each chunk, have the *separate API* model
   generate a **generic, style-agnostic** prompt that would elicit that chunk, *not* a
   topic-specific one. Optionally add a back-and-forth rewrite/filter pass. Generate multiple
   questions per chunk to augment scarce data.
5. **(Optional) low-weight next-token mix** — a small amount of raw next-token modeling on
   held-out real passages can help cadence; keep the weight low.
6. **Preference pairs (Stage-B)** — reserve held-out questions; at Stage B the Stage-A model
   generates the "rejected" answer and the user's real chunk is the "chosen" answer.
7. **Split** train / validation / **blind test** (never seen during training or dev).

## 5. Phased build

### Phase 0 — Baseline & harness (before any tuning)
- Stand up the base model with **few-shot style prompting** as the baseline to beat.
- Build the **eval harness first**: authorship verifier + stylometry dashboard + a small
  blind-test set. *You cannot improve what you can't measure.*
- **Deliverable:** baseline scores. Expect decent results on formal text, poor on informal — this
  is the target gap.

### Phase 1 — Stage A: LoRA SFT via instruction backtranslation
- Build the Google-Docs → chunk → scrub → backtranslated-question pipeline (§4).
- Train LoRA in bf16 targeting **attention + MLP** (`all-linear`), `r=16, alpha=32`,
  `use_dora=True`, `neftune_noise_alpha=5`, on (question → chunk) pairs plus a small next-token
  mix.
- Watch for **overfitting** (val loss up while train loss down) and **catastrophic forgetting**
  (fluency probe each checkpoint). Early stopping, LR 1e-4–2e-4, 1–3 epochs.
- **Deliverable:** an adapter that beats the prompting baseline on the authorship verifier for
  informal text, with fluency intact.

### Phase 2 — Ablations on Stage A
- Ablate: attention-only vs attention+MLP; LoRA rank (and split attention/MLP ranks); DoRA on/off;
  next-token mix weight; generic vs specific backtranslated questions; with/without entity
  scrubbing; all-layers vs subset.
- **Deliverable:** documented Stage-A recipe + ablation table (the core research result — this
  quantifies what actually moves style vs. what leaks content).

### Phase 3 — Stage B: on-policy DPO sharpening
- With the Stage-A adapter, prompt held-out questions; the model's generation is "rejected," the
  user's real matching chunk is "chosen." Match length/format so DPO learns *voice*, not
  formatting artifacts.
- Short DPO pass (start **plain DPO**; later upgrade to **AV-reward-filtered** pairs). The user's
  real text is the target in every pair, anchoring against collapse.
- **Deliverable:** measurable gain on the hardest (informal) cases and reduced "generic AI voice,"
  fluency preserved.

### Phase 4 — (Optional) Style-vector steering & HMM stylometric check
- **Style vector (training-free):** mean hidden-state direction separating author vs neutral text;
  add at inference as an adjustable "voice strength" dial, composable with the adapter.
- **HMM — non-training use only:** fit a *stylometric* HMM over abstracted features (POS tags,
  sentence-length buckets, function-word classes) and use it as an **eval metric** or a light
  decode-time rerank — **never** as a training loss (see §8).
- **Deliverable:** a voice-intensity control and an extra cadence check.

### Phase 5 — Productize the loop
- One-command pipeline: `connect Google Docs → scrub → pairs → Stage-A SFT → Stage-B DPO → eval
  report → servable model`.
- Per-user adapter storage + hot-swap serving. Privacy: on-device or per-user isolated training;
  adapters are the only artifact and are deletable.

## 6. Evaluation (define "it works" up front)

- **Automated authorship verification** — does an AV model attribute generations to the true
  author? (Primary metric; mirrors the 2509.14543 benchmark methodology.)
- **Stylometry match** — distributional distance on sentence length, burstiness, function words,
  punctuation, POS n-grams. (A stylometric HMM can live here as one such metric.)
- **Content-leakage check** — verify it did *not* memorize private facts (paraphrase test on
  held-out topics).
- **General-fluency regression** — small standard benchmark to confirm no catastrophic forgetting.
- **Human blind test** — the real bar: can people who know the author's writing tell which is
  which?

## 7. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Too little data | Google-Docs first-party corpus; backtranslation manufactures paired supervision; multiple questions per chunk; small model + LoRA; low epochs + early stop |
| Learns topics, not style | Generic backtranslated questions; entity scrubbing of targets; lower MLP rank if leaking; content-leakage eval |
| Catastrophic forgetting | Low rank/LR, DoRA, fluency probe per checkpoint, small next-token mix |
| "Generic AI voice" collapse | Stage-B on-policy DPO; style-vector steering; NEFTune; ablate toward what moves the verifier |
| Model collapse (self-training) | User's real text is always the DPO "chosen"/target; question-generator is a *separate* API model, not the trainee |
| Privacy | Read-only Docs scope, per-user isolation, scrub PII, adapter-only artifact, deletable, ideally on-device |

## 8. Design decision: why not the parallel-HMM distillation loss

A rejected alternative was to fit an HMM on the user's text and add a cross-entropy term pulling
the transformer toward it. Excluded as a *training signal* for four reasons: (1) an HMM's single
discrete hidden state collapses to a class-based bigram model, so it captures only the thinnest,
most local slice of style; (2) on a small personal corpus its biggest deviations are
*content/topic* words, so the nudge is mostly content leakage, not voice; (3) the cross-entropy
term is globally minimized when the transformer *becomes* the HMM — its fixed point is a weaker,
flatter model, so it pulls capability down rather than adding a style offset; (4) mass-covering a
diffuse HMM injects its errors, degrading fluency. The only sound uses of an HMM here are the
*inverse* — tractable decode-time control (à la GeLaTo) or a stylometric eval metric — both kept
in Phase 4/§6, never as a loss.

## 9. First week (concrete)

1. Rent a 24 GB GPU; set up **Qwen2.5-3B-Instruct** in **bf16 LoRA**,
   `target_modules="all-linear"`, `use_dora=True`, `neftune_noise_alpha=5`.
2. Wire up Google Docs read-only ingest on a **test author** (use your own writing); build chunk +
   scrub + backtranslation (questions from the **API** model).
3. Build the eval harness (AV model + stylometry) and record the **prompting baseline**.
4. Run **Stage A** (attention+MLP LoRA on backtranslated pairs); compare to baseline on the blind
   set.
5. If Stage A beats baseline, run a small **plain-DPO Stage B** pass and re-measure; write up what
   moved the needle.

## 10. Reference reading
- Self-Alignment with Instruction Backtranslation ("Humpback") — arXiv 2308.06259 (Stage A)
- Better Alignment with Instruction Back-and-Forth Translation — EMNLP Findings 2024 (pair quality)
- Authorship Style Transfer with Policy Optimization — arXiv 2403.08043 (Stage B rationale)
- On-Policy Distillation — Thinking Machines Lab (2025); survey — arXiv 2604.00626
- AI models collapse when trained on recursively generated data — Nature 2024; The Curse of
  Recursion — arXiv 2305.17493 (why the user's real text must stay the anchor)
- QLoRA — arXiv 2305.14314 (adapt all linear layers)
- DoRA — arXiv 2402.09353
- NEFTune — arXiv 2310.05914
- LoRA+ — arXiv 2402.12354
- Tractable Control for Autoregressive Language Generation (GeLaTo) — arXiv 2304.07438
- Catch Me If You Can? Not Yet — arXiv 2509.14543 (limits of style imitation; benchmark method)
- Steering LLMs with Register Analysis — arXiv 2505.00679; Representation Tuning — arXiv 2409.06927
- Prior art to study: Personal.ai, HyperWrite, Sudowrite, Compose AI, Writer.com brand voice
