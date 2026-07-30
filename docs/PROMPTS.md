# How to prompt Claude Code through this project

The scaffold already exists, so your prompts are no longer "build me X." They're "run this phase,
here's what counts as done." That shift is most of the answer to your question.

## The four habits that matter more than any individual prompt

**1. Put the plan and the invariants on disk, not in the chat.** `CLAUDE.md` is auto-loaded into
every Claude Code session in this repo. `docs/PLAN.md` is one `@`-mention away. Because both are
there, a prompt like "run Phase 2" already carries the whole design — you never re-explain that
splits are by document or that an HMM is not a loss. If you notice yourself repeating a
correction twice, that's a signal to write it into `CLAUDE.md` rather than say it a third time.

**2. Give a success criterion, not a task list.** "Train Stage A" invites a green checkmark on a
finished training run. "Train Stage A and beat the baseline's informal-register AV attribution
rate, with the fluency probe passing" invites the model to actually check, and to tell you when
it didn't. Every phase in `PLAN.md` §5 already has a deliverable defined this way — quote it.

**3. Plan mode for anything touching the training loop; just-do-it for pipeline glue.** Press
`Shift+Tab` twice to enter plan mode when the change is architectural (a new loss, a new stage, a
change to how pairs are built). Read the plan, argue with it, then approve. For "add a CLI flag"
or "fix this traceback," skip planning.

**4. Verify with a subagent, not with yourself.** After a phase, `use a subagent to independently
check whether runs/stage_a/report.json actually supports the claim that Stage A beat baseline`.
A fresh context is much more willing to say no than the one that just did the work.

## Two things worth setting up once

**A `/phase` command.** Put this in `.claude/commands/phase.md`:

```markdown
Run phase $1 of docs/PLAN.md in this repo.

Before doing anything: read the phase's entry in PLAN.md §5 and state its deliverable back to me
in one line. Then check runs/ for the previous phase's report.json so you know what you're
comparing against.

Rules: one variable per run. Config changes go in configs/, never hardcoded. Do not report the
phase complete unless the deliverable metric actually moved in the right direction AND the
fluency and leakage checks pass. If it didn't, say so and tell me the three most likely causes
ranked by how cheap they are to test.
```

Then each session is just `/phase 2`.

**Hooks so you don't have to ask for tests.** In `.claude/settings.json`, a `PostToolUse` hook on
`Edit|Write` matching `src/wlm/**` that runs `pytest -q && ruff check src` turns "did you run the
tests" into something you never say again.

## Phase-by-phase prompts

Copy these more or less verbatim. They're written to be pasted at the start of a session.

---

### Phase 0 — baseline and harness

The harness is already built. What's left is fitting it to *your* corpus, which is the part that
can silently go wrong.

```
Read CLAUDE.md and docs/PLAN.md §5 Phase 0.

My corpus is in data/raw/author/ (~N words, registers: informal + formal) and distractors are in
data/raw/distractor/.

Run the CPU pipeline end to end: ingest -> chunk -> scrub -> backtranslate -> split, then fit the
authorship verifier. Then stop and give me a go/no-go on the DATA, not the code:

- chunk health: what fraction start mid-thought, and is the word distribution sane?
- scrub summary: what got removed, and did any chunks get dropped for placeholder density?
- split summary: does each split have both registers, and is document overlap empty?
- verifier held-out AUC: is it above 0.75, and if it's suspiciously high (>0.97), is that because
  my distractors are register-mismatched rather than because the verifier is good?

Do not proceed to the GPU. I want to see these numbers first.
```

That last paragraph is the important one. A verifier with an AUC of 0.99 because your distractors
are Victorian novels will happily tell you Stage A worked when it didn't, and this is the one
point in the project where that error is cheap to catch.

Then, on the GPU box:

```
Record the Phase-0 prompting baseline: wlm baseline, then wlm eval run with --run-name baseline.
Report the AV attribution rate broken down by register, the stylometry distance, and the top 5
stylometry gaps. Per PLAN.md I expect decent numbers on formal text and poor ones on informal —
tell me whether that pattern actually holds, because the informal gap is the thing this whole
project is trying to close. If it doesn't hold, that changes what Phase 1 should target.
```

---

### Phase 1 — Stage A

```
Run Phase 1: Stage-A LoRA SFT with configs/stage_a.yaml.

Deliverable per PLAN.md: an adapter that beats the prompting baseline on the authorship verifier
for INFORMAL text, with fluency intact.

While it trains, watch two things and tell me if either fires:
- val loss rising while train loss falls (overfitting — the corpus is small, this is expected at
  some point; I want to know the step)
- the fluency probe in runs/stage_a/fluency_log.jsonl exceeding the 15% regression budget

When it's done: generate on the blind split, run the eval with --baseline pointed at the baseline
report.json, and give me the delta table. Then tell me plainly whether the deliverable was met.
If the AV rate went up but the leakage check also went up, say that first — it's the most likely
way this result is fake.
```

---

### Phase 2 — ablations (the actual research result)

This is where prompt structure earns the most, because it's 14 runs and you don't want to
supervise each one.

```
Run the Phase-2 ablation grid. configs/ablations/ has 14 configs; each is a one-line diff from
stage_a.yaml.

For each: train, generate on blind, eval, and record one row in docs/ABLATIONS.md — AV
attribution (overall and informal), stylometry distance, verbatim leakage rate, fluency
regression, wall-clock, and trainable param count.

Order them cheapest-first: a13 (1.5B) and a03 (r8) before a04 (r32) and a12 (embeddings).

Two things I care about more than the winner:
1. a01 vs a02 vs the baseline recipe — does the MLP actually contribute what PLAN.md §3 claims
   (word choice) while attention contributes cadence? Use the per-feature stylometry gaps to
   answer this, not just the overall number. This is the interesting result.
2. a14 (scrubbing off) is a DIAGNOSTIC — do not ship its adapter. Its job is to quantify how much
   of my AV gain is content leak rather than style. Report that number prominently.

Run them sequentially and check in after every 3 with a partial table. If two configs come out
within noise of each other, say "within noise" rather than ranking them.
```

The "within noise" instruction is worth including every time. With one person's corpus, most
ablation deltas will be smaller than seed variance, and the default failure mode is a confident
ranking of indistinguishable runs.

---

### Phase 3 — Stage B

```
Run Phase 3: Stage-B on-policy DPO on top of the winning Stage-A adapter.

Build the pairs first (wlm dpo-pairs) and show me the stats BEFORE training. Specifically: how
many candidates were dropped for length mismatch, and how many survived? If fewer than ~40
survived, stop and tell me — DPO on that few is noise, and I'd rather generate more held-out
questions than train on it.

Then plain sigmoid DPO only. Do not turn on av_reward_filter or change loss_type in the same run;
I want to know what plain DPO buys before stacking anything.

Deliverable: measurable gain on the informal cases and reduced generic-AI-voice, fluency
preserved. For "reduced generic voice," use the specific stylometry features that moved — hedge
rate, burstiness, sentence-opener distribution — not a vibe.

If reward accuracy comes out near 1.0, treat that as a bug signal and inspect the pairs before
believing the gain.
```

---

### Phase 4 — style vectors (optional)

```
Implement Phase 4's style vector: mean hidden-state difference between author and neutral text,
added at inference as an adjustable voice-strength dial, composable with the adapter.

Put it in src/wlm/steering.py with a `strength` parameter, and sweep strength over
[0, 0.5, 1, 1.5, 2] on the blind split, reporting AV attribution and fluency at each. I expect a
knee where voice keeps rising and fluency starts falling — find it.

Reminder from PLAN.md §8: if you also build the stylometric HMM, it is an EVAL METRIC or a
decode-time rerank only. Never a training loss. If you think that's wrong, argue it in chat
before writing code.
```

---

## Anti-prompts — things not to say

| Don't | Why |
|---|---|
| "make the style match better" | No measurable target, so any change looks like progress. Name the metric. |
| "try a bunch of hyperparameters" | Produces a multi-variable sweep you can't interpret. One variable per run. |
| "the loss is going down, keep going" | Loss is not the deliverable; the AV/leakage/fluency triple is. |
| "just make the tests pass" | Invites weakening the test. Say "fix the cause; the test encodes an invariant." |
| "add an HMM loss to pull it toward my style" | Explicitly rejected in §8 for four reasons. |

## When a session goes sideways

- `/clear` between phases. Context from Phase 1's debugging actively hurts Phase 2's judgment.
- `Esc` twice to rewind to an earlier message and re-prompt, rather than arguing forward.
- If the model reports success and you're unsure, the highest-value follow-up is one sentence:
  **"what would have to be true for this result to be fake?"**
