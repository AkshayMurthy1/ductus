# Phase-2 ablation table

The core research result. Fill one row per config in `configs/ablations/`. Every row is a
one-variable diff from the Stage-A recipe, so a delta is attributable.

> The RQ2 subset of this grid (a01 / a02 / a05 vs the reference) is driven automatically by
> `scripts/run_matrix.py --only rq2` and lands, with the style-per-unit-leakage ratio, in
> `runs/results/tables.md` via `scripts/assemble_results.py`. Pair it with
> `scripts/adapter_anatomy.py` on each adapter to see *where* the update mass moved, not just
> what it scored.

## How to read this table

- **AV informal** is the headline. Formal prose is the easy case; the informal column is where
  style tuning either earns its keep or doesn't.
- **Leak** is the veto. A row with the best AV score and a leakage rate above ~5% did not learn
  style, it learned content. Report it as a failure.
- **Fluency Δ** is the second veto. Budget is +15% perplexity vs adapter-off.
- **"within noise"** is a legitimate entry. With one person's corpus, most deltas here will be
  smaller than seed variance. Write `≈` rather than ranking indistinguishable runs, and if a
  result matters, re-run it with 2–3 seeds before believing it.

## Results

| # | Config | Variable changed | Trainable params | AV overall | AV informal | Stylometry dist ↓ | Leak rate ↓ | Fluency Δ | Wall clock | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| — | few-shot prompting | (Phase-0 baseline) | 0 | | | | | 0 | | the number to beat |
| — | `stage_a.yaml` | (reference recipe) | | | | | | | | all-linear r16 DoRA NEFTune5 |
| a01 | `a01_attention_only` | q,k,v,o only | | | | | | | | expect: cadence yes, word choice no |
| a02 | `a02_mlp_only` | gate,up,down only | | | | | | | | expect: idiom yes, rhythm no |
| a03 | `a03_rank8` | r=8 | | | | | | | | often matches r16 with less overfit |
| a04 | `a04_rank32` | r=32 + rsLoRA | | | | | | | | watch leak, not just val loss |
| a05 | `a05_split_rank_low_mlp` | attn r16 / MLP r8 | | | | | | | | the prescribed fix if a14 shows leak |
| a06 | `a06_no_dora` | plain LoRA | | | | | | | | isolates DoRA's contribution |
| a07 | `a07_no_neftune` | NEFTune off | | | | | | | | should cost something at this data size |
| a08 | `a08_no_raw_mix` | raw mix 0.0 | | | | | | | | does the raw mix buy cadence or leak? |
| a09 | `a09_heavy_raw_mix` | raw mix 0.4 | | | | | | | | expect better cadence AND worse leak |
| a10 | `a10_upper_layers_only` | layers 12–35 | | | | | | | | fewer params, less overfit |
| a11 | `a11_loraplus` | LoRA+ ratio 4.0 | | | | | | | | near-free if it helps |
| a12 | `a12_embeddings` | embed/lm_head saved | | | | | | | | only for distinctive tokens; overfits |
| a13 | `a13_1p5b` | Qwen2.5-1.5B | | | | | | | | iteration speed; not a headline number |
| a14 | `a14_no_scrubbing` | entity scrubbing off | | | | | | | | **DIAGNOSTIC — do not ship this adapter** |

## The three questions this table has to answer

1. **Does the MLP contribute what the plan claims?** Compare a01 / a02 / reference on
   *per-feature* stylometry gaps, not just the overall score. The plan's claim is specific:
   attention carries structure and rhythm, the MLP carries lexical choice and idiom. If a01 has a
   small `sent_len_hist` gap but a large `function_words` gap, that's confirmation. If both
   ablations look the same, the claim is wrong at this scale and that's a more interesting result.

2. **How much of the AV gain is content leak?** a14 vs reference. This is the number that decides
   whether the project did what it says. If a14 scores much higher on AV, scrubbing is carrying a
   lot of weight and the reference recipe's honest gain is the smaller one.

3. **Where does rank stop buying anything?** a03 / reference / a04. Expect a plateau well before
   r=32 on one person's corpus, with leakage rising after the plateau.

## Findings

_Write conclusions here as prose, not just the table. What actually moved style, what only leaked
content, and what recipe you're locking in for Stage B._
