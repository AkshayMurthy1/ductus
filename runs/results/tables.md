# Results — assembled from run records

_54 record(s) under `/Users/akshaymurthy/Developer/ductus/runs`. Regenerate with `python scripts/assemble_results.py` — never edit by hand._

Verifier held-out AUC (the gate on everything below): 0.8957

AV₂ = second instrument (StyleDistance/styledistance, held-out AUC 0.9659); rank agreement with the primary: Spearman 0.9274 over 48 runs. AV₂ reads adapters systematically higher, so compare *orderings* across instruments, not absolute values — details in `results/instruments.md`.

## Table 1 — scaling (RQ1)

| corpus (words) | docs | few-shot AV | adapter AV | AV₂ | Δ | verbatim leak | entity/gen | sem. echo | stylometry ↓ | fluency Δ | train (s) | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3,156 | 3 | 0.0040 | 0.0000 | 0.0000 | -0.0040 | 0.0000 | 3.1430 | -0.1011 | 0.0623 | -0.0008 | 92 | PASS |
| 5,909 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.2980 | -0.0869 | 0.0591 | 0.0006 | 116 | PASS |
| 10,226 | 8 | 0.0000 | 0.0040 | 0.0119 | 0.0040 | 0.0000 | 2.4760 | -0.1009 | 0.0724 | -0.0035 | 154 | PASS |
| 15,905 **← crossover** | 12 | 0.0000 | 0.5238 | 0.5040 | 0.5238 | 0.0000 | 2.7060 | -0.0948 | 0.0755 | 0.0029 | 214 | PASS |
| 20,257 | 15 | 0.0000 | 0.5675 | 0.5714 | 0.5675 | 0.0000 | 2.7540 | -0.0852 | 0.0700 | 0.0022 | 250 | PASS |
| 27,729 | 19 | 0.0000 | 0.7183 | 0.8016 | 0.7183 | 0.0000 | 2.6940 | -0.0540 | 0.0732 | -0.0128 | 378 | PASS |
| 50,020 | 32 | 0.0000 | 0.8254 | 0.9524 | 0.8254 | 0.0000 | 2.0520 | -0.0402 | 0.0666 | -0.0014 | 647 | PASS |
| 100,411 | 60 | 0.0000 | 0.7976 | 0.9246 | 0.7976 | 0.0000 | 1.8060 | -0.0370 | 0.0733 | 0.0057 | 1232 | PASS |
| 178,708 | 105 | 0.0040 | 0.7897 | 0.9365 | 0.7857 | 0.0000 | 1.8730 | -0.0338 | 0.0635 | 0.0185 | 1670 | PASS |
Crossover (first arm where the adapter beats its own few-shot baseline with both gates passing): **15k**.

## Table 2 — adaptation locus (RQ2)

SPL = (AV − base floor 0.0000) / (verbatim rate + 0.01); higher = more style per unit of leakage.

| locus | trainable params | AV | AV₂ | verbatim leak | sem. echo | stylometry ↓ | fluency Δ | SPL ↑ | wall-clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| both (reference, full arm) | 30,965,760 | 0.7897 | 0.9365 | 0.0000 | -0.0338 | 0.0635 | 0.0185 | 78.97 | 1670 |
| attention only | 7,538,688 | 0.8571 | 0.9325 | 0.0000 | -0.0397 | 0.0715 | -0.0053 | 85.71 | 611 |
| q,k only | 3,769,344 | 0.8373 | 0.9325 | 0.0000 | -0.0480 | 0.0734 | 0.0175 | 83.73 | 504 |
| v,o only | 3,769,344 | 0.7817 | 0.9365 | 0.0000 | -0.0290 | 0.0661 | -0.0089 | 78.17 | 496 |
| MLP only | 23,427,072 | 0.7937 | 0.9484 | 0.0000 | -0.0323 | 0.0666 | 0.0254 | 79.37 | 1534 |
| both, MLP r8 | 19,685,376 | 0.8095 | 0.9286 | 0.0000 | -0.0256 | 0.0706 | 0.0201 | 80.95 | 1941 |

## Table 3 — stage (RQ3)

Train (s) is the listed stage's own wall-clock; a Stage-B time excludes the Stage-A run it starts from.

| corpus | stage | AV | AV₂ | verbatim leak | sem. echo | stylometry ↓ | fluency Δ | train (s) | gates |
|---|---|---|---|---|---|---|---|---|---|
| 10,226 | A | 0.0040 | 0.0119 | 0.0000 | -0.1009 | 0.0724 | -0.0035 | 154 | PASS |
| 10,226 | A+B | 0.1667 | 0.1984 | 0.0000 | -0.0891 | 0.0834 | -0.0021 | 380 | PASS |
| 178,708 | A | 0.7897 | 0.9365 | 0.0000 | -0.0338 | 0.0635 | 0.0185 | 1670 | PASS |
| 178,708 | A+B | 0.8452 | 0.9206 | 0.0000 | -0.0231 | 0.0549 | 0.0154 | 388 | PASS |
| full | attn-only A + B (a15) | 0.8016 | 0.9603 | 0.0000 | -0.0281 | 0.0602 | -0.0053 | 113 | PASS |

The a15 interaction cell reads against attention-only Stage A (0.857): DPO's gain does **not** stack on the attention-only adapter.

## Table 4 — contamination control

| measure | public author (Chesterton) | private control |
|---|---|---|
| base-model familiarity ratio (ppl author / ppl distractor) | 0.8996 | — |
| base-model attribution floor | 0.0000 | — |
| few-shot AV | 0.0040 | — |
| adapter AV | 0.7897 | — |
| adapter verbatim leak | 0.0000 | — |
_Control column empty: run the identical protocol under a second WLM_ROOT and pass --control-runs._

## Table 5 — cross-model replication

Same corpus, splits, verifier and recipe; only the base model changes. Each model gets its own few-shot floor and contamination probe.

| model | arm | few-shot AV | few-shot leak | adapter AV | AV₂ | adapter leak | fluency Δ | train (s) | contamination ratio |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (reference) | 10k | 0.0000 | 0.1032 | 0.0040 | 0.0119 | 0.0000 | -0.0035 | 154 | 0.8996 |
| Qwen2.5-3B (reference) | 25k | 0.0000 | 0.0675 | 0.7183 | 0.8016 | 0.0000 | -0.0128 | 378 | 0.8996 |
| Qwen2.5-3B (reference) | full | 0.0040 | 0.0357 | 0.7897 | 0.9365 | 0.0000 | 0.0185 | 1670 | 0.8996 |
| a13_1p5b | 10k | 0.0000 | 0.0159 | 0.0317 | 0.0516 | 0.0000 | 0.0103 | 83 | 0.8634 |
| a13_1p5b | 25k | 0.0000 | 0.0040 | 0.7857 | 0.7619 | 0.0000 | 0.0100 | 200 | 0.8634 |
| a13_1p5b | full | 0.0000 | 0.0079 | 0.8849 | 0.9683 | 0.0000 | 0.0464 | 888 | 0.8634 |
| a16_llama3b | 10k | 0.0000 | 0.0595 | 0.0079 | 0.0238 | 0.0000 | -0.0047 | 132 | 0.8972 |
| a16_llama3b | 25k | 0.0000 | 0.0198 | 0.2103 | 0.5198 | 0.0000 | 0.0233 | 326 | 0.8972 |
| a16_llama3b | full | 0.0000 | 0.0437 | 0.7659 | 0.9921 | 0.0000 | 0.0612 | 1447 | 0.8972 |

## Noise floor (seed variance)

- **2k / stage_a** (3 seeds): AV mean 0.0013, sd 0.0023, range 0.0040
- **2k / baseline** (3 seeds): AV mean 0.0066, sd 0.0083, range 0.0159
- **full / stage_a** (3 seeds): AV mean 0.7857, sd 0.0069, range 0.0119
- **full / baseline** (3 seeds): AV mean 0.0053, sd 0.0023, range 0.0039

Any between-condition difference smaller than the ranges above is **within noise** and must be reported as a tie (brief §8).

## Compute & provenance

Hardware: 1× NVIDIA L40S (48 GB, compute capability 8.9). All wall-clock numbers in these tables were measured on this card (`training.train_runtime_s` in each record; eval-only records train nothing).

Total recorded training wall-clock: **5.3 h** across 28 trained runs. Records span 2026-08-01 → 2026-08-05.

| phase | records | trained | GPU wall-clock | first record | last record |
|---|---|---|---|---|---|
| floor | 1 | 0 | — | 2026-08-01T01:45 | 2026-08-01T01:45 |
| sweep | 28 | 15 | 2.5 h | 2026-08-01T02:04 | 2026-08-05T20:35 |
| matrix/rq2 | 5 | 5 | 1.4 h | 2026-08-01T17:52 | 2026-08-04T05:52 |
| matrix/a14_no_scrubbing | 1 | 1 | 28 min | 2026-08-03T13:59 | 2026-08-03T13:59 |
| trajectory | 6 | 0 | — | 2026-08-03T16:10 | 2026-08-04T00:48 |
| matrix/rq2x | 1 | 1 | 2 min | 2026-08-04T04:47 | 2026-08-04T04:47 |
| matrix/models | 12 | 6 | 51 min | 2026-08-04T06:13 | 2026-08-04T15:51 |
