# Results — assembled from run records

_35 record(s) under `/Users/akshaymurthy/Developer/ductus/runs`. Regenerate with `python scripts/assemble_results.py` — never edit by hand._

Verifier held-out AUC (the gate on everything below): 0.8957

## Table 1 — scaling (RQ1)

| corpus (words) | docs | few-shot AV | adapter AV | Δ | verbatim leak | entity/gen | sem. echo | stylometry ↓ | fluency Δ | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| 3,156 | 3 | 0.0040 | 0.0000 | -0.0040 | 0.0000 | 3.1430 | -0.1011 | 0.0623 | -0.0008 | PASS |
| 5,909 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.2980 | -0.0869 | 0.0591 | 0.0006 | PASS |
| 10,226 **← crossover** | 8 | 0.0000 | 0.0040 | 0.0040 | 0.0000 | 2.4760 | -0.1009 | 0.0724 | -0.0035 | PASS |
| 15,905 | 12 | — | — | — | — | — | — | — | — | — |
| 20,257 | 15 | — | — | — | — | — | — | — | — | — |
| 27,729 | 19 | 0.0000 | 0.7183 | 0.7183 | 0.0000 | 2.6940 | -0.0540 | 0.0732 | -0.0128 | PASS |
| 50,020 | 32 | 0.0000 | 0.8254 | 0.8254 | 0.0000 | 2.0520 | -0.0402 | 0.0666 | -0.0014 | PASS |
| 100,411 | 60 | 0.0000 | 0.7976 | 0.7976 | 0.0000 | 1.8060 | -0.0370 | 0.0733 | 0.0057 | PASS |
| 178,708 | 105 | 0.0040 | 0.7897 | 0.7857 | 0.0000 | 1.8730 | -0.0338 | 0.0635 | 0.0185 | PASS |
Crossover (first arm where the adapter beats its own few-shot baseline with both gates passing): **10k**.

## Table 2 — adaptation locus (RQ2)

SPL = (AV − base floor 0.0000) / (verbatim rate + 0.01); higher = more style per unit of leakage.

| locus | trainable params | AV | verbatim leak | sem. echo | stylometry ↓ | fluency Δ | SPL ↑ | wall-clock (s) |
|---|---|---|---|---|---|---|---|---|
| both (reference, full arm) | 30,965,760 | 0.7897 | 0.0000 | -0.0338 | 0.0635 | 0.0185 | 78.97 | 1670 |
| attention only | 7,538,688 | 0.8571 | 0.0000 | -0.0397 | 0.0715 | -0.0053 | 85.71 | 611 |
| MLP only | 23,427,072 | 0.7937 | 0.0000 | -0.0323 | 0.0666 | 0.0254 | 79.37 | 1534 |
| both, MLP r8 | 19,685,376 | 0.8095 | 0.0000 | -0.0256 | 0.0706 | 0.0201 | 80.95 | 1941 |

## Table 3 — stage (RQ3)

| corpus | stage | AV | verbatim leak | sem. echo | stylometry ↓ | fluency Δ | gates |
|---|---|---|---|---|---|---|---|
| 10,226 | A | 0.0040 | 0.0000 | -0.1009 | 0.0724 | -0.0035 | PASS |
| 10,226 | A+B | 0.1667 | 0.0000 | -0.0891 | 0.0834 | -0.0021 | PASS |
| 178,708 | A | 0.7897 | 0.0000 | -0.0338 | 0.0635 | 0.0185 | PASS |
| 178,708 | A+B | 0.8452 | 0.0000 | -0.0231 | 0.0549 | 0.0154 | PASS |

## Table 4 — contamination control

| measure | public author (Chesterton) | private control |
|---|---|---|
| base-model familiarity ratio (ppl author / ppl distractor) | 0.8996 | — |
| base-model attribution floor | 0.0000 | — |
| few-shot AV | 0.0040 | — |
| adapter AV | 0.7897 | — |
| adapter verbatim leak | 0.0000 | — |
_Control column empty: run the identical protocol under a second WLM_ROOT and pass --control-runs._

## Noise floor (seed variance)

- **2k / stage_a** (3 seeds): AV mean 0.0013, sd 0.0023, range 0.0040
- **2k / baseline** (3 seeds): AV mean 0.0066, sd 0.0083, range 0.0159
- **full / stage_a** (3 seeds): AV mean 0.7857, sd 0.0069, range 0.0119
- **full / baseline** (3 seeds): AV mean 0.0053, sd 0.0023, range 0.0039

Any between-condition difference smaller than the ranges above is **within noise** and must be reported as a tie (brief §8).
