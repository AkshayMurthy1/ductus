# Results — assembled from run records

_9 record(s) under `authors/twain/runs`. Regenerate with `python scripts/assemble_results.py` — never edit by hand._

Verifier held-out AUC (the gate on everything below): 0.8385

AV₂ = second instrument (StyleDistance/styledistance, held-out AUC 0.9507); rank agreement with the primary: Spearman 0.5714 over 9 runs. AV₂ reads adapters systematically higher, so compare *orderings* across instruments, not absolute values — details in `results/instruments.md`.

## Table 1 — scaling (RQ1)

| corpus (words) | docs | few-shot AV | adapter AV | AV₂ | Δ | verbatim leak | entity/gen | sem. echo | stylometry ↓ | fluency Δ | train (s) | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13,727 | 9 | 0.5407 | 0.4296 | 0.6889 | -0.1111 | 0.0000 | 2.8590 | -0.0649 | 0.0805 | -0.0009 | 138 | PASS |
| 27,011 **← crossover** | 19 | 0.4667 | 0.6963 | 0.8444 | 0.2296 | 0.0000 | 1.6810 | -0.0491 | 0.0695 | -0.0124 | 273 | PASS |
| 52,854 | 35 | 0.5259 | 0.9037 | 0.9407 | 0.3778 | 0.0000 | 0.9560 | -0.0194 | 0.0688 | 0.0047 | 464 | PASS |
| 104,340 | 68 | 0.4370 | 0.9037 | 0.9704 | 0.4667 | 0.0000 | 0.9560 | -0.0216 | 0.0684 | 0.0052 | 907 | PASS |
Crossover (first arm where the adapter beats its own few-shot baseline with both gates passing): **25k**.

## Table 2 — adaptation locus (RQ2)

SPL = (AV − base floor 0.4667) / (verbatim rate + 0.01); higher = more style per unit of leakage.

| locus | trainable params | AV | AV₂ | verbatim leak | sem. echo | stylometry ↓ | fluency Δ | SPL ↑ | wall-clock (s) |
|---|---|---|---|---|---|---|---|---|---|
| both (reference, full arm) | 30,965,760 | 0.9037 | 0.9704 | 0.0000 | -0.0216 | 0.0684 | 0.0052 | 43.70 | 907 |
| attention only | — | — | — | — | — | — | — | — | — |
| q,k only | — | — | — | — | — | — | — | — | — |
| v,o only | — | — | — | — | — | — | — | — | — |
| MLP only | — | — | — | — | — | — | — | — | — |
| both, MLP r8 | — | — | — | — | — | — | — | — | — |

## Table 3 — stage (RQ3)

_No Stage-B runs recorded yet._

## Table 4 — contamination control

| measure | public author (Chesterton) | private control |
|---|---|---|
| base-model familiarity ratio (ppl author / ppl distractor) | 1.1655 | — |
| base-model attribution floor | 0.4667 | — |
| few-shot AV | 0.4370 | — |
| adapter AV | 0.9037 | — |
| adapter verbatim leak | 0.0000 | — |
_Control column empty: run the identical protocol under a second WLM_ROOT and pass --control-runs._


## Noise floor (seed variance)

_Fewer than 3 seeds per cell so far — run `run_matrix.py --only seeds`. Until this section is populated, report every ranking as provisional._

Any between-condition difference smaller than the ranges above is **within noise** and must be reported as a tie (brief §8).

## Compute & provenance

Hardware: 1× NVIDIA L40S (48 GB, compute capability 8.9). All wall-clock numbers in these tables were measured on this card (`training.train_runtime_s` in each record; eval-only records train nothing).

Total recorded training wall-clock: **30 min** across 4 trained runs. Records span 2026-08-06 → 2026-08-06.

| phase | records | trained | GPU wall-clock | first record | last record |
|---|---|---|---|---|---|
| floor | 1 | 0 | — | 2026-08-06T02:34 | 2026-08-06T02:34 |
| sweep | 8 | 4 | 30 min | 2026-08-06T02:51 | 2026-08-06T08:28 |
