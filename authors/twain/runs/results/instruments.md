# Dual-instrument check

Second instrument: `StyleDistance/styledistance` (held-out AUC 0.9507)

Rank agreement over 9 runs: Spearman **0.5714** (disagreement 0.2143, mean |Δ| 0.0905)

| run | primary AV | second AV | Δ |
|---|---|---|---|
| floor | 0.4667 | 0.6074 | -0.1407 |
| sweep/10k/baseline | 0.5407 | 0.5481 | -0.0074 |
| sweep/10k/stage_a | 0.4296 | 0.6889 | -0.2593 |
| sweep/25k/baseline | 0.4667 | 0.5037 | -0.0370 |
| sweep/25k/stage_a | 0.6963 | 0.8444 | -0.1481 |
| sweep/50k/baseline | 0.5259 | 0.4963 | +0.0296 |
| sweep/50k/stage_a | 0.9037 | 0.9407 | -0.0370 |
| sweep/full/baseline | 0.4370 | 0.5259 | -0.0889 |
| sweep/full/stage_a | 0.9037 | 0.9704 | -0.0667 |

Real held-out author text under the second ruler: attribution **0.9185** (mean score 0.7015). Compare adapter rows against this the same way the primary is compared against 0.714.

Runs the two instruments disagree on (hand-inspect these first):
- `sweep/10k/stage_a`: primary 0.4296 vs second 0.6889
