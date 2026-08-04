# Dual-instrument check

Second instrument: `StyleDistance/styledistance` (held-out AUC 0.9659)

Rank agreement over 29 runs: Spearman **0.8968** (disagreement 0.0516, mean |Δ| 0.0464)

| run | primary AV | second AV | Δ |
|---|---|---|---|
| floor | 0.0000 | 0.0000 | +0.0000 |
| matrix/a14_no_scrubbing | 0.8532 | 0.8532 | +0.0000 |
| matrix/rq2/a01_attention_only | 0.8571 | 0.9325 | -0.0754 |
| matrix/rq2/a02_mlp_only | 0.7937 | 0.9484 | -0.1547 |
| matrix/rq2/a05_split_rank_low_mlp | 0.8095 | 0.9286 | -0.1191 |
| sweep/100k/baseline | 0.0000 | 0.0000 | +0.0000 |
| sweep/100k/stage_a | 0.7976 | 0.9246 | -0.1270 |
| sweep/10k/baseline | 0.0000 | 0.0040 | -0.0040 |
| sweep/10k/stage_a | 0.0040 | 0.0119 | -0.0079 |
| sweep/10k/stage_b | 0.1667 | 0.1984 | -0.0317 |
| sweep/25k/baseline | 0.0000 | 0.0000 | +0.0000 |
| sweep/25k/stage_a | 0.7183 | 0.8016 | -0.0833 |
| sweep/2k/baseline | 0.0040 | 0.0635 | -0.0595 |
| sweep/2k/seed29/baseline | 0.0159 | 0.0437 | -0.0278 |
| sweep/2k/seed29/stage_a | 0.0000 | 0.0079 | -0.0079 |
| sweep/2k/seed43/baseline | 0.0000 | 0.0000 | +0.0000 |
| sweep/2k/seed43/stage_a | 0.0040 | 0.0000 | +0.0040 |
| sweep/2k/stage_a | 0.0000 | 0.0000 | +0.0000 |
| sweep/50k/baseline | 0.0000 | 0.0000 | +0.0000 |
| sweep/50k/stage_a | 0.8254 | 0.9524 | -0.1270 |
| sweep/5k/baseline | 0.0000 | 0.0000 | +0.0000 |
| sweep/5k/stage_a | 0.0000 | 0.0000 | +0.0000 |
| sweep/full/baseline | 0.0040 | 0.0040 | +0.0000 |
| sweep/full/seed29/baseline | 0.0040 | 0.0079 | -0.0039 |
| sweep/full/seed29/stage_a | 0.7778 | 0.9087 | -0.1309 |
| sweep/full/seed43/baseline | 0.0079 | 0.0040 | +0.0039 |
| sweep/full/seed43/stage_a | 0.7897 | 0.9444 | -0.1547 |
| sweep/full/stage_a | 0.7897 | 0.9365 | -0.1468 |
| sweep/full/stage_b | 0.8452 | 0.9206 | -0.0754 |

Real held-out author text under the second ruler: attribution **0.7381** (mean score 0.6590). Compare adapter rows against this the same way the primary is compared against 0.714.

Runs the two instruments disagree on (sample the human panel here):
- `matrix/rq2/a02_mlp_only`: primary 0.7937 vs second 0.9484
- `matrix/rq2/a05_split_rank_low_mlp`: primary 0.8095 vs second 0.9286
- `sweep/100k/stage_a`: primary 0.7976 vs second 0.9246
- `sweep/50k/stage_a`: primary 0.8254 vs second 0.9524
- `sweep/full/seed29/stage_a`: primary 0.7778 vs second 0.9087
- `sweep/full/seed43/stage_a`: primary 0.7897 vs second 0.9444
- `sweep/full/stage_a`: primary 0.7897 vs second 0.9365
