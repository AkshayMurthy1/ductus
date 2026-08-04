"""Human blind panel — the only check the verifier cannot game.

Protocol (docs/EXPANSION.md): each item shows the rater one labeled *reference* excerpt (real
author text from TRAIN) and one blinded *candidate*, and asks "how likely is it that the same
person wrote both?" on a 1-5 scale. Candidates are drawn in equal numbers from three sources —
real held-out author text, adapter generations, and an era/register-matched distractor — and
the rater never learns the mix. This is the anchored-comparison design standard in authorship
verification; a 3-way forced choice without an anchor would test the rater's prior familiarity
with the author, not the text.

What it adjudicates: the 0.825 > 0.714 anomaly (STATUS §2, curiosity a). If raters score
adapter text at or below real text while the verifier scores it above, the verifier is being
gamed and the paper must say so; if raters agree with the verifier, "hyper-typical" survives
as the reading.

Pure CPU logic, deterministic under seed. The key (item -> source) is written separately from
the sheets and must never travel with them.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

SOURCES = ("real", "adapter", "distractor")


def build_panel(
    *,
    real: list[str],
    adapter: list[str],
    distractor: list[str],
    references: list[str],
    n_per_source: int = 12,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Blinded rating items + the separate answer key.

    Returns (items, key): items are [{item_id, reference, candidate}] in a shuffled order that
    never reveals source grouping; key maps item_id -> source. Same seed -> identical panel,
    so a sheet can be regenerated to verify no post-hoc editing.
    """
    pools = {"real": real, "adapter": adapter, "distractor": distractor}
    for name, pool in pools.items():
        if len(pool) < n_per_source:
            raise ValueError(f"need >= {n_per_source} {name} texts, got {len(pool)}")
    if not references:
        raise ValueError("need at least one reference excerpt")

    rng = random.Random(seed)
    picks: list[tuple[str, str]] = []
    for name in SOURCES:  # fixed source order, so the seed fully determines the draw
        picks += [(name, t) for t in rng.sample(pools[name], n_per_source)]
    rng.shuffle(picks)

    items, key = [], {}
    for i, (source, text) in enumerate(picks, 1):
        item_id = f"P{i:03d}"
        items.append({
            "item_id": item_id,
            "reference": rng.choice(references),
            "candidate": text,
        })
        key[item_id] = source
    return items, key


def score_panel(
    key: dict[str, str],
    ratings: dict[str, float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Per-source rating summary + the deltas the study actually needs.

    The headline number is `adapter_minus_real`: zero (CI spanning 0) means raters cannot tell
    the adapter from the real author; clearly negative means the verifier overrates the
    adapter; clearly positive would replicate the verifier's "hyper-typical" reading in humans.
    """
    by_source: dict[str, list[float]] = {s: [] for s in SOURCES}
    unmatched = 0
    for item_id, rating in ratings.items():
        src = key.get(item_id)
        if src is None:
            unmatched += 1
            continue
        by_source[src].append(float(rating))

    summary: dict[str, Any] = {"n_rated": sum(len(v) for v in by_source.values()),
                               "n_unmatched": unmatched}
    for s in SOURCES:
        vals = by_source[s]
        summary[s] = {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 4) if vals else None,
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        }

    def delta(a: str, b: str) -> dict[str, Any] | None:
        va, vb = by_source[a], by_source[b]
        if not va or not vb:
            return None
        d = statistics.mean(va) - statistics.mean(vb)
        rng = random.Random(seed)
        boots = sorted(
            statistics.mean(rng.choices(va, k=len(va)))
            - statistics.mean(rng.choices(vb, k=len(vb)))
            for _ in range(n_bootstrap)
        )
        lo, hi = boots[int(0.025 * n_bootstrap)], boots[int(0.975 * n_bootstrap) - 1]
        return {"delta": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    summary["adapter_minus_real"] = delta("adapter", "real")
    summary["adapter_minus_distractor"] = delta("adapter", "distractor")
    summary["real_minus_distractor"] = delta("real", "distractor")
    return summary
