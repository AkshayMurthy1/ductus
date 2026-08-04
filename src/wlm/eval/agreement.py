"""Between-instrument agreement — do two independently-trained rulers order the runs the same?

Why this exists: every style number in the study reads through one embedder + one head. An
adapter can look style-perfect by matching that verifier's *features* rather than style in any
human sense (the Goodhart reading of the 0.825 > 0.714 anomaly, STATUS §2). The defense is a
second instrument fit on the same data with a different embedder: if both rulers rank the runs
the same way, "one classifier says so" becomes a measurement; where they disagree is exactly
where to look for verifier gaming.

Pure functions over score lists — no models, no network, so the CPU test suite can hold them to
the metric contract (zero on identical inputs, larger on genuinely different ones).
"""

from __future__ import annotations

from typing import Any


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties shared — plain Spearman prep without scipy."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation in [-1, 1]. Returns 1.0 for degenerate constant inputs:
    two instruments that both report "everything is equal" are in perfect agreement."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length lists with >= 2 entries")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    da, db = [x - ma for x in ra], [x - mb for x in rb]
    denom = (sum(x * x for x in da) * sum(x * x for x in db)) ** 0.5
    if denom == 0:
        return 1.0
    return sum(x * y for x, y in zip(da, db, strict=True)) / denom


def rank_disagreement(a: list[float], b: list[float]) -> float:
    """(1 - spearman) / 2 — 0 when both instruments rank identically, 1 when exactly reversed.

    This is the number the dual-instrument table leads with: a small value licenses reading the
    primary verifier's orderings as instrument-independent; a large one means the headline
    claims are claims about an embedder.
    """
    return (1.0 - spearman(a, b)) / 2.0


def instrument_agreement(
    runs: list[dict[str, Any]], key_a: str = "av_primary", key_b: str = "av_second"
) -> dict[str, Any]:
    """Summary block for a set of runs scored under two instruments.

    Each run dict needs key_a and key_b (attribution rates). Flags the runs whose absolute
    disagreement exceeds twice the mean — those are the cells to hand-inspect first.
    """
    pairs = [(r[key_a], r[key_b]) for r in runs if r.get(key_a) is not None
             and r.get(key_b) is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "note": "need >= 2 dual-scored runs for an agreement number"}
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    abs_diff = [abs(x - y) for x, y in zip(a, b, strict=True)]
    mean_abs = sum(abs_diff) / len(abs_diff)
    flagged = [
        {"run": r.get("run", "?"), key_a: r[key_a], key_b: r[key_b]}
        for r in runs
        if r.get(key_a) is not None and r.get(key_b) is not None
        and abs(r[key_a] - r[key_b]) > max(2 * mean_abs, 1e-9) and mean_abs > 0
    ]
    return {
        "n": len(pairs),
        "spearman": round(spearman(a, b), 4),
        "rank_disagreement": round(rank_disagreement(a, b), 4),
        "mean_abs_diff": round(mean_abs, 4),
        "max_abs_diff": round(max(abs_diff), 4),
        "flagged_runs": flagged,
    }
