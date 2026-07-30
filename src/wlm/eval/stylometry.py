"""Classical stylometry: the interpretable half of the eval.

The authorship verifier tells you *whether* the imitation works. Stylometry tells you *what*
is off, which is the only thing you can act on. Every feature here is deliberately
content-independent -- function words, punctuation, sentence-length distribution, POS shape --
because a feature that responds to topic will reward content leak.

Features:
  - sentence length: mean, sd, and burstiness (Wolfram's measure: (sd-mean)/(sd+mean))
  - word length distribution
  - function-word relative frequencies (the classic Mosteller-Wallace signal)
  - punctuation histogram, incl. the tells: em dash, semicolon, parenthetical, ellipsis
  - contraction rate, first/second-person rate, hedging rate, sentence-opener distribution
  - type-token ratio (length-normalized via MSTTR)
  - optional POS trigram distribution when spaCy is available

Distance is a mean of per-block normalized distances (Jensen-Shannon for distributions,
standardized absolute difference for scalars) so it stays in [0, 1] and is comparable
across runs.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from wlm.chunk import split_sentences

# The 150-ish highest-frequency English function words. Topic-blind by construction.
FUNCTION_WORDS = """
a about above after all also although am an and another any are as at be because been before
being below between both but by can cannot could did do does doing done down during each either
else even ever every few for from further get got had has have having he her here hers herself
him himself his how however i if in into is it its itself just least less like made make many
may me might mine more most much must my myself neither never no nor not now of off on once one
only onto or other others ought our ours ourselves out over own perhaps quite rather really same
seem seemed she should since so some still such than that the their theirs them themselves then
there these they this those though through thus to too toward under until up upon us very was
we well were what whatever when where whether which while who whom whose why will with within
without would yet you your yours yourself
""".split()

PUNCT = [",", ".", ";", ":", "!", "?", "—", "–", "-", "(", ")", "\"", "'", "…", "/"]

HEDGES = {
    "maybe", "probably", "perhaps", "sort", "kind", "somewhat", "fairly", "pretty", "basically",
    "arguably", "seems", "suppose", "guess", "think", "feel", "might", "could", "possibly",
    "apparently", "roughly", "generally", "mostly", "honestly", "actually",
}
INTENSIFIERS = {"very", "really", "so", "extremely", "incredibly", "totally", "absolutely",
                "quite", "highly", "deeply", "wildly"}
CONTRACTION = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.I)
WORD = re.compile(r"[A-Za-z']+")

_NLP = None
_NLP_TRIED = False


def _nlp():
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception:
        _NLP = None
    return _NLP


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def _dist(counter: Counter, keys: list) -> np.ndarray:
    """Renormalized distribution of `counter` over `keys` (keys may be words or POS tuples)."""
    total = sum(counter.get(k, 0) for k in keys)
    if total == 0:
        return np.full(len(keys), 1.0 / len(keys))
    return np.array([counter.get(k, 0) / total for k in keys], dtype=float)


def msttr(tokens: list[str], window: int = 50) -> float:
    """Mean segmental type-token ratio. Plain TTR is a length artifact; this isn't."""
    if len(tokens) < window:
        return len(set(tokens)) / max(1, len(tokens))
    ratios = [
        len(set(tokens[i : i + window])) / window
        for i in range(0, len(tokens) - window + 1, window)
    ]
    return float(np.mean(ratios))


def stylometry_vector(text: str) -> dict[str, Any]:
    """Feature blocks for one text (a corpus concatenation, or a single generation)."""
    sents = split_sentences(text)
    toks = _tokens(text)
    n_tok = max(1, len(toks))
    sent_lens = np.array([len(_tokens(s)) for s in sents if s.strip()], dtype=float)
    if sent_lens.size == 0:
        sent_lens = np.array([0.0])

    mean_sl = float(sent_lens.mean())
    sd_sl = float(sent_lens.std())
    burstiness = (sd_sl - mean_sl) / (sd_sl + mean_sl) if (sd_sl + mean_sl) > 0 else 0.0

    # Sentence-length histogram in fixed buckets -- captures rhythm, not just its average.
    buckets = [0, 5, 10, 15, 20, 25, 30, 40, 60, 10**6]
    sl_hist = np.histogram(sent_lens, bins=buckets)[0].astype(float)
    sl_hist = sl_hist / max(1.0, sl_hist.sum())

    wl_hist = np.histogram(
        np.array([len(w) for w in toks], dtype=float), bins=[1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 100]
    )[0].astype(float)
    wl_hist = wl_hist / max(1.0, wl_hist.sum())

    tok_counts = Counter(toks)
    fw = _dist(tok_counts, FUNCTION_WORDS)

    punct_counts = Counter(ch for ch in text if ch in PUNCT)
    punct = np.array([punct_counts.get(p, 0) / n_tok * 100 for p in PUNCT], dtype=float)
    punct_dist = punct / max(1e-9, punct.sum())

    openers = Counter()
    for s in sents:
        w = _tokens(s)
        if w:
            openers[w[0] if w[0] in FUNCTION_WORDS else "<content>"] += 1
    opener_keys = [*FUNCTION_WORDS[:60], "<content>"]
    opener_dist = _dist(openers, opener_keys)

    scalars = {
        "sent_len_mean": mean_sl,
        "sent_len_sd": sd_sl,
        "burstiness": burstiness,
        "word_len_mean": float(np.mean([len(w) for w in toks])) if toks else 0.0,
        "msttr": msttr(toks),
        "contraction_rate": len(CONTRACTION.findall(text)) / n_tok * 100,
        "hedge_rate": sum(tok_counts[h] for h in HEDGES) / n_tok * 100,
        "intensifier_rate": sum(tok_counts[i] for i in INTENSIFIERS) / n_tok * 100,
        "first_person_rate": (tok_counts["i"] + tok_counts["me"] + tok_counts["my"]
                              + tok_counts["we"] + tok_counts["our"]) / n_tok * 100,
        "second_person_rate": (tok_counts["you"] + tok_counts["your"]) / n_tok * 100,
        "em_dash_per_100": punct_counts.get("—", 0) / n_tok * 100,
        "semicolon_per_100": punct_counts.get(";", 0) / n_tok * 100,
        "paren_per_100": punct_counts.get("(", 0) / n_tok * 100,
        "question_per_100": punct_counts.get("?", 0) / n_tok * 100,
        "exclaim_per_100": punct_counts.get("!", 0) / n_tok * 100,
        "para_len_sents": len(sents) / max(1, text.count("\n\n") + 1),
    }

    out: dict[str, Any] = {
        "scalars": scalars,
        "dists": {
            "sent_len_hist": sl_hist,
            "word_len_hist": wl_hist,
            "function_words": fw,
            "punctuation": punct_dist,
            "openers": opener_dist,
        },
        "n_tokens": len(toks),
        "n_sents": int(sent_lens.size),
    }

    nlp = _nlp()
    if nlp is not None and toks:
        doc = nlp(text[:200_000])
        tags = [t.pos_ for t in doc if not t.is_space]
        # Keep the *full* counts, not this text's top-N. The POS tagset is ~18 tags, so the
        # support is bounded at ~6k trigrams and usually far smaller. Truncating per-text would
        # make the key set depend on which text you looked at first, which is how this feature
        # was previously both asymmetric and blind to trigrams only the generation produces.
        out["pos_trigrams"] = Counter(zip(tags, tags[1:], tags[2:], strict=False))
    return out


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2 -> already in [0, 1]."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))  # noqa: E731
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


# Rough within-author scale for each scalar, used to normalize differences into [0,1].
# Values are "a difference this large is a full unit of stylistic distance".
_SCALAR_SCALE = {
    "sent_len_mean": 8.0, "sent_len_sd": 6.0, "burstiness": 0.25, "word_len_mean": 0.8,
    "msttr": 0.12, "contraction_rate": 2.0, "hedge_rate": 1.5, "intensifier_rate": 1.0,
    "first_person_rate": 3.0, "second_person_rate": 2.0, "em_dash_per_100": 0.8,
    "semicolon_per_100": 0.5, "paren_per_100": 0.8, "question_per_100": 1.0,
    "exclaim_per_100": 1.0, "para_len_sents": 2.5,
}


def stylometry_distance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per-feature and aggregate distance between two stylometry vectors. Lower is better."""
    dists = {k: _js(a["dists"][k], b["dists"][k]) for k in a["dists"]}
    if "pos_trigrams" in a and "pos_trigrams" in b:
        # Compare over the union of observed trigrams. The union is order-independent (so the
        # distance is symmetric) and it retains trigrams present in only one side (so a
        # generation's unfamiliar syntax is penalized rather than normalized away).
        keys = sorted(set(a["pos_trigrams"]) | set(b["pos_trigrams"]))
        if keys:
            dists["pos_trigrams"] = _js(
                _dist(a["pos_trigrams"], keys), _dist(b["pos_trigrams"], keys)
            )

    scal = {}
    for k, scale in _SCALAR_SCALE.items():
        d = abs(a["scalars"].get(k, 0.0) - b["scalars"].get(k, 0.0)) / scale
        scal[k] = min(1.0, d)

    overall = float(np.mean(list(dists.values()) + [float(np.mean(list(scal.values())))]))
    return {
        "overall": round(overall, 4),
        "distributions": {k: round(v, 4) for k, v in dists.items()},
        "scalars_normalized": {k: round(v, 4) for k, v in scal.items()},
        "scalars_raw": {
            k: (round(a["scalars"].get(k, 0.0), 3), round(b["scalars"].get(k, 0.0), 3))
            for k in _SCALAR_SCALE
        },
    }


def biggest_gaps(dist: dict[str, Any], k: int = 6) -> list[tuple[str, float]]:
    """What to fix next, ranked. This is the actionable output of the whole module."""
    items = [(f"dist:{n}", v) for n, v in dist["distributions"].items()]
    items += [(f"scalar:{n}", v) for n, v in dist["scalars_normalized"].items()]
    return sorted(items, key=lambda x: -x[1])[:k]


def corpus_text(rows: list[dict[str, Any]], field: str = "response") -> str:
    return "\n\n".join(r[field] for r in rows if r.get(field))


def bootstrap_ci(
    texts_a: list[str], texts_b: list[str], *, n: int = 200, seed: int = 17
) -> tuple[float, float]:
    """95% CI on overall distance by resampling texts. Without this you will over-read noise."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sa = "\n\n".join(rng.choice(texts_a, size=len(texts_a), replace=True))
        sb = "\n\n".join(rng.choice(texts_b, size=len(texts_b), replace=True))
        vals.append(stylometry_distance(stylometry_vector(sa), stylometry_vector(sb))["overall"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
