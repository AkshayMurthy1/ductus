"""Pretraining-contamination probe — the floor the whole public-author arm sits on.

Chesterton (the reproducible dev author) is in the base model's pretraining data, so part of any
apparent style gain on him is *recall*, not learning. RESEARCH_BRIEF §5 requires recording the
untuned base model's relationship to the author before believing any adapter number. Two
measurements do that:

  1. **Perplexity familiarity ratio** (this module): base-model perplexity on held-out author
     text vs on era/genre-matched distractor text. The distractors are the control — both sides
     are period prose the model may know, so a ratio well below 1.0 means the model knows *this
     author specifically*, beyond knowing the period.
  2. **Attribution floor** (driver step, not this module): `wlm generate` with no adapter and no
     few-shot exemplars, evaluated like any other run. That is the zero of the style axis.

The Chesterton-vs-private-author gap on these numbers is Table 4's interpretive key.
"""

from __future__ import annotations

from typing import Any

# Cap the probe so it stays a minutes-scale check, not an eval epoch. The cap is logged in the
# output (brief §8: log every bound the pipeline imposes).
DEFAULT_MAX_TEXTS = 48


def contamination_report(
    cfg,
    author_texts: list[str],
    distractor_texts: list[str],
    *,
    max_texts: int = DEFAULT_MAX_TEXTS,
) -> dict[str, Any]:
    """Base-model (no adapter) perplexity on author vs distractor text. GPU-friendly, CPU-capable."""
    from wlm.eval.fluency import perplexity
    from wlm.train.common import load_base_model, load_tokenizer

    author = [t for t in author_texts if t.strip()][:max_texts]
    distractor = [t for t in distractor_texts if t.strip()][:max_texts]
    if len(author) < 5 or len(distractor) < 5:
        raise ValueError(
            f"need at least 5 texts per side (got {len(author)} author, {len(distractor)} "
            "distractor) — a ratio from fewer is noise."
        )

    tok = load_tokenizer(cfg)
    model = load_base_model(cfg, for_training=False)
    model.eval()

    ppl_author = perplexity(model, tok, author)
    ppl_distractor = perplexity(model, tok, distractor)
    ratio = ppl_author / ppl_distractor if ppl_distractor else None

    return {
        "model": cfg.model.name,
        "ppl_author": round(ppl_author, 4),
        "ppl_distractor": round(ppl_distractor, 4),
        "familiarity_ratio": round(ratio, 4) if ratio else None,
        "n_author_texts": len(author),
        "n_distractor_texts": len(distractor),
        "max_texts_cap": max_texts,
        "interpretation": (
            "ratio ≈ 1.0: the base model knows this author no better than matched period prose. "
            "ratio << 1.0: pretraining contamination — discount the public-author arm's gains "
            "by comparison with the private-author control (Table 4)."
        ),
    }
