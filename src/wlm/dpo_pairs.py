"""Stage-B preference-pair construction: on-policy, anchored on the person's real text.

The pair is:
    prompt   = a held-out backtranslated question
    chosen   = the person's REAL passage for that question
    rejected = what the Stage-A model actually generates for it

Why on-policy: SFT teaches the model what a good answer looks like, but it never sees its own
failure mode. The residual "generic AI voice" survives Stage A precisely because SFT never
penalizes it. Sampling from the Stage-A model surfaces that failure mode, and DPO pushes away
from it -- while `chosen` stays real human text, which is the anchor that keeps this from being
self-training. (Recursive training on generated data collapses; see Nature 2024 / 2305.17493.)

Two filters that matter more than they look:

  1. **Length matching.** If `chosen` is 250 words and `rejected` is 90, DPO learns "be long",
     not "sound like this person". Any pair outside the tolerance is regenerated or dropped.
  2. **Already-good rejection.** If the verifier scores the model's own output above the real
     passage, this pair teaches the model to get worse. Drop it (`av_reward_filter`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wlm.config import Config
from wlm.paths import read_jsonl, write_jsonl


def length_ok(chosen: str, rejected: str, tol: float) -> bool:
    lc, lr = len(chosen.split()), len(rejected.split())
    if min(lc, lr) == 0:
        return False
    return abs(lc - lr) / max(lc, lr) <= tol


def build_dpo_pairs(
    cfg: Config,
    *,
    adapter: str | Path,
    split_path: str | Path,
    out_path: str | Path,
    av_dir: str | Path | None = None,
    max_retries: int = 2,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    from wlm.eval.fluency import repetition_score
    from wlm.generate import generate, load_for_generation

    refs = read_jsonl(split_path)
    stats = {
        "candidates": len(refs),
        "dropped_length": 0,
        "dropped_av": 0,
        "dropped_degenerate": 0,
        "kept": 0,
        "retries": 0,
    }

    verifier = None
    if cfg.dpo.av_reward_filter and av_dir and Path(av_dir).exists():
        from wlm.eval.av import AuthorshipVerifier

        verifier = AuthorshipVerifier.load(av_dir)

    # Nudge the model toward the real passage's length so length matching is achievable rather
    # than a coin flip.
    # Load the model once. `generate()` would otherwise re-load a 3B checkpoint on every retry
    # attempt, which is three full loads per invocation of this function.
    loaded = load_for_generation(cfg, adapter)

    pairs: list[dict[str, Any]] = []
    pending = refs
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if attempt:
            stats["retries"] += len(pending)
        prompts = [
            {**r, "prompt": _with_length_hint(r["prompt"], len(r["response"].split()))}
            for r in pending
        ]
        gens = generate(cfg, prompts, adapter=adapter, n_per_prompt=1, loaded=loaded)
        gen_by_id = {g.get("pair_id"): g for g in gens}
        still_pending = []
        for r in pending:
            g = gen_by_id.get(r.get("pair_id"))
            if not g or not g["completion"]:
                stats["dropped_degenerate"] += 1
                continue
            if repetition_score(g["completion"]) > 0.3:
                stats["dropped_degenerate"] += 1
                continue
            if not length_ok(r["response"], g["completion"], cfg.dpo.length_match_tolerance):
                if attempt < max_retries:
                    still_pending.append(r)
                else:
                    stats["dropped_length"] += 1
                continue
            pairs.append(
                {
                    "pair_id": r.get("pair_id"),
                    "register": r.get("register", "unknown"),
                    # Keep the original question in the training pair, not the length hint --
                    # the hint is a sampling aid, not something to teach.
                    "prompt": r["prompt"],
                    "chosen": r["response"],
                    "rejected": g["completion"],
                }
            )
        pending = still_pending

    if verifier is not None:
        chosen_scores = verifier.score([p["chosen"] for p in pairs])
        rejected_scores = verifier.score([p["rejected"] for p in pairs])
        filtered = []
        for p, cs, rs in zip(pairs, chosen_scores, rejected_scores, strict=False):
            if cs - rs < cfg.dpo.av_margin:
                stats["dropped_av"] += 1
                continue
            p["av_chosen"], p["av_rejected"] = round(float(cs), 4), round(float(rs), 4)
            filtered.append(p)
        pairs = filtered

    stats["kept"] = len(pairs)

    # Hold back a slice so Stage B can run eval -- which is what enables early visibility on
    # reward margins and, more importantly, the fluency probe.
    out_path = Path(out_path)
    if val_frac > 0 and len(pairs) >= 20:
        import random

        rng = random.Random(cfg.dpo.seed)
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        val_pairs, train_pairs = shuffled[:n_val], shuffled[n_val:]
        val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
        write_jsonl(val_path, val_pairs)
        write_jsonl(out_path, train_pairs)
        stats["train_pairs"], stats["val_pairs"] = len(train_pairs), len(val_pairs)
        stats["val_path"] = str(val_path)
    else:
        write_jsonl(out_path, pairs)
        stats["train_pairs"], stats["val_pairs"] = len(pairs), 0

    if stats["kept"] < 40:
        print(
            f"[dpo-pairs] only {stats['kept']} pairs survived. DPO on this few is mostly noise -- "
            "loosen length_match_tolerance, generate more held-out questions, or reuse val."
        )
    return stats


def _with_length_hint(prompt: str, n_words: int) -> str:
    lo, hi = int(n_words * 0.8), int(n_words * 1.25)
    return f"{prompt}\n\n(Aim for roughly {lo}-{hi} words.)"
