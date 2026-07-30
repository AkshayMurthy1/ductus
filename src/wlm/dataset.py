"""Splits and training-format conversion.

The one rule that matters: split by DOCUMENT, not by pair. Two chunks from the same document
share topic, register and often whole phrases. Putting one in train and one in blind lets the
model score well by recall, and every downstream number becomes a lie.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from wlm.paths import write_jsonl


def split_pairs(
    pairs: list[dict[str, Any]],
    *,
    val_frac: float = 0.15,
    blind_frac: float = 0.15,
    by: str = "document",
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    if by not in {"document", "pair"}:
        raise ValueError("by must be 'document' or 'pair'")
    rng = random.Random(seed)

    if by == "pair":
        rows = pairs[:]
        rng.shuffle(rows)
        n_val = int(len(rows) * val_frac)
        n_blind = int(len(rows) * blind_frac)
        return {
            "blind": rows[:n_blind],
            "val": rows[n_blind : n_blind + n_val],
            "train": rows[n_blind + n_val :],
        }

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_doc[p["doc_id"]].append(p)

    # Stratify by register so the informal set -- the hard case the plan cares about -- is
    # actually represented in val and blind instead of landing entirely in train by luck.
    doc_register = {d: rows[0].get("register", "unknown") for d, rows in by_doc.items()}
    buckets: dict[str, list[str]] = defaultdict(list)
    for d, reg in doc_register.items():
        buckets[reg].append(d)

    out: dict[str, list[dict]] = {"train": [], "val": [], "blind": []}
    for reg in sorted(buckets):
        docs = sorted(buckets[reg])
        rng.shuffle(docs)
        n = len(docs)
        if n >= 4:
            n_blind = max(1, round(n * blind_frac))
            n_val = max(1, round(n * val_frac))
        elif n == 3:
            # Early stopping and best-checkpoint restore both need a val set, so at three
            # documents spend one on val rather than leaving it empty.
            n_blind, n_val = 1, 1
        elif n == 2:
            n_blind, n_val = 1, 0
        else:
            n_blind, n_val = 0, 0
        assign = (
            [("blind", d) for d in docs[:n_blind]]
            + [("val", d) for d in docs[n_blind : n_blind + n_val]]
            + [("train", d) for d in docs[n_blind + n_val :]]
        )
        for split, d in assign:
            out[split].extend(by_doc[d])

    for k in out:
        rng.shuffle(out[k])
    return out


def split_summary(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for name, rows in splits.items():
        regs: dict[str, int] = defaultdict(int)
        for r in rows:
            regs[r.get("register", "unknown")] += 1
        s[name] = {
            "pairs": len(rows),
            "docs": len({r["doc_id"] for r in rows}),
            "words": sum(r.get("n_words", 0) for r in rows),
            "registers": dict(regs),
        }
    # Overlap check -- cheap insurance against a future refactor breaking the split.
    ids = {k: {r["doc_id"] for r in v} for k, v in splits.items()}
    s["_doc_overlap"] = {
        "train_val": sorted(ids.get("train", set()) & ids.get("val", set())),
        "train_blind": sorted(ids.get("train", set()) & ids.get("blind", set())),
        "val_blind": sorted(ids.get("val", set()) & ids.get("blind", set())),
    }
    return s


def to_prompt_completion(
    pair: dict[str, Any], system_prompt: str | None = None
) -> dict[str, Any]:
    """trl conversational *prompt-completion* format — the shape this project actually wants.

    Why not `messages` + `assistant_only_loss`: that path needs a chat template annotated with
    `{% generation %}` so the tokenizer can return an assistant mask, and the Qwen2.5 templates
    ship without it. A prompt-completion dataset gets completion-only loss from trl directly, no
    template surgery, no silent fallback to training on the machine-written question.

    The system prompt is included here whenever generation will include it, so the adapter is
    tuned on exactly the prompt format it will see at eval and serve time.
    """
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": pair["prompt"]})
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": pair["response"]}],
        "register": pair.get("register", "unknown"),
        "pair_id": pair.get("pair_id", ""),
    }


CONTINUE_INSTRUCTION = "Continue the passage below in the same voice.\n\n"


def to_self_continuation(
    pair: dict[str, Any],
    *,
    lead_frac: float = 0.2,
    min_lead_words: int = 12,
    system_prompt: str | None = None,
) -> dict[str, Any] | None:
    """Plan §4.5's next-token mix, as **lead-continuation** on the person's own prose.

    Honest description of what this is and isn't: §4.5 calls for raw next-token modeling with no
    instruction wrapper. trl cannot mix a plain-text language-modeling dataset with a
    prompt-completion one in a single run, so this expresses the same idea one step removed — the
    person's own opening becomes the prompt, the remainder is the completion, and loss falls only
    on their words. There *is* a chat wrapper and a short explicit instruction, so the model is
    learning "continue this prose" rather than pure unconditional LM. That instruction is stated
    outright so the task never looks like a bare user turn the model should be continuing instead
    of answering.

    The same system prompt as the Stage-A pairs is applied, so the training set has one prompt
    shape rather than two.

    Keep the mix weight low. No question abstracts away the topic here, so this is the most
    memorization-prone part of the recipe.
    """
    from wlm.chunk import split_sentences

    text = pair["response"]
    sents = split_sentences(text)
    if len(sents) < 3:
        return None

    target = max(min_lead_words, int(len(text.split()) * lead_frac))
    # Cut on a character offset in the original string rather than re-joining sentences: joining
    # with spaces would flatten paragraph breaks, and paragraph shape is one of the style features
    # this mix is supposed to teach.
    cursor, cut, taken = 0, 0, 0
    for s in sents[:-1]:
        idx = text.find(s, cursor)
        if idx < 0:
            return None
        cursor = idx + len(s)
        cut = cursor
        taken += len(s.split())
        if taken >= target:
            break

    lead, rest = text[:cut], text[cut:].strip()
    if not lead.strip() or len(rest.split()) < min_lead_words:
        return None
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": CONTINUE_INSTRUCTION + lead})
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": rest}],
        "register": pair.get("register", "unknown"),
        "pair_id": f"raw-{pair.get('pair_id', '')}",
    }


def write_splits(splits: dict[str, list[dict[str, Any]]], outdir) -> dict[str, int]:
    from pathlib import Path

    outdir = Path(outdir)
    counts = {}
    for name, rows in splits.items():
        counts[name] = write_jsonl(outdir / f"{name}.jsonl", rows)
    return counts
