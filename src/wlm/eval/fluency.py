"""Fluency probe — catastrophic-forgetting guard, run at every checkpoint.

A style adapter that wins the verifier by degrading the model is not a win. This measures
perplexity on a small fixed slice of general text with the adapter attached vs. detached; the
delta is your forgetting signal. Cheap enough to run inside the training callback.

Deliberately not a big benchmark: at 1-3B with a rank-16 adapter, MMLU-style evals are noise
next to the effect size you're looking for, and they cost minutes per checkpoint. Perplexity
delta on fixed text plus a short generation smoke test catches real degradation immediately.
"""

from __future__ import annotations

from typing import Any

# Fixed, license-clean general-domain probe text. Same text every run so numbers compare.
PROBE_TEXTS = [
    "The mitochondrion is a double-membraned organelle found in most eukaryotic cells. It "
    "generates most of the cell's supply of adenosine triphosphate, which is used as a source "
    "of chemical energy.",
    "In computer science, a hash table is a data structure that implements an associative array, "
    "mapping keys to values. A hash function computes an index into an array of buckets, from "
    "which the desired value can be found.",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War in the Holy Roman "
    "Empire and the Eighty Years' War between Spain and the Dutch Republic. Historians often "
    "treat it as the origin of the modern system of sovereign states.",
    "To make a simple vinaigrette, whisk together one part acid, usually vinegar or citrus "
    "juice, with three parts oil. Season with salt and pepper, and emulsify by adding the oil "
    "slowly while whisking continuously.",
    "A limit of a function describes the value that the function approaches as the input "
    "approaches some value. Limits are essential to calculus and are used to define continuity, "
    "derivatives, and integrals.",
]

SMOKE_PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "List three steps for changing a bicycle tire.",
    "What is the difference between a simile and a metaphor?",
]


def perplexity(model, tokenizer, texts: list[str], *, device: str | None = None) -> float:
    """Mean token-level perplexity over `texts`. Works with or without an active adapter."""
    import torch

    model.eval()
    device = device or (str(next(model.parameters()).device))
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            n = int(enc["input_ids"].numel()) - 1
            if n <= 0:
                continue
            total_nll += float(out.loss) * n
            total_tokens += n
    if total_tokens == 0:
        return float("nan")
    import math

    return math.exp(total_nll / total_tokens)


def fluency_report(
    model,
    tokenizer,
    *,
    probe_texts: list[str] | None = None,
    max_regression: float = 0.15,
) -> dict[str, Any]:
    """Adapter-on vs adapter-off perplexity on the fixed probe.

    Uses peft's `disable_adapter()` so this needs exactly one model in memory -- no second
    base-model load, which matters on a 24 GB card mid-training.
    """
    texts = probe_texts or PROBE_TEXTS
    ppl_on = perplexity(model, tokenizer, texts)

    ppl_off = None
    disable = getattr(model, "disable_adapter", None)
    if callable(disable):
        try:
            with model.disable_adapter():
                ppl_off = perplexity(model, tokenizer, texts)
        except Exception as e:
            print(f"[fluency] could not disable adapter for baseline: {e}")

    regression = None
    verdict = "unknown"
    if ppl_off and ppl_off > 0:
        regression = (ppl_on - ppl_off) / ppl_off
        verdict = "PASS" if regression <= max_regression else "FAIL: catastrophic forgetting"

    return {
        "ppl_adapter_on": round(ppl_on, 4),
        "ppl_adapter_off": round(ppl_off, 4) if ppl_off else None,
        "relative_regression": round(regression, 4) if regression is not None else None,
        "max_allowed": max_regression,
        "verdict": verdict,
        "n_probe_texts": len(texts),
    }


def smoke_generations(model, tokenizer, *, max_new_tokens: int = 80) -> list[dict[str, str]]:
    """Three general questions. Read these -- degeneration is obvious to a human before it is
    obvious in a number (repetition loops, dropped syntax, answering in the author's voice when
    the question is factual)."""
    import torch

    out = []
    device = str(next(model.parameters()).device)
    for p in SMOKE_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            pad = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
            )
        gen = tokenizer.decode(ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out.append({"prompt": p, "completion": gen.strip()})
    return out


def repetition_score(text: str, *, n: int = 4) -> float:
    """Fraction of 4-grams that are repeats. >0.25 means the model is looping."""
    toks = text.lower().split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)
