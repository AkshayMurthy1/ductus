"""Stage-A pair construction by instruction backtranslation (Humpback, arXiv 2308.06259).

For each real passage the person wrote, ask a *separate hosted API model* for the prompt that
would have elicited it. Training pair = (generated question -> the person's real passage).

Two invariants, both load-bearing:

  1. The generator is never the trainee. A separate API model breaks the self-generation
     feedback loop that drives model collapse (Nature 2024, arXiv 2305.17493).
  2. Questions must be GENERIC and style-agnostic. "Write a short reflection on a recent work
     setback" is right. "Explain why the Q3 vendor migration slipped" is wrong -- a topic-loaded
     question makes the pair solvable from content, and the adapter learns topics.

The optional back-and-forth pass (EMNLP Findings 2024) has the generator critique and rewrite
its own question before the pair is kept, which measurably raises pair quality.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from tqdm import tqdm

GEN_SYSTEM = """\
You write training prompts for a writing assistant. You will be shown a passage that a real \
person wrote. Your job is to recover the *generic instruction* that would have caused someone \
to write a passage like it.

Hard rules:
- The instruction must NOT mention any specific name, company, place, project, product, number, \
or event from the passage. Placeholders like <PERSON> or <ORG> must not appear either.
- The instruction must describe the TASK and FORM only: genre, length, register, stance, \
audience. Example: "Write two short paragraphs reflecting on a decision you second-guessed, \
in a casual, thinking-out-loud register."
- Someone who has never read the passage must be able to follow the instruction and write \
something structurally comparable.
- Do NOT describe the author's writing style in detail. The model is learning the style from \
the target text, so telling it the style in the prompt teaches it to obey instructions instead.
- No preamble. Output only the instructions, one per line, no numbering."""

REWRITE_SYSTEM = """\
You audit training prompts. You are given a passage and a candidate instruction. Reply with \
JSON only: {"leaks_content": bool, "too_stylistic": bool, "verdict": "keep"|"rewrite"|"drop", \
"rewritten": "<improved instruction, or empty>"}.

- leaks_content: the instruction references a specific topic, entity, number or event from the \
passage.
- too_stylistic: the instruction spells out the author's voice/tone in detail rather than the task.
- verdict "drop": the passage is not a plausible response to any generic instruction (it is a \
fragment, a list, boilerplate, or incoherent out of context)."""

# Words that mean the question leaked the passage's topic.
_TOPIC_LEAK = re.compile(r"<[A-Z]+>|\b(?:the (?:above|following|passage|text|excerpt))\b", re.I)


class Backtranslator:
    """Wraps the API generator. `offline=True` uses deterministic templates (demo/CI only)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        offline: bool = False,
        max_tokens: int = 400,
        temperature: float = 1.0,
    ) -> None:
        self.model = model or os.environ.get("WLM_BACKTRANSLATE_MODEL", "claude-sonnet-4-5")
        self.offline = offline
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        if not offline:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Set it in .env, or pass offline=True to use "
                    "template questions (fine for smoke tests, not for a real run)."
                )
            import anthropic

            self._client = anthropic.Anthropic()

    # ---------------------------------------------------------------- generation
    def _call(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def questions_for(self, passage: str, n: int = 2, register: str = "unknown") -> list[str]:
        if self.offline:
            return _template_questions(passage, n=n, register=register)
        user = (
            f"Register of the passage: {register}.\n"
            f"Produce {n} different generic instructions that could each have elicited a passage "
            f"like this one.\n\n<passage>\n{passage}\n</passage>"
        )
        raw = self._call(GEN_SYSTEM, user)
        qs = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in raw.split("\n")]
        qs = [q for q in qs if len(q.split()) >= 4 and not _TOPIC_LEAK.search(q)]
        return qs[:n]

    def audit(self, passage: str, question: str) -> dict[str, Any]:
        if self.offline:
            return {"verdict": "keep", "rewritten": "", "leaks_content": False}
        raw = self._call(
            REWRITE_SYSTEM,
            f"<passage>\n{passage}\n</passage>\n\n<instruction>\n{question}\n</instruction>",
        )
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"verdict": "keep", "rewritten": "", "leaks_content": False}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"verdict": "keep", "rewritten": "", "leaks_content": False}


def _template_questions(passage: str, *, n: int, register: str) -> list[str]:
    """Offline fallback. Deterministic, content-blind, keyed on shape only."""
    words = len(passage.split())
    paras = passage.count("\n\n") + 1
    length = "a short paragraph" if words < 110 else (
        "two or three paragraphs" if words < 240 else "several paragraphs"
    )
    tone = {
        "informal": "in a casual, thinking-out-loud register",
        "formal": "in a considered, professional register",
    }.get(register, "in your natural register")
    bank = [
        f"Write {length} {tone} reflecting on something you have been turning over lately.",
        f"Respond in {length} {tone}, explaining your current thinking on a decision you face.",
        f"Write {length} {tone} describing an experience and what you took away from it.",
        f"In {paras} paragraph(s) {tone}, argue for a position you hold and acknowledge the "
        "strongest objection to it.",
        f"Write {length} {tone} giving someone advice on a problem you have dealt with before.",
    ]
    idx = len(passage) % len(bank)
    return [bank[(idx + i) % len(bank)] for i in range(min(n, len(bank)))]


def build_pairs(
    chunks: list[dict[str, Any]],
    *,
    n_per_chunk: int = 2,
    offline: bool = False,
    model: str | None = None,
    audit: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """chunks -> (question, chunk) pairs. Multiple questions per chunk augments scarce data."""
    bt = Backtranslator(model=model, offline=offline)
    pairs: list[dict[str, Any]] = []
    stats = {"chunks": len(chunks), "dropped_by_audit": 0, "rewritten": 0, "no_question": 0}

    for c in tqdm(chunks, desc="backtranslate"):
        try:
            qs = bt.questions_for(c["text"], n=n_per_chunk, register=c.get("register", "unknown"))
        except Exception as e:  # keep going; one bad chunk shouldn't kill a corpus run
            print(f"[backtranslate] {c['chunk_id']}: {type(e).__name__}: {e}")
            continue
        if not qs:
            stats["no_question"] += 1
            continue
        for j, q in enumerate(qs):
            if audit:
                a = bt.audit(c["text"], q)
                if a.get("verdict") == "drop":
                    stats["dropped_by_audit"] += 1
                    continue
                if a.get("verdict") == "rewrite" and a.get("rewritten"):
                    q = a["rewritten"]
                    stats["rewritten"] += 1
            pairs.append(
                {
                    "pair_id": f"{c['chunk_id']}-q{j}",
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "register": c.get("register", "unknown"),
                    "prompt": q,
                    "response": c["text"],
                    "n_words": c.get("n_words", len(c["text"].split())),
                    "generator": "offline-template" if offline else bt.model,
                }
            )
    stats["pairs"] = len(pairs)
    return pairs, stats
