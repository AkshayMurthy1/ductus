"""Segment documents into response-sized units.

Plan §4.2: chunk quality bounds target quality, so this is evaluated, not assumed. A chunk
should read like a complete answer someone could have written to a single prompt -- if it
starts mid-argument, the backtranslated question will be wrong and the pair is noise.

Strategy, cheapest-first:
  1. respect blank-line paragraph boundaries
  2. greedily pack paragraphs to chunk_min_words..chunk_max_words
  3. never split a sentence; prefer breaking where a discourse marker starts a paragraph
"""

from __future__ import annotations

import re
from typing import Any

# Paragraphs opening with these are continuations -- gluing them to the previous chunk keeps
# an argument intact instead of handing the trainer a fragment.
CONTINUATION_MARKERS = re.compile(
    r"^\s*(but|and|so|however|that said|still|also|besides|moreover|furthermore|then|"
    r"which is why|either way|anyway|of course|in other words|for example|for instance|"
    r"more importantly|the point is|it|this|that|they|he|she|we)\b",
    re.I,
)

# Python's re requires fixed-width lookbehind, so the optional closing quote/bracket is written
# as two alternatives rather than `[.!?]["')\]]?`.
SENT_END = re.compile(r"(?<=[.!?][\"')\]])\s+(?=[A-Z\"'(\[])|(?<=[.!?])\s+(?=[A-Z\"'(\[])")


# Abbreviations whose final period is never a sentence end.
_ABBR_FULL = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "e.g.", "i.e.", "cf.",
    "Inc.", "Ltd.", "No.", "Fig.", "approx.", "Ave.", "Rd.",
)
# Abbreviations with *internal* periods whose final period MAY end a sentence
# ("...at 9 a.m. She was late."). Only the internal dots are protected.
_ABBR_INTERNAL = ("a.m.", "p.m.", "U.S.", "U.K.", "U.N.", "D.C.", "Ph.D.", "M.D.")


def split_sentences(text: str) -> list[str]:
    """Regex sentence split with abbreviation guards. Good enough; keeps deps light.

    Known limitation: an internal-period abbreviation followed by a capitalized word
    ("in the U.S. Congress") splits incorrectly. Rare in first-person prose, and the cost is one
    short chunk rather than a wrong training pair.
    """
    protected = text
    for abbr in _ABBR_FULL:
        protected = protected.replace(abbr, abbr.replace(".", "\x00"))
    for abbr in _ABBR_INTERNAL:
        head, _, _ = abbr.rpartition(".")
        protected = protected.replace(abbr, head.replace(".", "\x00") + ".")
    sents = [s.replace("\x00", ".").strip() for s in SENT_END.split(protected) if s]
    return [s for s in sents if s]


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Single-newline-separated prose is common in Docs exports; if we got one giant paragraph,
    # fall back to sentence packing.
    if len(paras) == 1 and len(paras[0].split()) > 400:
        sents = split_sentences(paras[0])
        paras = [" ".join(sents[i : i + 5]) for i in range(0, len(sents), 5)]
    return paras


def chunk_document(
    text: str,
    *,
    min_words: int = 60,
    max_words: int = 320,
) -> list[str]:
    paras = _paragraphs(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf, buf_words = [], 0

    for para in paras:
        pw = len(para.split())

        # A single paragraph longer than max_words gets sentence-packed on its own.
        if pw > max_words:
            flush()
            sents = split_sentences(para)
            cur: list[str] = []
            cw = 0
            for s in sents:
                sw = len(s.split())
                if cw + sw > max_words and cw >= min_words:
                    chunks.append(" ".join(cur))
                    cur, cw = [], 0
                cur.append(s)
                cw += sw
            if cur:
                if cw < min_words and chunks:
                    chunks[-1] = chunks[-1] + " " + " ".join(cur)
                else:
                    chunks.append(" ".join(cur))
            continue

        starts_continuation = bool(CONTINUATION_MARKERS.match(para))
        if buf_words >= min_words and not starts_continuation and buf_words + pw > max_words:
            flush()
        buf.append(para)
        buf_words += pw
        if buf_words >= max_words:
            flush()

    flush()

    # Drop or merge runts: a 20-word chunk teaches formatting, not voice.
    out: list[str] = []
    for c in chunks:
        if len(c.split()) < min_words and out:
            out[-1] = out[-1] + "\n\n" + c
        elif len(c.split()) >= min_words:
            out.append(c)
    return out


def chunk_records(
    docs: list[dict[str, Any]],
    *,
    min_words: int = 60,
    max_words: int = 320,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in docs:
        for i, c in enumerate(chunk_document(d["text"], min_words=min_words, max_words=max_words)):
            rows.append(
                {
                    "chunk_id": f"{d['doc_id']}-{i:03d}",
                    "doc_id": d["doc_id"],
                    "source": d.get("source", ""),
                    "register": d.get("register", "unknown"),
                    "n_words": len(c.split()),
                    "text": c,
                }
            )
    return rows


def chunk_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report you should actually read before training. Bad chunking is invisible downstream."""
    if not rows:
        return {"n": 0}
    words = [r["n_words"] for r in rows]
    starts_mid = sum(1 for r in rows if CONTINUATION_MARKERS.match(r["text"]))
    ends_mid = sum(1 for r in rows if not re.search(r"[.!?][\"')\]]?\s*$", r["text"]))
    return {
        "n": len(rows),
        "words_mean": round(sum(words) / len(words), 1),
        "words_min": min(words),
        "words_max": max(words),
        "total_words": sum(words),
        "frac_starts_mid_thought": round(starts_mid / len(rows), 3),
        "frac_ends_unterminated": round(ends_mid / len(rows), 3),
        "n_docs": len({r["doc_id"] for r in rows}),
        "registers": {
            reg: sum(1 for r in rows if r["register"] == reg)
            for reg in sorted({r["register"] for r in rows})
        },
    }
