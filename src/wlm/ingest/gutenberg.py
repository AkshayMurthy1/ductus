"""Project Gutenberg extraction helpers, shared by the corpus builders.

Generalized from scripts/build_dev_corpus.py (which stays frozen as the exact producer of the
committed Chesterton fixture). CPU-only, stdlib + wlm.chunk only. Every function here exists to
keep one property: **nothing that is not the named author's own prose enters a corpus** —
editorial front matter, transcriber notes, tables of contents, verse, and footnotes all carry
someone else's voice, and a corpus that mixes voices poisons both the training signal and the
verifier's negative class.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

SKIP_HEADINGS = {"PREFACE", "CONTENTS", "CONTENTS:", "INDEX", "FOOTNOTES", "THE AUTHOR",
                 "SOURCE", "TABLE OF CONTENTS", "INTRODUCTION", "NOTE", "DEDICATION",
                 "TRANSCRIBER'S NOTE", "APPENDIX", "POSTSCRIPT", "PREFATORY NOTE",
                 "ILLUSTRATIONS", "LIST OF ILLUSTRATIONS"}
ROMAN = re.compile(r"^[IVXLC]{1,6}\.\s+\S")
CHAPTER = re.compile(r"^CHAPTER [IVXLC0-9]{1,8}\.?(\s|$)")
ALLCAP = re.compile(r"^[A-Z][A-Z0-9 ’'“”,&\-—:\.\?!]{4,60}$")
# Front matter written by editors and transcribers is not the named author's voice.
EDITORIAL = re.compile(
    r"this (volume|text|edition|collection) (contains|includes)|the present (edition|volume|text)"
    r"|reliability\s*:|transcriber|project gutenberg|e-?text|proofread|scanned|public domain"
    r"|editor'?s? (note|preface)|copyright|all rights reserved", re.I)
FOOTNOTE_LINE = re.compile(r"^\s*(\[|footnote|produced by)", re.I)


def fetch(pid: int, cache_dir: Path) -> str:
    """Fetch one book's plain text, cached — Gutenberg rate-limits bursts, so retry gently."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    hit = cache_dir / f"pg{pid}.txt"
    if hit.exists():
        return hit.read_text(encoding="utf-8", errors="replace")
    url = f"https://www.gutenberg.org/cache/epub/{pid}/pg{pid}.txt"
    req = Request(url, headers={"User-Agent": "ductus-corpus-builder/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
            hit.write_text(raw, encoding="utf-8")
            return raw
        except Exception as e:  # noqa: PERF203 — transient network errors
            if attempt == 2:
                raise RuntimeError(f"could not fetch PG #{pid}: {e}") from e
            time.sleep(2 * (attempt + 1))
    return ""


def verify_title(raw: str, pid: int, expected: str) -> None:
    """Fail loudly if an ID points at the wrong book — a silently-wrong distractor adds a
    phantom author to the negative class, and the AUC would never tell you."""
    head = raw[:2000].lower()
    if expected.lower() not in head:
        raise RuntimeError(f"PG #{pid}: expected title {expected!r} not found in header")


def strip_pg(raw: str) -> str:
    if (m := re.search(r"\*\*\*\s*START OF.*?\*\*\*", raw, re.S)):
        raw = raw[m.end():]
    if (m := re.search(r"\*\*\*\s*END OF.*?\*\*\*", raw, re.S)):
        raw = raw[:m.start()]
    return raw


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or s.upper() in SKIP_HEADINGS or len(s) > 62:
        return False
    if s.startswith(("“", '"', "‘", "_", ".")) or set(s) <= set(". "):
        return False
    return bool(ROMAN.match(s) or CHAPTER.match(s.upper()) or ALLCAP.match(s))


def is_prose(block: str) -> bool:
    """Reject tables of contents and verse: prose has sentences, not a stack of short lines."""
    lines = [line for line in block.split("\n") if line.strip()]
    if not lines or sum(1 for line in lines if len(line.split()) <= 5) / len(lines) > 0.5:
        return False
    words = block.split()
    if (block.count(".") + block.count("?") + block.count("!")) < len(words) / 45:
        return False
    return not EDITORIAL.search(block)


def _clean(text: str, drop_indented_matching: re.Pattern | None = None) -> str:
    # Bracketed editorial insertions ("[Dictated December 1906.]", "[Laughter.]") are not the
    # author's prose; footnote paragraphs likewise.
    text = re.sub(r"^\[[^\]\n]{1,120}\]\s*$", "", text, flags=re.M)
    paras = [p for p in re.split(r"\n\s*\n", text) if not FOOTNOTE_LINE.match(p)]
    if drop_indented_matching is not None:
        # Editors annotate collected volumes with indented third-person notes ("Mr. Clemens
        # was introduced by..."). Drop indented paragraphs matching the caller's third-person
        # pattern — indentation alone is not enough, because authors also indent their own
        # quoted letters, which ARE their prose.
        paras = [
            p for p in paras
            if not (all(ln.startswith((" ", "\t")) for ln in p.split("\n") if ln.strip())
                    and drop_indented_matching.search(p))
        ]
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(paras)).strip()


def split_sections(body: str, *, min_words: int = 250, max_words: int = 6000,
                   drop_indented_matching: re.Pattern | None = None
                   ) -> list[tuple[str, str]]:
    """(title, text) per chapter/essay/speech. Sections outside the word bounds are dropped:
    under it they are stubs, over it the heading pattern missed the volume and the 'section'
    is the whole book, which would dominate a document-level split."""
    lines = body.split("\n")
    marks = [(i, ln.strip()) for i, ln in enumerate(lines)
             if is_heading(ln) and (i == 0 or not lines[i - 1].strip())]
    # Each heading appears twice — once in the table of contents, once in the body. Keeping the
    # later occurrence drops the whole TOC without needing to locate it.
    last: dict[str, tuple[int, str]] = {}
    for i, title in marks:
        last[re.sub(r"\s+", " ", title).lower()] = (i, title)

    out = []
    ordered = sorted(last.values())
    for (a, title), (b, _) in zip(ordered, ordered[1:] + [(len(lines), "")], strict=True):
        text = _clean("\n".join(lines[a + 1:b]), drop_indented_matching)
        if min_words <= len(text.split()) <= max_words and is_prose(text):
            out.append((title, text))
    return out


def segment_fallback(body: str, *, target_words: int = 1800,
                     drop_indented_matching: re.Pattern | None = None
                     ) -> list[tuple[str, str]]:
    """When a volume's headings don't match any pattern, fall back to ~target_words segments at
    paragraph boundaries. Segments are honest 'documents' for split purposes — what matters
    downstream is that sibling chunks share a doc_id, not that a title exists."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", _clean(body, drop_indented_matching))
             if p.strip() and is_prose(p) and not is_heading(p.split("\n")[0])]
    out, buf, n = [], [], 0
    for p in paras:
        buf.append(p)
        n += len(p.split())
        if n >= target_words:
            out.append((f"segment {len(out) + 1:02d}", "\n\n".join(buf)))
            buf, n = [], 0
    if n >= 250:
        out.append((f"segment {len(out) + 1:02d}", "\n\n".join(buf)))
    return out


def distractor_windows(body: str, who: str, *, per_author: int = 20,
                       min_words: int = 60, max_words: int = 320) -> list[tuple[str, str]]:
    """Evenly-spaced prose windows from one book — the dev corpus's negative-class recipe."""
    from wlm.chunk import chunk_document

    paras = [p for p in re.split(r"\n\s*\n", strip_pg(body) if "*** START" in body else body)
             if len(p.split()) >= 40 and not is_heading(p.strip().split("\n")[0])
             and is_prose(p) and not FOOTNOTE_LINE.match(p.strip())]
    paras = paras[max(2, len(paras) // 8):]  # skip prefaces and dedications
    if not paras:
        return []
    step = max(1, len(paras) // (per_author + 6))
    out: list[tuple[str, str]] = []
    for j in range(0, len(paras), step):
        windows = [c.strip() for c in
                   chunk_document("\n\n".join(paras[j:j + 3]),
                                  min_words=min_words, max_words=max_words)
                   if min_words <= len(c.split()) <= max_words]
        if windows:
            out.append((f"{who}-{len(out) + 1:02d}", windows[0]))
        if len(out) >= per_author:
            break
    return out
