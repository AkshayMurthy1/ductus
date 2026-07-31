#!/usr/bin/env python3
"""Build the DEVELOPMENT FIXTURE corpus: G.K. Chesterton vs 21 other essayists.

This is not a real corpus and produces nothing shippable. It exists so the CPU pipeline and the
eval harness can be exercised at realistic scale without anyone's private writing -- roughly
256k author words across 148 documents, against 420 windows from 21 distinct public-domain
essayists.

Why a fixture this large: the authorship verifier is only meaningful when it is stable. On a
5k-word corpus the AUC swung 0.26 between the balanced and unbalanced fits; here the two agree
to within 0.003. If you are changing anything that touches the AV, verify against this first.

Why Chesterton: a very distinctive voice (paradox, antithesis, long periodic sentences) attached
to a large body of *short* essays, so one essay maps cleanly to one document.

Why these distractors: same era, same genre, 21 different authors. A negative class that differs
by century or genre gives a high AUC for the wrong reason -- see data/raw/distractor/README.md.

    python scripts/build_dev_corpus.py            # writes into data/raw/{author,distractor}
    python scripts/build_dev_corpus.py --clean    # remove the fixture instead

Everything it writes is gitignored. Delete it before a real run; a corpus mixing one real person
with Chesterton is worse than either alone, because the adapter learns the average of two voices.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
AUTHOR = ROOT / "data/raw/author/formal"
DISTR = ROOT / "data/raw/distractor"
CACHE = ROOT / "data/interim/_gutenberg_cache"

# Chesterton essay collections. Volumes whose headings do not match either pattern below come
# through as one enormous "essay" and are dropped by the word-count bound.
AUTHOR_BOOKS = {
    470: "heretics", 2015: "miscellany-of-men", 8092: "tremendous-trifles",
    11505: "all-things-considered", 12245: "the-defendant",
    60057: "uses-of-diversity", 60164: "fancies-versus-fads",
}

# One collection per author, so every distractor file has a known and distinct provenance.
DISTRACTOR_BOOKS = {
    10343: "lamb", 386: "stevenson", 7432: "belloc", 77244: "arnold", 4652: "benson",
    59430: "repplier", 70: "twain", 2791: "addison", 1276: "meynell", 5177: "burroughs",
    11675: "legallienne", 12244: "birrell", 1463: "gissing", 4036: "pater",
    13088: "chapman", 39447: "more", 17452: "quiller-couch", 9846: "thoreau",
    2944: "emerson", 3020: "hazlitt", 18948: "mencken",
}

SKIP_HEADINGS = {"PREFACE", "CONTENTS", "CONTENTS:", "INDEX", "FOOTNOTES", "THE AUTHOR",
                 "SOURCE", "TABLE OF CONTENTS", "INTRODUCTION", "NOTE", "DEDICATION",
                 "TRANSCRIBER'S NOTE", "APPENDIX", "POSTSCRIPT"}
ROMAN = re.compile(r"^[IVXLC]{1,6}\.\s+\S")
ALLCAP = re.compile(r"^[A-Z][A-Z0-9 ’'“”,&\-—:\.\?!]{4,60}$")
# Front matter written by editors and transcribers is not the named author's voice; letting it
# through would add a phantom extra "author" to the negative class.
EDITORIAL = re.compile(
    r"this (volume|text|edition|collection) (contains|includes)|the present (edition|volume|text)"
    r"|reliability\s*:|transcriber|project gutenberg|e-?text|proofread|scanned|public domain"
    r"|editor'?s? (note|preface)|was the last male descendant|is best known", re.I)


def fetch(pid: int) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = CACHE / f"pg{pid}.txt"
    if hit.exists():
        return hit.read_text(encoding="utf-8", errors="replace")
    url = f"https://www.gutenberg.org/cache/epub/{pid}/pg{pid}.txt"
    req = Request(url, headers={"User-Agent": "ductus-dev-corpus/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
            hit.write_text(raw, encoding="utf-8")
            return raw
        except Exception as e:  # transient; Gutenberg rate-limits bursts
            if attempt == 2:
                raise RuntimeError(f"could not fetch PG #{pid}: {e}") from e
            time.sleep(2 * (attempt + 1))
    return ""


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
    return bool(ROMAN.match(s) or ALLCAP.match(s))


def is_prose(block: str) -> bool:
    """Reject tables of contents and verse: prose has sentences, not a stack of short lines."""
    lines = [line for line in block.split("\n") if line.strip()]
    if not lines or sum(1 for line in lines if len(line.split()) <= 5) / len(lines) > 0.5:
        return False
    words = block.split()
    if (block.count(".") + block.count("?") + block.count("!")) < len(words) / 45:
        return False
    return not EDITORIAL.search(block)


def split_essays(body: str) -> list[tuple[str, str]]:
    lines = body.split("\n")
    marks = [(i, ln.strip()) for i, ln in enumerate(lines)
             if is_heading(ln) and (i == 0 or not lines[i - 1].strip())]
    # Each heading appears twice -- once in the table of contents, once in the body. Keep the
    # later one, which drops the whole TOC without needing to locate it.
    last: dict[str, tuple[int, str]] = {}
    for i, title in marks:
        last[re.sub(r"\s+", " ", title).lower()] = (i, title)

    out = []
    ordered = sorted(last.values())
    for (a, title), (b, _) in zip(ordered, ordered[1:] + [(len(lines), "")], strict=True):
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines[a + 1:b]).strip())
        n = len(text.split())
        # Over 6k words means the heading pattern missed this volume and we captured the whole
        # book as one "essay" -- that would dominate the document-level split.
        if 250 <= n <= 6000:
            out.append((title, text))
    return out


def build() -> None:
    from wlm.chunk import chunk_document

    AUTHOR.mkdir(parents=True, exist_ok=True)
    DISTR.mkdir(parents=True, exist_ok=True)
    for f in AUTHOR.glob("*.md"):
        f.unlink()
    for f in DISTR.glob("*.txt"):
        f.unlink()

    total = 0
    for pid, slug in AUTHOR_BOOKS.items():
        essays = split_essays(strip_pg(fetch(pid)))
        for title, text in essays:
            name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or f"essay{total}"
            path = AUTHOR / f"{slug}__{name}.md"
            if not path.exists():
                path.write_text(text + "\n", encoding="utf-8")
                total += 1
        print(f"  {slug:<24} {len(essays):>3} essays")
    words = sum(len(p.read_text(encoding='utf-8').split()) for p in AUTHOR.glob("*.md"))
    print(f"AUTHOR (Chesterton): {total} documents, {words:,} words -> {AUTHOR}")

    items: list[tuple[str, str]] = []
    for pid, who in DISTRACTOR_BOOKS.items():
        body = strip_pg(fetch(pid))
        paras = [p for p in re.split(r"\n\s*\n", body)
                 if len(p.split()) >= 40 and not is_heading(p.strip().split("\n")[0])
                 and is_prose(p)
                 and not re.match(r"^\s*(\[|footnote|produced by)", p.strip(), re.I)]
        paras = paras[max(2, len(paras) // 8):]      # skip prefaces and dedications
        step = max(1, len(paras) // 26)
        taken = 0
        for j in range(0, len(paras), step):
            windows = [c.strip() for c in
                       chunk_document("\n\n".join(paras[j:j + 3]), min_words=60, max_words=320)
                       if 60 <= len(c.split()) <= 320]
            if windows:
                items.append((f"{who}-{taken + 1:02d}", windows[0]))
                taken += 1
            if taken >= 20:
                break

    authors = {slug.rsplit("-", 1)[0] for slug, _ in items}
    assert not any("chesterton" in a for a in authors), \
        "the author must never appear in their own negative class"
    # Interleave so alphabetical order mixes authors rather than clustering them.
    seen: dict[str, int] = {}
    ranked = []
    for slug, text in items:
        who = slug.rsplit("-", 1)[0]
        k = seen.get(who, 0)
        seen[who] = k + 1
        ranked.append(((k + 0.5) / 20, who, slug, text))
    ranked.sort(key=lambda r: (r[0], r[1]))
    for i, (_, _, slug, text) in enumerate(ranked, 1):
        (DISTR / f"{i:03d}_{slug}.txt").write_text(text + "\n", encoding="utf-8")

    dwords = sum(len(t.split()) for _, t in items)
    print(f"DISTRACTOR: {len(items)} windows, {dwords:,} words, "
          f"{len(authors)} authors -> {DISTR}")
    print("\nThis is a development fixture. Delete it before training on real writing.")


def clean() -> None:
    n = 0
    for pattern, base in ((("*.md",), AUTHOR), (("*.txt",), DISTR)):
        for pat in pattern:
            for f in base.glob(pat):
                f.unlink()
                n += 1
    print(f"removed {n} fixture file(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clean", action="store_true", help="delete the fixture instead of building")
    args = ap.parse_args()
    clean() if args.clean else build()
