#!/usr/bin/env python3
"""Build a registry-defined extra-author corpus into a WLM_ROOT tree (docs/EXPANSION.md Tier 2).

The generality expansion needs corpora that are as reproducible as the Chesterton fixture:
every author here is US public domain, every book is a pinned Project Gutenberg ID with a
verified title, and the extraction is deterministic — so an extra-author result can be audited
from this script alone, exactly like `build_dev_corpus.py` (which stays frozen as the producer
of the committed dev fixture).

    python scripts/new_author.py ~/authors/twain --name "Mark Twain" --register informal
    python scripts/build_author_corpus.py twain --root ~/authors/twain

Why Twain is the first expansion author: he varies every axis the dev author holds fixed —
informal first-person American prose (autobiography, travel narration, speeches) against
Chesterton's formal Edwardian essay — while having a large, cleanly-extractable body of work.
The distractor class is register-matched the same way the dev corpus's is: American
first-person memoir and travel writing of the same broad era, one book per author, so the
verifier has to learn *this* voice rather than "memoir-shaped prose". Franklin, Irving,
Thoreau, Parkman and Dana sit earlier than the rest — if the fitted AUC is suspiciously high,
drop them first ("learned the era" is the confound the dev corpus README warns about).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.ingest.gutenberg import (  # noqa: E402
    distractor_windows,
    fetch,
    segment_fallback,
    split_sections,
    strip_pg,
    verify_title,
)

# One registry entry per expansion author. Books are (gutenberg_id, slug, expected_title);
# titles are asserted at fetch time so a wrong ID fails loudly instead of poisoning a class.
AUTHORS: dict[str, dict] = {
    "twain": {
        "display": "Mark Twain",
        "register": "informal",
        # The 1910 Speeches volume frames each speech with an indented third-person editor's
        # note; the autobiography's indented blocks are Twain's own quoted letters and stay.
        "editorial_third_person": r"Mr\. Clemens|Mark Twain|the following (speech|address|"
                                  r"remarks)|was introduced by|responded to the toast",
        # Priority order: the most register-pure sources first; the build stops adding books
        # once --max-author-words is reached, keeping whole documents.
        "author_books": [
            (19987, "autobiography", "Chapters from My Autobiography"),
            (3188, "speeches", "Mark Twain's Speeches"),
            (3176, "innocents-abroad", "The Innocents Abroad"),
            (3177, "roughing-it", "Roughing It"),
            (245, "life-on-mississippi", "Life on the Mississippi"),
            (119, "tramp-abroad", "A Tramp Abroad"),
        ],
        "distractor_books": [
            (23, "douglass", "Narrative of the Life of Frederick Douglass"),
            (2376, "washington", "Up from Slavery"),
            (2397, "keller", "The Story of My Life"),
            (2044, "adams", "The Education of Henry Adams"),
            (3335, "roosevelt", "Theodore Roosevelt: An Autobiography"),
            (17976, "carnegie", "Autobiography of Andrew Carnegie"),
            (2055, "dana", "Two Years Before the Mast"),
            (148, "franklin", "The Autobiography of Benjamin Franklin"),
            (20885, "antin", "The Promised Land"),
            (8813, "whitman", "Complete Prose Works"),
            (4367, "grant", "Personal Memoirs of U. S. Grant"),
            (32540, "muir", "My First Summer in the Sierra"),
            (2293, "larcom", "A New England Girlhood"),
            (1015, "parkman", "The Oregon Trail"),
            (205, "thoreau", "Walden"),
            (2048, "irving", "The Sketch-Book of Geoffrey Crayon"),
            (4389, "moodie", "Roughing It in the Bush"),
        ],
    },
}


def build(author: str, root: Path, max_author_words: int) -> int:
    spec = AUTHORS[author]
    author_dir = root / "data" / "raw" / "author" / spec["register"]
    distr_dir = root / "data" / "raw" / "distractor"
    cache = root / "data" / "interim" / "_gutenberg_cache"
    author_dir.mkdir(parents=True, exist_ok=True)
    distr_dir.mkdir(parents=True, exist_ok=True)

    assert not any(author in slug for _, slug, _ in spec["distractor_books"]), \
        "the author must never appear in their own negative class"
    editorial = re.compile(spec["editorial_third_person"]) \
        if spec.get("editorial_third_person") else None

    for f in author_dir.glob("*.md"):  # rebuilds are clean, never additive
        f.unlink()
    total_docs, total_words = 0, 0
    for pid, slug, expected in spec["author_books"]:
        if total_words >= max_author_words:
            print(f"  {slug:<24} skipped — word budget reached")
            continue
        raw = fetch(pid, cache)
        verify_title(raw, pid, expected)
        sections = split_sections(strip_pg(raw), drop_indented_matching=editorial)
        if len(sections) < 3:  # headings didn't match this volume's layout
            sections = segment_fallback(strip_pg(raw), drop_indented_matching=editorial)
        n_docs = 0
        for title, text in sections:
            if total_words >= max_author_words:
                break
            name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] \
                or f"doc{total_docs}"
            path = author_dir / f"{slug}__{name}.md"
            if not path.exists():
                path.write_text(text + "\n", encoding="utf-8")
                total_docs += 1
                n_docs += 1
                total_words += len(text.split())
        print(f"  {slug:<24} {n_docs:>3} docs")
    print(f"AUTHOR ({spec['display']}): {total_docs} documents, {total_words:,} words "
          f"-> {author_dir}")

    items: list[tuple[str, str]] = []
    for pid, who, expected in spec["distractor_books"]:
        raw = fetch(pid, cache)
        verify_title(raw, pid, expected)
        got = distractor_windows(strip_pg(raw), who)
        if len(got) < 8:
            print(f"  WARNING {who}: only {len(got)} windows — thin representation")
        items += got

    # Interleave so alphabetical order mixes authors rather than clustering them (dev recipe).
    seen: dict[str, int] = {}
    ranked = []
    for slug, text in items:
        who = slug.rsplit("-", 1)[0]
        k = seen.get(who, 0)
        seen[who] = k + 1
        ranked.append(((k + 0.5) / 20, who, slug, text))
    ranked.sort(key=lambda r: (r[0], r[1]))
    for f in distr_dir.glob("*.txt"):
        f.unlink()
    for i, (_, _, slug, text) in enumerate(ranked, 1):
        (distr_dir / f"{i:03d}_{slug}.txt").write_text(text + "\n", encoding="utf-8")

    dwords = sum(len(t.split()) for _, t in items)
    authors = {slug.rsplit("-", 1)[0] for slug, _ in items}
    print(f"DISTRACTOR: {len(items)} windows, {dwords:,} words, {len(authors)} authors "
          f"-> {distr_dir}")
    if total_words < 60_000:
        print("WARNING author corpus under 60k words — the 50k sweep arm will not exist and "
              "the cliff can only be bracketed, not located.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("author", choices=sorted(AUTHORS))
    ap.add_argument("--root", required=True,
                    help="author root from scripts/new_author.py (outside the repo)")
    ap.add_argument("--max-author-words", type=int, default=150_000,
                    help="stop adding author documents past this budget (API cost control: "
                         "backtranslation is ~2 paid calls per chunk)")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if root == REPO or REPO in root.parents:
        print("author roots live outside the repo — see scripts/new_author.py")
        return 1
    if not (root / "RUNBOOK.md").exists():
        print(f"{root} is not a scaffolded author root — run scripts/new_author.py first")
        return 1
    return build(args.author, root, args.max_author_words)


if __name__ == "__main__":
    raise SystemExit(main())
