#!/usr/bin/env python3
"""Build an informal-register author corpus from the Blog Authorship Corpus (Schler et al. 2006).

This is the expansion's true informal-register cell (docs/EXPANSION.md): real casual
first-person prose — the product's hard case — from the corpus the closest accepted
style-imitation work used. It is NOT public domain (research use only), and the bloggers are
real, possibly-living people, so the rules differ from the Gutenberg builders:

  - the corpus root must be scaffolded with `scripts/new_author.py <root> --private` and live
    OUTSIDE the repo; nothing under it is ever committed anywhere;
  - author identity stays the corpus's anonymous numeric ID — never attempt deanonymization;
  - reproducibility is by recipe, not by data: this script + the archive checksum + the
    blogger ID fully determine the corpus, and the archive is the standard research release
    (hosted at huggingface.co/datasets/barilan/blog_authorship_corpus).

Archive format: blogs.zip contains one pseudo-XML file per blogger, named
<id>.<gender>.<age>.<industry>.<sign>.xml, with <post>...</post> bodies (dirty encodings,
`urlLink` artifacts, raw HTML entities — cleaned here before anything downstream sees them).

    python scripts/build_bac_corpus.py --root ~/authors/blogger --list          # rank candidates
    python scripts/build_bac_corpus.py --root ~/authors/blogger --blogger 12345 # build one

Distractors are other high-volume bloggers from the same corpus — register-matched by
construction, which is exactly what the verifier needs its negative class to be.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARCHIVE_URL = ("https://huggingface.co/datasets/barilan/blog_authorship_corpus/"
               "resolve/main/data/blogs.zip")
POST = re.compile(r"<post>(.*?)</post>", re.S)
# Corpus artifacts: the crawler replaced links with the literal token `urlLink`.
NOISE = re.compile(r"\burlLink\b|&nbsp;?")
FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'll|I'd|me|my|mine|myself)\b")
MIN_POST_WORDS = 60     # below the chunker's minimum, a post can't become a training unit
MAX_POST_WORDS = 6000   # runaway concatenation artifacts, not real posts
MIN_AGE = 18            # minors are excluded outright — author AND distractor pools
# First-person pronouns per 100 words. Diary-style blogs run ~3-6; quote-blogs (posts that
# are mostly copied hymns, creeds, news, lyrics) run under ~1.5 and would teach the QUOTED
# authors' voices, not the blogger's. The 942828 case that motivated this: a devotional blog
# whose posts were largely Newman and Southwell quotations.
MIN_FP_RATE = 2.5


def clean(text: str) -> str:
    text = html.unescape(NOISE.sub(" ", text))
    text = re.sub(r"<[^>]{1,80}>", " ", text)      # stray inline HTML
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def posts_from(zf: zipfile.ZipFile, name: str) -> list[str]:
    raw = zf.read(name).decode("latin-1", errors="replace")  # encodings are unreliable
    out = []
    for m in POST.finditer(raw):
        p = clean(m.group(1))
        if MIN_POST_WORDS <= len(p.split()) <= MAX_POST_WORDS:
            out.append(p)
    return out


def scan(archive: Path, cache: Path) -> list[dict]:
    """Rank every blogger by usable words. Cached — the scan reads the whole 800MB corpus."""
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith(".xml")]
        for i, name in enumerate(names):
            if i % 2000 == 0:
                print(f"  scanning {i}/{len(names)}...", flush=True)
            parts = Path(name).name.split(".")
            if len(parts) < 5 or not parts[2].isdigit() or int(parts[2]) < MIN_AGE:
                continue
            posts = posts_from(zf, name)
            words = sum(len(p.split()) for p in posts)
            if words >= 5000:  # below this, not even a distractor candidate
                text = " ".join(posts)
                fp = round(100 * len(FIRST_PERSON.findall(text)) / max(words, 1), 2)
                rows.append({"id": parts[0], "gender": parts[1], "age": parts[2],
                             "industry": parts[3], "file": name,
                             "posts": len(posts), "words": words, "fp_rate": fp})
    rows.sort(key=lambda r: -r["words"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def build(root: Path, archive: Path, blogger_id: str | None, max_words: int,
          n_distractors: int) -> int:
    from wlm.ingest.gutenberg import distractor_windows

    cache = root / "data" / "interim" / "_bac_cache"
    rows = scan(archive, cache / "ranking.json")
    if not rows:
        print("no usable bloggers found in the archive — is it the right blogs.zip?")
        return 1

    if blogger_id:
        row = next((r for r in rows if r["id"] == blogger_id), None)
        if row is None:
            print(f"blogger {blogger_id} not found, a minor, or under the 5k-word floor")
            return 1
        if row["fp_rate"] < MIN_FP_RATE:
            print(f"NOTE blogger {blogger_id} has first-person rate {row['fp_rate']} "
                  f"(< {MIN_FP_RATE}) — likely a quote-blog; inspect the posts before "
                  "spending API budget on it.")
    else:
        row = next((r for r in rows if r["fp_rate"] >= MIN_FP_RATE), None)
        if row is None:
            print("no candidate clears the first-person floor — inspect with --list")
            return 1

    author_dir = root / "data" / "raw" / "author" / "informal"
    distr_dir = root / "data" / "raw" / "distractor"
    author_dir.mkdir(parents=True, exist_ok=True)
    distr_dir.mkdir(parents=True, exist_ok=True)
    for f in [*author_dir.glob("*.md"), *distr_dir.glob("*.txt")]:
        f.unlink()

    with zipfile.ZipFile(archive) as zf:
        posts = posts_from(zf, row["file"])
        total_docs, total_words = 0, 0
        # Each post is one document: posts are the natural authored unit, and the
        # document-level split needs many documents, not a few concatenated slabs.
        for i, p in enumerate(posts, 1):
            if total_words >= max_words:
                break
            (author_dir / f"post{i:04d}.md").write_text(p + "\n", encoding="utf-8")
            total_docs += 1
            total_words += len(p.split())
        print(f"AUTHOR (blogger {row['id']}, {row['gender']}/{row['age']}/"
              f"{row['industry']}): {total_docs} posts, {total_words:,} words -> {author_dir}")

        # Distractors: the next-most-prolific bloggers, author excluded. Same-corpus
        # negatives are register-matched by construction; the first-person floor applies
        # here too (a quote-blog negative would teach the verifier "quotes vs prose", not
        # "this blogger vs other bloggers"). Interleave like the dev recipe.
        items: list[tuple[str, str]] = []
        pool = [r for r in rows
                if r["id"] != row["id"] and r["fp_rate"] >= MIN_FP_RATE][: n_distractors * 3]
        for r in pool:
            if len({s.rsplit("-", 1)[0] for s, _ in items}) >= n_distractors:
                break
            body = "\n\n".join(posts_from(zf, r["file"]))
            items += distractor_windows(body, f"b{r['id']}")

    seen: dict[str, int] = {}
    ranked = []
    for slug, text in items:
        who = slug.rsplit("-", 1)[0]
        k = seen.get(who, 0)
        seen[who] = k + 1
        ranked.append(((k + 0.5) / 20, who, slug, text))
    ranked.sort(key=lambda r: (r[0], r[1]))
    for i, (_, _, slug, text) in enumerate(ranked, 1):
        (distr_dir / f"{i:03d}_{slug}.txt").write_text(text + "\n", encoding="utf-8")
    dwords = sum(len(t.split()) for _, t in items)
    authors = {s.rsplit("-", 1)[0] for s, _ in items}
    print(f"DISTRACTOR: {len(items)} windows, {dwords:,} words, {len(authors)} bloggers "
          f"-> {distr_dir}")

    (root / "PROVENANCE.md").write_text(
        f"# Provenance\n\nBlog Authorship Corpus (Schler, Koppel, Argamon & Pennebaker 2006),\n"
        f"archive: {ARCHIVE_URL}\nAuthor: anonymous blogger ID {row['id']} "
        f"({row['gender']}, {row['age']}, {row['industry']}).\nDistractors: "
        f"{len(authors)} other high-volume bloggers from the same corpus.\n\n"
        "License: research use only — NEVER commit, redistribute, or attempt to\n"
        "deanonymize. This root stays outside version control (repo data policy).\n",
        encoding="utf-8")
    if total_words < 60_000:
        print("WARNING author corpus under 60k words — the 50k sweep arm will not exist.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="author root scaffolded by `new_author.py <root> --private`")
    ap.add_argument("--archive", default=None,
                    help="path to blogs.zip (default: <root>/data/interim/_bac_cache/blogs.zip)")
    ap.add_argument("--blogger", default=None, help="blogger ID (default: most prolific)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="print the top candidate bloggers and exit")
    ap.add_argument("--top", type=int, default=25, help="rows to print with --list")
    ap.add_argument("--max-author-words", type=int, default=150_000)
    ap.add_argument("--n-distractors", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if root == REPO or REPO in root.parents:
        print("BAC roots must live outside the repo — this corpus is never committed.")
        return 1
    if not (root / "RUNBOOK.md").exists():
        print(f"{root} is not a scaffolded author root — run "
              "`python scripts/new_author.py <root> --register informal --private` first")
        return 1
    archive = Path(args.archive).expanduser() if args.archive \
        else root / "data" / "interim" / "_bac_cache" / "blogs.zip"
    if not archive.exists():
        print(f"missing {archive}\nDownload the archive first:\n  curl -L -o {archive} "
              f"'{ARCHIVE_URL}'")
        return 1

    if args.list_only:
        rows = scan(archive, root / "data" / "interim" / "_bac_cache" / "ranking.json")
        print(f"{'id':>10} {'gender':>7} {'age':>4} {'industry':>16} {'posts':>6} "
              f"{'words':>9} {'fp/100w':>8}")
        for r in rows[: args.top]:
            flag = "" if r["fp_rate"] >= MIN_FP_RATE else "  <- likely quote-blog"
            print(f"{r['id']:>10} {r['gender']:>7} {r['age']:>4} {r['industry']:>16} "
                  f"{r['posts']:>6} {r['words']:>9,} {r['fp_rate']:>8}{flag}")
        return 0
    return build(root, archive, args.blogger, args.max_author_words, args.n_distractors)


if __name__ == "__main__":
    raise SystemExit(main())
