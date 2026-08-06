#!/usr/bin/env python3
"""Preview what `data/` and `runs/` would commit, and refuse to let private writing through.

`.gitignore` allow-lists by *path*, not by content — `data/raw/author/**` is committable because
it currently holds a public-domain fixture, not because git can tell. Swap in first-party prose
and the same rule would publish it. This is the guard for that: run it before any commit.

    make check-data          # or: python scripts/check_data.py

Exits non-zero if anything looks like private writing, or if a file exceeds GitHub's limits.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB_HARD_LIMIT = 100 * 1024 * 1024      # per-file rejection
GITHUB_WARN_LIMIT = 50 * 1024 * 1024       # per-file warning

# A public-domain corpus should not be full of first-person contemporary prose. These are the
# markers of the private corpora this project has actually held, not a general PII detector.
PRIVATE_MARKERS = ("_originals", "/private/")


def tracked_or_untracked(pathspec: str) -> list[Path]:
    """Files under `pathspec` that git would include (tracked + untracked, ignores applied)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", pathspec],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    return [ROOT / p for p in out if p.strip()]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def main() -> int:
    files = sorted(set(tracked_or_untracked("data") + tracked_or_untracked("runs")
                       + tracked_or_untracked("authors")))
    if not files:
        print("nothing under data/ or runs/ would be committed.")
        return 0

    by_top: dict[str, list[Path]] = {}
    for f in files:
        rel = f.relative_to(ROOT)
        key = "/".join(rel.parts[:2])
        by_top.setdefault(key, []).append(f)

    total = 0
    print("Would commit:\n")
    for key in sorted(by_top):
        group = by_top[key]
        size = sum(f.stat().st_size for f in group if f.exists())
        total += size
        print(f"  {key:<34} {len(group):>5} files   {human(size)}")
    print(f"\n  {'TOTAL':<34} {len(files):>5} files   {human(total)}")

    problems: list[str] = []

    private = [f for f in files if any(m in str(f.relative_to(ROOT)) for m in PRIVATE_MARKERS)]
    if private:
        problems.append(
            f"{len(private)} file(s) under a private path would be committed — "
            f"first is {private[0].relative_to(ROOT)}"
        )

    oversize = [f for f in files if f.exists() and f.stat().st_size > GITHUB_HARD_LIMIT]
    if oversize:
        problems.append(
            f"{len(oversize)} file(s) exceed GitHub's 100MB hard limit — "
            f"largest is {max(oversize, key=lambda p: p.stat().st_size).relative_to(ROOT)}"
        )

    warn = [f for f in files
            if f.exists() and GITHUB_WARN_LIMIT < f.stat().st_size <= GITHUB_HARD_LIMIT]
    if warn:
        print(f"\n  note: {len(warn)} file(s) over 50MB — GitHub warns but accepts these.")

    # The author fixture is allow-listed by path. Say plainly what is in it right now.
    author = ROOT / "data/raw/author"
    if author.exists():
        n = len([p for p in author.rglob("*") if p.is_file() and p.suffix in {".md", ".txt"}])
        print(f"\n  data/raw/author currently holds {n} document(s). The allow-list keys on the")
        print("  path, not the contents — confirm this is the public-domain fixture and not your")
        print("  own writing. See data/README.md.")

    if problems:
        print("\nBLOCKED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nOK — nothing private, nothing over the limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
