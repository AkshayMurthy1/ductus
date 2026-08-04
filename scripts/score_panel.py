#!/usr/bin/env python3
"""Score filled human-panel sheets against the key.

Takes any number of filled sheet.csv files (one per rater), pools the ratings, and writes
panel_results.json: per-source rating means and the deltas that decide the verifier-gaming
question. Read `adapter_minus_real` first:

    CI spans 0     raters cannot tell the adapter from the real author — the verifier's
                   ordering stands, "hyper-typical Chesterton" survives as the reading
    clearly < 0    raters rank adapter text BELOW real text while the verifier ranks it
                   above: verifier gaming, and the paper must report it as such
    clearly > 0    humans replicate the hyper-typicality reading

`real_minus_distractor` is the sanity row: if raters cannot separate the real author from the
distractor either, the panel lacked power and none of the rows above mean anything.

    python scripts/score_panel.py --ratings rater1.csv rater2.csv --key runs/human_panel/key.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.eval.panel import score_panel  # noqa: E402


def read_ratings(paths: list[str]) -> tuple[dict[str, float], int]:
    """Pooled {item_id -> rating} per rater-file suffix; returns (ratings, n_raters).
    Each rater's item P001 stays distinct (P001#0, P001#1, ...) so raters pool as samples."""
    ratings: dict[str, float] = {}
    for i, p in enumerate(paths):
        with Path(p).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get("rating") or "").strip()
                if not raw:
                    continue
                val = float(raw)
                if not 1 <= val <= 5:
                    raise ValueError(f"{p}: rating {raw!r} for {row['item_id']} not in 1-5")
                ratings[f"{row['item_id']}#{i}"] = val
    return ratings, len(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratings", nargs="+", required=True, help="filled sheet.csv per rater")
    ap.add_argument("--key", required=True, help="key.json from make_panel_sheets.py")
    ap.add_argument("--out", default=None,
                    help="default: panel_results.json next to the key")
    args = ap.parse_args()

    key_blob = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key = key_blob["key"]
    ratings, n_raters = read_ratings(args.ratings)
    # Expand the key to the per-rater item ids produced by read_ratings.
    expanded = {f"{item}#{i}": src for item, src in key.items() for i in range(n_raters)}

    result = score_panel(expanded, ratings)
    result["n_raters"] = n_raters
    result["source_run"] = key_blob.get("gen")

    out = Path(args.out) if args.out else Path(args.key).parent / "panel_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    if result["n_rated"] == 0:
        print("no ratings found — are the rating cells filled in?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
