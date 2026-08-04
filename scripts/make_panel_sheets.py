#!/usr/bin/env python3
"""Build blinded human-panel rating sheets from a run's generations.

Draws equal numbers of real held-out passages (blind split), adapter generations, and
era-matched distractor chunks; pairs each with a labeled reference excerpt from TRAIN; and
writes, under --out:

    sheet.csv       item_id, reference, candidate, rating (empty — the rater fills 1-5)
    sheet.md        the same items formatted for reading
    key.json        item_id -> source. NEVER send this with the sheets; scoring reads it.
    protocol.md     verbatim rater instructions

Which run to sample: the highest-AV adapter run whose score sits above the real-text rate
(the 0.825 > 0.714 anomaly is the thing being adjudicated), plus any runs flagged by
scripts/second_instrument.py. Same seed -> byte-identical sheets, so the panel is auditable.

    python scripts/make_panel_sheets.py --gen runs/sweep/50k/stage_a/gen.jsonl --out runs/human_panel
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.config import Config  # noqa: E402
from wlm.eval.av import load_distractor_texts  # noqa: E402
from wlm.eval.panel import build_panel  # noqa: E402
from wlm.paths import PROCESSED, read_jsonl  # noqa: E402

PROTOCOL = """\
# Rater instructions

Each numbered item shows a REFERENCE passage, which is genuinely by the author, and a
CANDIDATE passage. Rate how likely you think it is that the same person wrote both:

    1  almost certainly a different writer
    2  probably a different writer
    3  cannot tell
    4  probably the same writer
    5  almost certainly the same writer

Judge the *writing* — sentence rhythm, grammar, word choice, how arguments move — not the
topic. Two passages by one author can discuss completely different subjects. Do not research
the passages or the author. Enter the 1-5 rating in the `rating` column and nothing else.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen", required=True, help="gen.jsonl of the adapter run to adjudicate")
    ap.add_argument("--blind", default=str(PROCESSED / "blind.jsonl"))
    ap.add_argument("--train", default=str(PROCESSED / "train.jsonl"),
                    help="reference-excerpt source (real author text, labeled)")
    ap.add_argument("--distractor", default=str(REPO / "data/raw/distractor"))
    ap.add_argument("--config", default=str(REPO / "configs/stage_a.yaml"))
    ap.add_argument("--n-per-source", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="runs/human_panel")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    adapter = [g["completion"] for g in read_jsonl(args.gen) if g.get("completion")]
    real = [r["response"] for r in read_jsonl(args.blind) if r.get("response")]
    # References must never collide with the real candidates: train vs blind are split by
    # document upstream, so drawing references from TRAIN keeps the two pools disjoint.
    references = [r["response"] for r in read_jsonl(args.train) if r.get("response")]
    distractor = load_distractor_texts(
        args.distractor, min_words=cfg.data.chunk_min_words,
        max_words=cfg.data.chunk_max_words)

    items, key = build_panel(
        real=real, adapter=adapter, distractor=distractor, references=references,
        n_per_source=args.n_per_source, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "reference", "candidate", "rating"])
        for it in items:
            w.writerow([it["item_id"], it["reference"], it["candidate"], ""])
    md = ["# Panel sheet", "", "See protocol.md for instructions.", ""]
    for it in items:
        md += [f"## {it['item_id']}", "", "**Reference (by the author):**", "",
               it["reference"], "", "**Candidate:**", "", it["candidate"], "",
               "Rating (1-5): ____", ""]
    (out / "sheet.md").write_text("\n".join(md), encoding="utf-8")
    (out / "protocol.md").write_text(PROTOCOL, encoding="utf-8")
    (out / "key.json").write_text(json.dumps(
        {"seed": args.seed, "gen": args.gen, "key": key}, indent=2), encoding="utf-8")

    n = len(items)
    print(f"{n} items ({args.n_per_source} per source) -> {out}")
    print("Send sheet.csv (or sheet.md) + protocol.md to raters. Do NOT send key.json.")
    print("Score with: python scripts/score_panel.py --ratings <filled csvs...> "
          f"--key {out / 'key.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
