#!/usr/bin/env python3
"""Publication figures from committed run records — never hand-transcribed.

    python scripts/make_figures.py            # writes runs/results/figures/
    python scripts/make_figures.py --out DIR

One figure per established finding (docs/STATUS.md):

    fig1_frontier          R2  the L-shaped style-vs-leakage frontier
    fig2_phase_transition  R1/R8/R9  the cliff, across base models and authors
    fig3_locus             R3  the voice lives in attention
    fig4_stage_interaction R4/R5  DPO helps the full locus, hurts attention-only
    fig5_entity_emission   R6  entity hygiene is the pipeline's (a14 diagnostic)
    fig6_instruments       R7  two independent rulers rank the runs the same way
    fig7_trajectory        supporting  the voice is fully formed by the first checkpoint

Word counts come from the sweep manifests; every metric comes from a report.json.
Palette: colorblind-validated categorical set (adjacent-pair CVD dE >= 8).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED, LIGHTBLUE = "#9a9891", "#0b0b0b", "#52514e", "#b9cfe9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c9c7c0", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e8e6e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.facecolor": "white",
    "legend.edgecolor": "#e0ded8", "legend.framealpha": 1.0,
})

ARMS = ["2k", "5k", "10k", "15k", "20k", "25k", "50k", "100k", "full"]


def words(manifest_path: Path) -> dict[str, int]:
    return {m["arm"]: m["actual_words"] for m in json.loads(manifest_path.read_text())}


def rec(path: str) -> dict:
    r = json.loads((ROOT / path / "report.json").read_text())
    t = r.get("training", {})
    return {
        "av": r["av"]["attribution_rate"],
        "real": r["av"].get("real_reference_attribution_rate"),
        "leak": r["leakage"]["verbatim"]["rate"],
        "ents": (r["leakage"].get("entities") or {}).get("per_generation"),
        "params": (t.get("trainable") or {}).get("trainable_params"),
        "secs": t.get("train_runtime_s"),
    }


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# ------------------------------------------------------------------ fig1: the L-frontier (R2)
def fig1(out: Path) -> None:
    ad = [rec(f"runs/sweep/{a}/stage_a") for a in ARMS]
    fs = [rec(f"runs/sweep/{a}/baseline") for a in ARMS]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([f["leak"] * 100 for f in fs], [f["av"] for f in fs], "o", color=ORANGE,
            ms=7, mec="white", mew=1.2, label="few-shot prompting", zorder=3)
    ax.plot([a["leak"] * 100 for a in ad], [a["av"] for a in ad], "o", color=BLUE,
            ms=7, mec="white", mew=1.2, label="LoRA adapter (Stage A)", zorder=4)
    labels = {"15k": ("15k words", (7, -3)), "25k": ("25k", (7, -3)),
              "50k": ("50k", (7, 2)), "full": ("full (179k)", (7, -9))}
    for a, r in zip(ARMS, ad):
        if a in labels:
            txt, off = labels[a]
            ax.annotate(txt, (r["leak"] * 100, r["av"]), textcoords="offset points",
                        xytext=off, fontsize=8, color=MUTED)
    ax.annotate("adapters: every arm at 0.0% leak", (0.15, 0.30), fontsize=8, color=BLUE)
    ax.annotate("prompting leaks, style flat", (12.4, 0.075), fontsize=8, color=ORANGE,
                ha="right")
    ax.axvline(5, color=GRAY, lw=0.8, ls=":")
    ax.annotate("5% leakage gate", (5.15, 0.55), fontsize=7.5, color=MUTED,
                rotation=90, va="center")
    ax.set_xlabel("content leaked — generations with a verbatim training 12-gram (%)")
    ax.set_ylabel("style acquired — AV attribution rate")
    ax.set_xlim(-0.45, 13)
    ax.set_ylim(-0.04, 1.0)
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, out, "fig1_frontier")


# ------------------------------------- fig2: the phase transition replicates (R1 / R8 / R9)
def fig2(out: Path) -> None:
    W = words(ROOT / "data/processed/sweep/manifest.json")
    TW = words(ROOT / "authors/twain/data/processed/sweep/manifest.json")
    q3 = [(W[a], rec(f"runs/sweep/{a}/stage_a")["av"]) for a in ARMS]
    q15 = [(W[a], rec(f"runs/matrix/models/a13_1p5b/{a}/stage_a")["av"])
           for a in ("10k", "25k", "full")]
    ll3 = [(W[a], rec(f"runs/matrix/models/a16_llama3b/{a}/stage_a")["av"])
           for a in ("10k", "25k", "full")]
    tw = [(TW[a], rec(f"authors/twain/runs/sweep/{a}/stage_a")["av"])
          for a in ("10k", "25k", "50k", "full")]
    real = rec("runs/sweep/full/stage_a")["real"]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for pts, color, ls, label in [
        (q3, BLUE, "-", "Qwen2.5-3B × Chesterton (reference)"),
        (q15, ORANGE, "-", "Qwen2.5-1.5B × Chesterton"),
        (ll3, AQUA, "-", "Llama-3.2-3B × Chesterton"),
        (tw, YELLOW, "--", "Qwen2.5-3B × Twain (own verifier)"),
    ]:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=color, lw=2, marker="o", ms=5, mec="white", mew=1,
                label=label)
    ax.axhline(real, color=GRAY, lw=1, ls=":")
    ax.annotate(f"real held-out Chesterton ({real:.3f})", (185000, real - 0.02),
                fontsize=7.5, color=MUTED, ha="right", va="top")
    floor = rec("authors/twain/runs/floor")["av"]
    ax.axhline(floor, color=YELLOW, lw=0.9, ls=":", alpha=0.7)
    ax.annotate(f"Twain base-model floor ({floor:.3f})", (185000, floor - 0.02),
                fontsize=7.5, color=MUTED, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("training corpus size (words, log scale)")
    ax.set_ylabel("AV attribution rate")
    ax.set_ylim(-0.04, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, "fig2_phase_transition")


# ------------------------------------------------------- fig3: the voice lives in attention (R3)
def fig3(out: Path) -> None:
    loci = [
        ("attention only (q,k,v,o)", "runs/matrix/rq2/a01_attention_only"),
        ("q,k only", "runs/matrix/rq2/a17_qk_only"),
        ("both, MLP rank 8", "runs/matrix/rq2/a05_split_rank_low_mlp"),
        ("MLP only (gate,up,down)", "runs/matrix/rq2/a02_mlp_only"),
        ("both, r16 (reference)", "runs/sweep/full/stage_a"),
        ("v,o only", "runs/matrix/rq2/a18_vo_only"),
    ]
    rows = sorted(((lab, rec(p)) for lab, p in loci), key=lambda r: r[1]["av"])

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for y, (lab, r) in enumerate(rows):
        color = BLUE if lab.startswith("attention only") else LIGHTBLUE
        ax.barh(y, r["av"], height=0.62, color=color, zorder=3)
        ax.annotate(f'{r["av"]:.3f}   ·   {r["params"] / 1e6:.1f}M params, {r["secs"]:.0f}s',
                    (r["av"] + 0.01, y), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [lab for lab, _ in rows], fontsize=8.5)
    ax.set_xlim(0, 1.42)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("AV attribution rate (full corpus, identical recipe; one-variable diffs)",
                  fontsize=8.5)
    save(fig, out, "fig3_locus")


# --------------------------------------- fig4: the stage x locus interaction (R4 / R5)
def fig4(out: Path) -> None:
    seeds = [rec(p)["av"] for p in
             ("runs/sweep/full/stage_a", "runs/sweep/full/seed29/stage_a",
              "runs/sweep/full/seed43/stage_a")]
    mean = sum(seeds) / len(seeds)
    sd = (sum((s - mean) ** 2 for s in seeds) / (len(seeds) - 1)) ** 0.5
    groups = [
        ("10k corpus\n(below the cliff)", rec("runs/sweep/10k/stage_a")["av"],
         rec("runs/sweep/10k/stage_b")["av"], None),
        ("full corpus\nattention + MLP", rec("runs/sweep/full/stage_a")["av"],
         rec("runs/sweep/full/stage_b")["av"], sd),
        ("full corpus\nattention-only", rec("runs/matrix/rq2/a01_attention_only")["av"],
         rec("runs/matrix/rq2x/a15_attn_dpo")["av"], None),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    w = 0.34
    for i, (lab, a, b, err) in enumerate(groups):
        ax.bar(i - w / 2, a, w, color=BLUE, zorder=3,
               yerr=(err if err else None), ecolor=INK, capsize=3,
               label="Stage A (SFT)" if i == 0 else None)
        ax.bar(i + w / 2, b, w, color=ORANGE, zorder=3,
               label="Stage A+B (+on-policy DPO)" if i == 0 else None)
        ax.annotate(f"{a:.3f}", (i - w / 2, a), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK)
        ax.annotate(f"{b:.3f}", (i + w / 2, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK)
        delta = b - a
        ax.annotate(f"Δ {delta:+.3f}", (i, max(a, b)), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=8.5, fontweight="bold",
                    color=(AQUA if delta > 0 else "#e34948"))
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("AV attribution rate")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=8,
              frameon=False)
    save(fig, out, "fig4_stage_interaction")


# ------------------------------------------- fig5: entity hygiene is the pipeline's (R6)
def fig5(out: Path) -> None:
    W = words(ROOT / "data/processed/sweep/manifest.json")
    TW = words(ROOT / "authors/twain/data/processed/sweep/manifest.json")
    ch = [(W[a], rec(f"runs/sweep/{a}/stage_a")["ents"]) for a in ARMS]
    tw = [(TW[a], rec(f"authors/twain/runs/sweep/{a}/stage_a")["ents"])
          for a in ("10k", "25k", "50k", "full")]
    base = rec("runs/floor")["ents"]
    a14 = rec("runs/matrix/a14_no_scrubbing")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs, ys = zip(*ch)
    ax.plot(xs, ys, "-", color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1,
            label="adapter, scrubbed targets (Chesterton)")
    xs, ys = zip(*tw)
    ax.plot(xs, ys, "--", color=AQUA, lw=2, marker="o", ms=5, mec="white", mew=1,
            label="adapter, scrubbed targets (Twain)")
    ax.axhline(base, color=GRAY, lw=1, ls=":")
    ax.annotate(f"untuned base model ({base:.2f}/gen)", (3400, base - 0.12),
                fontsize=7.5, color=MUTED, va="top")
    ax.plot([W["full"]], [a14["ents"]], "D", color=ORANGE, ms=8, mec="white", mew=1.2,
            label="a14: unscrubbed targets (full)", zorder=5)
    ax.annotate("scrubbing off →\nentities absorbed", (W["full"] * 0.82, a14["ents"] + 0.04),
                fontsize=8, color=ORANGE, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("training corpus size (words, log scale)")
    ax.set_ylabel("author-corpus entities emitted per generation")
    ax.set_ylim(0, 4.9)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, out, "fig5_entity_emission")


# ----------------------------------------- fig6: two rulers, one ordering (R7)
def fig6(out: Path) -> None:
    pat = re.compile(r"^\| ([^|]+) \| (\d\.\d{4}) \| (\d\.\d{4}) \| ")
    pairs = []
    for line in (ROOT / "runs/results/instruments.md").read_text().splitlines():
        m = pat.match(line)
        if m and m.group(1).strip() != "run":
            pairs.append((float(m.group(2)), float(m.group(3))))
    meta = json.loads((ROOT / "runs/results/instruments.json").read_text())
    rho, n = meta["agreement"]["spearman"], meta["agreement"]["n"]
    real = (rec("runs/sweep/full/stage_a")["real"],
            meta["real_blind_under_second"]["attribution_rate"])
    assert len(pairs) == n, f"parsed {len(pairs)} rows, expected {n}"

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.9, ls=":", zorder=2)
    ax.plot(*zip(*pairs), "o", color=BLUE, ms=6, mec="white", mew=1, zorder=3,
            label=f"runs (n={n})")
    ax.plot(*real, "D", color=YELLOW, ms=8, mec="white", mew=1.2, zorder=4,
            label="real held-out Chesterton")
    ax.annotate(f"Spearman ρ = {rho:.3f}", (0.04, 0.93), fontsize=9, color=INK)
    ax.annotate("second ruler reads\nadapters higher", (0.60, 0.36), fontsize=8,
                color=MUTED, ha="center")
    ax.set_xlabel("primary verifier (Wegmann) — AV attribution")
    ax.set_ylabel("second verifier (StyleDistance) — AV attribution")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "fig6_instruments")


# ------------------------------- fig7: the voice forms before the first checkpoint (supporting)
def fig7(out: Path) -> None:
    steps = sorted(
        (int(p.name.split("-")[1]), rec(f"runs/trajectory/trajectory/{p.name}"))
        for p in (ROOT / "runs/trajectory/trajectory").iterdir() if p.name.startswith("step-")
    )
    real = rec("runs/sweep/full/stage_a")["real"]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xs = [s for s, _ in steps]
    ax.plot(xs, [r["av"] for _, r in steps], "-", color=BLUE, lw=2, marker="o", ms=5,
            mec="white", mew=1, label="adapter AV (full corpus)")
    ax.plot(xs, [r["leak"] for _, r in steps], "-", color=ORANGE, lw=2, marker="o", ms=5,
            mec="white", mew=1, label="verbatim leakage rate")
    ax.axhline(real, color=GRAY, lw=1, ls=":")
    ax.annotate(f"real held-out Chesterton ({real:.3f})", (150, real - 0.03),
                fontsize=7.5, color=MUTED, ha="right", va="top")
    ax.annotate("voice already formed\nat the first checkpoint", (xs[0], 0.60),
                fontsize=8, color=MUTED, va="top")
    ax.annotate("leakage: zero at every checkpoint", (xs[-1], 0.0),
                textcoords="offset points", xytext=(0, 8), fontsize=8, color=ORANGE,
                ha="right")
    ax.set_xlabel("training step (checkpoints of one full-corpus Stage-A run)")
    ax.set_ylabel("attribution / leakage rate")
    ax.set_xticks(xs)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(loc="center right", fontsize=8)
    save(fig, out, "fig7_trajectory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=ROOT / "runs/results/figures", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {args.out}")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(args.out)


if __name__ == "__main__":
    main()
