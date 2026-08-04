#!/usr/bin/env python3
"""Execute the RESEARCH_BRIEF run matrix unattended, resuming after any failure.

One driver for every arm the brief requires, so a multi-day GPU session can be restarted with
the same command and pick up where it stopped:

    floor   base model, no adapter, no exemplars — the zero of the style axis, plus the
            perplexity contamination probe (brief §5: record the untuned floor)
    rq1     per corpus-size arm: few-shot baseline + Stage-A adapter (needs make_size_sweep.py)
    rq2     adaptation locus at one fixed arm: attention-only / MLP-only / both / low-MLP-rank
    rq3     Stage B (on-policy DPO) on top of Stage A at 2-3 corpus sizes
    seeds   >=3 seeds at >=2 cells — the noise floor that must exist before any ranking
    rq2x    the RQ2×RQ3 interaction cell: DPO on top of the attention-only (a01) adapter —
            the recommended configuration must itself be a measured cell (docs/EXPANSION.md)
    models  cross-model replication: baseline + Stage A per corpus arm under a second base
            model (a13 within-family, a16 cross-family), plus that model's contamination probe

Resume logic: a unit is done when its report.json exists; done units are skipped. A failing
unit is recorded in runs/matrix_state.json and does not stop independent units, but anything
that requires its output (e.g. rq3 on a failed stage_a) is skipped with a reason.

Control author (Table 4): run this same driver a second time with WLM_ROOT pointing at a
directory holding the private author's data/ tree, after running the CPU pipeline there.
Every path below honors WLM_ROOT, so nothing else changes — which is the point: identical
protocol, different corpus.

    python scripts/run_matrix.py                 # everything
    python scripts/run_matrix.py --only rq1 rq2  # a subset
    python scripts/run_matrix.py --dry-run       # print the plan
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.paths import PROCESSED, ROOT, RUNS  # noqa: E402  (WLM_ROOT-aware)

SWEEP = PROCESSED / "sweep"
BLIND = PROCESSED / "blind.jsonl"
STATE = RUNS / "matrix_state.json"
CFG_A = str(REPO / "configs/stage_a.yaml")
CFG_B = str(REPO / "configs/stage_b.yaml")
RQ2_CONFIGS = [  # the brief's locus grid; the reference "both" cell comes from rq1's arm
    "a01_attention_only",
    "a02_mlp_only",
    "a05_split_rank_low_mlp",
]
# Per-matrix split of the attention cell (--rq2-configs to include): a17 q,k vs a18 v,o tests
# Savine 2507.21009's "V/O memorize most" against our zero-leakage attention result.
MODEL_CONFIGS = ["a16_llama3b"]  # --model-configs; a13_1p5b adds the within-family point
MODEL_ARMS = ["10k", "25k", "full"]  # cliff region + endpoint; --model-arms to widen
VARIANCE_SEEDS = [29, 43]  # plus the config default 17 from the main run = 3 seeds per cell


class Unit:
    def __init__(self, name: str, done: Path, cmds: list[list[str]],
                 requires: Path | None = None):
        self.name, self.done, self.cmds, self.requires = name, done, cmds, requires


def eval_cmd(gen_dir: Path, run_name: str, train: Path, baseline: Path | None = None,
             split_path: Path = BLIND) -> list[str]:
    cmd = ["wlm", "eval", "run", "--gen", str(gen_dir / "gen.jsonl"), "--ref", str(split_path),
           "--train", str(train), "--av", str(RUNS / "av"),
           "--out", str(gen_dir / "report.html"), "--run-name", run_name]
    if baseline is not None:
        cmd += ["--baseline", str(baseline)]
    return cmd


def sft_units(out: Path, train: Path, val: Path, run_name: str, *, config: str = CFG_A,
              seed: int | None = None, baseline_report: Path | None = None) -> Unit:
    seed_args = ["--seed", str(seed)] if seed is not None else []
    return Unit(
        run_name, out / "report.json",
        [["wlm", "train", "sft", "--config", config, "--train", str(train), "--val", str(val),
          "--out", str(out), *seed_args],
         ["wlm", "generate", "--config", config, "--adapter", str(out), "--split-path",
          str(BLIND), "--out", str(out / "gen.jsonl"), *seed_args],
         eval_cmd(out, run_name, train, baseline_report)])


def baseline_unit(out: Path, train: Path, run_name: str, *, config: str = CFG_A,
                  seed: int | None = None) -> Unit:
    seed_args = ["--seed", str(seed)] if seed is not None else []
    return Unit(
        run_name, out / "report.json",
        [["wlm", "baseline", "--config", config, "--train", str(train), "--split-path",
          str(BLIND), "--out", str(out), *seed_args],
         eval_cmd(out, run_name, train)])


def build_plan(args) -> list[Unit]:
    units: list[Unit] = []
    full_train = PROCESSED / "train.jsonl"
    arms = sorted([d for d in SWEEP.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime) if SWEEP.exists() else []
    # manifest order (small -> large) is the meaningful order when present
    manifest = SWEEP / "manifest.json"
    if manifest.exists():
        order = [m["arm"] for m in json.loads(manifest.read_text())]
        arms = [SWEEP / a for a in order if (SWEEP / a).is_dir()]

    if "floor" in args.only:
        floor = RUNS / "floor"
        units.append(Unit(
            "floor/base-model", floor / "report.json",
            [["wlm", "generate", "--config", CFG_A, "--split-path", str(BLIND),
              "--out", str(floor / "gen.jsonl")],
             eval_cmd(floor, "base-floor", full_train)]))
        units.append(Unit(
            "floor/contamination", RUNS / "contamination.json",
            [["wlm", "eval", "contamination", "--config", CFG_A, "--author", str(BLIND),
              "--out", str(RUNS / "contamination.json")]]))

    if "rq1" in args.only:
        if not arms:
            print(f"[rq1] no arms under {SWEEP} — run scripts/make_size_sweep.py first")
        for arm_dir in arms:
            arm = arm_dir.name
            base = RUNS / "sweep" / arm / "baseline"
            sft = RUNS / "sweep" / arm / "stage_a"
            units.append(baseline_unit(base, arm_dir / "train.jsonl", f"baseline-{arm}"))
            units.append(sft_units(sft, arm_dir / "train.jsonl", arm_dir / "val.jsonl",
                                   f"stage_a-{arm}", baseline_report=base / "report.json"))

    if "rq2" in args.only:
        arm_dir = SWEEP / args.rq2_arm
        train = arm_dir / "train.jsonl" if arm_dir.exists() else full_train
        val = arm_dir / "val.jsonl" if arm_dir.exists() else PROCESSED / "val.jsonl"
        for name in args.rq2_configs:
            cfg = str(REPO / "configs" / "ablations" / f"{name}.yaml")
            units.append(sft_units(RUNS / "matrix" / "rq2" / name, train, val, name, config=cfg))

    if "rq2x" in args.only:
        # DPO on top of the attention-only Stage-A adapter. Mirrors rq3's command chain, but the
        # adapter is a01's output and the baseline delta reads against a01, so the report answers
        # "what does DPO add to the cheapest configuration" directly.
        a01 = RUNS / "matrix" / "rq2" / "a01_attention_only"
        cfg = str(REPO / "configs" / "ablations" / "a15_attn_dpo.yaml")
        out = RUNS / "matrix" / "rq2x" / "a15_attn_dpo"
        pairs = out / "dpo.jsonl"
        arm_dir = SWEEP / args.rq2_arm
        train = arm_dir / "train.jsonl" if arm_dir.exists() else full_train
        units.append(Unit(
            "rq2x/a15_attn_dpo", out / "report.json",
            [["wlm", "dpo-pairs", "--config", cfg, "--adapter", str(a01), "--split", "val",
              "--out", str(pairs), "--av", str(RUNS / "av"), "--val-frac", "0.15"],
             ["wlm", "train", "dpo", "--config", cfg, "--adapter", str(a01),
              "--out", str(out), "--pairs", str(pairs),
              "--val-pairs", str(out / "dpo_val.jsonl")],
             ["wlm", "generate", "--config", cfg, "--adapter", str(out),
              "--split-path", str(BLIND), "--out", str(out / "gen.jsonl")],
             eval_cmd(out, "a15_attn_dpo", train, a01 / "report.json")],
            requires=a01 / "report.json"))

    if "models" in args.only:
        # Cross-model replication: per model config, the few-shot baseline AND Stage A per arm.
        # The baseline must move with the base model — scoring a Llama adapter against the Qwen
        # few-shot floor compares against the wrong zero. The verifier is text-side and shared;
        # the contamination probe is model-side, so each model gets its own.
        for name in args.model_configs:
            cfg = str(REPO / "configs" / "ablations" / f"{name}.yaml")
            mroot = RUNS / "matrix" / "models" / name
            units.append(Unit(
                f"models/{name}/contamination", mroot / "contamination.json",
                [["wlm", "eval", "contamination", "--config", cfg, "--author", str(BLIND),
                  "--out", str(mroot / "contamination.json")]]))
            for arm in args.model_arms:
                arm_dir = SWEEP / arm
                if not arm_dir.exists():
                    print(f"[models] no sweep arm {arm_dir} — run make_size_sweep.py first")
                    continue
                base = mroot / arm / "baseline"
                units.append(baseline_unit(base, arm_dir / "train.jsonl",
                                           f"baseline-{name}-{arm}", config=cfg))
                units.append(sft_units(mroot / arm / "stage_a", arm_dir / "train.jsonl",
                                       arm_dir / "val.jsonl", f"stage_a-{name}-{arm}",
                                       config=cfg, baseline_report=base / "report.json"))

    if "rq3" in args.only:
        for arm in args.rq3_arms:
            sft = RUNS / "sweep" / arm / "stage_a"
            out = RUNS / "sweep" / arm / "stage_b"
            pairs = out / "dpo.jsonl"
            units.append(Unit(
                f"stage_b-{arm}", out / "report.json",
                [["wlm", "dpo-pairs", "--config", CFG_B, "--adapter", str(sft), "--split", "val",
                  "--out", str(pairs), "--av", str(RUNS / "av"), "--val-frac", "0.15"],
                 ["wlm", "train", "dpo", "--config", CFG_B, "--adapter", str(sft),
                  "--out", str(out), "--pairs", str(pairs),
                  "--val-pairs", str(out / "dpo_val.jsonl")],
                 ["wlm", "generate", "--config", CFG_B, "--adapter", str(out),
                  "--split-path", str(BLIND), "--out", str(out / "gen.jsonl")],
                 eval_cmd(out, f"stage_b-{arm}", SWEEP / arm / "train.jsonl",
                          RUNS / "sweep" / arm / "stage_a" / "report.json")],
                requires=sft / "report.json"))

    if "seeds" in args.only:
        cells = args.seed_cells or ([arms[0].name, arms[-1].name] if len(arms) >= 2 else [])
        if not cells:
            print("[seeds] no sweep arms to place variance cells on — run make_size_sweep.py")
        for arm in cells:
            arm_dir = SWEEP / arm
            for sd in args.seeds:
                root = RUNS / "sweep" / arm / f"seed{sd}"
                units.append(baseline_unit(root / "baseline", arm_dir / "train.jsonl",
                                           f"baseline-{arm}-seed{sd}", seed=sd))
                units.append(sft_units(root / "stage_a", arm_dir / "train.jsonl",
                                       arm_dir / "val.jsonl", f"stage_a-{arm}-seed{sd}",
                                       seed=sd))
    return units


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=["floor", "rq1", "rq2", "rq3", "seeds"],
                    choices=["floor", "rq1", "rq2", "rq3", "seeds", "rq2x", "models"])
    ap.add_argument("--rq2-arm", default="full",
                    help="sweep arm whose corpus the locus grid trains on (one fixed size)")
    ap.add_argument("--rq2-configs", nargs="*", default=RQ2_CONFIGS,
                    help="ablation configs for the locus grid; add a17_qk_only a18_vo_only "
                         "for the per-matrix split")
    ap.add_argument("--model-configs", nargs="*", default=MODEL_CONFIGS,
                    help="base-model ablation configs for the cross-model section")
    ap.add_argument("--model-arms", nargs="*", default=MODEL_ARMS,
                    help="sweep arms each extra model runs at (default: cliff region + full)")
    ap.add_argument("--rq3-arms", nargs="*", default=["10k", "full"],
                    help="sweep arms that get a Stage-B pass on top of Stage A")
    ap.add_argument("--seeds", nargs="*", type=int, default=VARIANCE_SEEDS)
    ap.add_argument("--seed-cells", nargs="*", default=None,
                    help="sweep arms for the variance cells (default: smallest and largest)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (RUNS / "av").exists():
        print("missing runs/av — fit the verifier ONCE on the full train set first:\n"
              "  wlm eval fit-av --author data/processed/train.jsonl "
              "--distractor data/raw/distractor --out runs/av")
        return 1
    if not BLIND.exists():
        print(f"missing {BLIND} — run the CPU pipeline (scripts/00_phase0_cpu.sh) first")
        return 1

    units = build_plan(args)
    print(f"root: {ROOT}\nplan: {len(units)} unit(s)")
    state = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "units": {}}
    failed: set[Path] = set()

    for u in units:
        if u.done.exists():
            print(f"  [done] {u.name}")
            state["units"][u.name] = "done (cached)"
            continue
        if u.requires is not None and (u.requires in failed or not u.requires.exists()):
            print(f"  [skip] {u.name} — requires {u.requires}, which is missing or failed")
            state["units"][u.name] = f"skipped: requires {u.requires}"
            continue
        print(f"  [run ] {u.name}")
        if args.dry_run:
            for c in u.cmds:
                print("         $ " + " ".join(c))
            continue
        ok = True
        for c in u.cmds:
            print("         $ " + " ".join(c), flush=True)
            if subprocess.call(c) != 0:
                ok = False
                break
        state["units"][u.name] = "ok" if ok else "FAILED"
        if not ok:
            failed.add(u.done)
            print(f"  [FAIL] {u.name} — recorded; continuing with independent units. "
                  "Re-run this script to retry.")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    n_fail = sum(1 for v in state["units"].values() if v == "FAILED")
    print(f"\n{len(units)} unit(s); {n_fail} failed. State: {STATE}")
    if not args.dry_run:
        print("Assemble the deliverables:  python scripts/assemble_results.py")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
