#!/usr/bin/env python3
"""Where does the voice live? Per-layer, per-module magnitude of a trained LoRA update.

RQ2 asks whether attention vs MLP adaptation changes the style-per-leakage exchange rate. This
script gives that question a mechanistic companion figure: for every adapted matrix it computes
the Frobenius norm of the effective update ΔW = (α/r)·B·A — without materializing ΔW, via
‖BA‖_F² = tr((BᵀB)(AAᵀ)), which is r×r arithmetic — and reports how update mass distributes
across depth and across attention vs MLP.

Read against the stylometry per-feature gaps: if attention-only runs close the rhythm gap while
the reference run's update mass sits in upper-layer MLPs when the idiom gap closes, that is a
mechanistic account of the plan's "attention carries cadence, MLP carries lexicon" claim —
evidence stronger than the ablation deltas alone.

CPU-only; needs numpy + safetensors (both already required by the training stack).

    python scripts/adapter_anatomy.py runs/stage_a
    python scripts/adapter_anatomy.py runs/sweep/full/stage_a --json runs/results/anatomy.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LAYER = re.compile(r"\.layers\.(\d+)\.")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def module_scale(name: str, cfg: dict) -> float:
    """Effective LoRA scaling α/r for one module, honoring rank_pattern/alpha_pattern."""
    r = cfg.get("r", 8)
    alpha = cfg.get("lora_alpha", r)
    for pat, v in (cfg.get("rank_pattern") or {}).items():
        if re.search(pat, name):
            r = v
    for pat, v in (cfg.get("alpha_pattern") or {}).items():
        if re.search(pat, name):
            alpha = v
    return alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r


def analyze(adapter_dir: Path) -> dict:
    import numpy as np
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    st = adapter_dir / "adapter_model.safetensors"
    if not st.exists():
        raise FileNotFoundError(f"{st} not found — is this a saved PEFT adapter directory?")

    a_mats, b_mats, magnitudes = {}, {}, {}
    with safe_open(str(st), framework="numpy") as f:
        for key in f.keys():
            base = re.sub(r"\.(lora_A|lora_B|lora_magnitude_vector)\.?(weight|default)?.*$",
                          "", key)
            t = f.get_tensor(key)
            if ".lora_A" in key:
                a_mats[base] = np.asarray(t, dtype=np.float64)
            elif ".lora_B" in key:
                b_mats[base] = np.asarray(t, dtype=np.float64)
            elif "lora_magnitude_vector" in key:
                magnitudes[base] = np.asarray(t, dtype=np.float64)

    modules = []
    for base in sorted(set(a_mats) & set(b_mats)):
        A, B = a_mats[base], b_mats[base]
        # ‖BA‖_F² = tr((BᵀB)(AAᵀ)) — r×r products only.
        fro = float(np.sqrt(max(0.0, np.trace((B.T @ B) @ (A @ A.T)))))
        scale = module_scale(base, cfg)
        m = LAYER.search(base)
        kind = ("attention" if any(k in base for k in ATTN)
                else "mlp" if any(k in base for k in MLP) else "other")
        modules.append({
            "module": base.split("base_model.model.")[-1],
            "layer": int(m.group(1)) if m else None,
            "kind": kind,
            "proj": next((k for k in ATTN + MLP if k in base), "other"),
            "update_norm": round(scale * fro, 6),
            "dora_magnitude_drift": (
                round(float(np.abs(magnitudes[base] - 1.0).mean()), 6)
                if base in magnitudes else None),
        })

    by_kind: dict[str, float] = defaultdict(float)
    by_layer: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for mod in modules:
        by_kind[mod["kind"]] += mod["update_norm"]
        if mod["layer"] is not None:
            by_layer[mod["layer"]][mod["kind"]] += mod["update_norm"]
    total = sum(by_kind.values()) or 1.0

    return {
        "adapter": str(adapter_dir),
        "peft_config": {k: cfg.get(k) for k in
                        ("r", "lora_alpha", "use_dora", "use_rslora", "rank_pattern",
                         "alpha_pattern", "target_modules")},
        "share_by_kind": {k: round(v / total, 4) for k, v in sorted(by_kind.items())},
        "by_layer": {str(layer): {k: round(v, 4) for k, v in kinds.items()}
                     for layer, kinds in sorted(by_layer.items())},
        "top_modules": sorted(modules, key=lambda m: -m["update_norm"])[:12],
        "n_modules": len(modules),
    }


def print_depth_profile(rep: dict) -> None:
    layers = sorted(int(k) for k in rep["by_layer"])
    if not layers:
        return
    peak = max(sum(v.values()) for v in rep["by_layer"].values()) or 1.0
    print("\nupdate mass by depth (a=attention, m=mlp; bars scaled to the peak layer):")
    for layer in layers:
        kinds = rep["by_layer"][str(layer)]
        a, m = kinds.get("attention", 0.0), kinds.get("mlp", 0.0)
        bar = "a" * round(28 * a / peak) + "m" * round(28 * m / peak)
        print(f"  L{layer:>2} {bar:<28} {a + m:8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter", help="a saved adapter directory (or checkpoint-N directory)")
    ap.add_argument("--json", default=None, help="also write the full report here")
    args = ap.parse_args()

    rep = analyze(Path(args.adapter))
    print(f"{rep['n_modules']} adapted matrices in {rep['adapter']}")
    print(f"update-mass share: {rep['share_by_kind']}")
    print("top modules by update norm:")
    for m in rep["top_modules"][:8]:
        drift = (f"  dora-drift {m['dora_magnitude_drift']}"
                 if m["dora_magnitude_drift"] is not None else "")
        print(f"  {m['update_norm']:9.4f}  {m['module']}{drift}")
    print_depth_profile(rep)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
