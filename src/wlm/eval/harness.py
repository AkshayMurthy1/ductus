"""The harness: one function that turns generations into a verdict.

Built before any tuning (Phase 0) on purpose. Everything downstream -- the ablation table, the
Stage-B decision, the go/no-go on the whole approach -- is read off this output, so it exists
first and never changes shape between runs.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from wlm.eval import stylometry as sty
from wlm.eval.leakage import run_leakage_suite
from wlm.eval.report import SL_BUCKETS, write_report
from wlm.paths import read_jsonl


def evaluate(
    *,
    generations: list[dict[str, Any]],
    real_reference: list[dict[str, Any]],
    verifier=None,
    train_texts: list[str] | None = None,
    author_texts_raw: list[str] | None = None,
    fluency: dict[str, Any] | None = None,
    smoke: list[dict[str, str]] | None = None,
    config: dict[str, Any] | None = None,
    run_name: str = "run",
    split: str = "blind",
    model: str = "",
    bootstrap: bool = True,
) -> dict[str, Any]:
    """generations: [{prompt, completion, register?, pair_id?}]
    real_reference: the matching real passages, [{prompt, response, register?}]
    """
    gen_texts = [g["completion"] for g in generations if g.get("completion")]
    real_texts = [r["response"] for r in real_reference if r.get("response")]
    if not gen_texts or not real_texts:
        raise ValueError("need non-empty generations and real reference text")

    gen_vec = sty.stylometry_vector("\n\n".join(gen_texts))
    real_vec = sty.stylometry_vector("\n\n".join(real_texts))
    dist = sty.stylometry_distance(real_vec, gen_vec)
    if bootstrap and len(gen_texts) >= 5 and len(real_texts) >= 5:
        lo, hi = sty.bootstrap_ci(real_texts, gen_texts, n=120)
        dist["ci_lo"], dist["ci_hi"] = round(lo, 4), round(hi, 4)

    out: dict[str, Any] = {
        "run_name": run_name,
        "split": split,
        "model": model,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_generations": len(gen_texts),
        "stylometry": dist,
        "biggest_gaps": sty.biggest_gaps(dist, k=8),
        "sent_len_hist": {
            "buckets": SL_BUCKETS,
            "real": [round(float(x), 4) for x in real_vec["dists"]["sent_len_hist"]],
            "generated": [round(float(x), 4) for x in gen_vec["dists"]["sent_len_hist"]],
        },
        "config": config or {},
    }

    # --- authorship verification (primary metric)
    if verifier is not None:
        av = verifier.evaluate(gen_texts)
        av["real_reference_attribution_rate"] = round(
            verifier.attribution_rate(real_texts), 4
        )
        if verifier.metrics:
            av["verifier_auc"] = verifier.metrics.auc
            av["verifier_accuracy"] = verifier.metrics.accuracy
            av["embedder"] = verifier.metrics.embedder
            av["verifier_verdict"] = (
                "PASS" if verifier.metrics.auc >= 0.75 else "FAIL: verifier cannot separate author"
            )
        out["av"] = av
        # Per-register breakdown -- informal is the hard case the plan is aimed at.
        regs = {g.get("register", "unknown") for g in generations}
        if len(regs) > 1:
            out["av_by_register"] = {
                reg: verifier.evaluate(
                    [g["completion"] for g in generations if g.get("register") == reg]
                )
                for reg in sorted(regs)
                if any(g.get("register") == reg for g in generations)
            }

    # --- leakage
    if train_texts:
        out["leakage"] = run_leakage_suite(gen_texts, train_texts, author_texts_raw)

    if fluency:
        out["fluency"] = fluency
    if smoke:
        out["smoke"] = smoke

    # --- paired samples for the human read
    out["samples"] = [
        {
            "prompt": g.get("prompt", ""),
            "generated": g.get("completion", "")[:1400],
            "real": next(
                (
                    r["response"][:1400]
                    for r in real_reference
                    if r.get("pair_id") == g.get("pair_id")
                ),
                real_texts[i % len(real_texts)][:1400],
            ),
        }
        for i, g in enumerate(generations[:6])
    ]
    return out


def compare_to_baseline(result: dict[str, Any], baseline_json: str | Path) -> dict[str, Any]:
    """Attach a baseline delta block. Sign convention: positive delta = this run is better."""
    b = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    cmp: dict[str, Any] = {}

    def pair(name, bv, tv, higher_better):
        if bv is None or tv is None:
            return
        cmp[name] = {
            "baseline": bv,
            "this_run": tv,
            "delta": round((tv - bv) if higher_better else (bv - tv), 4),
        }

    pair(
        "AV attribution rate",
        (b.get("av") or {}).get("attribution_rate"),
        (result.get("av") or {}).get("attribution_rate"),
        True,
    )
    pair(
        "stylometry distance (lower better)",
        (b.get("stylometry") or {}).get("overall"),
        (result.get("stylometry") or {}).get("overall"),
        False,
    )
    pair(
        "verbatim leakage rate (lower better)",
        ((b.get("leakage") or {}).get("verbatim") or {}).get("rate"),
        ((result.get("leakage") or {}).get("verbatim") or {}).get("rate"),
        False,
    )
    for reg in (result.get("av_by_register") or {}):
        pair(
            f"AV attribution — {reg}",
            ((b.get("av_by_register") or {}).get(reg) or {}).get("attribution_rate"),
            (result["av_by_register"][reg] or {}).get("attribution_rate"),
            True,
        )
    result["baseline_comparison"] = cmp
    return result


def load_fluency_log(path: str | Path) -> dict[str, Any] | None:
    """Last entry of a run's fluency_log.jsonl.

    The fluency probe runs during training (it needs the live model with and without the adapter),
    so `wlm eval run` reads its verdict from the log rather than recomputing it. Without this the
    report has no fluency block, and "done" is gated on a verdict the report never shows.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "fluency_log.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return rows[-1] if rows else None


def run_from_files(
    *,
    gen_path: str | Path,
    ref_path: str | Path,
    av_dir: str | Path | None = None,
    train_path: str | Path | None = None,
    out_html: str | Path = "runs/report.html",
    baseline_json: str | Path | None = None,
    run_name: str = "run",
    model: str = "",
    split: str = "blind",
    fluency_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gens = read_jsonl(gen_path)
    refs = read_jsonl(ref_path)
    verifier = None
    if av_dir and Path(av_dir).exists():
        from wlm.eval.av import AuthorshipVerifier

        verifier = AuthorshipVerifier.load(av_dir)
    else:
        print(
            f"[eval] no fitted verifier at {av_dir} — the report will have stylometry and leakage "
            "but no authorship-verification number, which is the primary metric. Run "
            "`wlm eval fit-av` first."
        )
    train_texts = [r["response"] for r in read_jsonl(train_path)] if train_path else None

    # Pick up the run's own config and fluency log when they sit next to the generations.
    run_dir = Path(gen_path).parent
    if config is None and (run_dir / "run_meta.json").exists():
        config = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")).get("config")
    fluency = load_fluency_log(fluency_path or run_dir)

    result = evaluate(
        generations=gens,
        real_reference=refs,
        verifier=verifier,
        train_texts=train_texts,
        fluency=fluency,
        config=config,
        run_name=run_name,
        split=split,
        model=model,
    )
    if baseline_json and Path(baseline_json).exists():
        result = compare_to_baseline(result, baseline_json)
    write_report(result, out_html)

    failures = [
        v for v in (
            (result.get("fluency") or {}).get("verdict"),
            (result.get("leakage") or {}).get("verdict"),
            (result.get("av") or {}).get("verifier_verdict"),
        )
        if v and v.upper().startswith("FAIL")
    ]
    if failures:
        print(f"[eval] VETO — this run is not a success regardless of its AV score: {failures}")
    return result
