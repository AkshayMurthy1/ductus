"""wlm — one CLI for the whole pipeline.

    wlm ingest local|gdocs      corpus -> data/interim/docs.jsonl
    wlm chunk                   docs -> response-sized chunks (+ health report)
    wlm scrub                   PII/entity scrubbing of targets (+ audit log)
    wlm backtranslate           chunks -> (question, chunk) pairs via the API generator
    wlm split                   pairs -> train/val/blind, split BY DOCUMENT
    wlm eval fit-av             fit the authorship verifier on train vs distractors
    wlm baseline                Phase-0 few-shot prompting baseline  [GPU]
    wlm train sft|dpo           Stage A / Stage B                    [GPU]
    wlm generate                generate on a split with an adapter  [GPU]
    wlm dpo-pairs               Stage-A model -> preference pairs     [GPU]
    wlm eval run                generations -> HTML report + verdicts
    wlm pipeline                Phase-5 one-command run
    wlm demo                    CPU smoke test on the bundled fixture author
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wlm import paths
from wlm.config import Config
from wlm.paths import INTERIM, PROCESSED, RUNS, read_jsonl, write_jsonl


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _p(msg: str) -> None:
    print(msg, flush=True)


def _dump(label: str, obj) -> None:
    _p(f"\n[{label}]\n{json.dumps(obj, indent=2, default=str)}\n")


# --------------------------------------------------------------------------- ingest
def cmd_ingest(a) -> int:
    paths.ensure_dirs()
    if a.source == "local":
        from wlm.ingest.local import ingest_local

        docs = ingest_local(
            a.input, register=a.register, keep_templates=getattr(a, "keep_templates", False)
        )
    else:
        from wlm.ingest.gdocs import ingest_gdocs

        docs = ingest_gdocs(
            folder_id=a.folder_id,
            typed_only=a.typed_only,
            limit=a.limit,
            register=a.register or "unknown",
        )
    if not docs:
        _p("no documents ingested — nothing downstream will work. Check the messages above.")
        return 1
    n = write_jsonl(a.out, docs)
    words = sum(len(d["text"].split()) for d in docs)
    _p(f"wrote {n} document(s), {words:,} words -> {a.out}")
    if words < 15_000:
        _p(
            f"NOTE {words:,} words is a small corpus. Expect the informal register to be the "
            "hard case and the val curve to be noisy. More first-party prose is the single "
            "highest-leverage thing you can add to this project."
        )
    return 0


# --------------------------------------------------------------------------- chunk
def cmd_chunk(a) -> int:
    from wlm.chunk import chunk_health, chunk_records

    cfg = Config.load(a.config) if a.config else Config()
    docs = read_jsonl(a.input)
    rows = chunk_records(
        docs, min_words=cfg.data.chunk_min_words, max_words=cfg.data.chunk_max_words
    )
    write_jsonl(a.out, rows)
    health = chunk_health(rows)
    _dump("chunk health", health)
    if health.get("frac_starts_mid_thought", 0) > 0.35:
        _p(
            "WARNING >35% of chunks open mid-thought. Their backtranslated questions will be "
            "wrong, and the resulting pairs are noise. Chunk quality bounds target quality — "
            "fix this before spending API budget on backtranslation."
        )
    _p(f"wrote {len(rows)} chunk(s) -> {a.out}")
    return 0


# --------------------------------------------------------------------------- scrub
def cmd_scrub(a) -> int:
    from wlm.scrub import scrub_records

    cfg = Config.load(a.config) if a.config else Config()
    rows = read_jsonl(a.input)
    terms = [t.strip() for t in (a.terms or "").split(",") if t.strip()]
    if a.terms_file:
        terms += [
            ln.strip() for ln in Path(a.terms_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    kept, log, summary = scrub_records(
        rows,
        entities=cfg.data.scrub_entities and not a.no_entities,
        extra_terms=terms,
        never_scrub=cfg.data.never_scrub,
    )
    write_jsonl(a.out, kept)
    write_jsonl(paths.SCRUB_LOG, log)
    _dump("scrub summary", summary)
    _p(f"wrote {len(kept)} chunk(s) -> {a.out}; audit log -> {paths.SCRUB_LOG}")
    if not summary["spacy_ner"]:
        _p(
            "spaCy NER was unavailable, so only regex rules ran. Entity scrubbing is the main "
            "defense against learning content instead of style — install the model before a "
            "real run: python -m spacy download en_core_web_sm"
        )
    return 0


# --------------------------------------------------------------------------- backtranslate
def cmd_backtranslate(a) -> int:
    _load_env()
    from wlm.backtranslate import build_pairs

    cfg = Config.load(a.config) if a.config else Config()
    chunks = read_jsonl(a.input)
    pairs, stats = build_pairs(
        chunks,
        n_per_chunk=a.n or cfg.data.questions_per_chunk,
        offline=a.offline,
        model=a.model,
        audit=a.audit,
        use_supplied_prompts=cfg.data.use_supplied_prompts,
        supplied_prompt_scope=cfg.data.supplied_prompt_scope,
    )
    write_jsonl(a.out, pairs)
    _dump("backtranslation stats", stats)
    _p(f"wrote {len(pairs)} pair(s) -> {a.out}")
    return 0


# --------------------------------------------------------------------------- split
def cmd_split(a) -> int:
    from wlm.dataset import split_pairs, split_summary, write_splits

    cfg = Config.load(a.config) if a.config else Config()
    pairs = read_jsonl(a.input)
    splits = split_pairs(
        pairs,
        val_frac=cfg.data.val_frac,
        blind_frac=cfg.data.blind_frac,
        by=cfg.data.split_by,
        seed=cfg.data.seed,
        group_near_duplicates=cfg.data.group_near_duplicates,
        near_dup_threshold=cfg.data.near_dup_threshold,
    )
    counts = write_splits(splits, a.outdir)
    summary = split_summary(splits)
    _dump("split summary", summary)
    bad = [k for k, v in summary["_doc_overlap"].items() if v]
    if bad:
        _p(f"ERROR document overlap between splits: {bad}. Every number downstream is invalid.")
        return 1
    _p(f"wrote {counts} -> {a.outdir}")

    for name in ("val", "blind"):
        if not counts.get(name):
            _p(
                f"ERROR the {name} split is empty. Splits are allocated per register and need at "
                "least 3 documents in a register to fill both val and blind. Add documents, or "
                "for a throwaway debugging run only, set data.split_by: pair (it leaks — never "
                "report a number from it)."
            )
            return 1
    train_regs = set(summary["train"]["registers"])
    for name in ("val", "blind"):
        missing = train_regs - set(summary[name]["registers"])
        if missing:
            _p(
                f"WARNING register(s) {sorted(missing)} appear in train but not in {name}. The "
                "informal register is the hard case this project targets — if it is missing from "
                "blind, the headline metric cannot measure it."
            )
    return 0


# --------------------------------------------------------------------------- eval: fit AV
def cmd_eval_fit_av(a) -> int:
    from wlm.eval.av import AuthorshipVerifier, StyleEmbedder, load_distractor_texts

    cfg = Config.load(a.config) if a.config else Config()
    rows = read_jsonl(a.author)
    # Dedupe on the way in: several questions per chunk means several rows share one response.
    by_text: dict[str, str] = {}
    for r in rows:
        by_text.setdefault(r["response"].strip(), r.get("doc_id", "unknown"))
    author = list(by_text.keys())
    author_groups = list(by_text.values())

    distractor = load_distractor_texts(
        a.distractor,
        min_words=cfg.data.chunk_min_words,
        max_words=cfg.data.chunk_max_words,
        limit=a.limit,
    )
    # Balance the classes so the threshold isn't dominated by whichever side is bigger.
    if a.balance:
        k = min(len(author), len(distractor))
        author, author_groups, distractor = author[:k], author_groups[:k], distractor[:k]

    v = AuthorshipVerifier(StyleEmbedder(cfg.eval.style_embedder), head=cfg.eval.av_classifier)
    metrics = v.fit(author, distractor, seed=cfg.data.seed, author_groups=author_groups)
    v.save(a.out)
    _dump("verifier metrics", metrics.__dict__)
    _p(f"saved verifier -> {a.out}")
    return 0 if metrics.auc >= 0.75 else 1


# --------------------------------------------------------------------------- eval: run
def cmd_eval_run(a) -> int:
    from wlm.eval.harness import run_from_files

    result = run_from_files(
        gen_path=a.gen,
        ref_path=a.ref or (PROCESSED / f"{a.split}.jsonl"),
        av_dir=a.av,
        train_path=a.train or (PROCESSED / "train.jsonl"),
        out_html=a.out,
        baseline_json=a.baseline,
        run_name=a.run_name,
        model=a.model or "",
        split=a.split,
        fluency_path=a.fluency,
    )
    _dump(
        "headline",
        {
            "av": result.get("av", {}).get("attribution_rate"),
            "av_by_register": {
                k: v.get("attribution_rate")
                for k, v in (result.get("av_by_register") or {}).items()
            },
            "stylometry_overall": result.get("stylometry", {}).get("overall"),
            "fluency": (result.get("fluency") or {}).get("verdict"),
            "leakage": result.get("leakage", {}).get("verdict"),
            "top_gaps": result.get("biggest_gaps", [])[:4],
        },
    )
    _p(f"report -> {a.out}")
    # Non-zero exit on a veto so a shell loop over ablations stops rather than logging a
    # "success" that failed the fluency or leakage gate.
    for block in ("fluency", "leakage"):
        v = (result.get(block) or {}).get("verdict", "")
        if v.upper().startswith("FAIL"):
            return 2
    return 0


# --------------------------------------------------------------------------- GPU commands
def cmd_baseline(a) -> int:
    from wlm.generate import run_generation

    cfg = Config.load(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = run_generation(
        cfg,
        split_path=PROCESSED / f"{a.split}.jsonl",
        out_path=out / "gen.jsonl",
        adapter=None,
        fewshot_from=PROCESSED / "train.jsonl",  # exemplars from TRAIN only
        n_shots=a.n_shots,
    )
    _p(f"{n} baseline generation(s) -> {out / 'gen.jsonl'}")
    _p("now: wlm eval run --gen " + str(out / "gen.jsonl") + f" --split {a.split} --out "
       + str(out / "report.html") + " --run-name baseline")
    return 0


def cmd_train(a) -> int:
    cfg = Config.load(a.config)
    if a.stage == "sft":
        from wlm.train.stage_a_sft import train_stage_a

        m = train_stage_a(cfg, outdir=a.out, train_path=a.train, val_path=a.val)
    else:
        from wlm.train.stage_b_dpo import train_stage_b

        if not a.adapter:
            _p("`wlm train dpo` requires --adapter pointing at the Stage-A adapter directory.")
            return 2
        if not Path(a.adapter).exists():
            _p(f"--adapter {a.adapter} does not exist.")
            return 2
        m = train_stage_b(
            cfg,
            outdir=a.out,
            stage_a_adapter=a.adapter,
            pairs_path=a.pairs,
            val_pairs_path=a.val_pairs,
        )
    _dump("training metrics", {k: v for k, v in m.items() if k != "log_history"})
    return 0


def cmd_generate(a) -> int:
    from wlm.generate import run_generation

    cfg = Config.load(a.config)
    n = run_generation(
        cfg,
        split_path=a.split_path or (PROCESSED / f"{a.split}.jsonl"),
        out_path=a.out,
        adapter=a.adapter,
    )
    _p(f"{n} generation(s) -> {a.out}")
    return 0


def cmd_dpo_pairs(a) -> int:
    from wlm.dpo_pairs import build_dpo_pairs

    cfg = Config.load(a.config)
    stats = build_dpo_pairs(
        cfg,
        adapter=a.adapter,
        split_path=a.split_path or (PROCESSED / f"{a.split}.jsonl"),
        out_path=a.out,
        av_dir=a.av,
        val_frac=a.val_frac,
    )
    _dump("dpo pair stats", stats)
    if stats["kept"] < 40:
        _p("Too few surviving pairs for DPO to be more than noise. Read the drop counts above.")
        return 1
    return 0


def cmd_pipeline(a) -> int:
    """Phase 5: connect -> scrub -> pairs -> Stage A -> Stage B -> eval report."""
    import subprocess

    steps = [
        ["wlm", "ingest", "local", "--in", str(paths.AUTHOR_RAW), "--out",
         str(INTERIM / "docs.jsonl")],
        ["wlm", "chunk", "--in", str(INTERIM / "docs.jsonl"), "--out",
         str(INTERIM / "chunks.jsonl"), "--config", a.config],
        ["wlm", "scrub", "--in", str(INTERIM / "chunks.jsonl"), "--out",
         str(INTERIM / "scrubbed.jsonl"), "--config", a.config],
        ["wlm", "backtranslate", "--in", str(INTERIM / "scrubbed.jsonl"), "--out",
         str(INTERIM / "pairs.jsonl"), "--config", a.config],
        ["wlm", "split", "--in", str(INTERIM / "pairs.jsonl"), "--outdir", str(PROCESSED),
         "--config", a.config],
        ["wlm", "eval", "fit-av", "--author", str(PROCESSED / "train.jsonl"),
         "--distractor", str(paths.DISTRACTOR_RAW), "--out", str(RUNS / "av")],
        ["wlm", "baseline", "--config", a.config, "--out", str(RUNS / "baseline")],
        ["wlm", "eval", "run", "--gen", str(RUNS / "baseline" / "gen.jsonl"),
         "--av", str(RUNS / "av"), "--out", str(RUNS / "baseline" / "report.html"),
         "--run-name", "baseline"],
        ["wlm", "train", "sft", "--config", a.config, "--out", str(RUNS / "stage_a")],
        ["wlm", "generate", "--config", a.config, "--adapter", str(RUNS / "stage_a"),
         "--out", str(RUNS / "stage_a" / "gen.jsonl")],
        ["wlm", "eval", "run", "--gen", str(RUNS / "stage_a" / "gen.jsonl"),
         "--av", str(RUNS / "av"), "--out", str(RUNS / "stage_a" / "report.html"),
         "--run-name", "stage_a", "--baseline", str(RUNS / "baseline" / "report.json")],
    ]
    if a.with_dpo:
        steps += [
            ["wlm", "dpo-pairs", "--config", "configs/stage_b.yaml", "--adapter",
             str(RUNS / "stage_a"), "--split", "val", "--out", str(PROCESSED / "dpo.jsonl")],
            ["wlm", "train", "dpo", "--config", "configs/stage_b.yaml", "--adapter",
             str(RUNS / "stage_a"), "--out", str(RUNS / "stage_b"),
             "--pairs", str(PROCESSED / "dpo.jsonl"),
             "--val-pairs", str(PROCESSED / "dpo_val.jsonl")],
            ["wlm", "generate", "--config", "configs/stage_b.yaml", "--adapter",
             str(RUNS / "stage_b"), "--out", str(RUNS / "stage_b" / "gen.jsonl")],
            ["wlm", "eval", "run", "--gen", str(RUNS / "stage_b" / "gen.jsonl"),
             "--av", str(RUNS / "av"), "--out", str(RUNS / "stage_b" / "report.html"),
             "--run-name", "stage_b", "--baseline", str(RUNS / "baseline" / "report.json")],
        ]
    for step in steps:
        _p(f"\n$ {' '.join(step)}")
        if a.dry_run:
            continue
        rc = subprocess.call(step)
        if rc != 0:
            _p(f"step failed with rc={rc}; stopping.")
            return rc
    return 0


# --------------------------------------------------------------------------- demo
def cmd_demo(a) -> int:
    """CPU-only end-to-end smoke test. No API key, no GPU, no network."""
    from wlm.chunk import chunk_health, chunk_records
    from wlm.dataset import split_pairs, split_summary
    from wlm.eval import stylometry as sty
    from wlm.eval.leakage import run_leakage_suite
    from wlm.eval.report import write_report
    from wlm.ingest.local import ingest_local
    from wlm.scrub import scrub_records

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

    docs = ingest_local(fixture / "author", register="informal")
    _p(f"[demo] {len(docs)} document(s)")
    chunks = chunk_records(docs, min_words=40, max_words=200)
    _dump("chunk health", chunk_health(chunks))
    kept, _log, summary = scrub_records(chunks)
    _dump("scrub summary", summary)

    from wlm.backtranslate import build_pairs

    pairs, stats = build_pairs(kept, n_per_chunk=2, offline=True)
    _dump("backtranslation (offline templates)", stats)

    splits = split_pairs(pairs, val_frac=0.2, blind_frac=0.2)
    _dump("split summary", split_summary(splits))

    # Stand in for model output with a deliberately generic-voiced paraphrase so the report has
    # something to render and the stylometry gap is visibly non-zero.
    ref = splits["blind"] or splits["train"]
    fake_gen = [
        {
            "pair_id": r.get("pair_id"),
            "register": r.get("register"),
            "prompt": r["prompt"],
            "completion": _genericize(r["response"]),
        }
        for r in ref
    ]
    real_vec = sty.stylometry_vector("\n\n".join(r["response"] for r in ref))
    gen_vec = sty.stylometry_vector("\n\n".join(g["completion"] for g in fake_gen))
    dist = sty.stylometry_distance(real_vec, gen_vec)

    from wlm.eval.report import SL_BUCKETS

    result = {
        "run_name": "demo (synthetic generations)",
        "split": "blind",
        "model": "none — CPU smoke test",
        "n_generations": len(fake_gen),
        "generated_at": "demo",
        "stylometry": dist,
        "biggest_gaps": sty.biggest_gaps(dist, k=8),
        "sent_len_hist": {
            "buckets": SL_BUCKETS,
            "real": [round(float(x), 4) for x in real_vec["dists"]["sent_len_hist"]],
            "generated": [round(float(x), 4) for x in gen_vec["dists"]["sent_len_hist"]],
        },
        "leakage": run_leakage_suite(
            [g["completion"] for g in fake_gen], [r["response"] for r in splits["train"]]
        ),
        "samples": [
            {"prompt": g["prompt"], "generated": g["completion"], "real": r["response"]}
            for g, r in zip(fake_gen[:4], ref[:4], strict=False)
        ],
        "config": {"note": "demo run; no training happened"},
    }
    write_report(result, out / "report.html")
    _p(f"\n[demo] report -> {out / 'report.html'}")
    _p("[demo] the CPU half of the pipeline works. Add your own corpus to data/raw/author "
       "and an ANTHROPIC_API_KEY to do it for real.")
    return 0


def _genericize(text: str) -> str:
    """Flatten a passage toward 'generic AI voice' -- the demo needs a nonzero style gap."""
    import re

    t = re.sub(r"\s*—\s*", ", ", text)
    t = t.replace(";", ".").replace("(", "").replace(")", "")
    t = re.sub(r"\b(\w+)n['’]t\b", r"\1 not", t)
    t = re.sub(r"\b(I|we|you)['’](m|re|ve|ll|d)\b", r"\1 would", t)
    t = re.sub(r"\b(really|honestly|basically|actually|pretty|kind of|sort of)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    return "It is important to note that " + t[0].lower() + t[1:] if t else t


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wlm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="pull a corpus")
    ing_sub = ing.add_subparsers(dest="source", required=True)
    for name in ("local", "gdocs"):
        s = ing_sub.add_parser(name)
        s.add_argument("--out", default=str(INTERIM / "docs.jsonl"))
        s.add_argument("--register", default=None, choices=[None, "formal", "informal", "unknown"])
        if name == "local":
            s.add_argument("--in", dest="input", default=str(paths.AUTHOR_RAW))
            # The template filter is tuned for forms and outlines; short-line prose such as a
            # letter trips it. Keep the escape hatch visible rather than losing real writing.
            s.add_argument("--keep-templates", action="store_true", default=False)
        else:
            s.add_argument("--folder-id", default=None)
            s.add_argument("--typed-only", action="store_true", default=True)
            s.add_argument("--allow-pasted", dest="typed_only", action="store_false")
            s.add_argument("--limit", type=int, default=None)
    ing.set_defaults(func=cmd_ingest)

    c = sub.add_parser("chunk")
    c.add_argument("--in", dest="input", default=str(INTERIM / "docs.jsonl"))
    c.add_argument("--out", default=str(INTERIM / "chunks.jsonl"))
    c.add_argument("--config", default=None)
    c.set_defaults(func=cmd_chunk)

    s = sub.add_parser("scrub")
    s.add_argument("--in", dest="input", default=str(INTERIM / "chunks.jsonl"))
    s.add_argument("--out", default=str(INTERIM / "scrubbed.jsonl"))
    s.add_argument("--config", default=None)
    s.add_argument("--terms", default=None, help="comma-separated always-scrub terms")
    s.add_argument("--terms-file", default=None)
    s.add_argument("--no-entities", action="store_true", help="ablation a14 only")
    s.set_defaults(func=cmd_scrub)

    b = sub.add_parser("backtranslate")
    b.add_argument("--in", dest="input", default=str(INTERIM / "scrubbed.jsonl"))
    b.add_argument("--out", default=str(INTERIM / "pairs.jsonl"))
    b.add_argument("--config", default=None)
    b.add_argument("-n", type=int, default=None, help="questions per chunk")
    b.add_argument("--model", default=None)
    b.add_argument("--offline", action="store_true", help="template questions; smoke tests only")
    b.add_argument("--audit", action="store_true", help="back-and-forth rewrite/filter pass")
    b.set_defaults(func=cmd_backtranslate)

    sp = sub.add_parser("split")
    sp.add_argument("--in", dest="input", default=str(INTERIM / "pairs.jsonl"))
    sp.add_argument("--outdir", default=str(PROCESSED))
    sp.add_argument("--config", default=None)
    sp.set_defaults(func=cmd_split)

    ev = sub.add_parser("eval")
    ev_sub = ev.add_subparsers(dest="which", required=True)
    fa = ev_sub.add_parser("fit-av")
    fa.add_argument("--author", default=str(PROCESSED / "train.jsonl"))
    fa.add_argument("--distractor", default=str(paths.DISTRACTOR_RAW))
    fa.add_argument("--out", default=str(RUNS / "av"))
    fa.add_argument("--config", default=None)
    fa.add_argument("--limit", type=int, default=None)
    fa.add_argument("--balance", action="store_true", default=True)
    fa.add_argument("--no-balance", dest="balance", action="store_false")
    fa.set_defaults(func=cmd_eval_fit_av)
    er = ev_sub.add_parser("run")
    er.add_argument("--gen", required=True)
    er.add_argument("--ref", default=None)
    er.add_argument("--split", default="blind")
    er.add_argument("--av", default=str(RUNS / "av"))
    er.add_argument("--train", default=None)
    er.add_argument("--out", default=str(RUNS / "report.html"))
    er.add_argument("--baseline", default=None, help="baseline report.json for the delta block")
    er.add_argument("--run-name", default="run")
    er.add_argument("--model", default=None)
    er.add_argument("--fluency", default=None,
                    help="run dir or fluency_log.jsonl; defaults to the generations' directory")
    er.set_defaults(func=cmd_eval_run)

    bl = sub.add_parser("baseline", help="Phase-0 few-shot prompting baseline [GPU]")
    bl.add_argument("--config", default=str(paths.CONFIGS / "stage_a.yaml"))
    bl.add_argument("--out", default=str(RUNS / "baseline"))
    bl.add_argument("--split", default="blind")
    bl.add_argument("--n-shots", type=int, default=None)
    bl.set_defaults(func=cmd_baseline)

    tr = sub.add_parser("train", help="[GPU]")
    tr.add_argument("stage", choices=["sft", "dpo"])
    tr.add_argument("--config", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--train", default=None)
    tr.add_argument("--val", default=None)
    tr.add_argument("--adapter", default=None, help="Stage-A adapter, required for dpo")
    tr.add_argument("--pairs", default=None)
    tr.add_argument("--val-pairs", default=None, help="held-out preference pairs for dpo eval")
    tr.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="[GPU]")
    g.add_argument("--config", default=str(paths.CONFIGS / "stage_a.yaml"))
    g.add_argument("--adapter", default=None)
    g.add_argument("--split", default="blind")
    g.add_argument("--split-path", default=None)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    dp = sub.add_parser("dpo-pairs", help="[GPU]")
    dp.add_argument("--config", default=str(paths.CONFIGS / "stage_b.yaml"))
    dp.add_argument("--adapter", required=True)
    dp.add_argument("--split", default="val")
    dp.add_argument("--split-path", default=None)
    dp.add_argument("--out", default=str(PROCESSED / "dpo.jsonl"))
    dp.add_argument("--av", default=str(RUNS / "av"))
    dp.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction held back to <out>_val.jsonl so Stage B can run eval")
    dp.set_defaults(func=cmd_dpo_pairs)

    pl = sub.add_parser("pipeline", help="Phase-5 one-command run")
    pl.add_argument("--config", default=str(paths.CONFIGS / "stage_a.yaml"))
    pl.add_argument("--with-dpo", action="store_true")
    pl.add_argument("--dry-run", action="store_true")
    pl.set_defaults(func=cmd_pipeline)

    dm = sub.add_parser("demo", help="CPU smoke test on the bundled fixture author")
    dm.add_argument("--outdir", default="data/demo")
    dm.set_defaults(func=cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
