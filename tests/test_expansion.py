"""Tests for the venue-readiness expansion (docs/EXPANSION.md): new ablation configs, the
between-instrument agreement metric, and the new run_matrix sections.

All CPU, no network, no model downloads — the agreement metric is a pure function by design
so it can be held to the repo's metric contract here: zero on identical inputs, larger on
genuinely different ones.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wlm.config import Config  # noqa: E402
from wlm.eval.agreement import instrument_agreement, rank_disagreement, spearman  # noqa: E402

ABLATIONS = REPO / "configs" / "ablations"


# --------------------------------------------------------------------- new configs
def test_a15_extends_stage_b_and_only_renames():
    cfg = Config.load(ABLATIONS / "a15_attn_dpo.yaml")
    base = Config.load(REPO / "configs" / "stage_b.yaml")
    assert cfg.run_name == "a15_attn_dpo"
    # The DPO recipe must be untouched: the cell varies the adapter it sits on, nothing else.
    assert cfg.dpo == base.dpo
    assert cfg.model == base.model


def test_a16_changes_exactly_the_model():
    cfg = Config.load(ABLATIONS / "a16_llama3b.yaml")
    base = Config.load(REPO / "configs" / "stage_a.yaml")
    assert cfg.model.name == "meta-llama/Llama-3.2-3B-Instruct"
    assert (cfg.lora, cfg.sft, cfg.data) == (base.lora, base.sft, base.data)


@pytest.mark.parametrize(
    ("name", "modules"),
    [("a17_qk_only", ["q_proj", "k_proj"]), ("a18_vo_only", ["v_proj", "o_proj"])],
)
def test_per_matrix_ablations_change_exactly_the_target_modules(name, modules):
    cfg = Config.load(ABLATIONS / f"{name}.yaml")
    base = Config.load(REPO / "configs" / "stage_a.yaml")
    assert cfg.lora.target_modules == modules
    assert (cfg.model, cfg.sft, cfg.data) == (base.model, base.sft, base.data)


# --------------------------------------------------------------------- agreement metric
def test_rank_disagreement_zero_on_identical_larger_on_different():
    a = [0.1, 0.4, 0.7, 0.9]
    assert rank_disagreement(a, list(a)) == 0.0
    assert rank_disagreement(a, [0.11, 0.38, 0.72, 0.88]) == 0.0  # same ordering, same verdict
    shuffled = [0.4, 0.1, 0.9, 0.7]
    assert rank_disagreement(a, shuffled) > 0.0
    assert rank_disagreement(a, list(reversed(a))) == 1.0


def test_spearman_handles_ties_and_constant_inputs():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) == 1.0  # degenerate: no ordering to dispute
    with pytest.raises(ValueError):
        spearman([1.0], [1.0])


def test_instrument_agreement_flags_the_outlier_run():
    runs = [
        {"run": "a", "av_primary": 0.10, "av_second": 0.11},
        {"run": "b", "av_primary": 0.50, "av_second": 0.52},
        {"run": "c", "av_primary": 0.80, "av_second": 0.30},  # the disagreement
        {"run": "d", "av_primary": 0.70, "av_second": 0.69},
    ]
    out = instrument_agreement(runs)
    assert out["n"] == 4
    assert [f["run"] for f in out["flagged_runs"]] == ["c"]
    perfect = instrument_agreement(runs[:2])
    assert perfect["rank_disagreement"] == 0.0
    assert perfect["flagged_runs"] == []


# --------------------------------------------------------------------- driver + WLM_ROOT
def _fake_root(tmp_path: Path, with_sweep: bool) -> dict[str, str]:
    (tmp_path / "runs" / "av").mkdir(parents=True)
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "blind.jsonl").write_text('{"prompt": "q", "response": "a"}\n')
    (proc / "train.jsonl").write_text('{"prompt": "q", "response": "a", "doc_id": "d"}\n')
    if with_sweep:
        for arm in ("10k", "25k", "full"):
            d = proc / "sweep" / arm
            d.mkdir(parents=True)
            (d / "train.jsonl").write_text("{}\n")
            (d / "val.jsonl").write_text("{}\n")
    return {**os.environ, "WLM_ROOT": str(tmp_path)}


def test_run_matrix_dry_run_builds_rq2x_and_models_units(tmp_path):
    env = _fake_root(tmp_path, with_sweep=True)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_matrix.py"),
         "--dry-run", "--only", "rq2x", "models"],
        capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "a15_attn_dpo" in out.stdout
    assert "a16_llama3b" in out.stdout
    # rq2x depends on a01, which doesn't exist in the fake root -> skipped with a reason,
    # never silently dropped.
    assert "requires" in out.stdout
    # the models section carries its own baseline: an adapter must beat its own model's floor
    assert "baseline-a16_llama3b-10k" in out.stdout


def test_make_size_sweep_honors_wlm_root(tmp_path):
    env = {**os.environ, "WLM_ROOT": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_size_sweep.py")],
        capture_output=True, text=True, env=env, check=False)
    # Empty root: must fail by looking for train.jsonl under WLM_ROOT, not under the repo.
    assert out.returncode == 1
    assert str(tmp_path) in out.stdout


def test_new_author_scaffolds_outside_repo_only(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "new_author.py"), str(tmp_path / "orwell"),
         "--register", "informal", "--private"],
        capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    runbook = (tmp_path / "orwell" / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "PRIVATE AUTHOR" in runbook
    assert "WLM_ROOT" in runbook
    assert (tmp_path / "orwell" / "data" / "raw" / "author").is_dir()
    # Refuses to scaffold inside the repo.
    inside = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "new_author.py"), str(REPO / "data" / "x")],
        capture_output=True, text=True, check=False)
    assert inside.returncode == 1


def test_second_instrument_config_knob_exists():
    cfg = Config.load(REPO / "configs" / "stage_a.yaml")
    assert cfg.eval.second_style_embedder == "StyleDistance/styledistance"
    assert cfg.eval.second_style_embedder != cfg.eval.style_embedder


# --------------------------------------------------------------------- gutenberg extraction
def test_split_sections_drops_toc_and_editorial_preambles():
    from wlm.ingest.gutenberg import split_sections

    body_para = ("It was a bright morning and I walked out early. " * 40).strip()
    editorial = ("          Mr. Clemens was introduced by the president of the club.\n"
                 "          He spoke as follows that evening.")
    text = "\n\n".join([
        "CHAPTER I.", "CHAPTER II.",          # table of contents copies
        "CHAPTER I.", editorial, body_para,   # body: preamble + prose
        "CHAPTER II.", body_para + " " + body_para,
    ])
    import re
    secs = split_sections(text, min_words=100,
                          drop_indented_matching=re.compile(r"Mr\. Clemens"))
    assert [t for t, _ in secs] == ["CHAPTER I.", "CHAPTER II."]
    assert all("Clemens" not in body for _, body in secs)
    # TOC dedup kept the later (body) occurrence: section I holds prose, not the TOC gap.
    assert "bright morning" in secs[0][1]


def test_segment_fallback_covers_headingless_prose():
    from wlm.ingest.gutenberg import segment_fallback

    para = ("The river was quiet that year and we watched it from the porch every "
            "single evening without fail. " * 12).strip()
    text = "\n\n".join([para] * 12)
    segs = segment_fallback(text, target_words=400)
    assert len(segs) >= 3
    joined = " ".join(b for _, b in segs)
    assert "porch" in joined


def test_is_prose_rejects_verse_and_editorial():
    from wlm.ingest.gutenberg import is_prose

    prose = ("I remember the day well, because it rained and the roads were deep in mud. "
             "We argued about it for an hour and settled nothing at all. " * 5)
    verse = "\n".join(["the road goes on", "past the mill", "into the dark", "and on"] * 6)
    editorial = "This volume contains the collected speeches, proofread by the editor. " * 8
    assert is_prose(prose)
    assert not is_prose(verse)
    assert not is_prose(editorial)
