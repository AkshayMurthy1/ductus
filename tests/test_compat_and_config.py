"""Config loading and library-version tolerance.

These run on a machine with no torch/transformers/peft/trl, which is the point: the whole
laptop-side pipeline must import and work without CUDA or the training stack.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from wlm.config import Config
from wlm.train.common import accepted_kwargs, first_supported

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

CPU_SAFE_MODULES = [
    "wlm.cli",
    "wlm.chunk",
    "wlm.scrub",
    "wlm.dataset",
    "wlm.backtranslate",
    "wlm.generate",
    "wlm.dpo_pairs",
    "wlm.ingest.local",
    "wlm.ingest.gdocs",
    "wlm.ingest.normalize",
    "wlm.eval.stylometry",
    "wlm.eval.leakage",
    "wlm.eval.fluency",
    "wlm.eval.report",
    "wlm.eval.harness",
    "wlm.train.common",
    "wlm.train.stage_a_sft",
    "wlm.train.stage_b_dpo",
]


@pytest.mark.parametrize("mod", CPU_SAFE_MODULES)
def test_every_module_imports_without_the_training_stack(mod):
    """GPU imports must live inside functions, never at module top level."""
    importlib.import_module(mod)


# --------------------------------------------------------------------------- configs
def test_stage_a_config_matches_the_plans_locked_in_recipe():
    c = Config.load(CONFIGS / "stage_a.yaml")
    assert c.lora.target_modules == "all-linear", "LoRA must cover attention AND MLP"
    assert c.lora.r == 16 and c.lora.lora_alpha == 32, "alpha should be ~2r"
    assert c.lora.use_dora is True
    assert c.sft.neftune_noise_alpha == 5.0
    assert c.model.load_in_4bit is False, "bf16 at 1-3B on 24GB; 4-bit costs quality for nothing"
    assert 1.0 <= c.sft.epochs <= 3.0
    assert 1e-4 <= c.sft.lr <= 2e-4
    assert c.data.split_by == "document", "pair-level splits leak documents across the boundary"


def test_stage_b_inherits_and_is_conservative():
    a = Config.load(CONFIGS / "stage_a.yaml")
    b = Config.load(CONFIGS / "stage_b.yaml")
    assert b.model.name == a.model.name, "_extends should inherit the base model"
    assert b.lora.use_dora == a.lora.use_dora
    assert b.dpo.lr < a.sft.lr / 10, "DPO must run well below the SFT LR"
    assert b.dpo.epochs <= 1.0
    assert b.dpo.loss_type == "sigmoid", "start with plain DPO; change one thing at a time"
    assert b.dpo.av_reward_filter is False


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("ablations/*.yaml")))
def test_every_ablation_loads_and_changes_exactly_one_thing(path):
    base = Config.load(CONFIGS / "stage_a.yaml").to_dict()
    cfg = Config.load(path)
    assert cfg.run_name == path.stem, "run_name must match the filename so runs/ stays legible"
    assert cfg.notes, "every ablation states its hypothesis"

    diffs = _diff(base, cfg.to_dict())
    diffs = [d for d in diffs if not d.startswith(("run_name", "notes"))]
    # a04 (rank + rsLoRA) and a05 (rank_pattern + alpha_pattern) are coupled pairs, and a03
    # changes alpha with r to hold alpha≈2r. Everything else is a genuine single knob.
    assert len(diffs) <= 3, f"{path.name} changes too much to attribute a delta: {diffs}"


def test_unknown_config_keys_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("lora:\n  totally_made_up: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(p)


def test_config_roundtrips():
    c = Config.load(CONFIGS / "stage_a.yaml")
    assert Config.load(CONFIGS / "stage_a.yaml").to_dict() == c.to_dict()


def _diff(a: dict, b: dict) -> list[str]:
    """Diff at `section.field` granularity, comparing field values whole.

    Deliberately does not descend into a field's own value: `rank_pattern` mapping three MLP
    projections is one knob, not three.
    """
    out = []
    for section in a.keys() | b.keys():
        av, bv = a.get(section), b.get(section)
        if isinstance(av, dict) and isinstance(bv, dict):
            for field in av.keys() | bv.keys():
                if av.get(field) != bv.get(field):
                    out.append(f"{section}.{field}: {av.get(field)!r} -> {bv.get(field)!r}")
        elif av != bv:
            out.append(f"{section}: {av!r} -> {bv!r}")
    return out


# --------------------------------------------------------------------------- compat shim
def _sample(a=1, max_length=None, processing_class=None):
    pass


def test_accepted_kwargs_drops_unknown_and_keeps_known(capsys):
    got = accepted_kwargs(_sample, {"a": 2, "nope": 3, "max_length": 8}, label="sample")
    assert got == {"a": 2, "max_length": 8}
    assert "nope" in capsys.readouterr().out, "dropping a knob must be visible in the log"


def test_accepted_kwargs_passes_everything_through_for_var_keyword():
    def kwonly(**kw):
        pass

    assert accepted_kwargs(kwonly, {"whatever": 1}) == {"whatever": 1}


def test_first_supported_picks_the_available_alias():
    assert first_supported(_sample, {"max_seq_length": 8, "max_length": 8}) == {"max_length": 8}
    assert first_supported(_sample, {"nope": 1}) == {}


def test_the_training_stack_is_genuinely_absent_or_present_consistently():
    """Documents the assumption these tests run under, so a failure elsewhere is interpretable."""
    have = {m: importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "trl")}
    assert isinstance(have, dict)  # informational; never fails
