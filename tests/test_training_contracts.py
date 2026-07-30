"""Contracts between the CPU pipeline and the GPU trainers.

The training code itself can't run without CUDA, but its *inputs* can be checked here — which is
where most of the expensive mistakes live: a dataset in the wrong trl format, a prompt format that
differs between training and inference, a split that leaves val empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wlm.backtranslate import build_pairs
from wlm.chunk import chunk_records
from wlm.config import Config
from wlm.dataset import split_pairs, to_prompt_completion, to_self_continuation
from wlm.ingest.local import ingest_local
from wlm.scrub import scrub_records
from wlm.train.stage_a_sft import _pc_rows, _raw_rows
from wlm.train.stage_b_dpo import _to_conversational

FIXTURES = Path(__file__).parent / "fixtures" / "author"
SYSTEM = "You are writing as the author."


@pytest.fixture(scope="module")
def pairs():
    docs = ingest_local(FIXTURES, register="informal")
    chunks = chunk_records(docs, min_words=40, max_words=200)
    kept, _log, _s = scrub_records(chunks, entities=False)
    rows, _stats = build_pairs(kept, n_per_chunk=2, offline=True)
    return rows


# --------------------------------------------------------------------------- trl formats
def test_sft_rows_are_conversational_prompt_completion(pairs):
    """trl only applies the chat template when it sees message lists, and only computes
    completion-only loss when the row has separate prompt/completion keys."""
    rows = _pc_rows(pairs, SYSTEM)
    assert rows
    for r in rows:
        assert set(r) >= {"prompt", "completion"}
        assert isinstance(r["prompt"], list) and isinstance(r["completion"], list)
        assert [m["role"] for m in r["prompt"]] == ["system", "user"]
        assert [m["role"] for m in r["completion"]] == ["assistant"]
        assert all(isinstance(m["content"], str) and m["content"] for m in r["prompt"])


def test_sft_prompt_format_matches_generation_format(pairs):
    """The system prompt must be present in training exactly when generate() sends it.

    generate() always prepends cfg.gen.system_prompt, so training rows must carry it too —
    otherwise the adapter is tuned on a format it never sees at eval or serve time.
    """
    cfg = Config.load(Path(__file__).parents[1] / "configs" / "stage_a.yaml")
    row = to_prompt_completion({"prompt": "q", "response": "a"}, cfg.gen.system_prompt)
    assert row["prompt"][0] == {"role": "system", "content": cfg.gen.system_prompt}


def test_dpo_rows_are_conversational(pairs):
    cfg = Config.load(Path(__file__).parents[1] / "configs" / "stage_b.yaml")
    row = _to_conversational(
        {"prompt": "q", "chosen": "real passage", "rejected": "model passage"}, cfg
    )
    assert set(row) == {"prompt", "chosen", "rejected"}
    for key in ("prompt", "chosen", "rejected"):
        assert isinstance(row[key], list), f"{key} must be a message list, not a raw string"
    assert row["chosen"][0]["role"] == "assistant"
    assert row["prompt"][0]["role"] == "system"


def test_dpo_and_sft_share_the_same_system_prompt():
    """A different system prompt between stages makes Stage-B's reward incomparable to Stage A."""
    a = Config.load(Path(__file__).parents[1] / "configs" / "stage_a.yaml")
    b = Config.load(Path(__file__).parents[1] / "configs" / "stage_b.yaml")
    assert a.gen.system_prompt == b.gen.system_prompt


# --------------------------------------------------------------------------- raw mix
def test_self_continuation_splits_a_passage_losslessly(pairs):
    from wlm.dataset import CONTINUE_INSTRUCTION

    row = to_self_continuation(pairs[0])
    assert row is not None
    lead = row["prompt"][-1]["content"].removeprefix(CONTINUE_INSTRUCTION)
    rest = row["completion"][0]["content"]
    assert lead and rest
    source = pairs[0]["response"]
    assert lead in source, "the lead must be the author's own opening, verbatim"
    # Lossless: lead + rest reconstructs the passage, so paragraph breaks survive the cut.
    assert (lead + rest).replace(" ", "").replace("\n", "") == source.replace(
        " ", ""
    ).replace("\n", "")


def test_self_continuation_carries_the_same_system_turn_as_the_pairs(pairs):
    """Two prompt shapes in one training set is a silent format split."""
    row = to_self_continuation(pairs[0], system_prompt=SYSTEM)
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][0]["content"] == SYSTEM


def test_self_continuation_declines_short_passages():
    assert to_self_continuation({"prompt": "q", "response": "Too short. Really."}) is None


def test_raw_mix_is_deduped_and_seed_sampled(pairs):
    """build_pairs emits several pairs per chunk; the raw mix must not train on doubles."""
    rows = _raw_rows(pairs, 0.5, seed=17)
    leads = [r["prompt"][-1]["content"] for r in rows]
    assert len(leads) == len(set(leads)), "duplicate passages in the raw mix double memorization"
    again = [r["prompt"][-1]["content"] for r in _raw_rows(pairs, 0.5, seed=17)]
    assert leads == again, "raw mix must be deterministic for a given seed"
    other = [r["prompt"][-1]["content"] for r in _raw_rows(pairs, 0.5, seed=99)]
    assert isinstance(other, list)


def test_raw_mix_is_empty_when_disabled(pairs):
    assert _raw_rows(pairs, 0.0, seed=17) == []


def test_raw_mix_rows_have_the_same_schema_as_sft_rows(pairs):
    """They go into one Dataset; a schema mismatch is a pyarrow error mid-run."""
    sft = _pc_rows(pairs, SYSTEM)
    raw = _raw_rows(pairs, 0.3, seed=17, system_prompt=SYSTEM)
    assert raw
    assert set(sft[0]) == set(raw[0])
    # Same prompt shape too, not just the same keys.
    assert [m["role"] for m in sft[0]["prompt"]] == [m["role"] for m in raw[0]["prompt"]]


# --------------------------------------------------------------------------- splits
def test_three_documents_still_yield_a_val_split(pairs):
    """Early stopping, best-checkpoint restore and the fluency probe all key off eval_loss."""
    splits = split_pairs(pairs, by="document")
    assert splits["val"], "an empty val split makes the overfitting guards inoperative"
    assert splits["blind"]
    assert splits["train"]


def test_split_fractions_are_respected_on_a_larger_corpus():
    fake = [
        {"pair_id": f"p{i}", "doc_id": f"d{i // 2}", "register": "informal",
         "prompt": "q", "response": "a", "n_words": 10}
        for i in range(40)
    ]
    splits = split_pairs(fake, val_frac=0.2, blind_frac=0.2, by="document")
    docs = {k: {r["doc_id"] for r in v} for k, v in splits.items()}
    assert len(docs["val"]) == 4 and len(docs["blind"]) == 4 and len(docs["train"]) == 12
    assert not (docs["train"] & docs["val"]) and not (docs["train"] & docs["blind"])
