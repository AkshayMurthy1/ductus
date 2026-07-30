"""CPU tests for the parts of the pipeline that must be right before a GPU hour is spent."""

from __future__ import annotations

from pathlib import Path

import pytest

from wlm.backtranslate import build_pairs
from wlm.chunk import chunk_document, chunk_health, chunk_records, split_sentences
from wlm.dataset import split_pairs, split_summary
from wlm.ingest.local import ingest_local
from wlm.ingest.normalize import looks_like_template, normalize_document, prose_ratio
from wlm.scrub import placeholder_density, scrub_records, scrub_text

FIXTURES = Path(__file__).parent / "fixtures" / "author"


@pytest.fixture(scope="module")
def docs():
    return ingest_local(FIXTURES, register="informal")


@pytest.fixture(scope="module")
def chunks(docs):
    return chunk_records(docs, min_words=40, max_words=200)


# --------------------------------------------------------------------------- normalize
def test_signoff_and_sent_from_are_stripped(docs):
    joined = "\n".join(d["text"] for d in docs)
    assert "Sent from my iPhone" not in joined
    assert not joined.rstrip().endswith("Akshay")


def test_quoted_reply_is_stripped():
    text = "My actual paragraph here, long enough to matter.\n\n> their quoted line\n> more quote"
    out = normalize_document(text)
    assert "their quoted line" not in out
    assert "My actual paragraph" in out


def test_normalize_is_idempotent():
    raw = (FIXTURES / "notes_two.md").read_text(encoding="utf-8")
    once = normalize_document(raw)
    assert normalize_document(once) == once


def test_urls_become_placeholder():
    assert "<link>" in normalize_document("See https://example.com/x for the rest of it. " * 3)


def test_template_detection():
    assert looks_like_template("Name: ____\nDate: ____\nSignature: ____")
    assert not looks_like_template((FIXTURES / "essay_three.md").read_text(encoding="utf-8"))


def test_prose_ratio_prefers_prose():
    prose = (FIXTURES / "essay_three.md").read_text(encoding="utf-8")
    outline = "- one\n- two\n- three\n- four\n"
    assert prose_ratio(prose) > prose_ratio(outline)


# --------------------------------------------------------------------------- chunking
def test_sentence_split_handles_abbreviations():
    s = split_sentences("I met Dr. Chen at 9 a.m. She was late. Not by much.")
    assert len(s) == 3
    assert s[0].startswith("I met Dr. Chen")


def test_chunks_respect_word_bounds(chunks):
    assert chunks
    for c in chunks:
        assert c["n_words"] >= 40
        # Runt-merging can push a chunk past max; it must never be wildly over.
        assert c["n_words"] <= 200 * 2


def test_chunks_are_not_mostly_fragments(chunks):
    health = chunk_health(chunks)
    assert health["frac_starts_mid_thought"] < 0.5
    assert health["n_docs"] == 3


def test_oversized_paragraph_is_sentence_packed():
    para = " ".join(f"This is sentence number {i} and it has a few words in it." for i in range(60))
    out = chunk_document(para, min_words=40, max_words=120)
    assert len(out) > 1
    assert all(len(c.split()) <= 200 for c in out)


def test_chunk_ids_are_unique(chunks):
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


# --------------------------------------------------------------------------- scrubbing
def test_regex_pii_is_removed():
    text = (
        "Email me at akshay@example.com or call 512-555-0199. "
        "My key is sk-abcdefghijklmnopqrstuvwx and I live at 1200 Oak Street."
    )
    out, removals = scrub_text(text, entities=False)
    assert "akshay@example.com" not in out
    assert "512-555-0199" not in out
    assert "sk-abcdefghijklmnopqrstuvwx" not in out
    assert "1200 Oak Street" not in out
    kinds = {r["kind"] for r in removals}
    assert {"EMAIL", "PHONE", "SECRET", "ADDRESS"} <= kinds


def test_custom_terms_are_scrubbed():
    out, _ = scrub_text("Priya said the Northwind project slipped.", entities=False,
                        extra_terms=["Priya", "Northwind"])
    assert "Priya" not in out and "Northwind" not in out


def test_scrub_preserves_length_roughly():
    """Placeholders, not deletions -- cadence is the thing we're trying to learn."""
    text = "I told Priya at Acme that the launch in Denver was fine, more or less, honestly."
    out, _ = scrub_text(text, entities=False, extra_terms=["Priya", "Acme", "Denver"])
    assert abs(len(out.split()) - len(text.split())) <= 1


def test_scrub_log_carries_no_pii(chunks):
    _kept, log, _summary = scrub_records(
        chunks, entities=False, extra_terms=["Priya"]
    )
    blob = str(log)
    assert "Priya" not in blob, "the audit log must not become the PII file"
    assert all("removed" in row for row in log)


def test_overly_scrubbed_chunks_are_dropped():
    rows = [
        {"chunk_id": "x-000", "doc_id": "x", "text": "Priya Priya Priya " * 30, "n_words": 90,
         "register": "informal"}
    ]
    kept, _log, summary = scrub_records(rows, entities=False, extra_terms=["Priya"],
                                        max_placeholder_density=0.12)
    assert kept == []
    assert summary["dropped_over_dense"] == 1


def test_placeholder_density():
    assert placeholder_density("<PERSON> went to <PLACE> today") == pytest.approx(2 / 5)


# --------------------------------------------------------------------------- pairs & splits
def test_offline_backtranslation_produces_pairs(chunks):
    kept, _log, _s = scrub_records(chunks, entities=False)
    pairs, stats = build_pairs(kept, n_per_chunk=2, offline=True)
    assert stats["pairs"] == len(pairs) > 0
    assert all(p["prompt"] and p["response"] for p in pairs)
    # A backtranslated question must not carry the passage's content.
    assert all("<" not in p["prompt"] for p in pairs)


def test_split_never_leaks_a_document_across_splits(chunks):
    kept, _log, _s = scrub_records(chunks, entities=False)
    pairs, _ = build_pairs(kept, n_per_chunk=2, offline=True)
    splits = split_pairs(pairs, val_frac=0.2, blind_frac=0.2, by="document")
    summary = split_summary(splits)
    for _k, overlap in summary["_doc_overlap"].items():
        assert overlap == []


def test_split_is_deterministic(chunks):
    kept, _log, _s = scrub_records(chunks, entities=False)
    pairs, _ = build_pairs(kept, n_per_chunk=1, offline=True)
    a = split_pairs(pairs, seed=17)
    b = split_pairs(pairs, seed=17)
    assert [p["pair_id"] for p in a["train"]] == [p["pair_id"] for p in b["train"]]


# trl-format assertions live in tests/test_training_contracts.py, alongside the
# prompt-format-parity check they exist to protect.
