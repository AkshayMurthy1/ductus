"""CPU pipeline tests: supplied prompts must survive ingest -> chunk -> pair."""

import pytest


# --------------------------------------------------------------- supplied prompts
def test_front_matter_prompt_flows_to_pairs(tmp_path):
    """A document's real prompt must survive ingest -> chunk -> pair, unedited."""
    from wlm.backtranslate import build_pairs
    from wlm.chunk import chunk_records
    from wlm.ingest.local import ingest_local, split_front_matter

    prompt = "List five things that are important to you: be specific."
    body = " ".join(f"Sentence number {i} about a thing that matters to me." for i in range(40))
    (tmp_path / "a.md").write_text(f"---\nprompt: {prompt}\n---\n\n{body}\n", encoding="utf-8")

    meta, rest = split_front_matter(f"---\nprompt: {prompt}\n---\n\n{body}\n")
    assert meta["prompt"] == prompt
    assert "---" not in rest

    docs = ingest_local(tmp_path, register="formal")
    assert docs and docs[0]["prompt"] == prompt
    assert not docs[0]["text"].startswith("---")

    chunks = chunk_records(docs, min_words=60, max_words=320)
    assert chunks[0]["prompt"] == prompt and chunks[0]["chunk_index"] == 0

    pairs, stats = build_pairs(chunks, n_per_chunk=2, offline=True)
    first = [p for p in pairs if p["chunk_id"].endswith("-000")]
    assert first and first[0]["prompt"] == prompt
    assert first[0]["generator"] == "supplied-prompt"
    assert stats["supplied_prompt"] == 1
    # one supplied question, not n_per_chunk copies
    assert len(first) == 1


def test_supplied_prompt_can_be_disabled_and_scoped():
    from wlm.backtranslate import build_pairs

    chunks = [
        {"chunk_id": "d-000", "doc_id": "d", "chunk_index": 0, "prompt": "Real?", "text": "a " * 80},
        {"chunk_id": "d-001", "doc_id": "d", "chunk_index": 1, "prompt": "Real?", "text": "b " * 80},
    ]
    _, s_first = build_pairs(chunks, n_per_chunk=1, offline=True)
    assert s_first["supplied_prompt"] == 1 and s_first["generated"] == 1

    _, s_all = build_pairs(chunks, n_per_chunk=1, offline=True, supplied_prompt_scope="all_chunks")
    assert s_all["supplied_prompt"] == 2 and s_all["generated"] == 0

    _, s_off = build_pairs(chunks, n_per_chunk=1, offline=True, use_supplied_prompts=False)
    assert s_off["supplied_prompt"] == 0 and s_off["generated"] == 2

    with pytest.raises(ValueError):
        build_pairs(chunks, offline=True, supplied_prompt_scope="nonsense")


# --------------------------------------------------------- near-duplicate grouping
def _pair(doc, text, i=0):
    return {"pair_id": f"{doc}-{i}", "doc_id": doc, "register": "formal",
            "prompt": "q", "response": text, "n_words": len(text.split())}


def test_reused_passage_cannot_land_in_two_splits():
    """A passage recycled across two prompts must not be split into train and blind."""
    from wlm.dataset import near_duplicate_groups, split_pairs, split_summary

    shared = ("I first beheld the synergistic power of physics while exploring the heat "
              "equation with a finite volume solver and a neural network in tandem. ") * 4
    topics = ["glassblowing furnaces reach startling temperatures before dawn",
              "migratory geese navigate by magnetic inclination across continents",
              "sourdough starters demand patience nobody warns you about",
              "tidal turbines corrode faster than any brochure admits",
              "medieval scribes invented punctuation to survive long manuscripts",
              "desert beetles harvest fog on their ridged carapaces",
              "vinyl mastering engineers argue endlessly over groove spacing",
              "alpine lichens colonise bare rock over improbable centuries"]
    pairs = []
    for d, t in enumerate(topics):           # 8 genuinely unrelated documents
        pairs.append(_pair(f"doc{d}", " ".join(f"{t} in year {y}." for y in range(30))))
    pairs.append(_pair("reuse_a", shared))   # same passage, two prompts, two doc_ids
    pairs.append(_pair("reuse_b", shared + "A different closing sentence entirely."))

    groups = near_duplicate_groups(pairs)
    assert groups["reuse_a"] == groups["reuse_b"]
    assert len({groups[f"doc{d}"] for d in range(8)}) == 8   # distinct docs stay distinct

    splits = split_pairs(pairs, seed=3)
    where = {name: {r["doc_id"] for r in rows} for name, rows in splits.items()}
    landed = [n for n, ids in where.items() if {"reuse_a", "reuse_b"} & ids]
    assert len(landed) == 1, f"reused passage split across {landed}"
    assert not any(split_summary(splits)["_doc_overlap"].values())


def test_grouping_is_off_by_flag_and_zero_on_distinct_docs():
    from wlm.dataset import near_duplicate_groups

    pairs = [_pair("a", "The cat sat on a warm mat by the door. " * 20),
             _pair("b", "Quantum chromodynamics resists an intuitive picture entirely. " * 20)]
    g = near_duplicate_groups(pairs)
    assert g["a"] != g["b"]          # genuinely different text is never merged
    same = [_pair("x", "identical wording repeated here for the test. " * 20),
            _pair("y", "identical wording repeated here for the test. " * 20)]
    assert near_duplicate_groups(same)["x"] == near_duplicate_groups(same)["y"]


# ------------------------------------------------------------------ scrub span guards
@pytest.mark.parametrize(
    "text,must_keep",
    [
        ("But this innately-produced efficiency wasn't fictious, it was magic.", "wasn't"),
        ("But this innately-produced efficiency wasn’t fictious, it was magic.", "wasn’t"),
        ("When I learned the Euler-Lagrange equations, it seemed too good.", "Euler-Lagrange"),
        ("My devoutly-Hindu uncle rang the bells each morning without fail.", "devoutly-Hindu"),
        ("In my research I used ANSYS Fluent to construct a training dataset.", "ANSYS"),
        ("A Fourier transform turns the signal into something I can reason about.", "Fourier"),
    ],
)
def test_scrub_never_damages_a_word_the_author_wrote(text, must_keep):
    from wlm.scrub import scrub_text

    out, _ = scrub_text(text, entities=True)
    assert must_keep in out, f"scrubber damaged {must_keep!r}: {out!r}"


def test_scrub_still_removes_real_identifiers():
    """The guards must not blunt the actual privacy job."""
    from wlm.scrub import scrub_text

    out, _ = scrub_text("Email me at akshay@example.com or call 512-555-0100.", entities=False)
    assert "akshay@example.com" not in out and "<EMAIL>" in out
    assert "512-555-0100" not in out and "<PHONE>" in out

    out2, _ = scrub_text("I studied at UW-Milwaukee before moving on.", entities=True)
    assert "Milwaukee" not in out2, "a whole-compound institution name must still scrub"


def test_never_scrub_list_is_extensible():
    from wlm.scrub import scrub_text

    text = "The Kolmogorov bound is what my whole third chapter argues against."
    kept, _ = scrub_text(text, entities=True, never_scrub=["Kolmogorov"])
    assert "Kolmogorov" in kept


def test_em_dash_punctuation_does_not_shield_a_real_name():
    """An em dash is punctuation here, not a compound joiner -- names after it must still go."""
    from wlm.scrub import scrub_text

    out, _ = scrub_text("In the passage, Kevin Esvelt—the leader of the lab—raised the alarm.")
    assert "Esvelt" not in out and "<PERSON>" in out
