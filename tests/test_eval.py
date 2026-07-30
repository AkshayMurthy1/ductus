"""Tests for the eval harness. These matter most: a broken metric is worse than no metric,
because it produces confident wrong conclusions instead of an obvious blank."""

from __future__ import annotations

from pathlib import Path

import pytest

from wlm.eval.fluency import repetition_score
from wlm.eval.leakage import (
    build_fact_probes,
    score_fact_probes,
    verbatim_overlap,
)
from wlm.eval.report import grouped_bar_chart, hbar_chart, render_report
from wlm.eval.stylometry import (
    biggest_gaps,
    msttr,
    stylometry_distance,
    stylometry_vector,
)

FIXTURES = Path(__file__).parent / "fixtures" / "author"

TERSE = " ".join(["Short. Very short. Clipped. Fine. Done."] * 20)
FLOWING = " ".join(
    [
        "There is a particular sort of sentence that goes on for some while, gathering clauses "
        "as it moves, and only resolves once the reader has been carried a good distance from "
        "where it started, which is either a pleasure or a nuisance depending entirely on the "
        "day."
    ]
    * 12
)


# --------------------------------------------------------------------------- stylometry
def test_self_distance_is_near_zero():
    v = stylometry_vector(FLOWING)
    d = stylometry_distance(v, v)
    assert d["overall"] == pytest.approx(0.0, abs=1e-6)


def test_distinct_styles_are_far_apart():
    a = stylometry_vector(TERSE)
    b = stylometry_vector(FLOWING)
    assert stylometry_distance(a, b)["overall"] > 0.2


def test_distance_is_symmetric():
    a = stylometry_vector(TERSE)
    b = stylometry_vector(FLOWING)
    assert stylometry_distance(a, b)["overall"] == pytest.approx(
        stylometry_distance(b, a)["overall"], abs=1e-9
    )


def test_distance_is_bounded():
    d = stylometry_distance(stylometry_vector(TERSE), stylometry_vector(FLOWING))
    assert 0.0 <= d["overall"] <= 1.0
    for v in d["distributions"].values():
        assert 0.0 <= v <= 1.0


def test_burstiness_captures_variance_not_mean():
    even = stylometry_vector(" ".join(["Five words in this one." ] * 30))["scalars"]["burstiness"]
    mixed = stylometry_vector(
        ("Yes. " + "A rather longer sentence that keeps going for a while indeed. ") * 15
    )["scalars"]["burstiness"]
    assert mixed > even


def test_scalars_track_obvious_features():
    s = stylometry_vector("I don't think it's fine — really, I don't; you shouldn't either!")[
        "scalars"
    ]
    assert s["contraction_rate"] > 0
    assert s["em_dash_per_100"] > 0
    assert s["semicolon_per_100"] > 0
    assert s["second_person_rate"] > 0


def test_msttr_is_length_normalized():
    short = msttr("alpha beta gamma delta".split())
    assert 0.0 < short <= 1.0
    repeated = msttr(("the cat " * 80).split())
    assert repeated < 0.5


def test_biggest_gaps_is_sorted_and_labeled():
    d = stylometry_distance(stylometry_vector(TERSE), stylometry_vector(FLOWING))
    gaps = biggest_gaps(d, k=5)
    assert len(gaps) == 5
    assert [g[1] for g in gaps] == sorted((g[1] for g in gaps), reverse=True)
    assert all(g[0].startswith(("dist:", "scalar:")) for g in gaps)


def test_real_fixture_author_is_closer_to_self_than_to_other_style():
    author = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(FIXTURES.glob("*.md"))
    )
    half = len(author) // 2
    self_d = stylometry_distance(
        stylometry_vector(author[:half]), stylometry_vector(author[half:])
    )["overall"]
    other_d = stylometry_distance(stylometry_vector(author), stylometry_vector(TERSE))["overall"]
    assert self_d < other_d


# --------------------------------------------------------------------------- leakage
def test_verbatim_overlap_catches_memorization():
    train = ["the quick brown fox jumped over the lazy dog and then kept running down the road"]
    gen = ["the quick brown fox jumped over the lazy dog and then kept running down the road"]
    r = verbatim_overlap(gen, train, n=12)
    assert r["rate"] == 1.0
    assert r["longest_verbatim_run_tokens"] >= 12


def test_verbatim_overlap_clean_on_novel_text():
    train = ["one two three four five six seven eight nine ten eleven twelve thirteen"]
    gen = ["completely different words arranged in an entirely unrelated order for this test case"]
    assert verbatim_overlap(gen, train, n=12)["rate"] == 0.0


def test_fact_probes_build_and_score():
    probes = build_fact_probes(
        ["I flew to Reykjavik in 2019 and it rained the entire time I was there."]
    )
    assert probes
    assert "____" in probes[0]["probe"]
    perfect = score_fact_probes([p["answer"] for p in probes], probes)
    assert perfect["recall"] == 1.0
    clean = score_fact_probes(["somewhere else entirely"] * len(probes), probes)
    assert clean["recall"] == 0.0


def test_repetition_score_flags_loops():
    assert repetition_score("the same four words the same four words the same four words") > 0.25
    assert repetition_score(FLOWING[:400]) < 0.5


# --------------------------------------------------------------------------- report
def test_charts_render_svg():
    svg = hbar_chart([("sentence length", 0.3), ("function words", 0.12)])
    assert svg.startswith("<svg") and "rect" in svg and "sentence length" in svg
    g = grouped_bar_chart(["a", "b"], [("real", [0.4, 0.6]), ("model", [0.5, 0.5])])
    assert g.startswith("<svg") and g.count("<rect") == 4


def test_report_renders_self_contained_html():
    d = stylometry_distance(stylometry_vector(TERSE), stylometry_vector(FLOWING))
    html = render_report(
        {
            "run_name": "t",
            "stylometry": d,
            "biggest_gaps": biggest_gaps(d),
            "av": {"attribution_rate": 0.42, "verifier_auc": 0.9, "threshold": 0.5},
            "fluency": {"verdict": "PASS", "relative_regression": 0.02},
            "leakage": {"verdict": "PASS", "verbatim": {"rate": 0.0,
                                                        "longest_verbatim_run_tokens": 0}},
            "n_generations": 10,
        }
    )
    assert "<!doctype html>" in html
    assert "http://" not in html and "cdn" not in html.lower(), "report must be offline-safe"
    assert "42%" in html


def test_report_escapes_html_in_samples():
    html = render_report(
        {"run_name": "t", "stylometry": {"overall": 0.1},
         "samples": [{"prompt": "p", "real": "<script>alert(1)</script>", "generated": "x"}]}
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
