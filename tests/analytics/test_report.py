"""Tests for analytics.report — build a real DiffResult via the engine."""

from __future__ import annotations

import json

import pytest

from llmdiff import Corpus, evaluate
from llmdiff.analytics.report import build_report, dimension_report, render_json, render_markdown


def _corpus():
    return Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world foo bar",
         "candidate_output": "hello world foo bar"},
        {"id": "b", "prompt": "p", "baseline_output": "one two three four five",
         "candidate_output": "one two three four six"},
        {"id": "c", "prompt": "p", "baseline_output": "alpha beta gamma",
         "candidate_output": "alpha beta delta"},
    ])


def test_dimension_report_has_stats():
    result = evaluate(_corpus(), dimensions=("similarity",))
    rep = dimension_report(result, "similarity")
    assert rep["dimension"] == "similarity"
    assert "sign_test_p" in rep
    assert "cohens_d" in rep
    assert rep["baseline_scores"]["count"] == 3.0


def test_build_report_single_run():
    result = evaluate(_corpus(), dimensions=("similarity", "exact"))
    report = build_report([result])
    assert report["n_runs"] == 1
    assert set(report["dimensions"]) == {"similarity", "exact"}
    assert report["overall"]["verdict"] in ("win", "tie", "loss")


def test_build_report_multi_run_trend():
    r1 = evaluate(_corpus(), dimensions=("similarity",))
    r2 = evaluate(_corpus(), dimensions=("similarity",))
    report = build_report([r1, r2])
    trend = report["dimensions"]["similarity"]["trend"]
    assert trend["n_runs"] == 2
    assert "slope" in trend


def test_build_report_empty_raises():
    with pytest.raises(ValueError):
        build_report([])


def test_render_markdown_and_json():
    result = evaluate(_corpus(), dimensions=("similarity",))
    report = build_report([result])
    md = render_markdown(report)
    assert "# llmdiff analytics report" in md
    assert "similarity" in md
    parsed = json.loads(render_json(report))
    assert parsed["n_runs"] == 1
