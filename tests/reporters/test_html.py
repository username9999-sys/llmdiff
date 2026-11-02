"""Tests for the HTML reporter."""

from __future__ import annotations

import re

from llmdiff.reporters.html import HtmlReporter


def test_render_is_self_contained_html(evaluated_result):
    out = HtmlReporter().render(evaluated_result)
    assert out.startswith("<!DOCTYPE html>")
    assert "<style>" in out
    assert "</html>" in out
    assert "<script" not in out.lower()
    assert "<link" not in out.lower()


def test_contains_summary_cards_and_table(evaluated_result):
    out = HtmlReporter().render(evaluated_result)
    assert 'class="cards"' in out
    assert "<table>" in out
    assert "Verdict matrix" in out
    for summary in evaluated_result.matrix.dimensions:
        assert summary.dimension in out


def test_contains_inline_svg_chart(evaluated_result):
    out = HtmlReporter().render(evaluated_result)
    assert "<svg" in out
    assert "</svg>" in out
    assert 'aria-label="verdict bar chart"' in out


def test_expandable_case_details(handcrafted_result):
    out = HtmlReporter().render(handcrafted_result)
    assert "<details>" in out
    assert "regressed-case" in out
    assert "improved-case" in out


def test_escapes_script_payload(handcrafted_result):
    out = HtmlReporter().render(handcrafted_result)
    assert "<script>alert('xss')</script>" not in out
    assert "&lt;script&gt;" in out


def test_dark_mode_media_query_present(evaluated_result):
    out = HtmlReporter().render(evaluated_result)
    assert "prefers-color-scheme: dark" in out


def test_render_to_file(evaluated_result, tmp_path):
    path = HtmlReporter().render_to_file(evaluated_result, tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")


def test_title_is_escaped():
    from llmdiff.core.checks import CheckResult, Verdict
    from llmdiff.core.matrix import VerdictMatrix
    from llmdiff.core.result import CaseDiff, DiffResult

    check = CheckResult(dimension="d", verdict=Verdict.TIE, baseline_score=1.0, candidate_score=1.0)
    case = CaseDiff(case_id="c", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c"])
    result = DiffResult(matrix=matrix, cases=(case,), config=None)
    out = HtmlReporter(title="<b>bold</b>").render(result)
    assert "<title>&lt;b&gt;bold&lt;/b&gt;</title>" in out
    assert re.search(r"<h1>&lt;b&gt;bold&lt;/b&gt;</h1>", out)
