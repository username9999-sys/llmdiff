"""Tests for the unified-diff reporter."""

from __future__ import annotations

from llmdiff.core.checks import CheckResult, Verdict
from llmdiff.core.matrix import VerdictMatrix
from llmdiff.core.result import CaseDiff, DiffResult
from llmdiff.reporters.diff_reporter import DiffReporter


def _tie_only_result() -> DiffResult:
    check = CheckResult(dimension="d", verdict=Verdict.TIE, baseline_score=1.0, candidate_score=1.0)
    case = CaseDiff(case_id="c", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c"])
    return DiffResult(matrix=matrix, cases=(case,), config=None)


def test_render_fenced_diffs_for_regressed_cases(handcrafted_result, handcrafted_corpus):
    reporter = DiffReporter.from_corpus(handcrafted_corpus)
    out = reporter.render(handcrafted_result)
    assert out.startswith("# Output diffs (regressed cases)")
    assert "```diff" in out
    assert "regressed-case" in out
    assert "--- baseline/regressed-case" in out
    assert "+++ candidate/regressed-case" in out
    assert "-line two" in out
    assert "+line CHANGED" in out


def test_no_regressions_message():
    out = DiffReporter().render(_tie_only_result())
    assert "No regressed cases" in out


def test_render_plain_has_no_fences(handcrafted_result, handcrafted_corpus):
    reporter = DiffReporter.from_corpus(handcrafted_corpus)
    out = reporter.render_plain(handcrafted_result)
    assert "```" not in out
    assert "@@" in out


def test_max_cases_limits_output(handcrafted_result, handcrafted_corpus):
    reporter = DiffReporter.from_corpus(handcrafted_corpus, max_cases=0)
    out = reporter.render(handcrafted_result)
    assert "regressed-case" not in out


def test_missing_outputs_renders_empty_diff(handcrafted_result):
    out = DiffReporter().render(handcrafted_result)
    assert "regressed-case" in out
    assert "```diff" in out


def test_diff_case_lines(handcrafted_result, handcrafted_corpus):
    reporter = DiffReporter.from_corpus(handcrafted_corpus)
    case = handcrafted_result.case("regressed-case")
    lines = reporter.diff_case(case)
    assert any(line.startswith("---") for line in lines)
    assert any(line.startswith("+++") for line in lines)
    assert any(line.startswith("@@") for line in lines)
