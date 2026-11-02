"""Tests for the GitHub PR comment reporter."""

from __future__ import annotations

from llmdiff.core.checks import CheckResult, Verdict
from llmdiff.core.matrix import VerdictMatrix
from llmdiff.core.result import CaseDiff, DiffResult
from llmdiff.reporters.github import (
    CONCLUSION_FAILURE,
    CONCLUSION_NEUTRAL,
    CONCLUSION_SUCCESS,
    GithubReporter,
)


def _win_only_result() -> DiffResult:
    check = CheckResult(
        dimension="d", verdict=Verdict.WIN, baseline_score=0.0, candidate_score=1.0
    )
    case = CaseDiff(case_id="c", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c"])
    return DiffResult(matrix=matrix, cases=(case,), config=None)


def _loss_only_result() -> DiffResult:
    check = CheckResult(
        dimension="d", verdict=Verdict.LOSS, baseline_score=1.0, candidate_score=0.0
    )
    case = CaseDiff(case_id="c", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c"])
    return DiffResult(matrix=matrix, cases=(case,), config=None)


def test_render_has_header_and_details(evaluated_result):
    out = GithubReporter().render(evaluated_result)
    assert out.startswith("## ")
    assert "llmdiff" in out
    assert "<details>" in out
    assert "</details>" in out
    assert "Verdict matrix" in out
    assert "| Dimension | Verdict | W | T | L |" in out


def test_regressed_cases_collapsible(handcrafted_result):
    out = GithubReporter().render(handcrafted_result)
    assert "Regressed cases (1)" in out
    assert "regressed-case" in out
    assert "Improved cases (1)" in out
    assert "All cases" in out


def test_conclusion_mapping(handcrafted_result):
    assert GithubReporter().conclusion(handcrafted_result) == CONCLUSION_FAILURE
    assert GithubReporter().conclusion(_win_only_result()) == CONCLUSION_SUCCESS
    assert GithubReporter(fail_on_loss=False).conclusion(_loss_only_result()) == CONCLUSION_NEUTRAL


def test_status_payload(handcrafted_result):
    payload = GithubReporter(context="ci/llmdiff").status_payload(handcrafted_result)
    assert payload["state"] == CONCLUSION_FAILURE
    assert payload["context"] == "ci/llmdiff"
    assert len(payload["description"]) <= 140
    assert "W/" in payload["description"]
