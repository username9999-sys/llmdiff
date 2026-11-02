"""Tests for the markdown reporter."""

from __future__ import annotations

from llmdiff.reporters.markdown import MarkdownReporter


def test_render_contains_header_and_matrix(evaluated_result):
    out = MarkdownReporter().render(evaluated_result)
    assert out.startswith("# llmdiff report")
    assert "## Verdict matrix" in out
    assert "| Dimension | Verdict | Wins | Ties | Losses |" in out
    for summary in evaluated_result.matrix.dimensions:
        assert f"| {summary.dimension} |" in out
    assert "## Regressed cases" in out
    assert "## Configuration" in out


def test_render_shows_loss_markers(handcrafted_result):
    out = MarkdownReporter().render(handcrafted_result)
    assert "\u274c" in out  # loss emoji
    assert "regressed-case" in out
    assert "loss" in out
    assert "## Dimensions" in out
    assert "`exact`" in out


def test_render_summary_one_paragraph(handcrafted_result):
    summary = MarkdownReporter().render_summary(handcrafted_result)
    assert isinstance(summary, str) and len(summary) > 20
    # handcrafted overall verdict: 2W/5T/2L -> TIE
    assert "TIE" in summary
    assert "2 loss(es)" in summary
    assert "regressed-case" not in summary  # summary is aggregate only


def test_render_summary_evaluated(evaluated_result):
    summary = MarkdownReporter().render_summary(evaluated_result)
    assert "case(s)" in summary
    assert "%" in summary


def test_config_footer_can_be_disabled(evaluated_result):
    out = MarkdownReporter(show_config=False).render(evaluated_result)
    assert "## Configuration" not in out


def test_render_to_file(evaluated_result, tmp_path):
    path = MarkdownReporter().render_to_file(evaluated_result, tmp_path / "report.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# llmdiff report")


def test_pipe_in_dimension_is_escaped():
    from llmdiff.core.checks import CheckResult, Verdict
    from llmdiff.core.matrix import VerdictMatrix
    from llmdiff.core.result import CaseDiff, DiffResult

    check = CheckResult(
        dimension="we|ird", verdict=Verdict.LOSS,
        baseline_score=1.0, candidate_score=0.0, detail="a|b",
    )
    case = CaseDiff(case_id="c|1", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c|1"])
    result = DiffResult(matrix=matrix, cases=(case,), config=None)
    out = MarkdownReporter().render(result)
    assert "| we\\|ird |" in out
    assert "c\\|1" in out
