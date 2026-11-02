"""Tests for the CSV reporter."""

from __future__ import annotations

import csv
import io

import pytest

from llmdiff.core.checks import CheckResult, Verdict
from llmdiff.core.matrix import VerdictMatrix
from llmdiff.core.result import CaseDiff, DiffResult
from llmdiff.reporters.csv_reporter import CSV_HEADER, CsvReporter


def _rows(out: str):
    return list(csv.reader(io.StringIO(out)))


def _data_rows(out: str):
    return [r for r in _rows(out)[1:] if r and not r[0].startswith("#") and r[0] != ""]


def test_header_row(evaluated_result):
    out = CsvReporter().render(evaluated_result)
    assert tuple(_rows(out)[0]) == CSV_HEADER


def test_one_row_per_case_dimension(evaluated_result):
    out = CsvReporter().render(evaluated_result)
    expected = sum(len(case.results) for case in evaluated_result.cases)
    assert len(_data_rows(out)) == expected


def test_loss_rows_flagged(handcrafted_result):
    out = CsvReporter().render(handcrafted_result)
    rows = _data_rows(out)
    losses = [r for r in rows if r[2] == "loss"]
    assert len(losses) == 2
    assert all(r[6] == "1" for r in losses)
    wins = [r for r in rows if r[2] == "win"]
    assert len(wins) == 2
    assert all(r[6] == "0" for r in wins)


def test_summary_section_present(handcrafted_result):
    out = CsvReporter(include_summary=True).render(handcrafted_result)
    assert "# summary" in out
    assert "# losses,2" in out


def test_summary_can_be_disabled(handcrafted_result):
    out = CsvReporter(include_summary=False).render(handcrafted_result)
    assert "# summary" not in out


def test_detail_with_comma_is_quoted():
    check = CheckResult(
        dimension="d", verdict=Verdict.LOSS, baseline_score=1.0, candidate_score=0.0,
        detail='has, comma and "quotes"',
    )
    case = CaseDiff(case_id="c1", results=(check,))
    matrix = VerdictMatrix.from_results([check], case_ids=["c1"])
    result = DiffResult(matrix=matrix, cases=(case,), config=None)
    out = CsvReporter(include_summary=False).render(result)
    assert _rows(out)[1][7] == 'has, comma and "quotes"'


def test_custom_delimiter(handcrafted_result):
    out = CsvReporter(delimiter=";", include_summary=False).render(handcrafted_result)
    assert out.splitlines()[0].startswith("case_id;dimension;")


def test_invalid_delimiter():
    with pytest.raises(ValueError):
        CsvReporter(delimiter=";;")
