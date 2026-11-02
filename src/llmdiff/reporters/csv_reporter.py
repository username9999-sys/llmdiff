"""CSV report renderer.

Emits one row per (case, dimension) pair with the verdict and both scores,
plus a trailing summary section. Uses :mod:`csv` so quoting is correct for
values containing commas, quotes or newlines.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter

__all__ = ["CsvReporter", "CSV_HEADER"]

#: Column header of the per-(case, dimension) rows.
CSV_HEADER = (
    "case_id",
    "dimension",
    "verdict",
    "baseline_score",
    "candidate_score",
    "delta",
    "is_regression",
    "detail",
)


class CsvReporter(BaseReporter):
    """Render a :class:`DiffResult` as CSV rows."""

    name = "csv"

    def __init__(self, *, include_summary: bool = True, delimiter: str = ",") -> None:
        if len(delimiter) != 1:
            raise ValueError("delimiter must be a single character")
        self.include_summary = include_summary
        self.delimiter = delimiter

    def render(self, result: DiffResult) -> str:
        """Render ``result`` to CSV text."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=self.delimiter, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for case in result.cases:
            for check in case.results:
                writer.writerow(self._row(case.case_id, check))
        if self.include_summary:
            writer.writerow([])
            writer.writerow(["# summary"])
            matrix = result.matrix
            writer.writerow(["# total_cases", matrix.total_cases])
            writer.writerow(["# verdict", matrix.verdict])
            writer.writerow(["# wins", matrix.wins])
            writer.writerow(["# ties", matrix.ties])
            writer.writerow(["# losses", matrix.losses])
            writer.writerow(["# regression_rate", round(matrix.regression_rate, 6)])
            for summary in matrix.dimensions:
                writer.writerow(
                    [
                        "# dimension",
                        summary.dimension,
                        summary.wins,
                        summary.ties,
                        summary.losses,
                        round(summary.baseline_mean, 6),
                        round(summary.candidate_mean, 6),
                        round(summary.delta, 6),
                        summary.verdict,
                    ]
                )
        return buffer.getvalue()

    @staticmethod
    def _row(case_id: str, check: Any) -> list[Any]:
        return [
            case_id,
            check.dimension,
            check.verdict.value,
            round(check.baseline_score, 6),
            round(check.candidate_score, 6),
            round(check.delta, 6),
            1 if check.verdict.value == "loss" else 0,
            check.detail,
        ]
