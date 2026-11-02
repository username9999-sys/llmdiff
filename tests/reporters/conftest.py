"""Shared fixtures for reporter tests.

Two ways to build a DiffResult:

- ``evaluated_result``: run the real engine via ``evaluate()`` on a small
  corpus (mirrors tests/test_smoke.py).
- ``handcrafted_result``: build CheckResult/CaseDiff/VerdictMatrix directly
  with known regressions, including a ``<script>`` payload to prove HTML
  escaping.
"""

from __future__ import annotations

import json

import pytest

from llmdiff import Corpus, evaluate
from llmdiff.core.checks import CheckResult, Verdict
from llmdiff.core.matrix import DimensionSummary, VerdictMatrix
from llmdiff.core.result import CaseDiff, DiffResult

CORPUS_RECORDS = [
    {
        "id": "c1",
        "prompt": "hi",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m1", "tags": ["x"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "c2",
        "prompt": "yo",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
    {
        "id": "c3",
        "prompt": "same",
        "baseline_output": "identical output",
        "candidate_output": "identical output",
        "metadata": {},
    },
]


@pytest.fixture(scope="session")
def evaluated_result() -> DiffResult:
    """A real DiffResult from the engine on a 3-case corpus."""
    return evaluate(Corpus.from_records(CORPUS_RECORDS))


def _check(
    dimension: str, verdict: Verdict, base: float, cand: float, detail: str = ""
) -> CheckResult:
    return CheckResult(
        dimension=dimension,
        verdict=verdict,
        baseline_score=base,
        candidate_score=cand,
        detail=detail,
        metadata={"case_id": "x"},
    )


@pytest.fixture(scope="session")
def handcrafted_result() -> DiffResult:
    """A hand-built DiffResult with known wins/losses and a <script> payload."""
    evil = "<script>alert('xss')</script>"
    c1 = CaseDiff(
        case_id="regressed-case",
        results=(
            _check("exact", Verdict.LOSS, 1.0, 0.0, f"outputs differ {evil}"),
            _check("similarity", Verdict.LOSS, 1.0, 0.4, "similarity dropped"),
            _check("length", Verdict.TIE, 0.9, 0.9, "same length"),
        ),
    )
    c2 = CaseDiff(
        case_id="improved-case",
        results=(
            _check("exact", Verdict.WIN, 0.0, 1.0, "now exact"),
            _check("similarity", Verdict.WIN, 0.2, 1.0, "more similar"),
            _check("length", Verdict.TIE, 1.0, 1.0, "same length"),
        ),
    )
    c3 = CaseDiff(
        case_id="tied-case",
        results=(
            _check("exact", Verdict.TIE, 1.0, 1.0, "identical"),
            _check("similarity", Verdict.TIE, 1.0, 1.0, "identical"),
            _check("length", Verdict.TIE, 1.0, 1.0, "identical"),
        ),
    )
    all_results = [r for c in (c1, c2, c3) for r in c.results]
    matrix = VerdictMatrix.from_results(
        all_results, case_ids=["regressed-case", "improved-case", "tied-case"]
    )
    return DiffResult(matrix=matrix, cases=(c1, c2, c3), config=None)


@pytest.fixture(scope="session")
def handcrafted_corpus() -> Corpus:
    """The corpus matching handcrafted_result, for diff/trend reporters."""
    return Corpus.from_records(
        [
            {
                "id": "regressed-case",
                "prompt": "p1",
                "baseline_output": "line one\nline two\nline three\n",
                "candidate_output": "line one\nline CHANGED\nline three\nline four\n",
                "metadata": {},
            },
            {
                "id": "improved-case",
                "prompt": "p2",
                "baseline_output": "old output",
                "candidate_output": "new better output",
                "metadata": {},
            },
            {
                "id": "tied-case",
                "prompt": "p3",
                "baseline_output": "same",
                "candidate_output": "same",
                "metadata": {},
            },
        ]
    )


@pytest.fixture()
def dimension_summaries() -> tuple[DimensionSummary, ...]:
    """A couple of DimensionSummary objects for unit-level tests."""
    return (
        DimensionSummary(dimension="exact", wins=1, ties=1, losses=2,
                         baseline_mean=0.8, candidate_mean=0.5),
        DimensionSummary(dimension="similarity", wins=3, ties=0, losses=0,
                         baseline_mean=0.5, candidate_mean=0.9),
    )
