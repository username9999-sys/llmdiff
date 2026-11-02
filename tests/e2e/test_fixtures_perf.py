"""Tests for e2e.fixtures and e2e.perf."""

from __future__ import annotations

import json

from llmdiff import Corpus, evaluate
from llmdiff.e2e.fixtures import (
    FIXTURE_JSONL_TEXT,
    FIXTURE_REGRESSION,
    FIXTURE_SMALL,
    big_corpus_records,
    fixture_jsonl_text,
)
from llmdiff.e2e.perf import E2E_BUDGETS, perf_run


def test_fixture_jsonl_roundtrip():
    corpus = Corpus.from_jsonl_text(fixture_jsonl_text(FIXTURE_SMALL))
    assert len(corpus) == 2
    assert corpus[0].id == "f1"
    assert FIXTURE_JSONL_TEXT.strip().count("\n") == 1


def test_fixture_regression_detected():
    corpus = Corpus.from_records(FIXTURE_REGRESSION)
    result = evaluate(corpus, dimensions=("structure",))
    assert result.matrix.get("structure").losses == 2


def test_big_corpus_deterministic():
    a = big_corpus_records(50, seed=3)
    b = big_corpus_records(50, seed=3)
    assert a == b
    assert len(a) == 50


def test_perf_run_small_within_budgets():
    res = perf_run(300, dimensions=("exact", "similarity", "length", "content"))
    assert res.n_cases == 300
    assert res.n_dimensions == 4
    assert res.within_time_budget
    assert res.within_memory_budget
    assert res.cases_per_second > 0


def test_perf_budget_override():
    res = perf_run(50, dimensions=("exact",), budgets={"max_seconds": 0.0})
    assert res.within_time_budget is False
    assert E2E_BUDGETS["max_seconds"] == 60.0


def test_perf_json_serialisable():
    import dataclasses

    res = perf_run(100, dimensions=("exact",))
    json.dumps(dataclasses.asdict(res))
