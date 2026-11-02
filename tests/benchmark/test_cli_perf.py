"""Tests for benchmark.cli_hooks and benchmark.perf."""

from __future__ import annotations

import json

from llmdiff.benchmark.cli_hooks import bench, bench_scenario
from llmdiff.benchmark.perf import PERF_BUDGETS, perf_evaluate


def test_bench_scenario():
    out = bench_scenario("clean_json", dimensions=("exact",))
    assert out["name"] == "clean_json"
    assert out["n_cases"] == 50
    assert out["detected"] is False


def test_bench_full_writes_reports(tmp_path):
    md = tmp_path / "bench.md"
    js = tmp_path / "bench.json"
    db = tmp_path / "bench.db"
    out = bench(
        dimensions=("exact", "structure", "content"),
        out_md=md,
        out_json=js,
        db=db,
        label="test",
    )
    assert out["metrics"]["n_scenarios"] > 0
    assert md.read_text().startswith("# llmdiff self-benchmark")
    payload = json.loads(js.read_text())
    assert "metrics" in payload and "scenarios" in payload
    assert "db_row_id" in out


def test_perf_evaluate_small_within_budgets():
    res = perf_evaluate(200, kind="summarization", dimensions=("exact", "similarity", "length"))
    assert res.n_cases == 200
    assert res.n_dimensions == 3
    assert res.within_time_budget
    assert res.within_memory_budget
    assert res.cases_per_second > 0


def test_perf_budgets_override():
    # absurdly tight time budget -> within_time_budget must be False
    res = perf_evaluate(
        50,
        kind="classification",
        dimensions=("exact",),
        budgets={"max_seconds": 0.0},
    )
    assert res.within_time_budget is False
    assert PERF_BUDGETS["max_seconds"] == 30.0  # default untouched
