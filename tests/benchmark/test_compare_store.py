"""Tests for benchmark.compare and benchmark.results_store."""

from __future__ import annotations

import json

import pytest

from llmdiff.benchmark.compare import compare_runs, two_proportion_ztest
from llmdiff.benchmark.metrics import compute_metrics
from llmdiff.benchmark.results_store import (
    export_csv,
    export_json,
    history,
    load_results,
    store_results,
)
from llmdiff.benchmark.runner import ScenarioResult, run_all_scenarios


def _fake(name, detected, n=10):
    return ScenarioResult(
        name=name,
        n_cases=n,
        elapsed_s=0.01,
        peak_memory_bytes=100,
        verdict="loss" if detected else "tie",
        wins=0,
        ties=0 if detected else n,
        losses=n if detected else 0,
        regressed_dimensions=("structure",) if detected else (),
        detected=detected,
    )


def test_two_proportion_ztest_identical():
    z, p = two_proportion_ztest(5, 10, 5, 10)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_two_proportion_ztest_different():
    z, p = two_proportion_ztest(9, 10, 1, 10)
    assert abs(z) > 2
    assert p < 0.05


def test_compare_runs_significant():
    a = [_fake("clean_x", False), _fake("defect_a", True), _fake("defect_b", True)]
    b = [_fake("clean_x", False), _fake("defect_a", False), _fake("defect_b", False)]
    res = compare_runs(a, b, name_a="good", name_b="bad")
    assert res.detection_a == 1.0
    assert res.detection_b == 0.0
    assert res.significant
    assert res.p_value < 0.05


def test_compare_runs_empty_raises():
    with pytest.raises(ValueError):
        compare_runs([], [_fake("clean_x", False)])


def test_store_and_load_roundtrip(tmp_path):
    results = run_all_scenarios(dimensions=("exact", "structure"))
    metrics = compute_metrics(results)
    db = tmp_path / "bench.db"
    row_id = store_results(db, results, metrics, label="v1")
    assert row_id >= 1
    runs = load_results(db)
    assert len(runs) == 1
    assert runs[0]["label"] == "v1"
    assert runs[0]["metrics"]["n_scenarios"] == len(results)
    assert len(runs[0]["scenarios"]) == len(results)
    assert load_results(db, label="other") == []


def test_history_and_exports(tmp_path):
    results = run_all_scenarios(dimensions=("exact",))
    metrics = compute_metrics(results)
    db = tmp_path / "bench.db"
    store_results(db, results, metrics, label="v1")
    store_results(db, results, metrics, label="v1")
    hist = history(db, label="v1")
    assert len(hist) == 2
    assert hist[0][0] <= hist[1][0]
    csv_path = export_csv(db, tmp_path / "out.csv")
    assert csv_path.read_text().splitlines()[0].startswith("id,created_at")
    json_path = export_json(db, tmp_path / "out.json")
    assert len(json.loads(json_path.read_text())) == 2
