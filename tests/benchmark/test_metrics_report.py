"""Tests for benchmark.metrics and benchmark.report."""

from __future__ import annotations

import json

import pytest

from llmdiff.benchmark.metrics import (
    compute_metrics,
    detection_rate,
    false_positive_rate,
    throughput,
)
from llmdiff.benchmark.report import metrics_to_dict, render_benchmark_json, render_benchmark_markdown
from llmdiff.benchmark.runner import ScenarioResult, run_all_scenarios


def _fake(name, detected, n=10, elapsed=0.1, mem=1000):
    return ScenarioResult(
        name=name,
        n_cases=n,
        elapsed_s=elapsed,
        peak_memory_bytes=mem,
        verdict="loss" if detected else "tie",
        wins=0,
        ties=0 if detected else n,
        losses=n if detected else 0,
        regressed_dimensions=("structure",) if detected else (),
        detected=detected,
    )


def test_detection_rate():
    results = [_fake("clean_x", False), _fake("defect_a", True), _fake("defect_b", False)]
    assert detection_rate(results) == pytest.approx(0.5)


def test_false_positive_rate():
    results = [_fake("clean_a", True), _fake("clean_b", False), _fake("defect_c", True)]
    assert false_positive_rate(results) == pytest.approx(0.5)


def test_throughput():
    results = [
        _fake("clean_a", False, n=100, elapsed=0.5),
        _fake("defect_b", True, n=100, elapsed=0.5),
    ]
    assert throughput(results) == pytest.approx(200.0)


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([])


def test_compute_metrics_fields():
    results = [_fake("clean_a", False), _fake("defect_b", True)]
    m = compute_metrics(results)
    assert m.n_scenarios == 2
    assert m.detection_rate == 1.0
    assert m.false_positive_rate == 0.0
    assert m.total_cases == 20
    assert m.per_scenario == ("clean_a", "defect_b")


def test_metrics_to_dict_roundtrip():
    m = compute_metrics([_fake("clean_a", False), _fake("defect_b", True)])
    d = metrics_to_dict(m)
    json.dumps(d)
    assert d["n_scenarios"] == 2


def test_render_markdown_and_json_with_real_scenarios():
    results = run_all_scenarios(dimensions=("exact", "structure", "content"))
    md = render_benchmark_markdown(results)
    assert "# llmdiff self-benchmark" in md
    assert "detection rate" in md
    payload = json.loads(render_benchmark_json(results))
    assert payload["metrics"]["n_scenarios"] == len(results)
    assert len(payload["scenarios"]) == len(results)
