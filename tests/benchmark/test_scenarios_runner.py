"""Tests for benchmark.scenarios and benchmark.runner."""

from __future__ import annotations

import pytest

from llmdiff.benchmark.runner import run_all_scenarios, run_scenario
from llmdiff.benchmark.scenarios import (
    SCENARIOS,
    build_scenario,
    inject_field_drop,
    inject_latency_regression,
    inject_refusal,
    inject_token_bloat,
    inject_truncation,
)


def test_all_scenarios_build():
    for name in SCENARIOS:
        recs = build_scenario(name)
        assert len(recs) > 0
        assert all("baseline_output" in r and "candidate_output" in r for r in recs)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        build_scenario("nope")
    with pytest.raises(ValueError):
        run_scenario("nope")


def test_injectors():
    recs = [{"candidate_output": '{"a": 1, "b": 2}', "metadata": {}}]
    dropped = inject_field_drop(recs)
    assert "a" not in dropped[0]["candidate_output"] or "b" not in dropped[0]["candidate_output"]
    refused = inject_refusal(recs)
    assert "sorry" in refused[0]["candidate_output"].lower()
    trunc = inject_truncation([{"candidate_output": "x" * 100}])
    assert trunc[0]["candidate_output"].endswith("...")
    lat = inject_latency_regression(recs)
    assert lat[0]["metadata"]["candidate_ms"] == 500.0
    tok = inject_token_bloat(recs)
    assert tok[0]["metadata"]["candidate_tokens_out"] == 500


def test_run_scenario_detects_injected_regression():
    res = run_scenario("field_drop_json", dimensions=("structure",))
    assert res.detected
    assert res.losses > 0
    assert "structure" in res.regressed_dimensions
    assert res.elapsed_s >= 0
    assert res.n_cases == 50


def test_run_scenario_clean_has_no_losses():
    res = run_scenario("clean_json", dimensions=("exact", "structure"))
    assert res.losses == 0
    assert not res.detected


def test_run_all_scenarios():
    results = run_all_scenarios(dimensions=("exact", "structure", "content", "latency", "tokens"))
    assert len(results) == len(SCENARIOS)
    detected = {r.name for r in results if r.detected}
    assert "field_drop_json" in detected
    assert "refusal_all" in detected
