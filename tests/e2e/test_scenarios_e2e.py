"""Tests for e2e.scenarios_e2e — each scenario's expected verdict outcome."""

from __future__ import annotations

import pytest

from llmdiff.e2e.scenarios_e2e import E2E_SCENARIOS, run_all_e2e, run_scenario_e2e


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        run_scenario_e2e("nope")


def test_identical_scenario_no_losses():
    sc, res = run_scenario_e2e("identical")
    assert res.result.matrix.losses == 0


def test_reworded_scenario_detects_exact_losses():
    sc, res = run_scenario_e2e("reworded")
    dim = res.result.matrix.get(sc.expect_losses_on)
    assert dim is not None
    assert dim.losses >= sc.min_losses


def test_field_drop_detected_by_structure():
    sc, res = run_scenario_e2e("field_drop")
    dim = res.result.matrix.get("structure")
    assert dim is not None
    assert dim.losses == 3


def test_refusal_detected_by_content():
    sc, res = run_scenario_e2e("refusal")
    dim = res.result.matrix.get("content")
    assert dim is not None
    assert dim.losses >= 2


def test_truncation_detected_by_length():
    sc, res = run_scenario_e2e("truncation")
    dim = res.result.matrix.get("length")
    assert dim is not None
    assert dim.losses >= 2


def test_run_all_e2e():
    results = run_all_e2e()
    assert len(results) == len(E2E_SCENARIOS)
    names = {sc.name for sc, _ in results}
    assert names == set(E2E_SCENARIOS)
