"""Tests for analytics.trends — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.trends import (
    cusum_change_point,
    forecast_linear,
    linear_regression,
    moving_average,
    r_squared,
    run_series_from_results,
    slope,
)


def test_linear_regression_perfect_line():
    intercept, sl = linear_regression([1.0, 2.0, 3.0, 4.0])
    assert intercept == pytest.approx(1.0)
    assert sl == pytest.approx(1.0)


def test_slope_constant():
    assert slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_r_squared_perfect():
    assert r_squared([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r_squared_constant():
    assert r_squared([7.0, 7.0, 7.0]) == 1.0


def test_cusum_detects_change():
    series = [0.0] * 20 + [1.0] * 20
    idx = cusum_change_point(series, threshold=1.0)
    assert idx is not None
    assert 15 <= idx <= 25


def test_cusum_no_change():
    assert cusum_change_point([0.5] * 30, threshold=1.0) is None


def test_forecast_linear():
    fc = forecast_linear([1.0, 2.0, 3.0], 2)
    assert fc == pytest.approx([4.0, 5.0])
    assert forecast_linear([1.0], 0) == []


def test_moving_average():
    assert moving_average([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
    ma = moving_average([0.0, 0.0, 3.0, 0.0, 0.0], 3)
    assert ma[2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        moving_average([1.0], 0)


def test_run_series_from_results():
    class FakeSummary:
        loss_rate = 0.25
        win_rate = 0.5
        delta = -0.1
        candidate_mean = 0.8

    class FakeMatrix:
        def get(self, dim):
            return FakeSummary() if dim == "structure" else None

    class FakeResult:
        matrix = FakeMatrix()

    series = run_series_from_results([FakeResult(), FakeResult()], "structure", "loss_rate")
    assert series == [0.25, 0.25]
    assert run_series_from_results([FakeResult()], "missing") == []
    with pytest.raises(ValueError):
        run_series_from_results([FakeResult()], "structure", "nope")
