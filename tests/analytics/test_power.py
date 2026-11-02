"""Tests for analytics.power — sanity and monotonicity checks."""

from __future__ import annotations

import pytest

from llmdiff.analytics.power import (
    minimum_detectable_effect,
    normal_cdf,
    normal_ppf,
    power_for_win_rate,
    sample_size_for_win_rate,
)


def test_normal_cdf_known():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


def test_normal_ppf_inverse():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-6)
    with pytest.raises(ValueError):
        normal_ppf(0.0)


def test_sample_size_grows_with_smaller_effect():
    n_big = sample_size_for_win_rate(0.5, 0.10)
    n_small = sample_size_for_win_rate(0.5, 0.02)
    assert n_small > n_big > 0


def test_sample_size_validation():
    with pytest.raises(ValueError):
        sample_size_for_win_rate(0.0, 0.1)
    with pytest.raises(ValueError):
        sample_size_for_win_rate(0.5, 0.0)


def test_power_increases_with_n():
    p_small = power_for_win_rate(0.5, 0.1, n=50)
    p_big = power_for_win_rate(0.5, 0.1, n=500)
    assert p_big > p_small
    assert 0.0 <= p_small <= 1.0


def test_mde_decreases_with_n():
    mde_small_n = minimum_detectable_effect(0.5, n=50)
    mde_big_n = minimum_detectable_effect(0.5, n=2000)
    assert mde_big_n < mde_small_n


def test_mde_consistency():
    effect = 0.08
    n = sample_size_for_win_rate(0.5, effect)
    mde = minimum_detectable_effect(0.5, n)
    assert mde == pytest.approx(effect, rel=0.2)
