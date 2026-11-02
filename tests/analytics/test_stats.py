"""Tests for analytics.stats — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.stats import (
    bootstrap_ci,
    describe,
    histogram,
    mean,
    median,
    min_max,
    percentile,
    std,
    variance,
)


def test_mean():
    assert mean([1, 2, 3]) == 2.0
    assert mean([]) == 0.0


def test_variance_and_std():
    assert variance([1, 2, 3], sample=False) == pytest.approx(2.0 / 3.0)
    assert variance([5, 5, 5]) == 0.0
    assert std([1, 2, 3], sample=False) == pytest.approx((2.0 / 3.0) ** 0.5)


def test_median():
    assert median([3, 1, 2]) == 2.0
    assert median([4, 1, 2, 3]) == 2.5
    assert median([]) == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 0) == 1.0
    assert percentile([1, 2, 3, 4], 100) == 4.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_min_max():
    assert min_max([3, 1, 2]) == (1.0, 3.0)
    assert min_max([]) == (0.0, 0.0)


def test_describe():
    d = describe([1, 2, 3, 4])
    assert d["count"] == 4.0
    assert d["mean"] == 2.5
    assert d["min"] == 1.0
    assert d["max"] == 4.0
    assert d["median"] == 2.5


def test_bootstrap_ci_reproducible_and_contains_mean():
    xs = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.55]
    lo1, hi1 = bootstrap_ci(xs, resamples=200, seed=42)
    lo2, hi2 = bootstrap_ci(xs, resamples=200, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= mean(xs) <= hi1


def test_histogram():
    # bounds (0,1), 2 bins: 0.0 -> bucket 0; 0.5 and 1.0 -> bucket 1
    buckets = histogram([0.0, 0.5, 1.0], bins=2, bounds=(0.0, 1.0))
    assert [count for _, _, count in buckets] == [1, 2]
    assert histogram([]) == []
