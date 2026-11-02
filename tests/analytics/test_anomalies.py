"""Tests for analytics.anomalies — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.anomalies import (
    detect_outliers,
    distance_outliers,
    iqr_fences,
    iqr_outliers,
    mad_outliers,
    modified_zscore,
    zscore,
    zscore_outliers,
)


def test_zscore_basic():
    zs = zscore([1.0, 2.0, 3.0])
    assert zs[1] == pytest.approx(0.0)
    assert abs(zs[0]) == pytest.approx(abs(zs[2]))


def test_zscore_constant():
    assert zscore([5.0, 5.0]) == [0.0, 0.0]


def test_zscore_outliers():
    vals = [1.0] * 20 + [100.0]
    assert zscore_outliers(vals, threshold=3.0) == [20]


def test_modified_zscore():
    mz = modified_zscore([1.0, 2.0, 3.0, 100.0])
    assert abs(mz[3]) > abs(mz[0])


def test_mad_outliers():
    # varied base values so MAD > 0, plus one extreme outlier
    vals = [float(i % 5) for i in range(30)] + [1000.0]
    assert mad_outliers(vals) == [30]


def test_iqr_fences():
    lo, hi = iqr_fences([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    assert lo < 1
    assert hi < 100


def test_iqr_outliers():
    vals = [1.0] * 20 + [1000.0]
    assert iqr_outliers(vals) == [20]


def test_distance_outliers():
    pts = [[0.0], [0.1], [0.2], [50.0]]
    assert 3 in distance_outliers(pts, threshold=2.0)


def test_distance_outliers_small():
    assert distance_outliers([[1.0]]) == []


def test_detect_outliers_dispatch():
    vals = [1.0] * 20 + [100.0]
    assert detect_outliers(vals, method="iqr") == [20]
    with pytest.raises(ValueError):
        detect_outliers(vals, method="nope")
