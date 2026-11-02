"""Tests for analytics.correlations — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.correlations import (
    correlation_matrix,
    covariance_matrix,
    pca,
    pearson,
    rank,
    redundant_dimensions,
    spearman,
)


def test_pearson_perfect():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_zero_variance():
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_pearson_length_mismatch():
    with pytest.raises(ValueError):
        pearson([1, 2], [1])


def test_rank_ties():
    assert rank([10, 20, 20, 40]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_monotonic():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_covariance_matrix():
    cov = covariance_matrix([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    assert cov[0][0] == pytest.approx(1.0)
    assert cov[0][1] == pytest.approx(2.0)
    assert cov[1][1] == pytest.approx(4.0)


def test_correlation_matrix_diagonal():
    corr = correlation_matrix([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert corr[0][0] == pytest.approx(1.0)
    assert corr[0][1] == pytest.approx(-1.0)


def test_pca_explains_variance():
    # two perfectly correlated columns -> first component explains ~100%
    result = pca([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], components=2)
    assert result["explained_variance_ratio"][0] == pytest.approx(1.0, abs=1e-6)


def test_pca_empty():
    assert pca([]) == {"explained_variance_ratio": [], "components": [], "eigenvalues": []}


def test_redundant_dimensions():
    # only a and b are perfectly correlated; c is independent noise
    cols = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0],
        [5.0, 1.0, 4.0, 2.0, 3.0],
    ]
    names = ["a", "b", "c"]
    red = redundant_dimensions(cols, names, threshold=0.99)
    assert len(red) == 1
    assert red[0][0] == "a" and red[0][1] == "b"
    assert red[0][2] == pytest.approx(1.0)
