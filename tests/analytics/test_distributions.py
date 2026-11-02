"""Tests for analytics.distributions — hand-computed expected values."""

from __future__ import annotations

import math

import pytest

from llmdiff.analytics.distributions import (
    fit_beta,
    fit_lognormal,
    fit_normal,
    ks_statistic,
    ks_test_pvalue,
    log_likelihood,
    qq_data,
)


def test_fit_normal():
    mu, sigma = fit_normal([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu == pytest.approx(3.0)
    assert sigma == pytest.approx(math.sqrt(2.0))


def test_fit_normal_empty_raises():
    with pytest.raises(ValueError):
        fit_normal([])


def test_fit_lognormal():
    vals = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
    mu, sigma = fit_lognormal(vals)
    assert mu == pytest.approx(1.0)
    assert sigma > 0


def test_fit_lognormal_rejects_nonpositive():
    with pytest.raises(ValueError):
        fit_lognormal([0.0, -1.0])


def test_fit_beta():
    alpha, beta = fit_beta([0.2, 0.4, 0.5, 0.6, 0.8])
    assert alpha == pytest.approx(beta)
    assert alpha > 0


def test_fit_beta_rejects_out_of_range():
    with pytest.raises(ValueError):
        fit_beta([0.0, 1.0, 2.0])


def test_ks_statistic_perfect_fit():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    mu, sigma = fit_normal(vals)

    def cdf(x):
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

    assert ks_statistic(vals, cdf) < 0.2


def test_ks_test_pvalue_bad_fit():
    vals = [1.0] * 20 + [100.0]

    def cdf(x):
        return 0.5 * (1.0 + math.erf((x - 1.0) / math.sqrt(2.0)))

    assert ks_test_pvalue(vals, cdf) < 0.01


def test_qq_data_length():
    qq = qq_data([3.0, 1.0, 2.0], lambda p: p)
    assert len(qq) == 3
    assert qq[0][1] == 1.0


def test_log_likelihood():
    def pdf(x):
        return math.exp(-x) if x >= 0 else 0.0

    ll = log_likelihood([0.0, 1.0], pdf)
    assert ll == pytest.approx(math.log(1.0) + math.log(math.exp(-1.0)))
    assert log_likelihood([-1.0], pdf) == float("-inf")
