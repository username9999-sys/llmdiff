"""Distribution fitting and goodness-of-fit, stdlib only.

MLE fits for normal, lognormal, and beta distributions, plus a KS test and
QQ-plot data. All deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmdiff.analytics.stats import mean, std

__all__ = [
    "fit_normal",
    "fit_lognormal",
    "fit_beta",
    "ks_statistic",
    "ks_test_pvalue",
    "qq_data",
    "log_likelihood",
]


def _floats(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


def fit_normal(values: Sequence[float]) -> tuple[float, float]:
    """MLE normal fit: returns (mu, sigma)."""
    xs = _floats(values)
    if not xs:
        raise ValueError("empty sample")
    mu = mean(xs)
    sigma = std(xs, sample=False)
    return mu, sigma


def fit_lognormal(values: Sequence[float]) -> tuple[float, float]:
    """MLE lognormal fit on positive values: returns (mu_log, sigma_log)."""
    xs = [x for x in _floats(values) if x > 0]
    if len(xs) < 2:
        raise ValueError("need at least 2 positive values")
    logs = [math.log(x) for x in xs]
    return fit_normal(logs)


def fit_beta(values: Sequence[float]) -> tuple[float, float]:
    """Method-of-moments beta fit on values in (0, 1): returns (alpha, beta)."""
    xs = [x for x in _floats(values) if 0.0 < x < 1.0]
    if len(xs) < 2:
        raise ValueError("need at least 2 values in (0, 1)")
    m = mean(xs)
    v = std(xs, sample=False) ** 2
    if v == 0:
        raise ValueError("zero variance; beta fit undefined")
    common = m * (1 - m) / v - 1.0
    if common <= 0:
        raise ValueError("variance too large for a beta fit")
    return m * common, (1 - m) * common


def ks_statistic(values: Sequence[float], cdf) -> float:
    """Kolmogorov-Smirnov statistic against a callable CDF."""
    xs = sorted(_floats(values))
    n = len(xs)
    if n == 0:
        return 0.0
    d = 0.0
    for i, x in enumerate(xs):
        f = cdf(x)
        d = max(d, abs((i + 1) / n - f), abs(i / n - f))
    return d


def ks_test_pvalue(values: Sequence[float], cdf) -> float:
    """Approximate KS test p-value (Kolmogorov distribution asymptotic)."""
    n = len(values)
    if n == 0:
        return 1.0
    d = ks_statistic(values, cdf)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    total = 0.0
    for k in range(1, 101):
        total += (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    return min(1.0, max(0.0, 2.0 * total))


def qq_data(values: Sequence[float], cdf_inverse) -> list[tuple[float, float]]:
    """QQ-plot pairs (theoretical quantile, sample quantile) against an inverse CDF."""
    xs = sorted(_floats(values))
    n = len(xs)
    if n == 0:
        return []
    return [(cdf_inverse((i + 0.5) / n), x) for i, x in enumerate(xs)]


def log_likelihood(values: Sequence[float], pdf) -> float:
    """Sum of log pdf over the values (pdf must be strictly positive)."""
    total = 0.0
    for x in values:
        p = pdf(float(x))
        if p <= 0:
            return float("-inf")
        total += math.log(p)
    return total
