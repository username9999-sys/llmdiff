"""Statistical significance tests for paired comparisons, stdlib only.

Each test takes two equal-length series (baseline vs candidate scores) and
returns a p-value (plus effect size where applicable). Pure Python, so results
are deterministic and offline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmdiff.analytics.stats import mean, std

__all__ = [
    "sign_test",
    "wilcoxon_signed_rank",
    "paired_t_test",
    "cohens_d",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "correct_pvalues",
]


def _pairs(a: Sequence[float], b: Sequence[float]) -> list[tuple[float, float]]:
    if len(a) != len(b):
        raise ValueError("paired series must have equal length")
    return [(float(x), float(y)) for x, y in zip(a, b, strict=True)]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_sf(x: float) -> float:
    return 1.0 - _normal_cdf(x)


def sign_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided sign test p-value on the signs of (b - a).

    Exact binomial for small n (<= 25 non-zero diffs), normal approximation
    with continuity correction for larger n.
    """
    diffs = [y - x for x, y in _pairs(a, b) if y != x]
    n = len(diffs)
    if n == 0:
        return 1.0
    k = sum(1 for d in diffs if d > 0)
    if n <= 25:
        from math import comb

        p_one = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
        return min(1.0, 2.0 * p_one)
    z = (k - 0.5 - n / 2.0) / math.sqrt(n / 4.0)
    return min(1.0, 2.0 * _normal_sf(abs(z)))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approximation, tie-corrected)."""
    diffs = [(abs(y - x), 1.0 if y > x else -1.0) for x, y in _pairs(a, b) if y != x]
    n = len(diffs)
    if n == 0:
        return 1.0
    diffs.sort(key=lambda t: t[0])
    ranks: list[float] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and diffs[j + 1][0] == diffs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks.append(avg)
        i = j + 1
    w_plus = sum(r for r, (_, sign) in zip(ranks, diffs, strict=True) if sign > 0)
    mu = n * (n + 1) / 4.0
    tie_counts: dict[float, int] = {}
    for d, _ in diffs:
        tie_counts[d] = tie_counts.get(d, 0) + 1
    tie_term = sum(t * (t - 1) * (t + 1) for t in tie_counts.values())
    var = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
    if var <= 0:
        return 1.0
    z = (w_plus - mu) / math.sqrt(var)
    return min(1.0, 2.0 * _normal_sf(abs(z)))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    max_iter = 200
    eps = 3e-14
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """Survival function of Student's t distribution with ``df`` degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired t-test p-value via the t-distribution survival function."""
    diffs = [y - x for x, y in _pairs(a, b)]
    n = len(diffs)
    if n < 2:
        return 1.0
    m = mean(diffs)
    s = std(diffs, sample=True)
    if s == 0:
        return 1.0 if m == 0 else 0.0
    t = abs(m) / (s / math.sqrt(n))
    return min(1.0, 2.0 * _t_sf(t, n - 1))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d: mean(diff) / std(diff). Positive means b > a."""
    diffs = [y - x for x, y in _pairs(a, b)]
    if len(diffs) < 2:
        return 0.0
    s = std(diffs, sample=True)
    if s == 0:
        return 0.0
    return mean(diffs) / s


def bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Bonferroni correction: p * m, capped at 1."""
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotonic)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values (monotonic)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(1.0, pvalues[idx] * m / (rank + 1))
        running = min(running, val)
        adjusted[idx] = running
    return adjusted


def correct_pvalues(pvalues: Sequence[float], method: str = "holm") -> list[float]:
    """Dispatch to a multiple-comparison correction by name."""
    methods = {
        "bonferroni": bonferroni,
        "holm": holm,
        "bh": benjamini_hochberg,
        "fdr": benjamini_hochberg,
    }
    if method not in methods:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(methods)}")
    return methods[method](pvalues)
