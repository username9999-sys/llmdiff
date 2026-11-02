"""Correlation and dimensionality analysis over per-case score matrices.

Pure stdlib: Pearson/Spearman correlation, covariance matrix, and PCA via
power iteration on the covariance matrix (no numpy required).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from llmdiff.analytics.stats import mean, std

__all__ = [
    "pearson",
    "spearman",
    "covariance_matrix",
    "correlation_matrix",
    "rank",
    "pca",
    "redundant_dimensions",
]


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation coefficient in [-1, 1]; 0.0 when undefined."""
    if len(a) != len(b):
        raise ValueError("series must have equal length")
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    sa, sb = std(a, sample=False), std(b, sample=False)
    if sa == 0 or sb == 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / n
    return max(-1.0, min(1.0, cov / (sa * sb)))


def rank(values: Sequence[float]) -> list[float]:
    """Ranks (1-based) with average ranks for ties."""
    xs = [float(v) for v in values]
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson on the ranks."""
    if len(a) != len(b):
        raise ValueError("series must have equal length")
    return pearson(rank(a), rank(b))


def covariance_matrix(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Sample covariance matrix (n-1 denominator) of the given columns."""
    m = len(columns)
    if m == 0:
        return []
    n = len(columns[0])
    for c in columns:
        if len(c) != n:
            raise ValueError("all columns must have equal length")
    means = [mean(c) for c in columns]
    denom = max(1, n - 1)
    cov = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            s = sum((columns[i][k] - means[i]) * (columns[j][k] - means[j]) for k in range(n))
            cov[i][j] = cov[j][i] = s / denom
    return cov


def correlation_matrix(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Pearson correlation matrix of the given columns (diagonal = 1.0)."""
    m = len(columns)
    if m == 0:
        return []
    return [[pearson(columns[i], columns[j]) for j in range(m)] for i in range(m)]


def _power_iteration(
    mat: list[list[float]], *, iterations: int = 200, seed: int = 0
) -> tuple[float, list[float]]:
    """Largest eigenvalue/eigenvector of a symmetric matrix via power iteration."""
    n = len(mat)
    rng = random.Random(seed)
    v = [rng.random() for _ in range(n)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / norm for x in v]
    lam = 0.0
    for _ in range(iterations):
        w = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
        nw = math.sqrt(sum(x * x for x in w))
        if nw == 0:
            break
        v = [x / nw for x in w]
        lam = nw
    return lam, v


def pca(columns: Sequence[Sequence[float]], *, components: int = 2) -> dict[str, object]:
    """PCA over columns: explained variance ratios and component loadings.

    Uses power iteration with deflation on the covariance matrix.
    """
    m = len(columns)
    if m == 0:
        return {"explained_variance_ratio": [], "components": [], "eigenvalues": []}
    cov = covariance_matrix(columns)
    total = sum(cov[i][i] for i in range(m)) or 1.0
    remaining = [row[:] for row in cov]
    eigenvalues: list[float] = []
    comps: list[list[float]] = []
    k = min(components, m)
    for _ in range(k):
        lam, vec = _power_iteration(remaining)
        eigenvalues.append(lam)
        comps.append(vec)
        for i in range(m):
            for j in range(m):
                remaining[i][j] -= lam * vec[i] * vec[j]
    ratios = [max(0.0, lam) / total for lam in eigenvalues]
    return {
        "explained_variance_ratio": ratios,
        "components": comps,
        "eigenvalues": eigenvalues,
    }


def redundant_dimensions(
    columns: Sequence[Sequence[float]],
    names: Sequence[str],
    *,
    threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """Pairs of dimensions whose |correlation| >= threshold (likely redundant)."""
    if len(columns) != len(names):
        raise ValueError("columns and names must have equal length")
    corr = correlation_matrix(columns)
    out: list[tuple[str, str, float]] = []
    m = len(names)
    for i in range(m):
        for j in range(i + 1, m):
            r = corr[i][j]
            if abs(r) >= threshold:
                out.append((names[i], names[j], r))
    return sorted(out, key=lambda t: -abs(t[2]))
