"""Anomaly detection over score series, stdlib only.

Detects outlier cases/dimensions via z-score, modified z-score (MAD), IQR
fences, and distance-to-medoid. All deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmdiff.analytics.stats import mean, median, percentile, std

__all__ = [
    "zscore",
    "zscore_outliers",
    "modified_zscore",
    "mad_outliers",
    "iqr_outliers",
    "iqr_fences",
    "distance_outliers",
    "detect_outliers",
]


def zscore(values: Sequence[float]) -> list[float]:
    """Standard z-scores of the values (0.0 when std is 0)."""
    xs = [float(v) for v in values]
    if not xs:
        return []
    m = mean(xs)
    s = std(xs, sample=False)
    if s == 0:
        return [0.0] * len(xs)
    return [(x - m) / s for x in xs]


def zscore_outliers(values: Sequence[float], *, threshold: float = 3.0) -> list[int]:
    """Indices whose |z| >= threshold."""
    zs = zscore(values)
    return [i for i, z in enumerate(zs) if abs(z) >= threshold]


def modified_zscore(values: Sequence[float]) -> list[float]:
    """Modified z-score using MAD: 0.6745 * (x - median) / MAD."""
    xs = [float(v) for v in values]
    if not xs:
        return []
    med = median(xs)
    mad = median([abs(x - med) for x in xs])
    if mad == 0:
        return [0.0] * len(xs)
    return [0.6745 * (x - med) / mad for x in xs]


def mad_outliers(values: Sequence[float], *, threshold: float = 3.5) -> list[int]:
    """Indices whose |modified z| >= threshold (robust to outliers themselves)."""
    mz = modified_zscore(values)
    return [i for i, z in enumerate(mz) if abs(z) >= threshold]


def iqr_fences(values: Sequence[float]) -> tuple[float, float]:
    """Return (lower, upper) IQR fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR."""
    xs = [float(v) for v in values]
    if not xs:
        return 0.0, 0.0
    q1 = percentile(xs, 25)
    q3 = percentile(xs, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outliers(values: Sequence[float]) -> list[int]:
    """Indices outside the IQR fences."""
    lo, hi = iqr_fences(values)
    return [i for i, x in enumerate(values) if float(x) < lo or float(x) > hi]


def _medoid_index(points: Sequence[Sequence[float]]) -> int:
    """Index of the point with minimal total distance to all others."""
    n = len(points)
    if n == 0:
        return -1
    dim = len(points[0])
    best, best_total = 0, float("inf")
    for i in range(n):
        total = 0.0
        for j in range(n):
            if i == j:
                continue
            total += math.sqrt(sum((points[i][d] - points[j][d]) ** 2 for d in range(dim)))
        if total < best_total:
            best_total = total
            best = i
    return best


def distance_outliers(
    points: Sequence[Sequence[float]], *, threshold: float = 2.0
) -> list[int]:
    """Indices whose distance to the medoid exceeds threshold * MAD of distances."""
    n = len(points)
    if n < 2:
        return []
    med = _medoid_index(points)
    dim = len(points[0])
    dists = [
        math.sqrt(sum((points[i][d] - points[med][d]) ** 2 for d in range(dim)))
        for i in range(n)
    ]
    med_d = median(dists)
    mad_d = median([abs(d - med_d) for d in dists])
    if mad_d == 0:
        return []
    return [i for i, d in enumerate(dists) if abs(d - med_d) / (mad_d / 0.6745) >= threshold]


def detect_outliers(
    values: Sequence[float], *, method: str = "mad", threshold: float = 3.5
) -> list[int]:
    """Dispatch to a named outlier detector: ``zscore``, ``mad``, or ``iqr``."""
    if method == "zscore":
        return zscore_outliers(values, threshold=threshold)
    if method == "mad":
        return mad_outliers(values, threshold=threshold)
    if method == "iqr":
        return iqr_outliers(values)
    raise ValueError(f"unknown method {method!r}; choose zscore/mad/iqr")
