"""Descriptive statistics over check scores, stdlib only."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

__all__ = [
    "mean", "variance", "std", "median", "percentile",
    "min_max", "describe", "bootstrap_ci", "histogram",
]


def _floats(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; 0.0 for an empty series."""
    xs = _floats(values)
    return sum(xs) / len(xs) if xs else 0.0


def variance(values: Iterable[float], *, sample: bool = True) -> float:
    """Variance. Sample variance (n-1) by default; population when sample=False."""
    xs = _floats(values)
    n = len(xs)
    if n == 0:
        return 0.0
    denom = n - 1 if sample and n > 1 else n
    if denom == 0:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / denom


def std(values: Iterable[float], *, sample: bool = True) -> float:
    """Standard deviation (square root of variance)."""
    return math.sqrt(variance(values, sample=sample))


def median(values: Iterable[float]) -> float:
    """Median; 0.0 for empty input."""
    xs = sorted(_floats(values))
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear-interpolation percentile. ``pct`` is in [0, 100]."""
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be within [0, 100]")
    xs = sorted(_floats(values))
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    rank = (pct / 100.0) * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return xs[low]
    frac = rank - low
    return xs[low] + frac * (xs[high] - xs[low])


def min_max(values: Iterable[float]) -> tuple[float, float]:
    """Return (min, max); (0.0, 0.0) for empty input."""
    xs = _floats(values)
    if not xs:
        return 0.0, 0.0
    return min(xs), max(xs)


def describe(values: Iterable[float]) -> dict[str, float]:
    """Return count/mean/std/min/p25/median/p75/max as a dict."""
    xs = _floats(values)
    lo, hi = min_max(xs)
    return {
        "count": float(len(xs)),
        "mean": mean(xs),
        "std": std(xs),
        "min": lo,
        "p25": percentile(xs, 25),
        "median": median(xs),
        "p75": percentile(xs, 75),
        "max": hi,
    }


def bootstrap_ci(
    values: Iterable[float],
    *,
    stat: str = "mean",
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean or median (reproducible via seed)."""
    xs = _floats(values)
    if not xs:
        return 0.0, 0.0
    if stat not in ("mean", "median"):
        raise ValueError("stat must be 'mean' or 'median'")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    rng = random.Random(seed)
    n = len(xs)
    func = mean if stat == "mean" else median
    stats = sorted(func([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    alpha = 1.0 - confidence
    return percentile(stats, alpha / 2.0 * 100.0), percentile(stats, (1.0 - alpha / 2.0) * 100.0)


def histogram(
    values: Iterable[float],
    bins: int = 10,
    *,
    bounds: tuple[float, float] | None = None,
) -> list[tuple[float, float, int]]:
    """Return ``bins`` (low, high, count) buckets covering the values."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    xs = _floats(values)
    if not xs:
        return []
    lo, hi = bounds if bounds is not None else min_max(xs)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        idx = min(max(int((x - lo) / width), 0), bins - 1)
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]
