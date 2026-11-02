"""Trend analysis over a series of evaluation runs.

A "run series" is a list of per-dimension loss rates (or win rates) ordered by
time. All functions are stdlib-only and deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmdiff.analytics.stats import mean

__all__ = [
    "linear_regression",
    "slope",
    "r_squared",
    "cusum_change_point",
    "forecast_linear",
    "moving_average",
    "run_series_from_results",
]


def _xy(ys: Sequence[float]) -> tuple[list[float], list[float]]:
    xs = [float(i) for i in range(len(ys))]
    return xs, [float(y) for y in ys]


def linear_regression(ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares line y = intercept + slope * x over index positions.

    Returns ``(intercept, slope)``. Empty input gives (0.0, 0.0).
    """
    xs, yv = _xy(ys)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx, my = mean(xs), mean(yv)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, yv, strict=True)) / denom
    return my - sl * mx, sl


def slope(ys: Sequence[float]) -> float:
    """Just the slope of the regression line."""
    return linear_regression(ys)[1]


def r_squared(ys: Sequence[float]) -> float:
    """Coefficient of determination of the linear fit, in [0, 1]."""
    xs, yv = _xy(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    intercept, sl = linear_regression(ys)
    my = mean(yv)
    ss_tot = sum((y - my) ** 2 for y in yv)
    if ss_tot == 0:
        return 1.0
    ss_res = sum((y - (intercept + sl * x)) ** 2 for x, y in zip(xs, yv, strict=True))
    return max(0.0, 1.0 - ss_res / ss_tot)


def cusum_change_point(ys: Sequence[float], *, threshold: float = 1.0) -> int | None:
    """Detect the first index where the CUSUM statistic crosses the threshold.

    Returns the index of the strongest change point, or None when stable.
    """
    xs = [float(y) for y in ys]
    n = len(xs)
    if n < 3:
        return None
    overall = mean(xs)
    cusum = 0.0
    best_idx: int | None = None
    best_val = 0.0
    for i, x in enumerate(xs):
        cusum += x - overall
        if abs(cusum) > best_val:
            best_val = abs(cusum)
            best_idx = i
    if best_idx is not None and best_val >= threshold * math.sqrt(n) * 0.5:
        return best_idx
    return None


def forecast_linear(ys: Sequence[float], steps: int) -> list[float]:
    """Forecast ``steps`` future values by extending the regression line."""
    if steps <= 0:
        return []
    intercept, sl = linear_regression(ys)
    n = len(ys)
    return [intercept + sl * (n + i) for i in range(steps)]


def moving_average(ys: Sequence[float], window: int) -> list[float]:
    """Centered moving average with the given window size."""
    if window < 1:
        raise ValueError("window must be >= 1")
    xs = [float(y) for y in ys]
    n = len(xs)
    if window == 1 or n == 0:
        return list(xs)
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(mean(xs[lo:hi]))
    return out


def run_series_from_results(
    results: Sequence[object], dimension: str, metric: str = "loss_rate"
) -> list[float]:
    """Extract a per-run time series for one dimension from DiffResult-like objects.

    Args:
        results: Objects with ``.matrix.get(dimension)`` returning a summary
            with ``loss_rate``/``win_rate``/``delta``/``candidate_mean``.
        dimension: Dimension name to extract.
        metric: One of ``loss_rate``, ``win_rate``, ``delta``, ``candidate_mean``.
    """
    out: list[float] = []
    for result in results:
        summary = result.matrix.get(dimension)
        if summary is None:
            continue
        if metric == "loss_rate":
            out.append(summary.loss_rate)
        elif metric == "win_rate":
            out.append(summary.win_rate)
        elif metric == "delta":
            out.append(summary.delta)
        elif metric == "candidate_mean":
            out.append(summary.candidate_mean)
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return out
