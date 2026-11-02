"""llmdiff.analytics — statistical analysis of evaluation runs."""

from __future__ import annotations

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

__all__ = [
    "mean",
    "variance",
    "std",
    "median",
    "percentile",
    "min_max",
    "describe",
    "bootstrap_ci",
    "histogram",
]
