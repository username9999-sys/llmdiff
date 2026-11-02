"""A/B comparison of two benchmark runs (e.g. two llmdiff versions/configs).

Compares detection rates between two sets of ScenarioResult objects using a
two-proportion z-test (normal approximation, stdlib only).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llmdiff.analytics.power import normal_cdf
from llmdiff.benchmark.metrics import detection_rate, false_positive_rate
from llmdiff.benchmark.runner import ScenarioResult

__all__ = ["ABResult", "compare_runs", "two_proportion_ztest"]


@dataclass(frozen=True, slots=True)
class ABResult:
    """Outcome of comparing two benchmark runs."""

    name_a: str
    name_b: str
    detection_a: float
    detection_b: float
    fpr_a: float
    fpr_b: float
    z_stat: float
    p_value: float
    significant: bool


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return z, min(1.0, p)


def compare_runs(
    a: Sequence[ScenarioResult],
    b: Sequence[ScenarioResult],
    *,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> ABResult:
    """Compare detection/FPR of two benchmark runs with a significance test."""
    if not a or not b:
        raise ValueError("both runs need at least one scenario")
    defects_a = [r for r in a if not r.name.startswith("clean")]
    defects_b = [r for r in b if not r.name.startswith("clean")]
    k1 = sum(1 for r in defects_a if r.detected)
    k2 = sum(1 for r in defects_b if r.detected)
    z, p = two_proportion_ztest(k1, len(defects_a), k2, len(defects_b))
    return ABResult(
        name_a=name_a,
        name_b=name_b,
        detection_a=detection_rate(a),
        detection_b=detection_rate(b),
        fpr_a=false_positive_rate(a),
        fpr_b=false_positive_rate(b),
        z_stat=z,
        p_value=p,
        significant=p < alpha,
    )
