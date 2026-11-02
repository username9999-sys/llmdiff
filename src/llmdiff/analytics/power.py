"""Sample-size and power analysis for detecting a win-rate change, stdlib only.

Normal approximation for a two-proportion comparison (baseline win rate vs
candidate win rate). Deterministic and offline.
"""

from __future__ import annotations

import math

__all__ = [
    "normal_cdf",
    "normal_ppf",
    "sample_size_for_win_rate",
    "power_for_win_rate",
    "minimum_detectable_effect",
]


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be within (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def sample_size_for_win_rate(
    baseline_rate: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Sample size per arm needed to detect ``effect`` change in win rate.

    Args:
        baseline_rate: Expected win rate under the null, in (0, 1).
        effect: Absolute change to detect (candidate = baseline + effect).
        alpha: Significance level.
        power: Desired power.

    Returns:
        Required number of cases per arm (rounded up).
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be within (0, 1)")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be within (0, 1)")
    p1 = baseline_rate
    p2 = min(1.0, max(0.0, baseline_rate + effect))
    if p1 == p2:
        raise ValueError("effect must be non-zero")
    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    z_beta = normal_ppf(power)
    p_bar = (p1 + p2) / 2.0
    numerator = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_beta * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    n = (numerator**2) / ((p2 - p1) ** 2)
    return math.ceil(n)


def power_for_win_rate(
    baseline_rate: float,
    effect: float,
    n: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Achieved power for detecting ``effect`` with ``n`` cases per arm."""
    if n < 1:
        raise ValueError("n must be >= 1")
    p1 = baseline_rate
    p2 = min(1.0, max(0.0, baseline_rate + effect))
    if p1 == p2:
        raise ValueError("effect must be non-zero")
    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    p_bar = (p1 + p2) / 2.0
    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    z = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    return normal_cdf(z)


def minimum_detectable_effect(
    baseline_rate: float,
    n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest absolute win-rate change detectable with ``n`` cases per arm."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be within (0, 1)")
    lo, hi = 1e-9, 1.0 - baseline_rate
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if power_for_win_rate(baseline_rate, mid, n, alpha=alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
