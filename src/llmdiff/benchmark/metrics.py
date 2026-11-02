"""Benchmark metrics: detection rate, false-positive rate, throughput.

Given a list of ScenarioResult objects (from benchmark.runner), compute
aggregate quality metrics for llmdiff's own verdict matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llmdiff.benchmark.runner import ScenarioResult

__all__ = [
    "BenchmarkMetrics",
    "compute_metrics",
    "detection_rate",
    "false_positive_rate",
    "throughput",
]


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Aggregate benchmark metrics over a set of scenario results."""

    n_scenarios: int
    detection_rate: float
    false_positive_rate: float
    mean_elapsed_s: float
    total_cases: int
    cases_per_second: float
    mean_peak_memory_bytes: float
    per_scenario: tuple[str, ...] = ()


def detection_rate(
    results: Sequence[ScenarioResult],
    *,
    defect_scenarios: set[str] | None = None,
) -> float:
    """Fraction of defect scenarios where the matrix reported losses.

    By default every scenario whose name does not start with ``clean`` is
    treated as a defect scenario.
    """
    if defect_scenarios is None:
        defects = [r for r in results if not r.name.startswith("clean")]
    else:
        defects = [r for r in results if r.name in defect_scenarios]
    if not defects:
        return 0.0
    return sum(1 for r in defects if r.detected) / len(defects)


def false_positive_rate(results: Sequence[ScenarioResult]) -> float:
    """Fraction of clean scenarios where the matrix wrongly reported losses."""
    cleans = [r for r in results if r.name.startswith("clean")]
    if not cleans:
        return 0.0
    return sum(1 for r in cleans if r.detected) / len(cleans)


def throughput(results: Sequence[ScenarioResult]) -> float:
    """Total cases evaluated divided by total elapsed seconds."""
    total_cases = sum(r.n_cases for r in results)
    total_time = sum(r.elapsed_s for r in results)
    return total_cases / total_time if total_time > 0 else 0.0


def compute_metrics(results: Sequence[ScenarioResult]) -> BenchmarkMetrics:
    """Compute the full aggregate metric set over scenario results."""
    if not results:
        raise ValueError("at least one scenario result is required")
    n = len(results)
    total_cases = sum(r.n_cases for r in results)
    total_time = sum(r.elapsed_s for r in results)
    return BenchmarkMetrics(
        n_scenarios=n,
        detection_rate=detection_rate(results),
        false_positive_rate=false_positive_rate(results),
        mean_elapsed_s=total_time / n,
        total_cases=total_cases,
        cases_per_second=throughput(results),
        mean_peak_memory_bytes=sum(r.peak_memory_bytes for r in results) / n,
        per_scenario=tuple(r.name for r in results),
    )
