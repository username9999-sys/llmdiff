"""Benchmark report: markdown + JSON rendering of scenario results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from llmdiff.benchmark.metrics import BenchmarkMetrics, compute_metrics
from llmdiff.benchmark.runner import ScenarioResult

__all__ = ["render_benchmark_markdown", "render_benchmark_json", "metrics_to_dict"]


def metrics_to_dict(m: BenchmarkMetrics) -> dict[str, Any]:
    """JSON-serialisable representation of BenchmarkMetrics."""
    return {
        "n_scenarios": m.n_scenarios,
        "detection_rate": round(m.detection_rate, 6),
        "false_positive_rate": round(m.false_positive_rate, 6),
        "mean_elapsed_s": round(m.mean_elapsed_s, 6),
        "total_cases": m.total_cases,
        "cases_per_second": round(m.cases_per_second, 3),
        "mean_peak_memory_bytes": round(m.mean_peak_memory_bytes, 1),
        "per_scenario": list(m.per_scenario),
    }


def render_benchmark_markdown(results: Sequence[ScenarioResult]) -> str:
    """Render scenario results + aggregate metrics as a markdown report."""
    m = compute_metrics(results)
    lines = [
        "# llmdiff self-benchmark",
        "",
        f"- scenarios: {m.n_scenarios}",
        f"- detection rate (defect scenarios): **{m.detection_rate:.1%}**",
        f"- false positive rate (clean scenarios): **{m.false_positive_rate:.1%}**",
        f"- total cases: {m.total_cases} in {m.mean_elapsed_s * m.n_scenarios:.3f}s"
        f" ({m.cases_per_second:.1f} cases/s)",
        f"- mean peak memory: {m.mean_peak_memory_bytes / 1e6:.2f} MB",
        "",
        "| scenario | cases | verdict | W | T | L | detected | s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_cases} | {r.verdict} | {r.wins} | {r.ties} | {r.losses} | "
            f"{'yes' if r.detected else 'no'} | {r.elapsed_s:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_benchmark_json(results: Sequence[ScenarioResult]) -> str:
    """Render scenario results + aggregate metrics as pretty JSON."""
    m = compute_metrics(results)
    payload = {
        "metrics": metrics_to_dict(m),
        "scenarios": [
            {
                "name": r.name,
                "n_cases": r.n_cases,
                "elapsed_s": round(r.elapsed_s, 6),
                "peak_memory_bytes": r.peak_memory_bytes,
                "verdict": r.verdict,
                "wins": r.wins,
                "ties": r.ties,
                "losses": r.losses,
                "regressed_dimensions": list(r.regressed_dimensions),
                "detected": r.detected,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
