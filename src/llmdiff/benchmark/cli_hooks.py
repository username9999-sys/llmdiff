"""Hooks callable from the CLI ``llmdiff bench`` command (lazy imports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmdiff.benchmark.metrics import compute_metrics
from llmdiff.benchmark.report import render_benchmark_json, render_benchmark_markdown
from llmdiff.benchmark.results_store import store_results
from llmdiff.benchmark.runner import run_all_scenarios, run_scenario

__all__ = ["bench", "bench_scenario", "bench_report"]


def bench(
    *,
    dimensions: tuple[str, ...] | None = None,
    out_md: str | Path | None = None,
    out_json: str | Path | None = None,
    db: str | Path | None = None,
    label: str = "default",
) -> dict[str, Any]:
    """Run all benchmark scenarios, optionally write reports and store results.

    Returns a dict with metrics + rendered report paths (when requested).
    """
    results = run_all_scenarios(dimensions=dimensions)
    metrics = compute_metrics(results)
    out: dict[str, Any] = {
        "metrics": {
            "n_scenarios": metrics.n_scenarios,
            "detection_rate": metrics.detection_rate,
            "false_positive_rate": metrics.false_positive_rate,
            "cases_per_second": metrics.cases_per_second,
            "total_cases": metrics.total_cases,
        },
        "scenarios": [r.name for r in results],
    }
    if out_md is not None:
        Path(out_md).write_text(render_benchmark_markdown(results), encoding="utf-8")
        out["markdown_path"] = str(out_md)
    if out_json is not None:
        Path(out_json).write_text(render_benchmark_json(results), encoding="utf-8")
        out["json_path"] = str(out_json)
    if db is not None:
        out["db_row_id"] = store_results(db, results, metrics, label=label)
    return out


def bench_scenario(name: str, *, dimensions: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run a single named scenario and return its metrics as a dict."""
    r = run_scenario(name, dimensions=dimensions)
    return {
        "name": r.name,
        "n_cases": r.n_cases,
        "elapsed_s": r.elapsed_s,
        "peak_memory_bytes": r.peak_memory_bytes,
        "verdict": r.verdict,
        "wins": r.wins,
        "ties": r.ties,
        "losses": r.losses,
        "detected": r.detected,
        "regressed_dimensions": list(r.regressed_dimensions),
    }


def bench_report(results_md_path: str | Path) -> str:
    """Re-read an existing markdown report (pass-through helper for the CLI)."""
    return Path(results_md_path).read_text(encoding="utf-8")
