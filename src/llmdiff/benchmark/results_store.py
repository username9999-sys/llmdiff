"""SQLite storage for benchmark results, with query and export helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from llmdiff.benchmark.metrics import BenchmarkMetrics
from llmdiff.benchmark.runner import ScenarioResult

__all__ = ["store_results", "load_results", "export_csv", "export_json", "history"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    label TEXT NOT NULL,
    metrics TEXT NOT NULL,
    scenarios TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bench_label ON benchmark_runs(label);
"""


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def store_results(
    path: str | Path,
    results: list[ScenarioResult],
    metrics: BenchmarkMetrics,
    *,
    label: str = "default",
) -> int:
    """Store one benchmark run. Returns the new row id."""
    conn = _connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO benchmark_runs (created_at, label, metrics, scenarios) VALUES (?, ?, ?, ?)",
            (
                time.time(),
                label,
                json.dumps(
                    {
                        "n_scenarios": metrics.n_scenarios,
                        "detection_rate": metrics.detection_rate,
                        "false_positive_rate": metrics.false_positive_rate,
                        "cases_per_second": metrics.cases_per_second,
                        "mean_elapsed_s": metrics.mean_elapsed_s,
                        "total_cases": metrics.total_cases,
                        "mean_peak_memory_bytes": metrics.mean_peak_memory_bytes,
                    }
                ),
                json.dumps(
                    [
                        {
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
                        for r in results
                    ]
                ),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_results(
    path: str | Path, *, label: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """Load stored benchmark runs, newest first. Optionally filter by label."""
    conn = _connect(path)
    try:
        if label is None:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE label = ? ORDER BY created_at DESC LIMIT ?",
                (label, limit),
            ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "label": r["label"],
            "metrics": json.loads(r["metrics"]),
            "scenarios": json.loads(r["scenarios"]),
        }
        for r in rows
    ]


def history(path: str | Path, *, label: str = "default") -> list[tuple[float, float]]:
    """Return (created_at, detection_rate) history for a label, oldest first."""
    runs = load_results(path, label=label, limit=10_000)
    runs.reverse()
    return [(r["created_at"], r["metrics"]["detection_rate"]) for r in runs]


def export_csv(path: str | Path, out: str | Path) -> Path:
    """Export all stored runs to a CSV file. Returns the output path."""
    import csv

    runs = load_results(path, limit=100_000)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "created_at", "label", "detection_rate", "false_positive_rate", "cases_per_second", "n_scenarios"]
        )
        for r in runs:
            m = r["metrics"]
            writer.writerow(
                [r["id"], r["created_at"], r["label"], m["detection_rate"], m["false_positive_rate"], m["cases_per_second"], m["n_scenarios"]]
            )
    return target


def export_json(path: str | Path, out: str | Path) -> Path:
    """Export all stored runs to a JSON file. Returns the output path."""
    runs = load_results(path, limit=100_000)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return target
