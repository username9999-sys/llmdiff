"""Query helpers over the server's SQLite run storage, stdlib only.

Works with any SQLite file that has the ``runs`` table created by
``llmdiff.server.storage`` (id, created_at, config, result as JSON text).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RunRow",
    "query_runs",
    "run_history",
    "aggregate_by_model",
    "aggregate_by_tag",
    "load_result",
]


@dataclass(frozen=True, slots=True)
class RunRow:
    """One row of the runs table, with parsed JSON columns."""

    id: str
    created_at: str
    config: dict[str, Any]
    result: dict[str, Any]


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_run(row: sqlite3.Row) -> RunRow:
    return RunRow(
        id=row["id"],
        created_at=row["created_at"],
        config=json.loads(row["config"]),
        result=json.loads(row["result"]),
    )


def query_runs(
    path: str | Path,
    *,
    limit: int = 100,
    offset: int = 0,
    order: str = "DESC",
) -> list[RunRow]:
    """Return recent runs from the SQLite store, newest first by default."""
    if order.upper() not in ("ASC", "DESC"):
        raise ValueError("order must be ASC or DESC")
    conn = _connect(path)
    try:
        rows = conn.execute(
            f"SELECT id, created_at, config, result FROM runs "
            f"ORDER BY created_at {order.upper()} LIMIT ? OFFSET ?",
            (max(1, limit), max(0, offset)),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_run(r) for r in rows]


def run_history(
    path: str | Path, *, dimension: str, metric: str = "loss_rate"
) -> list[tuple[str, float]]:
    """Return ``(created_at, metric)`` per run for one dimension, oldest first."""
    runs = query_runs(path, limit=10_000, order="ASC")
    out: list[tuple[str, float]] = []
    for run in runs:
        dims = run.result.get("matrix", {}).get("dimensions", [])
        for d in dims:
            if d.get("dimension") == dimension:
                out.append((run.created_at, float(d.get(metric, 0.0))))
                break
    return out


def aggregate_by_model(path: str | Path) -> dict[str, dict[str, float]]:
    """Aggregate mean loss rate per model recorded in run configs."""
    runs = query_runs(path, limit=10_000)
    buckets: dict[str, list[float]] = {}
    for run in runs:
        model = run.config.get("model") or run.result.get("config", {}).get("model") or "unknown"
        loss_rate = float(run.result.get("matrix", {}).get("regression_rate", 0.0))
        buckets.setdefault(str(model), []).append(loss_rate)
    return {
        model: {"count": float(len(v)), "mean_loss_rate": sum(v) / len(v)}
        for model, v in buckets.items()
    }


def aggregate_by_tag(path: str | Path) -> dict[str, dict[str, float]]:
    """Aggregate mean per-case loss rate per tag found in case metadata."""
    runs = query_runs(path, limit=10_000)
    buckets: dict[str, list[float]] = {}
    for run in runs:
        for case in run.result.get("cases", []):
            meta = case.get("metadata") or {}
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            total = case.get("wins", 0) + case.get("ties", 0) + case.get("losses", 0)
            loss_rate = float(case.get("losses", 0)) / max(1, total)
            for tag in tags:
                buckets.setdefault(str(tag), []).append(loss_rate)
    return {
        tag: {"count": float(len(v)), "mean_loss_rate": sum(v) / len(v)}
        for tag, v in buckets.items()
    }


def load_result(path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Load the parsed result JSON for one run id, or None."""
    conn = _connect(path)
    try:
        row = conn.execute("SELECT result FROM runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return json.loads(row["result"])
