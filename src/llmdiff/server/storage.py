"""SQLite-backed storage for evaluation runs.

One table, one job: persist a DiffResult as JSON and read it back. Uses only
the standard library so the server works with zero extra dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["RunStorage", "RunRecord"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecord:
    """One stored evaluation run."""

    __slots__ = ("id", "created_at", "config", "result")

    def __init__(self, id: str, created_at: str, config: dict[str, Any], result: dict[str, Any]) -> None:
        self.id = id
        self.created_at = created_at
        self.config = config
        self.result = result
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "config": self.config,
            "result": self.result,
        }


class RunStorage:
    """Thread-safe SQLite storage for runs.

    Args:
        path: Database file path, or ``":memory:"`` for an in-memory database.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, result: dict[str, Any], config: dict[str, Any] | None = None) -> RunRecord:
        """Persist a DiffResult dict and return the stored record."""
        run_id = uuid.uuid4().hex[:12]
        created_at = _utcnow_iso()
        cfg = config if config is not None else result.get("config", {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, created_at, config, result) VALUES (?, ?, ?, ?)",
                (run_id, created_at, json.dumps(cfg), json.dumps(result)),
            )
            self._conn.commit()
        return RunRecord(run_id, created_at, cfg, result)

    def get(self, run_id: str) -> RunRecord | None:
        """Return the run with ``run_id``, or None."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunRecord(row["id"], row["created_at"], json.loads(row["config"]), json.loads(row["result"]))

    def list(self, limit: int = 100, offset: int = 0) -> list[RunRecord]:
        """Return recent runs, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            RunRecord(r["id"], r["created_at"], json.loads(r["config"]), json.loads(r["result"]))
            for r in rows
        ]

    def delete(self, run_id: str) -> bool:
        """Delete a run. Returns True when a row was removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        """Return the number of stored runs."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()
