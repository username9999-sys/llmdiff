"""Tests for analytics.db — uses a temp SQLite file with the runs schema."""

from __future__ import annotations

import json
import sqlite3

from llmdiff.analytics.db import (
    aggregate_by_model,
    aggregate_by_tag,
    load_result,
    query_runs,
    run_history,
)


def _make_db(tmp_path, runs):
    db = tmp_path / "runs.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            config TEXT NOT NULL,
            result TEXT NOT NULL
        );
        """
    )
    for run in runs:
        conn.execute(
            "INSERT INTO runs (id, created_at, config, result) VALUES (?, ?, ?, ?)",
            (run["id"], run["created_at"], json.dumps(run["config"]), json.dumps(run["result"])),
        )
    conn.commit()
    conn.close()
    return db


def _sample_runs():
    return [
        {
            "id": "r1",
            "created_at": "2026-11-01T10:00:00+00:00",
            "config": {"model": "gpt-4o-mini"},
            "result": {
                "matrix": {
                    "regression_rate": 0.1,
                    "dimensions": [{"dimension": "structure", "loss_rate": 0.2}],
                },
                "cases": [
                    {"case_id": "c1", "wins": 1, "ties": 1, "losses": 0,
                     "metadata": {"tags": ["refund"]}}
                ],
            },
        },
        {
            "id": "r2",
            "created_at": "2026-11-01T11:00:00+00:00",
            "config": {"model": "gpt-4o"},
            "result": {
                "matrix": {
                    "regression_rate": 0.3,
                    "dimensions": [{"dimension": "structure", "loss_rate": 0.4}],
                },
                "cases": [
                    {"case_id": "c2", "wins": 0, "ties": 0, "losses": 2,
                     "metadata": {"tags": ["refund", "json"]}}
                ],
            },
        },
    ]


def test_query_runs_order(tmp_path):
    db = _make_db(tmp_path, _sample_runs())
    runs = query_runs(db)
    assert [r.id for r in runs] == ["r2", "r1"]  # DESC default
    runs_asc = query_runs(db, order="ASC")
    assert [r.id for r in runs_asc] == ["r1", "r2"]


def test_run_history(tmp_path):
    db = _make_db(tmp_path, _sample_runs())
    hist = run_history(db, dimension="structure", metric="loss_rate")
    assert hist == [
        ("2026-11-01T10:00:00+00:00", 0.2),
        ("2026-11-01T11:00:00+00:00", 0.4),
    ]


def test_aggregate_by_model(tmp_path):
    db = _make_db(tmp_path, _sample_runs())
    agg = aggregate_by_model(db)
    assert agg["gpt-4o-mini"]["mean_loss_rate"] == 0.1
    assert agg["gpt-4o"]["count"] == 1.0


def test_aggregate_by_tag(tmp_path):
    db = _make_db(tmp_path, _sample_runs())
    agg = aggregate_by_tag(db)
    assert "refund" in agg
    assert agg["refund"]["count"] == 2.0


def test_load_result(tmp_path):
    db = _make_db(tmp_path, _sample_runs())
    result = load_result(db, "r1")
    assert result is not None
    assert result["matrix"]["regression_rate"] == 0.1
    assert load_result(db, "nope") is None
