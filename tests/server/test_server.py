"""Tests for the server package (offline, in-memory storage)."""

import json

import pytest

from llmdiff.server.storage import RunStorage

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from llmdiff.server.app import create_app  # noqa: E402

RECORDS = [
    {
        "id": "c1",
        "prompt": "hi",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m1", "tags": ["x"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "c2",
        "prompt": "yo",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
]


def test_storage_roundtrip():
    storage = RunStorage(":memory:")
    record = storage.save({"matrix": {"verdict": "loss"}, "cases": [], "config": {}}, {"tolerance": 0.0})
    assert storage.count() == 1
    loaded = storage.get(record.id)
    assert loaded is not None
    assert loaded.result["matrix"]["verdict"] == "loss"
    assert len(storage.list()) == 1
    assert storage.delete(record.id) is True
    assert storage.count() == 0
    assert storage.get(record.id) is None
    storage.close()


def test_api_endpoints():
    storage = RunStorage(":memory:")
    client = TestClient(create_app(storage))

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200

    resp = client.post("/api/runs", json={"records": RECORDS})
    assert resp.status_code == 201
    run = resp.json()
    run_id = run["id"]
    assert run["result"]["matrix"]["total_cases"] == 2

    listing = client.get("/api/runs").json()
    assert listing["count"] == 1
    assert listing["runs"][0]["id"] == run_id

    full = client.get(f"/api/runs/{run_id}").json()
    assert "matrix" in full["result"]

    matrix = client.get(f"/api/runs/{run_id}/matrix").json()
    assert matrix["total_cases"] == 2
    dims = {d["dimension"] for d in matrix["dimensions"]}
    assert "structure" in dims

    assert client.get("/api/runs/nope").status_code == 404
    assert client.delete(f"/api/runs/{run_id}").json() == {"deleted": True}
    assert client.get(f"/api/runs/{run_id}").status_code == 404
