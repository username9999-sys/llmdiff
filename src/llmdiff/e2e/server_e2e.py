"""Helpers for e2e server tests via FastAPI TestClient (offline, in-memory)."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from llmdiff.server.app import create_app  # noqa: E402
from llmdiff.server.storage import RunStorage  # noqa: E402

__all__ = ["make_client", "RECORDS"]


def make_client() -> TestClient:
    """Return a TestClient backed by in-memory storage."""
    return TestClient(create_app(RunStorage(":memory:")))


RECORDS = [
    {
        "id": "c1",
        "prompt": "p",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m", "tags": ["json"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "c2",
        "prompt": "p",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
]
