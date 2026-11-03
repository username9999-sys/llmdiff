"""Tests for server.static — web asset serving with path-traversal guard."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llmdiff.server.static import WEB_ROOT, mount_static, web_file_response  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    mount_static(app)
    return TestClient(app)


def test_web_root_exists():
    assert WEB_ROOT.is_dir()
    assert (WEB_ROOT / "index.html").is_file()


def test_web_file_response_index():
    assert web_file_response("index.html") is not None


def test_web_file_response_missing_returns_none():
    assert web_file_response("nope.js") is None


def test_web_file_response_traversal_blocked():
    resp = web_file_response("../pyproject.toml")
    assert resp is None or getattr(resp, "status_code", 200) == 403


def test_mounted_index_and_asset():
    client = _client()
    r = client.get("/app")
    assert r.status_code == 200
    assert "llmdiff" in r.text
    r2 = client.get("/app/static/css/app.css")
    assert r2.status_code == 200
    assert "{" in r2.text


def test_mounted_404_and_traversal():
    client = _client()
    assert client.get("/app/missing.js").status_code == 404
    r = client.get("/app/..%2F..%2Fpyproject.toml")
    assert r.status_code in (403, 404)
