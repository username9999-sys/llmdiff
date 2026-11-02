"""Tests for e2e.mock_llm_server — fully offline, local thread only."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from llmdiff.e2e.mock_llm_server import MockLLMServer, scripted_response


def _post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_scripted_response_exact_and_default():
    assert scripted_response("hi", {"hi": "yo"}) == "yo"
    assert scripted_response("other", {"hi": "yo"}) == "yo"  # hash pick of the only entry
    assert scripted_response("x") == "mock reply"


def test_server_openai_chat():
    with MockLLMServer({"hello": "world"}) as srv:
        code, data = _post(
            f"{srv.base_url}/v1/chat/completions",
            {"model": "mock", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert code == 200
        assert data["choices"][0]["message"]["content"] == "world"
        assert "usage" in data


def test_server_anthropic_messages():
    with MockLLMServer(default="bonjour") as srv:
        code, data = _post(
            f"{srv.base_url}/v1/messages",
            {"model": "mock", "messages": [{"role": "user", "content": "salut"}]},
        )
        assert code == 200
        assert data["content"][0]["text"] == "bonjour"


def test_server_health_and_models():
    with MockLLMServer() as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health", timeout=5) as r:
            assert json.loads(r.read()) == {"status": "ok"}
        with urllib.request.urlopen(f"{srv.base_url}/v1/models", timeout=5) as r:
            assert r.status == 200


def test_server_error_injection():
    with MockLLMServer(error_rate=1.0, error_code=429) as srv:
        code, data = _post(
            f"{srv.base_url}/v1/chat/completions",
            {"model": "mock", "messages": [{"role": "user", "content": "x"}]},
        )
        assert code == 429
        assert "error" in data


def test_server_404():
    with MockLLMServer() as srv:
        code, _ = _post(f"{srv.base_url}/nope", {})
        assert code == 404
