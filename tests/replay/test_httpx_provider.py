"""Tests for HttpProvider against a local in-thread HTTP server (offline)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmdiff.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)
from llmdiff.replay.httpx_provider import (
    AsyncHttpProvider,
    HttpProvider,
    build_chat_payload,
    parse_retry_after,
)


class _Handler(BaseHTTPRequestHandler):
    """Scripted handler: behaviour depends on the request path and headers."""

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        auth = self.headers.get("Authorization", "")
        if self.path == "/v1/chat/completions":
            if auth == "Bearer bad-key":
                self._send(401, {"error": "invalid api key"})
                return
            if self.headers.get("X-Test-RateLimit") == "1":
                self._send(429, {"error": "slow down"}, extra={"Retry-After": "3"})
                return
            if self.headers.get("X-Test-Error") == "1":
                self._send(500, {"error": "internal"})
                return
            prompt = body["messages"][-1]["content"]
            self._send(
                200,
                {
                    "id": "chatcmpl-1",
                    "model": body.get("model", "test-model"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": f"echo:{prompt}"}}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            )
            return
        self._send(404, {"error": "not found"})

    def _send(self, status, payload, extra=None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join(timeout=5)


def test_build_chat_payload_passthrough():
    payload = build_chat_payload("m", "hi", {"temperature": 0.3, "max_tokens": 10, "system": "sys"})
    assert payload["model"] == "m"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 10


def test_parse_retry_after_case_insensitive():
    assert parse_retry_after({"Retry-After": "2.5"}) == 2.5
    assert parse_retry_after({"retry-after": "7"}) == 7.0
    assert parse_retry_after({"Retry-After": "soon"}) is None
    assert parse_retry_after({}) is None


def test_http_provider_success(server):
    provider = HttpProvider(base_url=server, api_key="good", model="test-model")
    response = provider.complete("hello", temperature=0.2)
    assert response.text == "echo:hello"
    assert response.tokens_in == 5
    assert response.tokens_out == 2
    assert response.model == "test-model"
    assert response.latency_ms >= 0
    assert response.ok
    provider.close()


def test_http_provider_auth_error(server):
    provider = HttpProvider(base_url=server, api_key="bad-key", model="m")
    with pytest.raises(ProviderAuthError):
        provider.complete("hi")
    provider.close()


def test_http_provider_rate_limit_with_retry_after(server):
    import httpx

    provider = HttpProvider(base_url=server, api_key="good", model="m")
    # Inject the rate-limit trigger header via a custom client.
    client = httpx.Client(
        timeout=10.0, headers={"X-Test-RateLimit": "1"}, transport=httpx.HTTPTransport()
    )
    provider._client = client
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete("hi")
    assert exc_info.value.retry_after == 3.0
    provider.close()


def test_http_provider_server_error(server):
    import httpx

    provider = HttpProvider(base_url=server, api_key="good", model="m")
    client = httpx.Client(
        timeout=10.0, headers={"X-Test-Error": "1"}, transport=httpx.HTTPTransport()
    )
    provider._client = client
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.complete("hi")
    provider.close()


def test_http_provider_name_and_url(server):
    provider = HttpProvider(base_url=server + "/", api_key=None, model="m")
    assert provider.url == f"{server}/v1/chat/completions"
    assert provider.name == f"http:{server}:m"
    provider.close()


def test_async_http_provider_success(server):
    import asyncio

    async def main():
        provider = AsyncHttpProvider(base_url=server, api_key="good", model="amodel")
        response = await provider.acomplete("async-hi")
        await provider.aclose()
        return response

    response = asyncio.run(main())
    assert response.text == "echo:async-hi"
    assert response.tokens_in == 5
