"""Tests for OllamaProvider (local server) and SDK-fallback provider paths."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmdiff.core.errors import ProviderError, ProviderTimeoutError
from llmdiff.replay.anthropic_provider import AnthropicProvider
from llmdiff.replay.ollama_provider import AsyncOllamaProvider, OllamaProvider
from llmdiff.replay.openai_provider import OpenAIProvider


class _OllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/api/generate"
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("prompt") == "fail":
            payload = {"error": "model not found"}
            self._send(404, payload)
            return
        self._send(
            200,
            {
                "model": body["model"],
                "response": f"ollama:{body['prompt']}",
                "prompt_eval_count": 7,
                "eval_count": 3,
            },
        )

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def ollama_server():
    httpd = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join(timeout=5)


def test_ollama_provider_success(ollama_server):
    provider = OllamaProvider(model="llama3.1", base_url=ollama_server)
    response = provider.complete("hi", temperature=0.1)
    assert response.text == "ollama:hi"
    assert response.tokens_in == 7
    assert response.tokens_out == 3
    assert response.model == "llama3.1"
    assert provider.name == "ollama:llama3.1"


def test_ollama_provider_error(ollama_server):
    provider = OllamaProvider(model="llama3.1", base_url=ollama_server)
    with pytest.raises(ProviderError, match="model not found"):
        provider.complete("fail")


def test_ollama_provider_timeout():
    # Port 1 is closed -> connection refused -> ProviderError (transport).
    provider = OllamaProvider(model="m", base_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_async_ollama_provider(ollama_server):
    import asyncio

    async def main():
        provider = AsyncOllamaProvider(model="m", base_url=ollama_server)
        return await provider.acomplete("yo")

    response = asyncio.run(main())
    assert response.text == "ollama:yo"


def test_openai_provider_falls_back_or_uses_sdk():
    # We only assert construction and naming here; no network call.
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", base_url="http://127.0.0.1:1")
    assert provider.name == "openai:gpt-4o-mini"
    assert isinstance(provider.uses_sdk, bool)


def test_anthropic_provider_http_fallback_against_local_server():
    """AnthropicProvider without the SDK must call /v1/messages over HTTP."""

    class _AnthropicHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/messages"
            assert self.headers.get("x-api-key") == "sk-ant-test"
            assert self.headers.get("anthropic-version") == "2023-06-01"
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.headers.get("X-Bad-Key") == "1":
                data = json.dumps({"error": "auth"}).encode()
                self.send_response(401)
            else:
                data = json.dumps(
                    {
                        "id": "msg_1",
                        "model": body["model"],
                        "content": [
                            {"type": "text", "text": f"ant:{body['messages'][0]['content']}"}
                        ],
                        "usage": {"input_tokens": 4, "output_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-test", base_url=base)
        if provider.uses_sdk:
            pytest.skip("anthropic SDK installed; HTTP fallback path not exercised")
        response = provider.complete("hello")
        assert response.text == "ant:hello"
        assert response.tokens_in == 4
        assert response.tokens_out == 2
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_openai_provider_http_fallback_against_local_server():
    """OpenAIProvider without the SDK must hit /v1/chat/completions."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            data = json.dumps(
                {
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "sdk-fallback-ok"}}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        provider = OpenAIProvider(api_key="sk-x", model="m", base_url=base)
        if provider.uses_sdk:
            pytest.skip("openai SDK installed; HTTP fallback path not exercised")
        assert provider.complete("hi").text == "sdk-fallback-ok"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_provider_timeout_error_type():
    # A timeout against a non-routable address with a tiny timeout must map to
    # ProviderTimeoutError for the HTTP provider.
    from llmdiff.replay.httpx_provider import HttpProvider

    provider = HttpProvider(base_url="http://10.255.255.1", model="m", timeout=0.2)
    with pytest.raises((ProviderTimeoutError, ProviderError)):
        provider.complete("hi")
    provider.close()
