"""A configurable mock LLM server for offline end-to-end tests.

Serves OpenAI-compatible ``/v1/chat/completions`` and Anthropic-compatible
``/v1/messages`` on a local thread. Responses are scripted per prompt hash,
with optional latency and error injection. No external network is used.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["MockLLMServer", "scripted_response"]


def scripted_response(
    prompt: str, responses: dict[str, str] | None = None, default: str = "mock reply"
) -> str:
    """Deterministic response: exact prompt match, else hash-based pick, else default."""
    if responses and prompt in responses:
        return responses[prompt]
    if responses:
        keys = sorted(responses)
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        return responses[keys[h % len(keys)]]
    return default


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockLLM/1.0"

    def log_message(self, *args) -> None:  # silence default logging
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        srv = self.server
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        if srv.latency_s > 0:
            time.sleep(srv.latency_s)
        if srv.error_rate > 0:
            roll = int(hashlib.sha256(raw).hexdigest()[:4], 16) % 10000
            if roll < srv.error_rate * 10000:
                self._send_json(srv.error_code, {"error": f"injected {srv.error_code}"})
                return

        if self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            prompt = messages[-1].get("content", "") if messages else ""
            text = scripted_response(prompt, srv.responses, srv.default)
            self._send_json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt) // 4,
                        "completion_tokens": len(text) // 4,
                        "total_tokens": (len(prompt) + len(text)) // 4,
                    },
                },
            )
        elif self.path == "/v1/messages":
            messages = body.get("messages", [])
            prompt = messages[-1].get("content", "") if messages else ""
            if isinstance(prompt, list):
                prompt = "".join(
                    str(b.get("text", "")) for b in prompt if isinstance(b, dict)
                )
            text = scripted_response(prompt, srv.responses, srv.default)
            self._send_json(
                200,
                {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "model": body.get("model", "mock"),
                    "usage": {
                        "input_tokens": len(prompt) // 4,
                        "output_tokens": len(text) // 4,
                    },
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._send_json(404, {"error": "not found"})


class MockLLMServer:
    """A threaded local mock LLM server.

    Args:
        responses: Optional mapping prompt -> response text.
        default: Fallback response text.
        latency_s: Artificial latency per request.
        error_rate: Fraction of requests that fail with ``error_code``.
        error_code: HTTP status to inject on errors.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default: str = "mock reply",
        latency_s: float = 0.0,
        error_rate: float = 0.0,
        error_code: int = 500,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.latency_s = latency_s
        self.error_rate = error_rate
        self.error_code = error_code
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.responses = self.responses
        self._httpd.default = default
        self._httpd.latency_s = latency_s
        self._httpd.error_rate = error_rate
        self._httpd.error_code = error_code
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockLLMServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "MockLLMServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
