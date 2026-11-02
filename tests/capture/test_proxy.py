"""Tests for llmdiff.capture.proxy — local target server in a thread, offline."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llmdiff.capture.proxy import build_arg_parser, make_server
from llmdiff.capture.record import TrafficRecorder


class _TargetHandler(BaseHTTPRequestHandler):
    """Fake upstream LLM API."""

    def log_message(self, *args):
        pass

    def _respond(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            self._respond({"choices": [{"message": {"content": "proxied-answer"}}]})
        elif self.path == "/v1/messages":
            self._respond({"content": [{"type": "text", "text": "proxied-claude"}]})
        else:
            self._respond({"error": "not found"}, status=404)

    def do_GET(self):
        self._respond({"ok": True})


@pytest.fixture()
def target_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TargetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def proxy_server(target_server):
    recorder = TrafficRecorder()
    server = make_server(port=0, target=target_server, host="127.0.0.1", recorder=recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, recorder
    server.shutdown()
    server.server_close()


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_proxy_forwards_and_records_chat_completions(proxy_server, target_server):
    server, recorder = proxy_server
    port = server.server_address[1]
    status, payload = _post(
        port,
        "/v1/chat/completions",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "via proxy"}]},
    )
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "proxied-answer"
    assert len(recorder) == 1
    case = recorder.cases[0]
    assert "via proxy" in case.prompt
    assert case.candidate_output == "proxied-answer"
    assert case.metadata["model"] == "gpt-4o"
    assert case.metadata["endpoint"] == "/v1/chat/completions"
    assert case.metadata["via"] == "proxy"
    assert isinstance(case.metadata["latency_ms"], float)


def test_proxy_records_anthropic_messages(proxy_server):
    server, recorder = proxy_server
    port = server.server_address[1]
    status, payload = _post(
        port,
        "/v1/messages",
        {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200
    assert payload["content"][0]["text"] == "proxied-claude"
    assert recorder.cases[0].candidate_output == "proxied-claude"
    assert recorder.cases[0].metadata["endpoint"] == "/v1/messages"


def test_proxy_passes_through_non_captured_paths(proxy_server):
    server, recorder = proxy_server
    port = server.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode()) == {"ok": True}
    assert len(recorder) == 0


def test_proxy_upstream_error_returns_502():
    # target that refuses connections
    recorder = TrafficRecorder()
    server = make_server(port=0, target="http://127.0.0.1:1", host="127.0.0.1", recorder=recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b'{"messages":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 502
    finally:
        server.shutdown()
        server.server_close()


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.port == 8080
    assert args.target == "https://api.openai.com"
    assert args.out == "traffic.jsonl"
    args2 = build_arg_parser().parse_args(
        ["--port", "9000", "--target", "http://x", "--tag", "a", "--tag", "b"]
    )
    assert args2.port == 9000 and args2.tag == ["a", "b"]
