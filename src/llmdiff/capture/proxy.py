"""Runnable local HTTP capture proxy (stdlib only).

Forwards every request to a target base URL and records LLM exchanges
(paths ``/v1/chat/completions`` and ``/v1/messages``) into a
:class:`~llmdiff.capture.record.TrafficRecorder`, persisted as JSONL.

Run as a module::

    python -m llmdiff.capture.proxy --port 8080 \
        --target https://api.openai.com --out traffic.jsonl

Then point any OpenAI/Anthropic-compatible client at
``http://localhost:8080/v1`` with any dummy API key.

Implementation notes:

* ``http.server.ThreadingHTTPServer`` + a ``BaseHTTPRequestHandler`` subclass;
* forwarding uses ``http.client`` (stdlib) so redirects are NOT followed
  automatically (safer for capture);
* request/response bodies are fully buffered so they can be both forwarded
  and recorded; streaming responses are accumulated for recording while
  still being streamed to the client chunk-by-chunk;
* only stdlib is required, so the proxy works even without fastapi/httpx.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse
from collections.abc import Iterable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from llmdiff.capture.errors import ProxyError
from llmdiff.capture.middleware import extract_prompt, parse_sse_chunks
from llmdiff.capture.record import TrafficRecorder

__all__ = ["CaptureProxyHandler", "make_server", "run_proxy", "main", "build_arg_parser"]

CAPTURED_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
    }
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _extract_output_json(body: bytes) -> str:
    """Extract assistant text from a non-streaming JSON response body."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"]
    content = payload.get("content")
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    if isinstance(content, str):
        return content
    for key in ("output", "completion", "text", "response"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return ""


class CaptureProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that forwards requests and records captured exchanges."""

    server_version = "llmdiff-capture-proxy/0.1"
    protocol_version = "HTTP/1.1"

    # injected by make_server()
    recorder: TrafficRecorder
    target_host: str
    target_port: int
    target_scheme: str
    capture_paths: frozenset[str]
    extra_tags: tuple[str, ...]

    # ------------------------------------------------------------ plumbing
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Quiet by default; enable with --verbose on the server.
        server = self.server
        if getattr(server, "verbose", False):
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None:
            # chunked transfer encoding is not supported for capture; read none
            return b""
        try:
            n = int(length)
        except ValueError as exc:
            raise ProxyError(f"invalid Content-Length: {length!r}") from exc
        if n < 0:
            raise ProxyError("negative Content-Length")
        return self.rfile.read(n)

    def _forward(self, method: str, body: bytes) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
        """Forward the request to the target and return (status, headers, body)."""
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        if self.target_scheme == "https":
            conn_cls: type[http.client.HTTPConnection] = http.client.HTTPSConnection
        else:
            conn_cls = http.client.HTTPConnection
        conn = conn_cls(self.target_host, self.target_port, timeout=120)
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() != "host"
        }
        headers["Host"] = f"{self.target_host}:{self.target_port}"
        if body:
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(method, path, body=body if body else None, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_headers = [(k.lower().encode(), v.encode()) for k, v in resp.getheaders()]
            return resp.status, resp_headers, resp_body
        finally:
            conn.close()

    def _relay(self, method: str) -> None:
        try:
            body = self._read_body()
        except ProxyError as exc:
            self.send_error(400, str(exc))
            return
        started = time.perf_counter()
        try:
            status, resp_headers, resp_body = self._forward(method, body)
        except Exception as exc:
            self.send_error(502, f"proxy upstream error: {exc}")
            return
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        # ---- record if this is a captured LLM endpoint
        path_only = urllib.parse.urlsplit(self.path).path
        if path_only in self.capture_paths:
            self._record(path_only, body, status, resp_headers, resp_body, latency_ms)

        # ---- relay response to the client
        self.send_response(status)
        is_stream = False
        for key, value in resp_headers:
            if key == b"content-type" and b"event-stream" in value.lower():
                is_stream = True
            if key in HOP_BY_HOP:
                continue
            self.send_header(key.decode("latin-1"), value.decode("latin-1"))
        if is_stream:
            # stream through in chunks so the client sees SSE live
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk_size = 8192
            for i in range(0, len(resp_body), chunk_size):
                chunk = resp_body[i : i + chunk_size]
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            if resp_body:
                self.wfile.write(resp_body)

    def _record(
        self,
        path: str,
        req_body: bytes,
        status: int,
        resp_headers: list[tuple[bytes, bytes]],
        resp_body: bytes,
        latency_ms: float,
    ) -> None:
        prompt, extra = extract_prompt(path, req_body)
        is_stream = any(
            k == b"content-type" and b"event-stream" in v.lower() for k, v in resp_headers
        )
        if is_stream:
            output = parse_sse_chunks(resp_body)
        else:
            output = _extract_output_json(resp_body) if status < 400 else ""
        meta: dict[str, Any] = {
            "endpoint": path,
            "timestamp": _utcnow_iso(),
            "latency_ms": latency_ms,
            "status": status,
            "stream": is_stream,
            "via": "proxy",
        }
        meta.update(extra)
        try:
            self.recorder.record(prompt, output, tags=self.extra_tags, **meta)
        except Exception as exc:  # recording must never break the proxy
            sys.stderr.write(f"[llmdiff.capture.proxy] record failed: {exc}\n")

    # ------------------------------------------------------------ verbs
    def do_GET(self) -> None:  # noqa: N802
        self._relay("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._relay("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._relay("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._relay("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._relay("PATCH")


class CaptureProxyServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the capture configuration."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        addr: tuple[str, int],
        recorder: TrafficRecorder,
        target: str,
        *,
        capture_paths: Iterable[str] | None = None,
        extra_tags: Iterable[str] = (),
        verbose: bool = False,
    ) -> None:
        parsed = urllib.parse.urlsplit(target if "://" in target else "https://" + target)
        self.recorder = recorder
        self.target_scheme = parsed.scheme or "https"
        self.target_host = parsed.hostname or "localhost"
        self.target_port = parsed.port or (443 if self.target_scheme == "https" else 80)
        self.capture_paths = frozenset(capture_paths) if capture_paths else CAPTURED_PATHS
        self.extra_tags = tuple(extra_tags)
        self.verbose = verbose
        super().__init__(addr, CaptureProxyHandler)
        # bind config onto the handler class for each request
        CaptureProxyHandler.recorder = recorder  # type: ignore[attr-defined]
        CaptureProxyHandler.target_host = self.target_host  # type: ignore[attr-defined]
        CaptureProxyHandler.target_port = self.target_port  # type: ignore[attr-defined]
        CaptureProxyHandler.target_scheme = self.target_scheme  # type: ignore[attr-defined]
        CaptureProxyHandler.capture_paths = self.capture_paths  # type: ignore[attr-defined]
        CaptureProxyHandler.extra_tags = self.extra_tags  # type: ignore[attr-defined]


def make_server(
    port: int = 8080,
    target: str = "https://api.openai.com",
    *,
    host: str = "127.0.0.1",
    recorder: TrafficRecorder | None = None,
    out: str | Path | None = None,
    extra_tags: Iterable[str] = (),
    verbose: bool = False,
) -> CaptureProxyServer:
    """Create (but do not start) a capture proxy server.

    Args:
        port: Local listen port.
        target: Base URL of the upstream API (e.g. ``https://api.openai.com``).
        host: Local bind address.
        recorder: Recorder to use; a fresh one is created when omitted.
        out: When given, the recorder's output path (used by :func:`run_proxy`
            to save on shutdown).
        extra_tags: Tags added to every captured record.
        verbose: Log each request to stderr.
    """
    if recorder is None:
        recorder = TrafficRecorder()
    server = CaptureProxyServer(
        (host, port), recorder, target, extra_tags=extra_tags, verbose=verbose
    )
    server.out_path = Path(out) if out is not None else None  # type: ignore[attr-defined]
    return server


def run_proxy(
    port: int = 8080,
    target: str = "https://api.openai.com",
    *,
    host: str = "127.0.0.1",
    out: str | Path = "traffic.jsonl",
    recorder: TrafficRecorder | None = None,
    extra_tags: Iterable[str] = (),
    verbose: bool = False,
    serve_once: bool = False,
) -> CaptureProxyServer:
    """Create and serve a capture proxy (blocking).

    On KeyboardInterrupt the recorded traffic is saved to ``out``.
    With ``serve_once=True`` a single request is handled then the server
    stops (useful for tests).
    """
    server = make_server(
        port=port,
        target=target,
        host=host,
        recorder=recorder,
        out=out,
        extra_tags=extra_tags,
        verbose=verbose,
    )
    out_path = getattr(server, "out_path", None)
    try:
        if serve_once:
            server.handle_request()
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if out_path is not None:
            try:
                server.recorder.save(out_path)
            except Exception as exc:
                sys.stderr.write(f"[llmdiff.capture.proxy] save failed: {exc}\n")
        server.server_close()
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llmdiff.capture.proxy",
        description=(
            "Local capture proxy: forwards LLM API traffic to a target and records it as JSONL."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="local listen port (default 8080)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="local bind address (default 127.0.0.1)"
    )
    parser.add_argument(
        "--target",
        default="https://api.openai.com",
        help="upstream base URL, e.g. https://api.openai.com (default)",
    )
    parser.add_argument(
        "--out", default="traffic.jsonl", help="output JSONL path (default traffic.jsonl)"
    )
    parser.add_argument(
        "--tag", action="append", default=[], help="extra tag for captured records (repeatable)"
    )
    parser.add_argument("--verbose", action="store_true", help="log requests to stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(
        f"llmdiff capture proxy listening on http://{args.host}:{args.port} "
        f"-> {args.target} (out: {args.out})",
        file=sys.stderr,
    )
    run_proxy(
        port=args.port,
        target=args.target,
        host=args.host,
        out=args.out,
        extra_tags=tuple(args.tag),
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
