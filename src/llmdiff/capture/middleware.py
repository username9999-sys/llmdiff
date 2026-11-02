"""ASGI middleware that captures LLM API traffic passing through an app.

:class:`CaptureMiddleware` wraps any ASGI app and intercepts requests to
``/v1/chat/completions`` (OpenAI-style) and ``/v1/messages`` (Anthropic-style):

1. It reads the request body (buffered, then restored for the downstream app).
2. Extracts the prompt text from the JSON body (``messages`` for chat
   completions, ``messages`` + ``system`` for Anthropic).
3. Calls the downstream app, buffering the response.
4. Extracts the output text from the response JSON, or accumulates SSE
   chunks when the response is ``text/event-stream``.
5. Records a :class:`~llmdiff.core.case.Case` with metadata
   ``{model, endpoint, timestamp, latency_ms, ...}``.

The downstream response is passed through untouched: same status, headers
(minus ``content-length`` which is recomputed) and body bytes.

Works with any ASGI framework (Starlette/FastAPI, raw ASGI).  Stdlib-only.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from llmdiff.capture.errors import MiddlewareError
from llmdiff.capture.record import TrafficRecorder

__all__ = ["CaptureMiddleware", "extract_prompt", "extract_output", "parse_sse_chunks"]

Scope = MutableMapping = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

CAPTURED_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _messages_to_text(messages: Any) -> str:
    """Flatten a ``messages`` list into a single prompt string."""
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # multimodal content blocks: keep text blocks only
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    parts.append(f"{role}: {block.get('text', '')}")
    return "\n".join(parts)


def extract_prompt(path: str, body: bytes) -> tuple[str, dict[str, Any]]:
    """Extract ``(prompt_text, extra_meta)`` from a request body.

    Returns ``("", {})`` when the body is not JSON or has no messages.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", {}
    if not isinstance(payload, Mapping):
        return "", {}
    meta: dict[str, Any] = {}
    model = payload.get("model")
    if isinstance(model, str):
        meta["model"] = model
    if path == "/v1/messages":
        system = payload.get("system")
        if isinstance(system, str) and system:
            prompt = f"system: {system}\n" + _messages_to_text(payload.get("messages"))
        else:
            prompt = _messages_to_text(payload.get("messages"))
    else:
        prompt = _messages_to_text(payload.get("messages"))
    return prompt, meta


def extract_output(path: str, status: int, headers: list[tuple[bytes, bytes]], body: bytes) -> str:
    """Extract the assistant output text from a response.

    Handles OpenAI-style ``choices[0].message.content``, Anthropic-style
    ``content`` blocks, and plain ``{"output": ...}`` / ``{"completion": ...}``.
    Returns ``""`` for non-JSON bodies (the caller handles SSE separately).
    """
    if status >= 400:
        return ""
    content_type = b""
    for key, value in headers:
        if key.lower() == b"content-type":
            content_type = value.lower()
            break
    if b"event-stream" in content_type:
        return ""  # handled by parse_sse_chunks
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    # OpenAI chat completions
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    # Anthropic messages
    content = payload.get("content")
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        return "".join(texts)
    if isinstance(content, str):
        return content
    # Generic fallbacks
    for key in ("output", "completion", "text", "response"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def parse_sse_chunks(body: bytes) -> str:
    """Accumulate text deltas from an SSE body (``data: {...}`` lines).

    Understands OpenAI (``choices[0].delta.content``) and Anthropic
    (``content_block_delta.delta.text``) chunk shapes.
    """
    text_parts: list[str] = []
    for raw_line in body.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            delta = choices[0].get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            continue
        # Anthropic content_block_delta: {"type": "content_block_delta", "delta": {"text": ...}}
        if payload.get("type") == "content_block_delta":
            d = payload.get("delta")
            if isinstance(d, Mapping) and isinstance(d.get("text"), str):
                text_parts.append(d["text"])
            continue
        delta = payload.get("delta")
        if isinstance(delta, Mapping) and isinstance(delta.get("text"), str):
            text_parts.append(delta["text"])
    return "".join(text_parts)


class CaptureMiddleware:
    """ASGI middleware recording LLM traffic into a :class:`TrafficRecorder`.

    Args:
        app: The downstream ASGI app.
        recorder: Where captured exchanges are recorded.
        paths: Endpoint paths to capture (default: the two standard ones).
        capture_errors: When true, also record responses with status >= 400
            (output will be empty; the error body is kept in metadata).
        tags: Extra tags added to every captured record.
    """

    def __init__(
        self,
        app: ASGIApp,
        recorder: TrafficRecorder,
        *,
        paths: Iterable[str] | None = None,
        capture_errors: bool = False,
        tags: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(recorder, TrafficRecorder):
            raise MiddlewareError("recorder must be a TrafficRecorder")
        self.app = app
        self.recorder = recorder
        self.paths = frozenset(paths) if paths is not None else CAPTURED_PATHS
        self.capture_errors = capture_errors
        self.tags = tuple(tags or ())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("path", "") not in self.paths:
            await self.app(scope, receive, send)
            return

        # ---- buffer the request body, then replay it downstream
        body = b""
        more = True
        while more:
            message = await receive()
            if message.get("type") != "http.request":
                # client disconnected etc.; forward as-is
                await self.app(scope, receive, send)
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
        request_body = body  # keep a copy for recording; replay_receive drains `body`

        async def replay_receive() -> Message:
            nonlocal body
            if body:
                chunk, body = body, b""
                return {"type": "http.request", "body": chunk, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        # ---- capture the response
        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []
        response_body = b""

        async def capture_send(message: Message) -> None:
            nonlocal status_code, response_headers, response_body
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = [tuple(h) for h in message.get("headers", [])]
            elif message.get("type") == "http.response.body":
                response_body += message.get("body", b"")
                if not message.get("more_body", False):
                    await send(message)
                return
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, replay_receive, capture_send)
        except Exception:
            # Never let capture break the downstream app: if the app raised,
            # re-raise after recording nothing (response never started).
            raise
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        # ---- record (use the buffered copy; `body` was drained by replay_receive)
        path = scope.get("path", "")
        prompt, extra = extract_prompt(path, request_body)
        is_stream = any(
            k.lower() == b"content-type" and b"event-stream" in v.lower()
            for k, v in response_headers
        )
        if is_stream:
            output = parse_sse_chunks(response_body)
        else:
            output = extract_output(path, status_code, response_headers, response_body)
        if status_code >= 400 and not self.capture_errors:
            return
        meta: dict[str, Any] = {
            "endpoint": path,
            "timestamp": _utcnow_iso(),
            "latency_ms": latency_ms,
            "status": status_code,
            "stream": is_stream,
        }
        meta.update(extra)
        if status_code >= 400:
            meta["error_body"] = response_body.decode("utf-8", errors="replace")[:4000]
        self.recorder.record(prompt, output, tags=self.tags, **meta)

    def __repr__(self) -> str:
        return f"CaptureMiddleware(paths={sorted(self.paths)!r}, recorder={self.recorder!r})"
