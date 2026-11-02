"""Tests for llmdiff.capture.middleware — raw ASGI calls, no network."""

from __future__ import annotations

import json

from llmdiff.capture.middleware import (
    CaptureMiddleware,
    extract_output,
    extract_prompt,
    parse_sse_chunks,
)
from llmdiff.capture.record import TrafficRecorder


async def _call_asgi(app, path="/v1/chat/completions", body=b"{}", method="POST"):
    """Drive an ASGI app directly and return (status, headers, body)."""
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    sent = []
    received = {"body": body, "sent": False}

    async def receive():
        if not received["sent"]:
            received["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = 500
    headers = []
    out_body = b""
    for msg in sent:
        if msg["type"] == "http.response.start":
            status = msg["status"]
            headers = msg.get("headers", [])
        elif msg["type"] == "http.response.body":
            out_body += msg.get("body", b"")
    return status, headers, out_body


def _downstream_app(
    response_body: bytes, status: int = 200, content_type: bytes = b"application/json"
):
    async def app(scope, receive, send):
        await receive()  # consume request body
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", content_type)],
            }
        )
        await send({"type": "http.response.body", "body": response_body, "more_body": False})

    return app


def test_extract_prompt_chat_completions():
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi there"}]}
    ).encode()
    prompt, meta = extract_prompt("/v1/chat/completions", body)
    assert "user: hi there" in prompt
    assert meta["model"] == "gpt-4o"


def test_extract_prompt_anthropic_with_system():
    body = json.dumps(
        {
            "model": "claude-sonnet-4-5",
            "system": "be terse",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    prompt, meta = extract_prompt("/v1/messages", body)
    assert "system: be terse" in prompt
    assert "user: hello" in prompt
    assert meta["model"] == "claude-sonnet-4-5"


def test_extract_prompt_invalid_json():
    prompt, meta = extract_prompt("/v1/chat/completions", b"not json")
    assert prompt == "" and meta == {}


def test_extract_output_openai_shape():
    body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "hi!"}}]}
    ).encode()
    headers = [(b"content-type", b"application/json")]
    out = extract_output("/v1/chat/completions", 200, headers, body)
    assert out == "hi!"


def test_extract_output_anthropic_shape():
    body = json.dumps(
        {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}
    ).encode()
    headers = [(b"content-type", b"application/json")]
    out = extract_output("/v1/messages", 200, headers, body)
    assert out == "hello world"


def test_extract_output_error_status_returns_empty():
    body = json.dumps({"error": "bad"}).encode()
    headers = [(b"content-type", b"application/json")]
    out = extract_output("/v1/chat/completions", 400, headers, body)
    assert out == ""


def test_parse_sse_chunks_openai():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]
    body = ("\n\n".join(lines) + "\n\n").encode()
    assert parse_sse_chunks(body) == "Hello world"


def test_parse_sse_chunks_anthropic():
    lines = [
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"abc"}}',
    ]
    body = ("\n\n".join(lines) + "\n\n").encode()
    assert parse_sse_chunks(body) == "abc"


def test_middleware_records_chat_completion():
    import asyncio

    recorder = TrafficRecorder()
    req_body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "2+2?"}]}
    ).encode()
    resp_body = json.dumps(
        {"choices": [{"message": {"content": "4"}}]}
    ).encode()
    app = CaptureMiddleware(_downstream_app(resp_body), recorder)
    status, headers, out = asyncio.run(_call_asgi(app, body=req_body))
    assert status == 200
    assert out == resp_body  # passed through untouched
    assert len(recorder) == 1
    case = recorder.cases[0]
    assert "2+2?" in case.prompt
    assert case.candidate_output == "4"
    assert case.metadata["model"] == "gpt-4o"
    assert case.metadata["endpoint"] == "/v1/chat/completions"
    assert "timestamp" in case.metadata
    assert isinstance(case.metadata["latency_ms"], float)


def test_middleware_records_anthropic_messages():
    import asyncio

    recorder = TrafficRecorder()
    req_body = json.dumps(
        {
            "model": "claude-x",
            "system": "sys",
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode()
    resp_body = json.dumps({"content": [{"type": "text", "text": "pong"}]}).encode()
    app = CaptureMiddleware(_downstream_app(resp_body), recorder)
    status, _, _ = asyncio.run(_call_asgi(app, path="/v1/messages", body=req_body))
    assert status == 200
    case = recorder.cases[0]
    assert "ping" in case.prompt and "sys" in case.prompt
    assert case.candidate_output == "pong"
    assert case.metadata["endpoint"] == "/v1/messages"


def test_middleware_passes_through_non_captured_paths():
    import asyncio

    recorder = TrafficRecorder()
    app = CaptureMiddleware(_downstream_app(b'{"ok":true}'), recorder)
    status, _, out = asyncio.run(_call_asgi(app, path="/health", body=b"{}"))
    assert status == 200
    assert out == b'{"ok":true}'
    assert len(recorder) == 0


def test_middleware_handles_streaming_response():
    import asyncio

    recorder = TrafficRecorder()
    req_body = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    ).encode()
    sse = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    app = CaptureMiddleware(
        _downstream_app(sse, content_type=b"text/event-stream"), recorder
    )
    status, headers, out = asyncio.run(_call_asgi(app, body=req_body))
    assert status == 200
    assert out == sse  # body passed through
    case = recorder.cases[0]
    assert case.candidate_output == "ab"
    assert case.metadata["stream"] is True


def test_middleware_skips_error_responses_by_default():
    import asyncio

    recorder = TrafficRecorder()
    app = CaptureMiddleware(_downstream_app(b'{"error":"x"}', status=500), recorder)
    asyncio.run(_call_asgi(app, body=b'{"messages":[]}'))
    assert len(recorder) == 0


def test_middleware_capture_errors_option():
    import asyncio

    recorder = TrafficRecorder()
    app = CaptureMiddleware(
        _downstream_app(b'{"error":"boom"}', status=500), recorder, capture_errors=True
    )
    asyncio.run(_call_asgi(app, body=b'{"messages":[{"role":"user","content":"x"}]}'))
    assert len(recorder) == 1
    assert recorder.cases[0].metadata["status"] == 500
    assert "error_body" in recorder.cases[0].metadata


def test_middleware_requires_recorder():
    import pytest

    from llmdiff.capture.errors import MiddlewareError

    with pytest.raises(MiddlewareError):
        CaptureMiddleware(_downstream_app(b"{}"), recorder=object())  # type: ignore[arg-type]
