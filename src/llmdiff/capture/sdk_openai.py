"""Capturing wrapper for an OpenAI-style client (duck-typed).

:class:`CapturingOpenAI` wraps any client object that exposes
``client.chat.completions.create(**kwargs)`` (the real ``openai`` SDK, a
custom gateway client, a mock...).  The wrapper delegates the call, records
the exchange into a :class:`~llmdiff.capture.record.TrafficRecorder`, and
returns the original response object untouched.

No import of the ``openai`` package happens at module import time, so this
module works even when the SDK is not installed.

Usage::

    from openai import OpenAI
    from llmdiff.capture import CapturingOpenAI, TrafficRecorder

    recorder = TrafficRecorder()
    client = CapturingOpenAI(OpenAI(), recorder)
    resp = client.chat.completions.create(model="gpt-4o", messages=[...])
    recorder.save("traffic.jsonl")
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from llmdiff.capture.record import TrafficRecorder

__all__ = ["CapturingOpenAI", "CapturingCompletions", "CapturingChat"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _messages_to_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages) if messages is not None else ""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, Mapping) and b.get("type") == "text"
                )
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def _response_to_text(response: Any) -> str:
    """Best-effort extraction of the assistant text from a response object."""
    # SDK objects: choices[0].message.content
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
    # dict-shaped responses (gateways / mocks)
    if isinstance(response, Mapping):
        ch = response.get("choices")
        if isinstance(ch, list) and ch and isinstance(ch[0], Mapping):
            msg = ch[0].get("message")
            if isinstance(msg, Mapping) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(ch[0].get("text"), str):
                return ch[0]["text"]
        for key in ("output", "completion", "text"):
            v = response.get(key)
            if isinstance(v, str):
                return v
    return ""


class CapturingCompletions:
    """Wraps ``client.chat.completions`` and records every ``create`` call."""

    def __init__(self, inner: Any, recorder: TrafficRecorder, *, tags: Iterable[str] = ()) -> None:
        self._inner = inner
        self._recorder = recorder
        self._tags = tuple(tags)

    def create(self, **kwargs: Any) -> Any:
        prompt = _messages_to_prompt(kwargs.get("messages"))
        model = kwargs.get("model")
        started = time.perf_counter()
        try:
            response = self._inner.create(**kwargs)
        except Exception:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self._recorder.record(
                prompt,
                "",
                tags=self._tags,
                model=model,
                endpoint="/v1/chat/completions",
                timestamp=_utcnow_iso(),
                latency_ms=latency_ms,
                error=True,
            )
            raise
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        output = _response_to_text(response)
        meta: dict[str, Any] = {
            "endpoint": "/v1/chat/completions",
            "timestamp": _utcnow_iso(),
            "latency_ms": latency_ms,
        }
        if model is not None:
            meta["model"] = model
        # usage stats when present
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_dict = (
                usage.model_dump()
                if hasattr(usage, "model_dump")
                else (dict(usage) if isinstance(usage, Mapping) else {})
            )
            if usage_dict:
                meta["usage"] = usage_dict
        self._recorder.record(prompt, output, tags=self._tags, **meta)
        return response

    def __getattr__(self, name: str) -> Any:
        # Delegate any other attribute (e.g. .with_raw_response) untouched.
        return getattr(self._inner, name)


class CapturingChat:
    """Wraps ``client.chat`` exposing a capturing ``completions``."""

    def __init__(self, inner: Any, recorder: TrafficRecorder, *, tags: Iterable[str] = ()) -> None:
        self._inner = inner
        self.completions = CapturingCompletions(inner.completions, recorder, tags=tags)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CapturingOpenAI:
    """Drop-in wrapper around an OpenAI-style client.

    Args:
        client: Any object with ``client.chat.completions.create(**kwargs)``.
        recorder: The :class:`TrafficRecorder` to record into.
        tags: Extra tags for every record.
    """

    def __init__(self, client: Any, recorder: TrafficRecorder, *, tags: Iterable[str] = ()) -> None:
        if not isinstance(recorder, TrafficRecorder):
            raise TypeError("recorder must be a TrafficRecorder")
        if not hasattr(client, "chat"):
            raise TypeError("client must expose a .chat attribute (duck-typed OpenAI client)")
        self._client = client
        self.recorder = recorder
        self.chat = CapturingChat(client.chat, recorder, tags=tags)

    def __getattr__(self, name: str) -> Any:
        # Everything else (client.models, client.embeddings, ...) passes through.
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"CapturingOpenAI(client={self._client!r}, recorder={self.recorder!r})"
