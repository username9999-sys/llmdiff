"""Capturing wrapper for an Anthropic-style client (duck-typed).

:class:`CapturingAnthropic` wraps any client exposing
``client.messages.create(**kwargs)`` (the real ``anthropic`` SDK or a mock).
Calls are delegated, recorded into a
:class:`~llmdiff.capture.record.TrafficRecorder`, and the original response
is returned untouched.  The ``anthropic`` package is never imported at
module load, so this works without the SDK installed.

Usage::

    import anthropic
    from llmdiff.capture import CapturingAnthropic, TrafficRecorder

    recorder = TrafficRecorder()
    client = CapturingAnthropic(anthropic.Anthropic(), recorder)
    msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=64,
                                 messages=[{"role": "user", "content": "hi"}])
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from llmdiff.capture.record import TrafficRecorder

__all__ = ["CapturingAnthropic", "CapturingMessages"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def _messages_to_prompt(messages: Any, system: Any = None) -> str:
    parts: list[str] = []
    if isinstance(system, str) and system:
        parts.append(f"system: {system}")
    elif isinstance(system, list):
        parts.append("system: " + _content_to_text(system))
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, Mapping):
                role = msg.get("role", "user")
                parts.append(f"{role}: {_content_to_text(msg.get('content', ''))}")
            else:
                parts.append(str(msg))
    return "\n".join(parts)


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is not None:
        return _content_to_text(content)
    if isinstance(response, Mapping):
        return _content_to_text(response.get("content"))
    return ""


class CapturingMessages:
    """Wraps ``client.messages`` and records every ``create`` call."""

    def __init__(self, inner: Any, recorder: TrafficRecorder, *, tags: Iterable[str] = ()) -> None:
        self._inner = inner
        self._recorder = recorder
        self._tags = tuple(tags)

    def create(self, **kwargs: Any) -> Any:
        prompt = _messages_to_prompt(kwargs.get("messages"), kwargs.get("system"))
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
                endpoint="/v1/messages",
                timestamp=_utcnow_iso(),
                latency_ms=latency_ms,
                error=True,
            )
            raise
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        output = _response_to_text(response)
        meta: dict[str, Any] = {
            "endpoint": "/v1/messages",
            "timestamp": _utcnow_iso(),
            "latency_ms": latency_ms,
        }
        if model is not None:
            meta["model"] = model
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
        return getattr(self._inner, name)


class CapturingAnthropic:
    """Drop-in wrapper around an Anthropic-style client.

    Args:
        client: Any object with ``client.messages.create(**kwargs)``.
        recorder: The :class:`TrafficRecorder` to record into.
        tags: Extra tags for every record.
    """

    def __init__(self, client: Any, recorder: TrafficRecorder, *, tags: Iterable[str] = ()) -> None:
        if not isinstance(recorder, TrafficRecorder):
            raise TypeError("recorder must be a TrafficRecorder")
        if not hasattr(client, "messages"):
            raise TypeError(
                "client must expose a .messages attribute (duck-typed Anthropic client)"
            )
        self._client = client
        self.recorder = recorder
        self.messages = CapturingMessages(client.messages, recorder, tags=tags)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"CapturingAnthropic(client={self._client!r}, recorder={self.recorder!r})"
