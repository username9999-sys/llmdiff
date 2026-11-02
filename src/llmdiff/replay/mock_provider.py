"""Deterministic mock providers for tests and dry runs.

These never touch the network: :class:`MockProvider` answers from a mapping
or callable, :class:`EchoProvider` returns the prompt, and
:class:`ScriptedProvider` cycles through a fixed list of outputs so tests can
simulate failures, rate limits and varying latency.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping
from typing import Any

from llmdiff.replay.provider import ProviderResponse

__all__ = ["MockProvider", "EchoProvider", "ScriptedProvider"]

ResponseSpec = str | ProviderResponse | BaseException


class MockProvider:
    """Answers prompts from a dict or a callable.

    Args:
        responses: Either a mapping of ``prompt -> response`` (a response may be
            a string, a :class:`ProviderResponse`, or an exception instance to
            raise) or a callable ``(prompt, **params) -> response``.
        default: Fallback response for prompts not in the mapping.
        latency_ms: Artificial latency added to every call (simulated).
    """

    def __init__(
        self,
        responses: Mapping[str, ResponseSpec] | Callable[..., ResponseSpec] | None = None,
        *,
        default: ResponseSpec = "mock-response",
        latency_ms: float = 0.0,
        model: str = "mock-model",
    ) -> None:
        self.responses = responses
        self.default = default
        self.latency_ms = latency_ms
        self.model = model
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return f"mock:{self.model}"

    def _resolve(self, prompt: str, params: dict[str, Any]) -> ResponseSpec:
        if callable(self.responses):
            return self.responses(prompt, **params)
        if isinstance(self.responses, Mapping):
            if prompt in self.responses:
                return self.responses[prompt]
            for key, value in self.responses.items():
                if key != "*" and key in prompt:
                    return value
            if "*" in self.responses:
                return self.responses["*"]
        return self.default

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        self.calls.append((prompt, dict(params)))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        spec = self._resolve(prompt, params)
        if isinstance(spec, BaseException):
            raise spec
        if isinstance(spec, ProviderResponse):
            return spec
        return ProviderResponse(text=str(spec), model=self.model, latency_ms=self.latency_ms)


class EchoProvider:
    """Returns the prompt itself as the completion (useful for plumbing tests)."""

    def __init__(self, prefix: str = "", model: str = "echo") -> None:
        self.prefix = prefix
        self.model = model
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return f"echo:{self.model}"

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        self.calls.append(prompt)
        return ProviderResponse(text=f"{self.prefix}{prompt}", model=self.model)


class ScriptedProvider:
    """Cycles through a list of scripted responses, one per call.

    Each entry may be a string, a :class:`ProviderResponse`, or an exception
    instance (raised when that entry is reached). When the list is exhausted
    the last entry repeats. Deterministic: no randomness, no clock dependence
    unless a scripted response says so.
    """

    def __init__(self, script: list[ResponseSpec], *, model: str = "scripted") -> None:
        if not script:
            raise ValueError("ScriptedProvider requires a non-empty script")
        self.script = list(script)
        self.model = model
        self._cycle = itertools.cycle(self.script)
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return f"scripted:{self.model}"

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        self.calls.append(prompt)
        spec = next(self._cycle)
        if isinstance(spec, BaseException):
            raise spec
        if isinstance(spec, ProviderResponse):
            return spec
        return ProviderResponse(text=str(spec), model=self.model)
