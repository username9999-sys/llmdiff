"""The :class:`Provider` protocol and the :class:`ProviderResponse` record.

A *provider* is anything that can turn a prompt string into a completion.
Every concrete provider (HTTP, SDK wrappers, mocks) implements the same small
protocol so the replay engine can treat them interchangeably.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llmdiff.core.errors import ProviderError

__all__ = ["Provider", "ProviderResponse"]


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """One completion returned by a provider.

    Attributes:
        text: The completion text. Empty string when ``error`` is set.
        raw: The raw decoded response payload, when available.
        latency_ms: Wall-clock time of the call in milliseconds.
        tokens_in: Prompt tokens consumed, when the provider reported them.
        tokens_out: Completion tokens produced, when reported.
        model: The model identifier that actually served the call.
        error: Error message when the call failed, else ``None``.
    """

    text: str = ""
    raw: Mapping[str, Any] | None = None
    latency_ms: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the call succeeded (no recorded error)."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this response."""
        return {
            "text": self.text,
            "raw": dict(self.raw) if self.raw is not None else None,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model": self.model,
            "error": self.error,
        }


@runtime_checkable
class Provider(Protocol):
    """Protocol every replay provider implements.

    Implementations must raise :class:`ProviderError` subclasses (or return a
    :class:`ProviderResponse` with ``error`` set) on failure; they must never
    raise unrelated exceptions for expected failure modes such as auth errors,
    rate limits or timeouts.
    """

    @property
    def name(self) -> str:
        """Short, stable identifier used in cache keys and logs."""
        ...

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        """Complete ``prompt`` and return a :class:`ProviderResponse`.

        Args:
            prompt: The full prompt text to send.
            **params: Provider-specific generation parameters (temperature,
                max_tokens, ...). Unknown params should be ignored or raise
                :class:`ProviderError`.

        Raises:
            ProviderAuthError: On credential rejection.
            ProviderRateLimitError: On rate limiting (with ``retry_after``).
            ProviderTimeoutError: When the call exceeds its deadline.
            ProviderError: On any other provider failure.
        """
        ...


def ensure_response(value: ProviderResponse | str) -> ProviderResponse:
    """Coerce a provider return value into a :class:`ProviderResponse`.

    Providers written for convenience may return a bare string; this helper
    normalises that into a full response record.

    Raises:
        ProviderError: If ``value`` is neither a response nor a string.
    """
    if isinstance(value, ProviderResponse):
        return value
    if isinstance(value, str):
        return ProviderResponse(text=value)
    raise ProviderError(
        f"provider returned an unsupported type: {type(value).__name__}"
    )


def response_from_error(exc: BaseException, *, model: str | None = None) -> ProviderResponse:
    """Build an error :class:`ProviderResponse` from an exception."""
    return ProviderResponse(text="", error=f"{type(exc).__name__}: {exc}", model=model)


@dataclass(slots=True)
class CallRecord:
    """Bookkeeping for one provider call, used by routers and caches."""

    provider: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    response: ProviderResponse | None = None
