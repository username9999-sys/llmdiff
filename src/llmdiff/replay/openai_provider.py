"""OpenAI provider: prefers the official ``openai`` SDK, falls back to HTTP.

When the ``openai`` package is importable we use its client (which handles
retries and auth itself); otherwise we degrade to :class:`HttpProvider`
against the same endpoint so the package keeps working with only httpx.
"""

from __future__ import annotations

import time
from typing import Any

from llmdiff.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from llmdiff.replay.httpx_provider import HttpProvider
from llmdiff.replay.provider import ProviderResponse

__all__ = ["OpenAIProvider", "OPENAI_AVAILABLE"]

try:  # optional dependency
    import openai as _openai  # type: ignore[import-not-found]

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _openai = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False

DEFAULT_BASE_URL = "https://api.openai.com"


def _map_sdk_error(exc: BaseException) -> ProviderError:
    """Translate an ``openai`` SDK exception into the llmdiff hierarchy."""
    if _openai is None:  # pragma: no cover - defensive
        return ProviderError(str(exc))
    if isinstance(exc, _openai.AuthenticationError):
        return ProviderAuthError(str(exc))
    if isinstance(exc, _openai.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            value = response.headers.get("retry-after")
            if value is not None:
                try:
                    retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = None
        return ProviderRateLimitError(str(exc), retry_after=retry_after)
    if isinstance(exc, _openai.APITimeoutError):
        return ProviderTimeoutError(str(exc))
    if isinstance(exc, _openai.APIError):
        return ProviderError(str(exc))
    return ProviderError(f"{type(exc).__name__}: {exc}")


class OpenAIProvider:
    """Provider for OpenAI or any OpenAI-compatible endpoint via the SDK.

    Args:
        api_key: API key; falls back to the ``OPENAI_API_KEY`` env var when the
            SDK is used and no key is given.
        model: Default model.
        base_url: Override for compatible gateways.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self._client: Any = None
        self._fallback: HttpProvider | None = None
        if OPENAI_AVAILABLE:
            kwargs: dict[str, Any] = {"base_url": self.base_url + "/v1", "timeout": timeout}
            if api_key is not None:
                kwargs["api_key"] = api_key
            self._client = _openai.OpenAI(**kwargs)
        else:
            self._fallback = HttpProvider(
                base_url=self.base_url, api_key=api_key, model=model, timeout=timeout
            )

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    @property
    def uses_sdk(self) -> bool:
        return self._client is not None

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.pop("model", self.model))
        if self._fallback is not None:
            return self._fallback.complete(prompt, model=model, **params)
        system = params.pop("system", None)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model, messages=messages, **params
            )
        except Exception as exc:  # SDK raises its own hierarchy; map it
            raise _map_sdk_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            text=text,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            latency_ms=latency_ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            model=getattr(response, "model", model),
        )
