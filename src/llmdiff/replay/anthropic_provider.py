"""Anthropic provider: official ``anthropic`` SDK when present, else raw HTTP.

The HTTP fallback calls ``POST {base_url}/v1/messages`` with the ``x-api-key``
header and the ``anthropic-version`` header, mapping status codes onto the
llmdiff error hierarchy the same way the OpenAI-compatible provider does.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from llmdiff.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from llmdiff.replay.httpx_provider import parse_retry_after
from llmdiff.replay.provider import ProviderResponse

__all__ = ["AnthropicProvider", "ANTHROPIC_AVAILABLE"]

try:  # optional dependency
    import anthropic as _anthropic  # type: ignore[import-not-found]

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


def _map_sdk_error(exc: BaseException) -> ProviderError:
    if _anthropic is None:  # pragma: no cover - defensive
        return ProviderError(str(exc))
    if isinstance(exc, _anthropic.AuthenticationError):
        return ProviderAuthError(str(exc))
    if isinstance(exc, _anthropic.RateLimitError):
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
    if isinstance(exc, _anthropic.APITimeoutError):
        return ProviderTimeoutError(str(exc))
    if isinstance(exc, _anthropic.APIError):
        return ProviderError(str(exc))
    return ProviderError(f"{type(exc).__name__}: {exc}")


def _map_http_error(status: int, body: str, headers: dict[str, str]) -> ProviderError:
    if status in (401, 403):
        return ProviderAuthError(f"HTTP {status} from Anthropic: {body[:200]}")
    if status == 429:
        return ProviderRateLimitError(
            f"HTTP 429 from Anthropic: {body[:200]}", retry_after=parse_retry_after(headers)
        )
    return ProviderError(f"HTTP {status} from Anthropic: {body[:200]}")


class AnthropicProvider:
    """Provider for the Anthropic Messages API.

    Args:
        api_key: API key; falls back to ``ANTHROPIC_API_KEY`` when the SDK is used.
        model: Default model, e.g. ``claude-sonnet-4-5``.
        base_url: Override for proxies.
        timeout: Request timeout in seconds.
        max_tokens: Default ``max_tokens`` for the Messages API (required by it).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None
        if ANTHROPIC_AVAILABLE:
            kwargs: dict[str, Any] = {"base_url": self.base_url, "timeout": timeout}
            if api_key is not None:
                kwargs["api_key"] = api_key
            self._client = _anthropic.Anthropic(**kwargs)

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    @property
    def uses_sdk(self) -> bool:
        return self._client is not None

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.pop("model", self.model))
        system = params.pop("system", None)
        max_tokens = int(params.pop("max_tokens", self.max_tokens))
        if self._client is not None:
            return self._complete_sdk(prompt, model, system, max_tokens, params)
        return self._complete_http(prompt, model, system, max_tokens, params)

    def _complete_sdk(
        self, prompt: str, model: str, system: Any, max_tokens: int, params: dict[str, Any]
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = str(system)
        kwargs.update(params)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _map_sdk_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            text=text,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            latency_ms=latency_ms,
            tokens_in=getattr(usage, "input_tokens", None) if usage else None,
            tokens_out=getattr(usage, "output_tokens", None) if usage else None,
            model=getattr(response, "model", model),
        )

    def _complete_http(
        self, prompt: str, model: str, system: Any, max_tokens: int, params: dict[str, Any]
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = str(system)
        for key in ("temperature", "top_p", "stop_sequences"):
            if key in params and params[key] is not None:
                payload[key] = params[key]
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"request to Anthropic timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP transport error calling Anthropic: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise _map_http_error(response.status_code, response.text, dict(response.headers))
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Anthropic returned non-JSON body: {exc}") from exc
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return ProviderResponse(
            text=text,
            raw=body,
            latency_ms=latency_ms,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            model=body.get("model") or model,
        )
