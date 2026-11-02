"""Generic OpenAI-compatible HTTP provider built on httpx.

Talks to any server exposing ``POST {base_url}/v1/chat/completions`` (vLLM,
LM Studio, OpenRouter, Together, a local gateway, ...). Both a synchronous
:class:`HttpProvider` and an asynchronous :class:`AsyncHttpProvider` are
provided; the async variant shares the same request-building and error-mapping
logic.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from llmdiff.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from llmdiff.replay.provider import ProviderResponse

__all__ = ["HttpProvider", "AsyncHttpProvider", "build_chat_payload", "parse_retry_after"]

DEFAULT_TIMEOUT = 60.0


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Extract a ``Retry-After`` value (seconds) from response headers.

    Header lookup is case-insensitive because httpx normalises casing but
    tests may pass plain dicts.
    """
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def build_chat_payload(model: str, prompt: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Build the OpenAI chat-completions request body for a single prompt.

    Generation params are passed through, with a few aliases normalised:
    ``max_tokens`` is accepted alongside ``max_completion_tokens``.
    """
    messages = [{"role": "user", "content": prompt}]
    if system := params.get("system"):
        messages.insert(0, {"role": "system", "content": str(system)})
    payload: dict[str, Any] = {"model": model, "messages": messages}
    for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop"):
        if key in params and params[key] is not None:
            payload[key] = params[key]
    for key, value in params.items():
        if key not in payload and key != "system":
            payload[key] = value
    return payload


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _map_http_error(status: int, body: str, headers: Mapping[str, str]) -> ProviderError:
    """Map an HTTP error status onto the llmdiff error hierarchy."""
    if status in (401, 403):
        return ProviderAuthError(f"HTTP {status} from provider: {body[:200]}")
    if status == 429:
        return ProviderRateLimitError(
            f"HTTP 429 from provider: {body[:200]}",
            retry_after=parse_retry_after(headers),
        )
    return ProviderError(f"HTTP {status} from provider: {body[:200]}")


def _parse_chat_response(payload: Mapping[str, Any], model: str) -> ProviderResponse:
    """Extract text and token usage from a chat-completions response body."""
    try:
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(f"malformed chat-completions response: {exc}") from exc
    usage = payload.get("usage") or {}
    return ProviderResponse(
        text=text,
        raw=dict(payload),
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
        model=payload.get("model") or model,
    )


class HttpProvider:
    """Synchronous OpenAI-compatible provider using ``httpx.Client``.

    Args:
        base_url: Root URL of the server, e.g. ``https://api.openai.com``.
        api_key: Bearer token; may be ``None`` for keyless local servers.
        model: Default model name sent with every request.
        timeout: Per-request timeout in seconds.
        client: Optional pre-built ``httpx.Client`` (mainly for tests).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = DEFAULT_TIMEOUT,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return f"http:{self.base_url}:{self.model}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.pop("model", self.model))
        payload = build_chat_payload(model, prompt, params)
        started = time.perf_counter()
        try:
            response = self._get_client().post(
                self.url, json=payload, headers=_headers(self.api_key)
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"request to {self.url} timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP transport error calling {self.url}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise _map_http_error(response.status_code, response.text, response.headers)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned non-JSON body: {exc}") from exc
        result = _parse_chat_response(body, model)
        return ProviderResponse(
            text=result.text,
            raw=result.raw,
            latency_ms=latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            model=result.model,
        )


class AsyncHttpProvider:
    """Asynchronous variant of :class:`HttpProvider` using ``httpx.AsyncClient``.

    Used by :class:`~llmdiff.replay.batch.ReplayEngine` for concurrent replays.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = DEFAULT_TIMEOUT,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return f"http:{self.base_url}:{self.model}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def acomplete(self, prompt: str, **params: Any) -> ProviderResponse:
        """Async counterpart of :meth:`HttpProvider.complete`."""
        model = str(params.pop("model", self.model))
        payload = build_chat_payload(model, prompt, params)
        started = time.perf_counter()
        try:
            response = await self._get_client().post(
                self.url, json=payload, headers=_headers(self.api_key)
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"request to {self.url} timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP transport error calling {self.url}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise _map_http_error(response.status_code, response.text, response.headers)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned non-JSON body: {exc}") from exc
        result = _parse_chat_response(body, model)
        return ProviderResponse(
            text=result.text,
            raw=result.raw,
            latency_ms=latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            model=result.model,
        )

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        """Synchronous fallback: runs the async path on a fresh event loop."""
        import asyncio

        return asyncio.run(self.acomplete(prompt, **params))
