"""Local Ollama provider hitting ``POST /api/generate`` (no API key needed)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from llmdiff.core.errors import ProviderError, ProviderTimeoutError
from llmdiff.replay.provider import ProviderResponse

__all__ = ["OllamaProvider", "AsyncOllamaProvider"]

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Provider for a local Ollama server.

    Args:
        model: Model tag, e.g. ``llama3.1``.
        base_url: Ollama root URL.
        timeout: Per-request timeout in seconds (generation can be slow).
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/api/generate"

    def _payload(self, prompt: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for key in ("temperature", "top_p", "seed", "num_predict"):
            if key in params and params[key] is not None:
                options[key] = params[key]
        if "max_tokens" in params and params["max_tokens"] is not None:
            options["num_predict"] = params["max_tokens"]
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if system := params.get("system"):
            payload["system"] = str(system)
        return payload

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.pop("model", self.model))
        payload = self._payload(prompt, model, params)
        started = time.perf_counter()
        try:
            response = httpx.post(self.url, json=payload, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP transport error calling Ollama: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise ProviderError(
                f"HTTP {response.status_code} from Ollama: {response.text[:200]}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Ollama returned non-JSON body: {exc}") from exc
        if body.get("error"):
            raise ProviderError(f"Ollama error: {body['error']}")
        return ProviderResponse(
            text=body.get("response", ""),
            raw=body,
            latency_ms=latency_ms,
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
            model=body.get("model") or model,
        )


class AsyncOllamaProvider(OllamaProvider):
    """Async variant used by the replay engine's concurrent path."""

    async def acomplete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.pop("model", self.model))
        payload = self._payload(prompt, model, params)
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, json=payload)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"Ollama request timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"HTTP transport error calling Ollama: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise ProviderError(
                f"HTTP {response.status_code} from Ollama: {response.text[:200]}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Ollama returned non-JSON body: {exc}") from exc
        if body.get("error"):
            raise ProviderError(f"Ollama error: {body['error']}")
        return ProviderResponse(
            text=body.get("response", ""),
            raw=body,
            latency_ms=latency_ms,
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
            model=body.get("model") or model,
        )

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        import asyncio

        return asyncio.run(self.acomplete(prompt, **params))
