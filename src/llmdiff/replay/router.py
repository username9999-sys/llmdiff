"""Provider router: failover across an ordered list of providers.

:class:`ProviderRouter` tries each provider in order. Transient failures
(rate limit, timeout, generic provider error) trigger failover to the next
provider; auth errors do not (retrying a bad key against another endpoint is
pointless and can leak the key to a different host). Every call is recorded
in :attr:`ProviderRouter.history` so callers can see which provider actually
served each prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmdiff.core.errors import ProviderAuthError, ProviderError
from llmdiff.replay.provider import Provider, ProviderResponse

__all__ = ["ProviderRouter", "RouteRecord"]


@dataclass(slots=True)
class RouteRecord:
    """One routed call: the prompt, the provider that served it, and the response."""

    prompt: str
    provider: str
    response: ProviderResponse
    attempts: tuple[str, ...] = field(default_factory=tuple)


class ProviderRouter:
    """Try providers in order until one succeeds.

    Args:
        providers: Ordered providers; the first is primary.
        retry_auth: When true, auth errors also trigger failover (default
            false, since a bad key is usually bad everywhere).
    """

    def __init__(self, providers: list[Provider], *, retry_auth: bool = False) -> None:
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider")
        self.providers = list(providers)
        self.retry_auth = retry_auth
        self.history: list[RouteRecord] = []

    @property
    def name(self) -> str:
        return "router(" + ",".join(p.name for p in self.providers) + ")"

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        attempts: list[str] = []
        last_error: ProviderError | None = None
        for provider in self.providers:
            attempts.append(provider.name)
            try:
                response = provider.complete(prompt, **params)
            except ProviderAuthError as exc:
                last_error = exc
                if not self.retry_auth:
                    raise
                continue
            except ProviderError as exc:
                last_error = exc
                continue
            record = RouteRecord(
                prompt=prompt, provider=provider.name, response=response, attempts=tuple(attempts)
            )
            self.history.append(record)
            return response
        raise ProviderError(
            f"all providers failed for prompt ({len(prompt)} chars); "
            f"tried: {', '.join(attempts)}; last error: {last_error}"
        )

    def served_by(self) -> dict[str, int]:
        """Count of successful calls per provider name."""
        counts: dict[str, int] = {}
        for record in self.history:
            counts[record.provider] = counts.get(record.provider, 0) + 1
        return counts
