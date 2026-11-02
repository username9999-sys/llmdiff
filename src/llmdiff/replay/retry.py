"""Retry policy: exponential backoff with jitter for provider calls.

:func:`with_retry` wraps any callable and retries it according to a
:class:`RetryPolicy`. Rate-limit errors honour the provider's ``retry_after``
when present; timeouts and generic provider errors use exponential backoff.
Non-provider exceptions are never retried.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from llmdiff.core.errors import ProviderError, ProviderRateLimitError, ProviderTimeoutError

__all__ = ["RetryPolicy", "with_retry", "compute_backoff"]

T = TypeVar("T")

#: Error types that are safe to retry.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configuration for :func:`with_retry`.

    Attributes:
        max_attempts: Total attempts including the first call (>= 1).
        base_delay: Delay before the first retry, in seconds.
        max_delay: Upper bound for any computed delay, in seconds.
        backoff_factor: Multiplier applied to the delay after each attempt.
        jitter: When true, add uniform random jitter in ``[0, delay)``.
        retryable: Exception types that trigger a retry.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable: tuple[type[BaseException], ...] = RETRYABLE_ERRORS

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.backoff_factor < 1:
            raise ValueError("delays must be >= 0 and backoff_factor >= 1")


def compute_backoff(policy: RetryPolicy, attempt: int, retry_after: float | None = None) -> float:
    """Compute the delay before retry ``attempt`` (1-based).

    A provider-supplied ``retry_after`` wins over the computed backoff, but is
    still capped by ``max_delay`` unless it is smaller than the base delay.
    """
    delay = min(policy.base_delay * (policy.backoff_factor ** (attempt - 1)), policy.max_delay)
    if retry_after is not None:
        delay = max(delay, min(retry_after, policy.max_delay))
    if policy.jitter:
        delay += random.uniform(0, delay)
    return delay


def with_retry(
    fn: Callable[..., T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn`` retrying retryable failures according to ``policy``.

    Args:
        fn: The callable to invoke (usually ``provider.complete`` partially
            applied with prompt and params).
        policy: Retry configuration; defaults to a 3-attempt policy.
        sleep: Sleep function, injectable for tests.
        rng: Random source for jitter, injectable for deterministic tests.

    Raises:
        ProviderError: The last failure when attempts are exhausted.
        BaseException: Any non-retryable exception from ``fn``, unchanged.
    """
    policy = policy or RetryPolicy()
    if rng is not None:
        # Route jitter through the injected RNG by monkeypatching the module
        # function used by compute_backoff via a local random draw.
        original_uniform = random.uniform

        def _uniform(a: float, b: float) -> float:
            return rng.uniform(a, b)

        random.uniform = _uniform  # type: ignore[assignment]
        try:
            return _run(fn, policy, sleep)
        finally:
            random.uniform = original_uniform  # type: ignore[assignment]
    return _run(fn, policy, sleep)


def _run(fn: Callable[..., T], policy: RetryPolicy, sleep: Callable[[float], None]) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retryable as exc:
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            retry_after = (
                exc.retry_after if isinstance(exc, ProviderRateLimitError) else None
            )
            delay = compute_backoff(policy, attempt, retry_after)
            sleep(delay)
    assert last_error is not None
    raise last_error


def with_retry_async():  # pragma: no cover - placeholder kept for API symmetry
    raise NotImplementedError("use asyncio-aware retry inside batch.py")
