"""Tests for the retry policy: attempt counting, backoff, retry_after handling."""

from __future__ import annotations

import random

import pytest

from llmdiff.core.errors import ProviderError, ProviderRateLimitError, ProviderTimeoutError
from llmdiff.replay.retry import RetryPolicy, compute_backoff, with_retry


def test_with_retry_succeeds_first_try():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert with_retry(fn, RetryPolicy(max_attempts=3), sleep=lambda s: None) == "ok"
    assert calls["n"] == 1


def test_with_retry_counts_attempts_until_success():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderTimeoutError("slow")
        return "ok"

    result = with_retry(fn, RetryPolicy(max_attempts=5, jitter=False), sleep=lambda s: None)
    assert result == "ok"
    assert attempts["n"] == 3


def test_with_retry_exhausts_and_raises_last_error():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ProviderRateLimitError("limited", retry_after=1.0)

    with pytest.raises(ProviderRateLimitError):
        with_retry(fn, RetryPolicy(max_attempts=3, jitter=False), sleep=lambda s: None)
    assert attempts["n"] == 3


def test_with_retry_respects_retry_after_in_sleep():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ProviderRateLimitError("limited", retry_after=5.0)

    policy = RetryPolicy(max_attempts=2, base_delay=0.1, max_delay=30.0, jitter=False)
    with pytest.raises(ProviderRateLimitError):
        with_retry(fn, policy, sleep=sleeps.append)
    assert attempts["n"] == 2
    # retry_after (5.0) must dominate the computed backoff (0.1).
    assert sleeps == [5.0]


def test_with_retry_does_not_retry_non_provider_errors():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        with_retry(fn, RetryPolicy(max_attempts=5), sleep=lambda s: None)
    assert attempts["n"] == 1


def test_compute_backoff_grows_exponentially_and_caps():
    policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0, max_delay=10.0, jitter=False)
    assert compute_backoff(policy, 1) == 1.0
    assert compute_backoff(policy, 2) == 2.0
    assert compute_backoff(policy, 3) == 4.0
    assert compute_backoff(policy, 10) == 10.0  # capped at max_delay


def test_compute_backoff_retry_after_dominates():
    policy = RetryPolicy(base_delay=0.1, backoff_factor=2.0, max_delay=30.0, jitter=False)
    assert compute_backoff(policy, 1, retry_after=7.5) == 7.5


def test_with_retry_deterministic_with_rng():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ProviderError("x")
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=True)
    rng = random.Random(42)
    assert with_retry(fn, policy, sleep=sleeps.append, rng=rng) == "ok"
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] < 2.0  # base delay + jitter in [0, delay)


def test_retry_policy_validation():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_factor=0.5)
