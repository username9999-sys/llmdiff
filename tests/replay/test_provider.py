"""Tests for the provider protocol, mocks, and the provider router."""

from __future__ import annotations

import pytest

from llmdiff.core.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from llmdiff.replay.mock_provider import EchoProvider, MockProvider, ScriptedProvider
from llmdiff.replay.provider import Provider, ProviderResponse, ensure_response
from llmdiff.replay.router import ProviderRouter


def test_provider_response_ok_flag():
    assert ProviderResponse(text="hi").ok
    assert not ProviderResponse(error="boom").ok


def test_ensure_response_coercion():
    assert ensure_response("x").text == "x"
    resp = ProviderResponse(text="y")
    assert ensure_response(resp) is resp
    with pytest.raises(ProviderError):
        ensure_response(123)  # type: ignore[arg-type]


def test_provider_protocol_runtime_check():
    assert isinstance(MockProvider(), Provider)
    assert isinstance(EchoProvider(), Provider)


def test_mock_provider_mapping_and_default():
    provider = MockProvider({"hello": "world", "*": "fallback"})
    assert provider.complete("hello").text == "world"
    assert provider.complete("anything else").text == "fallback"
    assert provider.complete("say hello please").text == "world"  # substring match


def test_mock_provider_callable_and_exception():
    provider = MockProvider(lambda prompt, **kw: f"got:{prompt}")
    assert provider.complete("p").text == "got:p"
    failing = MockProvider({"bad": ProviderRateLimitError("slow down", retry_after=2.0)})
    with pytest.raises(ProviderRateLimitError):
        failing.complete("bad")


def test_echo_provider():
    echo = EchoProvider(prefix="> ")
    assert echo.complete("abc").text == "> abc"
    assert echo.calls == ["abc"]


def test_scripted_provider_cycles_and_raises():
    provider = ScriptedProvider(["a", ProviderTimeoutError("t"), "b"])
    assert provider.complete("x").text == "a"
    with pytest.raises(ProviderTimeoutError):
        provider.complete("x")
    assert provider.complete("x").text == "b"
    # Cycles: after exhausting, repeats from the start.
    assert provider.complete("x").text == "a"


def test_scripted_provider_requires_script():
    with pytest.raises(ValueError):
        ScriptedProvider([])


def test_router_failover_on_transient_error():
    primary = ScriptedProvider([ProviderError("down")])
    backup = EchoProvider()
    router = ProviderRouter([primary, backup])
    response = router.complete("ping")
    assert response.text == "ping"
    assert router.served_by() == {backup.name: 1}
    assert router.history[0].attempts == (primary.name, backup.name)


def test_router_does_not_retry_auth_by_default():
    primary = ScriptedProvider([ProviderAuthError("bad key")])
    backup = EchoProvider()
    router = ProviderRouter([primary, backup])
    with pytest.raises(ProviderAuthError):
        router.complete("ping")
    assert backup.calls == []


def test_router_auth_failover_when_enabled():
    primary = ScriptedProvider([ProviderAuthError("bad key")])
    backup = EchoProvider()
    router = ProviderRouter([primary, backup], retry_auth=True)
    assert router.complete("ping").text == "ping"


def test_router_raises_when_all_fail():
    router = ProviderRouter(
        [ScriptedProvider([ProviderError("a")]), ScriptedProvider([ProviderTimeoutError("b")])]
    )
    with pytest.raises(ProviderError, match="all providers failed"):
        router.complete("x")


def test_router_requires_provider():
    with pytest.raises(ValueError):
        ProviderRouter([])
