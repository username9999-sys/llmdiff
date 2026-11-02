"""Tests for the response cache backends and CachedProvider."""

from __future__ import annotations

import time

from llmdiff.replay.cache import (
    CachedProvider,
    MemoryCache,
    SQLiteCache,
    cache_key,
    canonical_params,
)
from llmdiff.replay.mock_provider import MockProvider
from llmdiff.replay.provider import ProviderResponse


def test_cache_key_is_deterministic_and_param_sensitive():
    k1 = cache_key("p", "m", "prompt", {"temperature": 0.5})
    k2 = cache_key("p", "m", "prompt", {"temperature": 0.5})
    assert k1 == k2
    assert k1 != cache_key("p", "m", "prompt", {"temperature": 0.7})
    assert k1 != cache_key("p", "m", "other", {"temperature": 0.5})
    assert k1 != cache_key("other", "m", "prompt", {"temperature": 0.5})
    # Param order must not matter.
    assert cache_key("p", "m", "x", {"a": 1, "b": 2}) == cache_key("p", "m", "x", {"b": 2, "a": 1})
    assert len(k1) == 64  # sha256 hex


def test_canonical_params_sorted():
    assert canonical_params({"b": 2, "a": 1}) == canonical_params({"a": 1, "b": 2})


def test_memory_cache_lru_eviction():
    cache = MemoryCache(max_entries=2)
    cache.put("a", ProviderResponse(text="a"))
    cache.put("b", ProviderResponse(text="b"))
    cache.get("a")  # touch a so b becomes oldest
    cache.put("c", ProviderResponse(text="c"))
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_memory_cache_ttl_expiry():
    cache = MemoryCache(ttl=0.05)
    cache.put("k", ProviderResponse(text="v"))
    assert cache.get("k") is not None
    time.sleep(0.08)
    assert cache.get("k") is None


def test_sqlite_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.db"
    cache = SQLiteCache(path)
    resp = ProviderResponse(text="hello", tokens_in=3, tokens_out=4, model="m", latency_ms=12.5)
    cache.put("key1", resp)
    loaded = cache.get("key1")
    assert loaded is not None
    assert loaded.text == "hello"
    assert loaded.tokens_in == 3
    assert loaded.tokens_out == 4
    assert loaded.model == "m"
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    cache.close()
    # Reopen: data persisted before clear is gone, but file is usable again.
    cache2 = SQLiteCache(path)
    assert cache2.get("key1") is None
    cache2.close()


def test_sqlite_cache_ttl(tmp_path):
    cache = SQLiteCache(tmp_path / "c.db", ttl=0.05)
    cache.put("k", ProviderResponse(text="v"))
    assert cache.get("k") is not None
    time.sleep(0.08)
    assert cache.get("k") is None
    cache.close()


def test_cached_provider_hit_avoids_second_call():
    inner = MockProvider({"p": "answer"})
    provider = CachedProvider(inner, MemoryCache())
    r1 = provider.complete("p", temperature=0.1)
    r2 = provider.complete("p", temperature=0.1)
    assert r1.text == r2.text == "answer"
    assert len(inner.calls) == 1  # second call served from cache
    assert provider.hits == 1
    assert provider.misses == 1


def test_cached_provider_different_params_miss():
    inner = MockProvider({"p": "answer"})
    provider = CachedProvider(inner, MemoryCache())
    provider.complete("p", temperature=0.1)
    provider.complete("p", temperature=0.9)
    assert len(inner.calls) == 2
    assert provider.misses == 2


def test_cached_provider_does_not_cache_errors():
    import contextlib

    from llmdiff.core.errors import ProviderError

    inner = MockProvider({"p": ProviderError("fail")})
    provider = CachedProvider(inner, MemoryCache())
    for _ in range(2):
        with contextlib.suppress(ProviderError):
            provider.complete("p")
    assert len(inner.calls) == 2  # errors are never cached
