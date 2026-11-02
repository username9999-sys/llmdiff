"""Response caching for replay providers.

The cache key is ``sha256(provider_name + model + prompt + canonical params)``
so identical calls never hit the network twice. Two stdlib backends are
provided: an in-process LRU (:class:`MemoryCache`) and a persistent SQLite
file (:class:`SQLiteCache`), both with optional TTL. :class:`CachedProvider`
wraps any provider and transparently serves hits from the cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from llmdiff.replay.provider import Provider, ProviderResponse

__all__ = [
    "CacheBackend",
    "MemoryCache",
    "SQLiteCache",
    "CachedProvider",
    "cache_key",
    "canonical_params",
]


def canonical_params(params: Mapping[str, Any]) -> str:
    """Serialise params deterministically for cache-key purposes."""
    return json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)


def cache_key(provider_name: str, model: str | None, prompt: str, params: Mapping[str, Any]) -> str:
    """Compute the sha256 cache key for one provider call."""
    raw = "\x1f".join(
        [provider_name, model or "", prompt, canonical_params(params)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheBackend(Protocol):
    """Storage protocol implemented by the cache backends."""

    def get(self, key: str) -> ProviderResponse | None: ...

    def put(self, key: str, response: ProviderResponse) -> None: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


class MemoryCache:
    """In-process LRU cache backed by an :class:`OrderedDict`.

    Args:
        max_entries: Maximum number of entries; oldest are evicted.
        ttl: Time-to-live in seconds; ``None`` means entries never expire.
    """

    def __init__(self, max_entries: int = 4096, ttl: float | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self.ttl = ttl
        self._store: OrderedDict[str, tuple[float, ProviderResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ProviderResponse | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stored_at, response = entry
            if self.ttl is not None and (time.monotonic() - stored_at) > self.ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return response

    def put(self, key: str, response: ProviderResponse) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), response)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class SQLiteCache:
    """Persistent cache stored in a SQLite file (stdlib ``sqlite3``).

    Args:
        path: Database file path; ``":memory:"`` for a transient database.
        ttl: Time-to-live in seconds; ``None`` means entries never expire.
    """

    def __init__(self, path: str | Path = ":memory:", ttl: float | None = None) -> None:
        self.path = str(path)
        self.ttl = ttl
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  payload TEXT NOT NULL,"
            "  stored_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> ProviderResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload, stored_at = row
        if self.ttl is not None and (time.time() - stored_at) > self.ttl:
            with self._lock:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
            return None
        data = json.loads(payload)
        return ProviderResponse(
            text=data["text"],
            raw=data.get("raw"),
            latency_ms=data.get("latency_ms", 0.0),
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
            model=data.get("model"),
            error=data.get("error"),
        )

    def put(self, key: str, response: ProviderResponse) -> None:
        payload = json.dumps(response.to_dict(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, payload, stored_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])


class CachedProvider:
    """Wrap a provider so identical calls are served from a cache.

    Cache hits are returned with ``latency_ms`` preserved from the original
    call; misses call through to the wrapped provider and store the result.
    Errors (responses with ``error`` set or raised exceptions) are not cached.
    """

    def __init__(self, provider: Provider, backend: CacheBackend) -> None:
        self.provider = provider
        self.backend = backend
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.provider.name

    def complete(self, prompt: str, **params: Any) -> ProviderResponse:
        model = str(params.get("model", getattr(self.provider, "model", None) or ""))
        key = cache_key(self.provider.name, model, prompt, params)
        cached = self.backend.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = self.provider.complete(prompt, **params)
        if response.ok:
            self.backend.put(key, response)
        return response
