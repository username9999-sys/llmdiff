"""llmdiff.replay - replay a prompt corpus against a candidate system.

This subpackage turns a captured corpus of real prompts into candidate
outputs by calling an LLM provider, with retries, caching, concurrency,
checkpointing, templating and parameter sweeps. The resulting corpus can be
merged with a baseline corpus via :mod:`llmdiff.replay.compare_runs` and fed
to ``llmdiff.evaluate``.
"""

from __future__ import annotations

from llmdiff.replay.batch import ReplayEngine, ReplayStats, load_checkpoint_ids, run_sync
from llmdiff.replay.cache import (
    CacheBackend,
    CachedProvider,
    MemoryCache,
    SQLiteCache,
    cache_key,
)
from llmdiff.replay.compare_runs import compare_runs, load_pair
from llmdiff.replay.httpx_provider import AsyncHttpProvider, HttpProvider
from llmdiff.replay.mock_provider import EchoProvider, MockProvider, ScriptedProvider
from llmdiff.replay.params import config_name, expand_sweep, sweep
from llmdiff.replay.prompt_templates import TemplateProvider, render_template
from llmdiff.replay.provider import Provider, ProviderResponse
from llmdiff.replay.retry import RetryPolicy, compute_backoff, with_retry
from llmdiff.replay.router import ProviderRouter, RouteRecord
from llmdiff.replay.runner import ReplayOptions, replay

__all__ = [
    "AsyncHttpProvider",
    "CacheBackend",
    "CachedProvider",
    "EchoProvider",
    "HttpProvider",
    "MemoryCache",
    "MockProvider",
    "Provider",
    "ProviderResponse",
    "ProviderRouter",
    "ReplayEngine",
    "ReplayOptions",
    "ReplayStats",
    "RetryPolicy",
    "RouteRecord",
    "SQLiteCache",
    "ScriptedProvider",
    "TemplateProvider",
    "cache_key",
    "compare_runs",
    "compute_backoff",
    "config_name",
    "expand_sweep",
    "load_checkpoint_ids",
    "load_pair",
    "render_template",
    "replay",
    "run_sync",
    "sweep",
    "with_retry",
]
