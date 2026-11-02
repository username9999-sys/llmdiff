"""High-level replay entry point.

:func:`replay` ties the pieces together: filter/sample the corpus, wrap the
provider with cache and retry, run the batch engine with checkpointing, and
return a new corpus whose candidate outputs were produced by the candidate
system. ``dry_run`` performs everything except the actual provider call
(each case gets a placeholder candidate) so pipelines can be smoke-tested
offline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llmdiff.core.corpus import Corpus
from llmdiff.replay.batch import ReplayEngine, ReplayStats, load_checkpoint_ids
from llmdiff.replay.cache import CacheBackend, CachedProvider
from llmdiff.replay.provider import Provider
from llmdiff.replay.retry import RetryPolicy

__all__ = ["replay", "ReplayOptions"]


class ReplayOptions:
    """Plain option bag for :func:`replay` (kept simple for CLI forwarding)."""

    def __init__(
        self,
        *,
        concurrency: int = 8,
        cache: CacheBackend | None = None,
        retry: RetryPolicy | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 25,
        resume: bool = True,
        sample: int | None = None,
        sample_seed: int = 0,
        tags: Iterable[str] | None = None,
        match_all_tags: bool = False,
        max_cases: int | None = None,
        dry_run: bool = False,
        progress: bool = True,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.concurrency = concurrency
        self.cache = cache
        self.retry = retry
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.sample = sample
        self.sample_seed = sample_seed
        self.tags = tags
        self.match_all_tags = match_all_tags
        self.max_cases = max_cases
        self.dry_run = dry_run
        self.progress = progress
        self.params = params or {}


def _select_cases(corpus: Corpus, opts: ReplayOptions) -> Corpus:
    """Apply tag filters, sampling and max_cases in a deterministic order."""
    selected = corpus
    if opts.tags:
        selected = selected.filter_by_tags(opts.tags, match_all=opts.match_all_tags)
    if opts.sample is not None:
        selected = selected.sample(opts.sample, seed=opts.sample_seed)
    if opts.max_cases is not None and opts.max_cases >= 0:
        selected = Corpus(list(selected)[: opts.max_cases])
    return selected


class _DryRunProvider:
    """Provider used for ``dry_run``: returns a deterministic placeholder."""

    def __init__(self, inner: Provider) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, prompt: str, **params: Any) -> Any:
        from llmdiff.replay.provider import ProviderResponse

        return ProviderResponse(
            text=f"[dry-run] {len(prompt)} chars", model="dry-run", latency_ms=0.0
        )


def replay(
    corpus: Corpus,
    provider: Provider,
    out_path: str | Path | None = None,
    **opts: Any,
) -> tuple[Corpus, ReplayStats]:
    """Replay ``corpus`` against ``provider`` and return ``(corpus, stats)``.

    Args:
        corpus: Input corpus; its prompts are sent to the provider.
        provider: The candidate system's provider.
        out_path: When given, the resulting corpus is written there as JSONL.
        **opts: Any :class:`ReplayOptions` field (concurrency, cache, retry,
            checkpoint_path, sample, tags, max_cases, dry_run, ...).

    Returns:
        A tuple of the replayed corpus (candidate outputs filled in) and the
        run statistics.
    """
    options = ReplayOptions(**opts)
    selected = _select_cases(corpus, options)

    effective = provider
    if options.dry_run:
        effective = _DryRunProvider(provider)  # type: ignore[assignment]
    if options.cache is not None:
        effective = CachedProvider(effective, options.cache)

    skip: set[str] = set()
    if options.resume and options.checkpoint_path is not None:
        skip = load_checkpoint_ids(options.checkpoint_path)

    engine = ReplayEngine(
        effective,
        concurrency=options.concurrency,
        checkpoint_path=options.checkpoint_path,
        checkpoint_every=options.checkpoint_every,
        progress=options.progress,
        retry_policy=options.retry,
    )
    result = engine.run_sync(selected, skip_ids=skip)
    if out_path is not None:
        result.to_jsonl(out_path)
    return result, engine.stats
