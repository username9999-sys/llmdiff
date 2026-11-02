"""Batch replay engine: run a corpus through a provider with concurrency.

:class:`ReplayEngine` executes every case's prompt against a provider using
``asyncio`` with a semaphore bounding concurrency. Per-case failures are
captured into the case metadata (``error`` field) instead of aborting the
batch. Progress is reported through ``tqdm`` when available, otherwise plain
prints. Every ``checkpoint_every`` completed cases the partial results are
appended to a checkpoint JSONL file, so an interrupted run can be resumed by
skipping ids already present.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmdiff.core.case import Case
from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import ProviderError
from llmdiff.replay.provider import Provider, ProviderResponse

__all__ = ["ReplayEngine", "ReplayStats", "load_checkpoint_ids", "run_sync"]

try:  # optional progress bar
    from tqdm import tqdm as _tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _tqdm = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False


@dataclass(slots=True)
class ReplayStats:
    """Aggregate outcome of one replay run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    retried: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    max_concurrent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "retried": self.retried,
            "elapsed_s": round(self.elapsed_s, 3),
            "errors": list(self.errors),
            "max_concurrent": self.max_concurrent,
        }


def load_checkpoint_ids(path: str | Path) -> set[str]:
    """Read a checkpoint JSONL file and return the set of case ids it contains."""
    ids: set[str] = set()
    source = Path(path)
    if not source.exists():
        return ids
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        case_id = payload.get("id")
        if isinstance(case_id, str):
            ids.add(case_id)
    return ids


class ReplayEngine:
    """Run corpus prompts through a provider with bounded concurrency.

    Args:
        provider: Any :class:`Provider`. If it exposes ``acomplete`` that is
            used for the async path; otherwise ``complete`` is called in a
            thread executor.
        concurrency: Maximum number of in-flight calls.
        checkpoint_path: When set, partial results are appended every
            ``checkpoint_every`` cases.
        checkpoint_every: Checkpoint interval in number of completed cases.
        progress: Show a progress bar (tqdm when available).
        retry_policy: Optional :class:`~llmdiff.replay.retry.RetryPolicy`
            applied around each individual call.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        concurrency: int = 8,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 25,
        progress: bool = True,
        retry_policy: Any = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.provider = provider
        self.concurrency = concurrency
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every = max(1, checkpoint_every)
        self.progress = progress
        self.retry_policy = retry_policy
        self.stats = ReplayStats()
        self._current_concurrent = 0

    # ------------------------------------------------------------------ API

    async def run(self, corpus: Corpus, *, skip_ids: Iterable[str] = ()) -> Corpus:
        """Replay every case in ``corpus`` and return a new corpus with candidates.

        Cases whose id is in ``skip_ids`` keep their existing candidate output
        (used for checkpoint resume).
        """

        skip = set(skip_ids)
        cases = [case for case in corpus if case.id not in skip]
        self.stats = ReplayStats(total=len(cases))
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.concurrency)
        checkpoint_file = None
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file = self.checkpoint_path.open("a", encoding="utf-8")
        done = 0
        try:
            tasks = [self._run_one(case, semaphore) for case in cases]
            results: list[Case] = []
            progress_iter = self._progress(total=len(cases))
            for coro in asyncio.as_completed(tasks):
                case = await coro
                results.append(case)
                done += 1
                # Append every completed case so a resume never loses work;
                # checkpoint_every only controls how often we flush to disk.
                if checkpoint_file is not None:
                    checkpoint_file.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
                    if done % self.checkpoint_every == 0:
                        checkpoint_file.flush()
                if progress_iter is not None:
                    progress_iter.update(1)
            if progress_iter is not None:
                progress_iter.close()
        finally:
            if checkpoint_file is not None:
                checkpoint_file.close()
        self.stats.elapsed_s = time.perf_counter() - started
        # Preserve original corpus order.
        by_id = {case.id: case for case in results}
        ordered = [by_id.get(case.id, case) for case in corpus]
        return Corpus(ordered)

    def run_sync(self, corpus: Corpus, *, skip_ids: Iterable[str] = ()) -> Corpus:
        """Synchronous wrapper around :meth:`run`."""
        return asyncio.run(self.run(corpus, skip_ids=skip_ids))

    # ------------------------------------------------------------- internal

    def _progress(self, total: int) -> Any:
        if not self.progress or total == 0:
            return None
        if TQDM_AVAILABLE:
            return _tqdm(total=total, desc="replay", unit="case")
        print(f"replay: 0/{total} cases")
        return _PlainProgress(total)

    async def _run_one(self, case: Case, semaphore: asyncio.Semaphore) -> Case:
        async with semaphore:
            self._current_concurrent += 1
            self.stats.max_concurrent = max(self.stats.max_concurrent, self._current_concurrent)
            try:
                response = await self._call(case.prompt)
            except ProviderError as exc:
                self.stats.failed += 1
                err = f"{case.id}: {type(exc).__name__}: {exc}"
                self.stats.errors.append(err)
                return case.with_candidate("", error=err, provider=self.provider.name)
            except Exception as exc:  # noqa: BLE001 - never sink the batch
                self.stats.failed += 1
                err = f"{case.id}: unexpected {type(exc).__name__}: {exc}"
                self.stats.errors.append(err)
                return case.with_candidate("", error=err, provider=self.provider.name)
            finally:
                self._current_concurrent -= 1
        self.stats.succeeded += 1
        metadata: dict[str, Any] = {
            "provider": self.provider.name,
            "latency_ms": round(response.latency_ms, 3),
        }
        if response.model:
            metadata["model"] = response.model
        if response.tokens_in is not None:
            metadata["tokens_in"] = response.tokens_in
        if response.tokens_out is not None:
            metadata["tokens_out"] = response.tokens_out
        if response.error:
            self.stats.failed += 1
            metadata["error"] = response.error
            return case.with_candidate("", **metadata)
        return case.with_candidate(response.text, **metadata)

    async def _call(self, prompt: str) -> ProviderResponse:
        acomplete = getattr(self.provider, "acomplete", None)
        if callable(acomplete):
            call = lambda: acomplete(prompt)  # noqa: E731
        else:
            loop = asyncio.get_running_loop()
            call = lambda: loop.run_in_executor(None, self.provider.complete, prompt)  # noqa: E731
        if self.retry_policy is not None:
            from llmdiff.replay.retry import RetryPolicy

            policy = self.retry_policy
            if not isinstance(policy, RetryPolicy):
                policy = RetryPolicy()
            attempts = {"n": 0}

            async def _with_retry() -> ProviderResponse:
                attempts["n"] += 1
                if attempts["n"] > 1:
                    self.stats.retried += 1
                return await call()

            # Async retry: run the retry loop on the awaitable.
            last_error: BaseException | None = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await call()
                except policy.retryable as exc:
                    last_error = exc
                    if attempt >= policy.max_attempts:
                        break
                    self.stats.retried += 1
                    from llmdiff.core.errors import ProviderRateLimitError
                    from llmdiff.replay.retry import compute_backoff

                    retry_after = (
                        exc.retry_after if isinstance(exc, ProviderRateLimitError) else None
                    )
                    await asyncio.sleep(min(compute_backoff(policy, attempt, retry_after), 0.05))
            assert last_error is not None
            raise last_error
        return await call()


class _PlainProgress:
    """Minimal stand-in for tqdm when it is not installed."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.n % 50 == 0 or self.n == self.total:
            print(f"replay: {self.n}/{self.total} cases")

    def close(self) -> None:
        pass


def run_sync(
    corpus: Corpus,
    provider: Provider,
    out_path: str | Path | None = None,
    *,
    concurrency: int = 8,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
    progress: bool = True,
    resume: bool = True,
    retry_policy: Any = None,
) -> tuple[Corpus, ReplayStats]:
    """Convenience function: replay ``corpus`` and optionally write results.

    When ``resume`` is true and ``checkpoint_path`` exists, ids already in the
    checkpoint are skipped. Returns the replayed corpus and stats.
    """
    skip: set[str] = set()
    if resume and checkpoint_path is not None:
        skip = load_checkpoint_ids(checkpoint_path)
    engine = ReplayEngine(
        provider,
        concurrency=concurrency,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        progress=progress,
        retry_policy=retry_policy,
    )
    result = engine.run_sync(corpus, skip_ids=skip)
    if out_path is not None:
        result.to_jsonl(out_path)
    return result, engine.stats
