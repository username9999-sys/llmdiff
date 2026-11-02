"""Tests for ReplayEngine: concurrency cap, error isolation, checkpoint resume."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import ProviderError
from llmdiff.replay.batch import ReplayEngine, load_checkpoint_ids, run_sync
from llmdiff.replay.mock_provider import MockProvider, ScriptedProvider
from llmdiff.replay.provider import ProviderResponse


def make_corpus(n: int) -> Corpus:
    return Corpus.from_records(
        [
            {
                "id": f"c{i}",
                "prompt": f"prompt-{i}",
                "baseline_output": f"base-{i}",
                "candidate_output": "",
                "metadata": {"tags": ["t1"] if i % 2 == 0 else ["t2"]},
            }
            for i in range(n)
        ]
    )


class _CountingProvider:
    """Provider that tracks current/max concurrency via a shared counter."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.current = 0
        self.max_seen = 0
        self._lock = threading.Lock()
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting"

    def complete(self, prompt: str, **params):
        with self._lock:
            self.calls += 1
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(self.delay)
        with self._lock:
            self.current -= 1
        return ProviderResponse(text=f"out:{prompt}", model="counting")


def test_engine_fills_candidate_outputs():
    corpus = make_corpus(5)
    engine = ReplayEngine(MockProvider({"*": "mocked"}), concurrency=4, progress=False)
    result = engine.run_sync(corpus)
    assert len(result) == 5
    for case in result:
        assert case.candidate_output == "mocked"
        assert case.metadata["provider"] == "mock:mock-model"
    assert engine.stats.succeeded == 5
    assert engine.stats.failed == 0


def test_engine_concurrency_respects_semaphore():
    corpus = make_corpus(12)
    provider = _CountingProvider(delay=0.03)
    engine = ReplayEngine(provider, concurrency=3, progress=False)
    engine.run_sync(corpus)
    assert provider.max_seen <= 3
    assert provider.max_seen >= 2  # actually ran concurrently
    assert engine.stats.max_concurrent <= 3


def test_engine_error_in_one_case_does_not_sink_batch():
    # Decide by prompt content, not call order, so the test is deterministic
    # under any concurrency.
    def fn(prompt, **params):
        if prompt == "prompt-0":
            raise ProviderError("boom")
        return f"ok:{prompt}"

    provider = MockProvider(fn)
    corpus = make_corpus(3)
    engine = ReplayEngine(provider, concurrency=2, progress=False)
    result = engine.run_sync(corpus)
    by_id = {c.id: c for c in result}
    assert by_id["c0"].candidate_output == ""
    assert "error" in by_id["c0"].metadata
    assert by_id["c1"].candidate_output == "ok:prompt-1"
    assert by_id["c2"].candidate_output == "ok:prompt-2"
    assert engine.stats.failed == 1
    assert engine.stats.succeeded == 2


def test_checkpoint_written_and_resume_skips_done(tmp_path):
    corpus = make_corpus(6)
    checkpoint = tmp_path / "ckpt.jsonl"
    provider = MockProvider({"*": "done"})
    engine = ReplayEngine(
        provider, concurrency=2, checkpoint_path=checkpoint, checkpoint_every=2, progress=False
    )
    result = engine.run_sync(corpus)
    assert checkpoint.exists()
    ids = load_checkpoint_ids(checkpoint)
    assert ids == {f"c{i}" for i in range(6)}
    # Resume: all ids already checkpointed -> no new provider calls. Skipped
    # cases keep the candidate outputs of the corpus passed in, so resume the
    # first run's result corpus.
    provider2 = MockProvider({"*": "again"})
    engine2 = ReplayEngine(
        provider2, concurrency=2, checkpoint_path=checkpoint, checkpoint_every=2, progress=False
    )
    result2 = engine2.run_sync(result, skip_ids=ids)
    assert len(provider2.calls) == 0
    assert all(c.candidate_output == "done" for c in result2)


def test_load_checkpoint_ids_handles_missing_and_garbage(tmp_path):
    assert load_checkpoint_ids(tmp_path / "nope.jsonl") == set()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\nnot json\n{"id": "b"}\n\n')
    assert load_checkpoint_ids(bad) == {"a", "b"}


def test_run_sync_writes_out_path(tmp_path):
    corpus = make_corpus(3)
    out = tmp_path / "out.jsonl"
    result, stats = run_sync(
        corpus, MockProvider({"*": "x"}), out_path=out, concurrency=2, progress=False
    )
    assert out.exists()
    loaded = Corpus.from_jsonl(out)
    assert len(loaded) == 3
    assert all(c.candidate_output == "x" for c in loaded)
    assert stats.total == 3


def test_engine_rejects_bad_concurrency():
    with pytest.raises(ValueError):
        ReplayEngine(MockProvider(), concurrency=0)


def test_engine_uses_acomplete_when_available():
    class AsyncOnly:
        def __init__(self):
            self.calls = 0

        @property
        def name(self):
            return "async-only"

        async def acomplete(self, prompt, **params):
            self.calls += 1
            await asyncio.sleep(0)
            return ProviderResponse(text=f"a:{prompt}")

    provider = AsyncOnly()
    corpus = make_corpus(4)
    engine = ReplayEngine(provider, concurrency=2, progress=False)
    result = engine.run_sync(corpus)
    assert provider.calls == 4
    assert all(c.candidate_output.startswith("a:") for c in result)


def test_engine_retry_policy_retries_transient_failures():
    from llmdiff.replay.retry import RetryPolicy

    # Fail twice with a retryable error, then succeed.
    script = [ProviderError("flaky"), ProviderError("flaky"), "recovered"]
    provider = ScriptedProvider(script)
    corpus = make_corpus(1)
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False)
    engine = ReplayEngine(provider, concurrency=1, progress=False, retry_policy=policy)
    result = engine.run_sync(corpus)
    assert result[0].candidate_output == "recovered"
    assert engine.stats.retried >= 1


def test_checkpoint_file_is_valid_jsonl(tmp_path):
    corpus = make_corpus(4)
    checkpoint = tmp_path / "ck.jsonl"
    engine = ReplayEngine(
        MockProvider({"*": "v"}), concurrency=1, checkpoint_path=checkpoint,
        checkpoint_every=1, progress=False,
    )
    engine.run_sync(corpus)
    for line in checkpoint.read_text().splitlines():
        payload = json.loads(line)
        assert {"id", "prompt", "baseline_output", "candidate_output", "metadata"} <= set(payload)
