"""Tests for the high-level replay() runner: filters, sample, dry_run, out_path."""

from __future__ import annotations

from llmdiff.core.corpus import Corpus
from llmdiff.replay.cache import MemoryCache
from llmdiff.replay.mock_provider import MockProvider
from llmdiff.replay.runner import replay


def make_corpus(n: int = 10) -> Corpus:
    return Corpus.from_records(
        [
            {
                "id": f"c{i}",
                "prompt": f"p{i}",
                "baseline_output": f"b{i}",
                "candidate_output": "",
                "metadata": {"tags": ["a"] if i % 2 == 0 else ["b"], "model": "m1"},
            }
            for i in range(n)
        ]
    )


def test_replay_basic_fills_candidates():
    corpus = make_corpus(6)
    result, stats = replay(corpus, MockProvider({"*": "out"}), concurrency=4, progress=False)
    assert stats.total == 6
    assert all(c.candidate_output == "out" for c in result)


def test_replay_tag_filter():
    corpus = make_corpus(10)
    result, stats = replay(
        corpus, MockProvider({"*": "x"}), tags=["a"], concurrency=2, progress=False
    )
    assert stats.total == 5
    assert all("a" in c.tags for c in result)


def test_replay_tag_filter_match_all():
    corpus = Corpus.from_records(
        [
            {"id": "1", "prompt": "p", "baseline_output": "b", "candidate_output": "",
             "metadata": {"tags": ["a", "b"]}},
            {"id": "2", "prompt": "p", "baseline_output": "b", "candidate_output": "",
             "metadata": {"tags": ["a"]}},
        ]
    )
    result, _ = replay(
        corpus, MockProvider({"*": "x"}), tags=["a", "b"], match_all_tags=True, progress=False
    )
    assert [c.id for c in result] == ["1"]


def test_replay_sample_is_deterministic():
    corpus = make_corpus(20)
    r1, s1 = replay(corpus, MockProvider({"*": "x"}), sample=5, sample_seed=7, progress=False)
    r2, s2 = replay(corpus, MockProvider({"*": "x"}), sample=5, sample_seed=7, progress=False)
    assert s1.total == 5
    assert [c.id for c in r1] == [c.id for c in r2]


def test_replay_max_cases():
    corpus = make_corpus(10)
    result, stats = replay(corpus, MockProvider({"*": "x"}), max_cases=3, progress=False)
    assert stats.total == 3
    assert len(result) == 3


def test_replay_dry_run_makes_no_provider_calls():
    inner = MockProvider({"*": "real"})
    corpus = make_corpus(4)
    result, stats = replay(corpus, inner, dry_run=True, progress=False)
    assert len(inner.calls) == 0
    assert all(c.candidate_output.startswith("[dry-run]") for c in result)
    assert stats.succeeded == 4


def test_replay_with_cache_avoids_duplicate_calls():
    inner = MockProvider({"*": "cached"})
    cache = MemoryCache()
    corpus = make_corpus(5)
    replay(corpus, inner, cache=cache, concurrency=1, progress=False)
    calls_after_first = len(inner.calls)
    assert calls_after_first == 5
    # Re-run with same cache: all hits, no new calls.
    replay(corpus, inner, cache=cache, concurrency=1, progress=False)
    assert len(inner.calls) == calls_after_first


def test_replay_writes_out_path(tmp_path):
    corpus = make_corpus(3)
    out = tmp_path / "result.jsonl"
    replay(corpus, MockProvider({"*": "z"}), out_path=out, progress=False)
    loaded = Corpus.from_jsonl(out)
    assert len(loaded) == 3
    assert all(c.candidate_output == "z" for c in loaded)


def test_replay_checkpoint_resume(tmp_path):
    corpus = make_corpus(4)
    ckpt = tmp_path / "ck.jsonl"
    inner = MockProvider({"*": "v1"})
    first, _ = replay(corpus, inner, checkpoint_path=ckpt, checkpoint_every=2, progress=False)
    # Second run resumes the first run's output: nothing new to do.
    inner2 = MockProvider({"*": "v2"})
    result, stats = replay(
        first, inner2, checkpoint_path=ckpt, checkpoint_every=2, resume=True, progress=False
    )
    assert len(inner2.calls) == 0
    assert all(c.candidate_output == "v1" for c in result)
