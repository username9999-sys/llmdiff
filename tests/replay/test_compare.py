"""Tests for compare_runs: merging baseline and candidate corpora."""

from __future__ import annotations

import pytest

from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import CorpusError
from llmdiff.replay.compare_runs import compare_runs, load_pair
from llmdiff.replay.mock_provider import MockProvider
from llmdiff.replay.runner import replay


def make_baseline() -> Corpus:
    return Corpus.from_records(
        [
            {"id": "a", "prompt": "p1", "baseline_output": "old-1", "candidate_output": "old-1",
             "metadata": {"tags": ["x"]}},
            {"id": "b", "prompt": "p2", "baseline_output": "old-2", "candidate_output": "old-2"},
        ]
    )


def test_compare_runs_merges_candidates():
    baseline = make_baseline()
    candidate = Corpus.from_records(
        [
            {"id": "a", "prompt": "p1", "baseline_output": "", "candidate_output": "new-1",
             "metadata": {"model": "v2"}},
            {"id": "b", "prompt": "p2", "baseline_output": "", "candidate_output": "new-2",
             "metadata": {"model": "v2"}},
        ]
    )
    merged = compare_runs(baseline, candidate)
    by_id = {c.id: c for c in merged}
    assert by_id["a"].baseline_output == "old-1"
    assert by_id["a"].candidate_output == "new-1"
    assert by_id["a"].metadata["model"] == "v2"
    assert by_id["b"].candidate_output == "new-2"


def test_compare_runs_rejects_extra_candidate_ids():
    baseline = make_baseline()
    candidate = Corpus.from_records(
        [
            {"id": "a", "prompt": "p", "baseline_output": "", "candidate_output": "n"},
            {"id": "zzz", "prompt": "p", "baseline_output": "", "candidate_output": "n"},
        ]
    )
    with pytest.raises(CorpusError, match="absent from baseline"):
        compare_runs(baseline, candidate)


def test_compare_runs_missing_candidate_id_raises():
    baseline = make_baseline()
    candidate = Corpus.from_records(
        [{"id": "a", "prompt": "p", "baseline_output": "", "candidate_output": "n"}]
    )
    with pytest.raises(CorpusError):
        compare_runs(baseline, candidate)


def test_load_pair_roundtrip(tmp_path):
    baseline = make_baseline()
    candidate = Corpus.from_records(
        [
            {"id": "a", "prompt": "p1", "baseline_output": "", "candidate_output": "new-1"},
            {"id": "b", "prompt": "p2", "baseline_output": "", "candidate_output": "new-2"},
        ]
    )
    bp = tmp_path / "base.jsonl"
    cp = tmp_path / "cand.jsonl"
    baseline.to_jsonl(bp)
    candidate.to_jsonl(cp)
    merged = load_pair(bp, cp)
    assert {c.id: c.candidate_output for c in merged} == {"a": "new-1", "b": "new-2"}


def test_end_to_end_replay_then_compare():
    """Full loop: replay a corpus with a mock provider, then compare to baseline."""
    baseline = make_baseline()
    provider = MockProvider({"p1": "fresh-1", "p2": "fresh-2"})
    candidate, stats = replay(baseline, provider, concurrency=2, progress=False)
    assert stats.succeeded == 2
    merged = compare_runs(baseline, candidate)
    by_id = {c.id: c for c in merged}
    assert by_id["a"].candidate_output == "fresh-1"
    assert by_id["a"].baseline_output == "old-1"
    assert by_id["b"].candidate_output == "fresh-2"
