"""Tests for llmdiff.capture.dedup — crafted prompts."""

from __future__ import annotations

import pytest

from llmdiff.capture.dedup import (
    DedupPipeline,
    ExactDeduper,
    SimHashDeduper,
    dedup_records,
    hamming,
    normalize_prompt,
    prompt_hash,
    simhash,
)
from llmdiff.capture.errors import DedupError


def _rec(prompt, i=0):
    return {
        "id": f"r{i}",
        "prompt": prompt,
        "baseline_output": "",
        "candidate_output": "o",
        "metadata": {},
    }


def test_normalize_prompt():
    assert normalize_prompt("Hello,  WORLD!") == "hello world"
    assert normalize_prompt("  a  b  ") == "a b"


def test_prompt_hash_stable_and_normalized():
    assert prompt_hash("Hello world") == prompt_hash("hello   world")
    assert prompt_hash("a") != prompt_hash("b")


def test_hamming():
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1111, 0b0000) == 4
    assert hamming(1, 3) == 1


def test_simhash_identical_prompts_equal():
    assert simhash("the quick brown fox") == simhash("the quick brown fox")


def test_simhash_similar_prompts_close():
    # measured: a one-token change in a longer prompt stays within hamming 3
    a = simhash(
        "summarize this long article about machine learning models and "
        "neural networks in production systems today"
    )
    b = simhash(
        "summarize this long article about machine learning models and "
        "neural networks in production systems now"
    )
    assert hamming(a, b) <= 3


def test_simhash_different_prompts_far():
    a = simhash("the quick brown fox jumps over the lazy dog")
    b = simhash(
        "quantum entanglement enables novel cryptographic protocols for distributed systems"
    )
    assert hamming(a, b) > 3


def test_simhash_invalid_bits():
    with pytest.raises(DedupError):
        simhash("x", bits=0)
    with pytest.raises(DedupError):
        simhash("x", bits=100)


def test_exact_deduper():
    d = ExactDeduper()
    recs = [_rec("Hello world", 1), _rec("hello   world", 2), _rec("other", 3)]
    kept = d.filter(recs)
    assert [r["id"] for r in kept] == ["r1", "r3"]
    assert d.dropped == 1


def test_simhash_deduper_near_dup():
    d = SimHashDeduper(threshold=3)
    base = (
        "summarize this long article about machine learning models and "
        "neural networks in production systems today"
    )
    near = (
        "summarize this long article about machine learning models and "
        "neural networks in production systems now"
    )  # one-token diff -> hamming 3 (measured)
    far = "write a poem about the ocean waves at sunset on the beach"
    recs = [_rec(base, 1), _rec(near, 2), _rec(far, 3)]
    kept = d.filter(recs)
    assert "r1" in [r["id"] for r in kept]
    assert "r3" in [r["id"] for r in kept]
    assert "r2" not in [r["id"] for r in kept]


def test_simhash_threshold_validation():
    with pytest.raises(DedupError):
        SimHashDeduper(threshold=64, bits=64)
    with pytest.raises(DedupError):
        SimHashDeduper(threshold=1, bits=64, blocks=3)  # 3 does not divide 64


def test_dedup_pipeline_exact_then_near():
    pipe = DedupPipeline(near_threshold=3)
    recs = [
        _rec("identical prompt", 1),
        _rec("identical prompt", 2),  # exact dup
        _rec("completely different topic about cooking recipes", 3),
    ]
    kept = pipe.filter(recs)
    assert [r["id"] for r in kept] == ["r1", "r3"]
    assert pipe.stats["exact_dropped"] == 1


def test_dedup_pipeline_no_near():
    pipe = DedupPipeline(near_threshold=None)
    recs = [_rec("alpha beta gamma delta", 1), _rec("alpha beta gamma delta epsilon", 2)]
    kept = pipe.filter(recs)
    assert len(kept) == 2  # near-dup disabled


def test_dedup_records_convenience():
    recs = [_rec("same", 1), _rec("same", 2)]
    assert len(dedup_records(recs, near_threshold=None)) == 1
