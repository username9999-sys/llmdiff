"""Tests for hash_sim dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.hash_sim import (
    Blake2bPrefixMatchCheck,
    CharShingleJaccardCheck,
    ContentHashEqualCheck,
    Crc32MatchCheck,
    FnvHashMatchCheck,
    HashEntropyCheck,
    HashPrefixLenMatchCheck,
    Md5PrefixMatchCheck,
    MinhashJaccardCheck,
    MinhashMinCountCheck,
    MinhashSignatureMatchCheck,
    RollingHashMatchCheck,
    Sha256PrefixMatchCheck,
    ShingleCountMatchCheck,
    ShingleSetJaccardCheck,
    SimhashBitFlipCountCheck,
    SimhashBucketCountCheck,
    SimhashHammingCheck,
    SimhashSimilarityCheck,
    WordShingleJaccardCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

TEXT = "the quick brown fox jumps over the lazy dog"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_simhash_sim():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashSimilarityCheck().score(TEXT, ctx) == 1.0
    assert SimhashSimilarityCheck().score("completely different text here", ctx) < 1.0


def test_simhash_hamming_and_popcount():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashHammingCheck().score(TEXT, ctx) == 1.0
    assert SimhashBucketCountCheck().score(TEXT, ctx) == 1.0


def test_minhash():
    ctx = _ctx(TEXT, TEXT)
    assert MinhashJaccardCheck().score(TEXT, ctx) == 1.0
    assert MinhashSignatureMatchCheck().score(TEXT, ctx) == 1.0
    assert MinhashMinCountCheck().score(TEXT, ctx) == 1.0


def test_content_hash_equal():
    ctx = _ctx(TEXT, TEXT)
    assert ContentHashEqualCheck().score(TEXT, ctx) == 1.0
    assert ContentHashEqualCheck().score("other", ctx) == 0.0


def test_hash_prefixes():
    ctx = _ctx(TEXT, TEXT)
    assert Md5PrefixMatchCheck().score(TEXT, ctx) == 1.0
    assert Sha256PrefixMatchCheck().score(TEXT, ctx) == 1.0
    assert Crc32MatchCheck().score(TEXT, ctx) == 1.0
    assert Blake2bPrefixMatchCheck().score(TEXT, ctx) == 1.0
    assert HashPrefixLenMatchCheck().score(TEXT, ctx) == 1.0
    assert FnvHashMatchCheck().score(TEXT, ctx) == 1.0


def test_rolling_hash():
    ctx = _ctx("a\nb\nc", "a\nb\nc")
    assert RollingHashMatchCheck().score("a\nb\nc", ctx) == 1.0


def test_shingles():
    ctx = _ctx(TEXT, TEXT)
    assert ShingleSetJaccardCheck().score(TEXT, ctx) == 1.0
    assert WordShingleJaccardCheck().score(TEXT, ctx) == 1.0
    assert CharShingleJaccardCheck().score(TEXT, ctx) == 1.0
    assert ShingleCountMatchCheck().score(TEXT, ctx) == 1.0
    assert ShingleSetJaccardCheck().score("zzz yyy xxx", ctx) < 1.0


def test_hash_entropy():
    ctx = _ctx(TEXT, TEXT)
    assert HashEntropyCheck().score(TEXT, ctx) == 1.0


def test_simhash_bitflip():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashBitFlipCountCheck().score(TEXT, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": TEXT,
         "candidate_output": TEXT},
        {"id": "b", "prompt": "p", "baseline_output": TEXT,
         "candidate_output": "totally unrelated content here"},
    ])
    result = evaluate(corpus, dimensions=("simhash_sim", "md5_equal"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["simhash_sim"].losses == 1
    assert dims["md5_equal"].losses == 1
