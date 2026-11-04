"""Tests for semantic_hash dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.semantic_hash import (
    BandSignatureEqualCheck,
    CharNgramLshJaccardCheck,
    CharShingleHashSetCheck,
    FeatureHashDimCheck,
    HashTrickBucketCountCheck,
    LshCollisionProbCheck,
    MinhashLshBandMatchCheck,
    RandomProjectionSignCheck,
    RpBucketJaccardCheck,
    RpDotProductSignCheck,
    SimhashBandCountCheck,
    SimhashBandMatchCheck,
    SimhashCosineCheck,
    SimhashHighBitsMatchCheck,
    SimhashLowBitsMatchCheck,
    SimhashNeighborCountCheck,
    SimhashPrefixBandCheck,
    SimhashSuffixBandCheck,
    WordNgramLshJaccardCheck,
    WordShingleHashSetCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

TEXT = "the quick brown fox jumps over the lazy dog near the river bank"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_simhash_bands():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashBandMatchCheck().score(TEXT, ctx) == 1.0
    assert SimhashBandCountCheck().score(TEXT, ctx) == 1.0
    assert SimhashPrefixBandCheck().score(TEXT, ctx) == 1.0
    assert SimhashSuffixBandCheck().score(TEXT, ctx) == 1.0
    assert BandSignatureEqualCheck().score(TEXT, ctx) == 1.0
    assert BandSignatureEqualCheck().score("zzz qqq", ctx) == 0.0


def test_simhash_bits():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashLowBitsMatchCheck().score(TEXT, ctx) == 1.0
    assert SimhashHighBitsMatchCheck().score(TEXT, ctx) == 1.0
    assert SimhashCosineCheck().score(TEXT, ctx) == 1.0
    assert SimhashCosineCheck().score("totally different", ctx) < 1.0


def test_rp_signs():
    ctx = _ctx(TEXT, TEXT)
    assert RandomProjectionSignCheck().score(TEXT, ctx) == 1.0
    assert RpDotProductSignCheck().score(TEXT, ctx) == 1.0
    assert RpBucketJaccardCheck().score(TEXT, ctx) == 1.0


def test_lsh_jaccards():
    ctx = _ctx(TEXT, TEXT)
    assert CharNgramLshJaccardCheck().score(TEXT, ctx) == 1.0
    assert WordNgramLshJaccardCheck().score(TEXT, ctx) == 1.0
    assert CharShingleHashSetCheck().score(TEXT, ctx) == 1.0
    assert WordShingleHashSetCheck().score(TEXT, ctx) == 1.0
    assert CharNgramLshJaccardCheck().score("zzz yyy xxx www", ctx) < 1.0


def test_minhash_band():
    ctx = _ctx(TEXT, TEXT)
    assert MinhashLshBandMatchCheck().score(TEXT, ctx) == 1.0


def test_bucket_counts():
    ctx = _ctx(TEXT, TEXT)
    assert HashTrickBucketCountCheck().score(TEXT, ctx) == 1.0
    assert FeatureHashDimCheck().score(TEXT, ctx) == 1.0


def test_collision_prob():
    ctx = _ctx(TEXT, TEXT)
    assert LshCollisionProbCheck().score(TEXT, ctx) == 1.0


def test_neighbors():
    ctx = _ctx(TEXT, TEXT)
    assert SimhashNeighborCountCheck().score(TEXT, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": TEXT,
         "candidate_output": TEXT},
        {"id": "b", "prompt": "p", "baseline_output": TEXT,
         "candidate_output": "completely unrelated words here"},
    ])
    result = evaluate(corpus, dimensions=("simhash_band_match", "simhash_cosine"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["simhash_band_match"].losses == 1
    assert dims["simhash_cosine"].losses == 1
