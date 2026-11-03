"""Tests for ngram_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.ngram_stats import (
    BigramEntropyCheck,
    BigramJaccardCheck,
    BigramOverlapCheck,
    CharBigramOverlapCheck,
    CharTrigramOverlapCheck,
    MaxRepeatRunCheck,
    MostCommonBigramMatchCheck,
    MostCommonTrigramMatchCheck,
    NgramCountMatchCheck,
    NgramDiversityCheck,
    RepeatedBigramRatioCheck,
    RepeatedTrigramRatioCheck,
    ShannonEntropyCheck,
    TrigramEntropyCheck,
    TrigramOverlapCheck,
    UnigramJaccardCheck,
    UnigramOverlapCheck,
    UniqueBigramRatioCheck,
    UniqueTrigramRatioCheck,
    ZipfSlopeCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_unigram_overlap():
    ctx = _ctx("the cat sat", "the cat sat")
    assert UnigramOverlapCheck().score("the cat sat", ctx) == 1.0
    assert UnigramOverlapCheck().score("dog runs fast", ctx) < 1.0


def test_bigram_overlap():
    ctx = _ctx("the cat sat on", "the cat sat on")
    assert BigramOverlapCheck().score("the cat sat on", ctx) == 1.0
    assert BigramOverlapCheck().score("a b c d", ctx) < 1.0


def test_trigram_overlap():
    ctx = _ctx("the cat sat on mat", "the cat sat on mat")
    assert TrigramOverlapCheck().score("the cat sat on mat", ctx) == 1.0


def test_unigram_jaccard():
    ctx = _ctx("cat dog", "cat dog")
    assert UnigramJaccardCheck().score("cat dog", ctx) == 1.0
    assert UnigramJaccardCheck().score("cat bird", ctx) < 1.0


def test_bigram_jaccard():
    ctx = _ctx("cat dog bird", "cat dog bird")
    assert BigramJaccardCheck().score("cat dog bird", ctx) == 1.0


def test_char_bigram_overlap():
    ctx = _ctx("hello", "hello")
    assert CharBigramOverlapCheck().score("hello", ctx) == 1.0
    assert CharBigramOverlapCheck().score("world", ctx) < 1.0


def test_char_trigram_overlap():
    ctx = _ctx("hello", "hello")
    assert CharTrigramOverlapCheck().score("hello", ctx) == 1.0


def test_top_bigram():
    ctx = _ctx("a b a b a b", "a b a b a b")
    assert MostCommonBigramMatchCheck().score("a b a b a b", ctx) == 1.0
    assert MostCommonBigramMatchCheck().score("x y x y z z", ctx) == 0.0


def test_top_trigram():
    ctx = _ctx("a b c a b c a b c", "a b c a b c a b c")
    assert MostCommonTrigramMatchCheck().score("a b c a b c a b c", ctx) == 1.0


def test_bigram_entropy():
    ctx = _ctx("a b c d e", "a b c d e")
    assert BigramEntropyCheck().score("a b c d e", ctx) == 1.0


def test_trigram_entropy():
    ctx = _ctx("a b c d e f", "a b c d e f")
    assert TrigramEntropyCheck().score("a b c d e f", ctx) == 1.0


def test_repeated_bigram_ratio():
    ctx = _ctx("a b a b a b", "a b a b a b")
    assert RepeatedBigramRatioCheck().score("a b a b a b", ctx) == 1.0
    assert RepeatedBigramRatioCheck().score("a b c d e f", ctx) < 1.0


def test_repeated_trigram_ratio():
    ctx = _ctx("a b c a b c", "a b c a b c")
    assert RepeatedTrigramRatioCheck().score("a b c a b c", ctx) == 1.0


def test_max_repeat_run():
    ctx = _ctx("go go go stop", "go go go stop")
    assert MaxRepeatRunCheck().score("go go go stop", ctx) == 1.0
    assert MaxRepeatRunCheck().score("go stop", ctx) < 1.0


def test_unique_bigram_ratio():
    ctx = _ctx("a b c d", "a b c d")
    assert UniqueBigramRatioCheck().score("a b c d", ctx) == 1.0


def test_unique_trigram_ratio():
    ctx = _ctx("a b c d e", "a b c d e")
    assert UniqueTrigramRatioCheck().score("a b c d e", ctx) == 1.0


def test_bigram_count():
    ctx = _ctx("a b c d", "a b c d")
    assert NgramCountMatchCheck().score("a b c d", ctx) == 1.0
    assert NgramCountMatchCheck().score("a b", ctx) < 1.0


def test_char_entropy():
    ctx = _ctx("abcdef", "abcdef")
    assert ShannonEntropyCheck().score("abcdef", ctx) == 1.0


def test_zipf_slope():
    text = " ".join(["the"] * 10 + [f"w{i}" for i in range(20)])
    ctx = _ctx(text, text)
    assert ZipfSlopeCheck().score(text, ctx) == 1.0


def test_ngram_diversity():
    ctx = _ctx("a b c d e", "a b c d e")
    assert NgramDiversityCheck().score("a b c d e", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "the cat sat on the mat",
         "candidate_output": "the cat sat on the mat"},
        {"id": "b", "prompt": "p", "baseline_output": "the cat sat on the mat",
         "candidate_output": "dogs run fast today"},
    ])
    result = evaluate(corpus, dimensions=("bigram_overlap", "bigram_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["bigram_overlap"].losses == 1
    assert dims["bigram_count"].losses == 1
