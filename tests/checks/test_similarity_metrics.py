"""Tests for similarity_metrics dimensions."""

from __future__ import annotations

import pytest

from llmdiff import Corpus, evaluate
from llmdiff.checks.similarity_metrics import (
    BhattacharyyaWordCheck,
    ContainmentRatioCheck,
    CosineCharTrigramCheck,
    CosineUnigramCheck,
    CosineWordVectorCheck,
    DamerauLevenshteinRatioCheck,
    DiceCoefficientCheck,
    EuclideanWordDistCheck,
    HammingSimilarityCheck,
    HellingerWordDistCheck,
    JaccardWordCheck,
    KLDivergenceWordCheck,
    LcsWordRatioCheck,
    LevenshteinRatioCheck,
    LongestCommonSubseqRatioCheck,
    ManhattanWordDistCheck,
    OverlapCoefficientCheck,
    RatcliffObershelpCheck,
    SmithWatermanRatioCheck,
    SorensenDiceBigramCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_levenshtein_ratio():
    ctx = _ctx("kitten", "kitten")
    assert LevenshteinRatioCheck().score("kitten", ctx) == 1.0
    assert LevenshteinRatioCheck().score("sitting", ctx) < 1.0


def test_damerau_ratio():
    ctx = _ctx("abcd", "abdc")
    assert DamerauLevenshteinRatioCheck().score("abdc", ctx) > 0.7
    assert DamerauLevenshteinRatioCheck().score("wxyz", ctx) < 1.0


def test_hamming_sim():
    ctx = _ctx("abc", "abc")
    assert HammingSimilarityCheck().score("abc", ctx) == 1.0
    assert HammingSimilarityCheck().score("axc", ctx) < 1.0


def test_jaccard_words():
    ctx = _ctx("cat dog bird", "cat dog bird")
    assert JaccardWordCheck().score("cat dog bird", ctx) == 1.0
    assert JaccardWordCheck().score("cat fish", ctx) < 1.0


def test_dice_words():
    ctx = _ctx("cat dog", "cat dog")
    assert DiceCoefficientCheck().score("cat dog", ctx) == 1.0
    assert DiceCoefficientCheck().score("cat fish", ctx) < 1.0


def test_cosine_words():
    ctx = _ctx("cat dog cat", "cat dog cat")
    assert CosineWordVectorCheck().score("cat dog cat", ctx) == pytest.approx(1.0)
    assert CosineWordVectorCheck().score("fish bird", ctx) < 1.0


def test_cosine_char_trigrams():
    ctx = _ctx("hello world", "hello world")
    assert CosineCharTrigramCheck().score("hello world", ctx) == 1.0


def test_overlap_coef():
    ctx = _ctx("cat dog", "cat dog bird fish")
    assert OverlapCoefficientCheck().score("cat dog bird fish", ctx) == 1.0


def test_dice_bigrams():
    ctx = _ctx("the cat sat", "the cat sat")
    assert SorensenDiceBigramCheck().score("the cat sat", ctx) == 1.0
    assert SorensenDiceBigramCheck().score("a b c d", ctx) < 1.0


def test_containment():
    ctx = _ctx("cat dog", "cat dog bird")
    assert ContainmentRatioCheck().score("cat dog bird", ctx) == 1.0
    assert ContainmentRatioCheck().score("cat", ctx) < 1.0


def test_lcs_char_ratio():
    ctx = _ctx("abcdef", "abcdef")
    assert LongestCommonSubseqRatioCheck().score("abcdef", ctx) == 1.0
    assert LongestCommonSubseqRatioCheck().score("xyz", ctx) < 1.0


def test_lcs_word_ratio():
    ctx = _ctx("the cat sat on mat", "the cat sat on mat")
    assert LcsWordRatioCheck().score("the cat sat on mat", ctx) == 1.0
    assert LcsWordRatioCheck().score("dog runs", ctx) < 1.0


def test_ratcliff_obershelp():
    ctx = _ctx("hello", "hello")
    assert RatcliffObershelpCheck().score("hello", ctx) == 1.0


def test_smith_waterman():
    ctx = _ctx("hello world", "hello world")
    assert SmithWatermanRatioCheck().score("hello world", ctx) == 1.0


def test_cosine_unigrams():
    ctx = _ctx("a b c", "a b c")
    assert CosineUnigramCheck().score("a b c", ctx) == pytest.approx(1.0)


def test_euclidean_words():
    ctx = _ctx("cat dog", "cat dog")
    assert EuclideanWordDistCheck().score("cat dog", ctx) == 1.0
    assert EuclideanWordDistCheck().score("cat fish", ctx) < 1.0


def test_manhattan_words():
    ctx = _ctx("cat dog", "cat dog")
    assert ManhattanWordDistCheck().score("cat dog", ctx) == 1.0


def test_hellinger_words():
    ctx = _ctx("cat dog", "cat dog")
    assert HellingerWordDistCheck().score("cat dog", ctx) == 1.0


def test_bhattacharyya_words():
    ctx = _ctx("cat dog", "cat dog")
    assert BhattacharyyaWordCheck().score("cat dog", ctx) == 1.0
    assert BhattacharyyaWordCheck().score("cat fish", ctx) < 1.0


def test_kl_words():
    ctx = _ctx("cat dog bird", "cat dog bird")
    assert KLDivergenceWordCheck().score("cat dog bird", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "the quick brown fox",
         "candidate_output": "the quick brown fox"},
        {"id": "b", "prompt": "p", "baseline_output": "the quick brown fox",
         "candidate_output": "lazy dogs sleep all day"},
    ])
    result = evaluate(corpus, dimensions=("levenshtein_ratio", "jaccard_words"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["levenshtein_ratio"].losses == 1
    assert dims["jaccard_words"].losses == 1
