"""Tests for the n-gram overlap dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.ngram import (
    NgramF1Check,
    NgramPrecisionCheck,
    NgramRecallCheck,
    TokenNgramCheck,
    ngrams,
    tokenize,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_tokenize_lowercase_alnum():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]
    assert tokenize("") == []


def test_ngrams_basic():
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"): 1, ("b", "c"): 1}
    assert ngrams(["a"], 2) == {}


def test_token_jaccard_identical_is_one():
    ctx = _ctx("the quick brown fox", "the quick brown fox")
    assert TokenNgramCheck().score("the quick brown fox", ctx) == 1.0


def test_token_jaccard_disjoint_is_zero():
    ctx = _ctx("aaa bbb", "ccc ddd")
    assert TokenNgramCheck().score("ccc ddd", ctx) == 0.0


def test_precision_recall_f1_partial_overlap():
    base = "the quick brown fox jumps over the lazy dog"
    cand = "the quick brown fox sleeps over the lazy dog"
    ctx = _ctx(base, cand)
    prec = NgramPrecisionCheck().score(cand, ctx)
    rec = NgramRecallCheck().score(cand, ctx)
    f1 = NgramF1Check().score(cand, ctx)
    assert 0.0 < prec < 1.0
    assert 0.0 < rec < 1.0
    assert 0.0 < f1 < 1.0
    assert f1 <= max(prec, rec)


def test_both_empty_is_perfect():
    ctx = _ctx("", "")
    for check in (TokenNgramCheck(), NgramPrecisionCheck(), NgramRecallCheck(), NgramF1Check()):
        assert check.score("", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p",
         "baseline_output": "one two three four five",
         "candidate_output": "one two three four five"},
        {"id": "b", "prompt": "p",
         "baseline_output": "one two three four five",
         "candidate_output": "zzz yyy xxx www vvv"},
    ])
    result = evaluate(corpus, dimensions=("token_jaccard", "ngram_f1"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    # identical output scores equal -> tie, not win
    assert dims["token_jaccard"].ties == 1
    assert dims["token_jaccard"].losses == 1
    assert dims["ngram_f1"].losses == 1
