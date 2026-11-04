"""Tests for tokenmix_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.tokenmix_deep import (
    AlphaTokenRatioCheck,
    AvgTokenFreqCheck,
    CamelTokenRatioCheck,
    DotTokenRatioCheck,
    HyphenTokenRatioCheck,
    LowerTokenRatioCheck,
    MixedAlnumTokenRatioCheck,
    NumericTokenRatioCheck,
    PunctTokenRatioCheck,
    RareTokenRatioCheck,
    SlashTokenRatioCheck,
    SnakeTokenRatioCheck,
    SpaceTokenRatioCheck,
    StopwordTokenRatioCheck,
    TokenLengthSkewCheck,
    TokenLengthStdCheck,
    TokenMixEntropyCheck,
    UnderscoreTokenRatioCheck,
    UpperTokenRatioCheck,
    VocabRichnessCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_alpha_numeric_ratio():
    ctx = _ctx("abc def", "abc def")
    assert AlphaTokenRatioCheck().score("abc def", ctx) == 1.0
    assert AlphaTokenRatioCheck().score("123 456", ctx) < 1.0
    nctx = _ctx("123 456", "123 456")
    assert NumericTokenRatioCheck().score("123 456", nctx) == 1.0


def test_punct_space_ratio():
    ctx = _ctx("a, b!", "a, b!")
    assert PunctTokenRatioCheck().score("a, b!", ctx) == 1.0
    sctx = _ctx("a b  c", "a b  c")
    assert SpaceTokenRatioCheck().score("a b  c", sctx) == 1.0


def test_mixed_alnum():
    ctx = _ctx("abc123 x9", "abc123 x9")
    assert MixedAlnumTokenRatioCheck().score("abc123 x9", ctx) == 1.0


def test_camel_snake_upper_lower():
    ctx = _ctx("myVar", "myVar")
    assert CamelTokenRatioCheck().score("myVar", ctx) == 1.0
    sctx = _ctx("my_var", "my_var")
    assert SnakeTokenRatioCheck().score("my_var", sctx) == 1.0
    uctx = _ctx("ABC", "ABC")
    assert UpperTokenRatioCheck().score("ABC", uctx) == 1.0
    lctx = _ctx("abc", "abc")
    assert LowerTokenRatioCheck().score("abc", lctx) == 1.0


def test_symbol_ratios():
    ctx = _ctx("a_b", "a_b")
    assert UnderscoreTokenRatioCheck().score("a_b", ctx) == 1.0
    hctx = _ctx("well-known", "well-known")
    assert HyphenTokenRatioCheck().score("well-known", hctx) == 1.0
    dctx = _ctx("file.txt", "file.txt")
    assert DotTokenRatioCheck().score("file.txt", dctx) == 1.0
    slctx = _ctx("a/b/c", "a/b/c")
    assert SlashTokenRatioCheck().score("a/b/c", slctx) == 1.0


def test_token_length_stats():
    ctx = _ctx("aa bbbb cc", "aa bbbb cc")
    assert TokenLengthStdCheck().score("aa bbbb cc", ctx) == 1.0
    assert TokenLengthSkewCheck().score("aa bbbb cc", ctx) == 1.0


def test_vocab_entropy_rare_stopword_freq():
    ctx = _ctx("cat dog bird", "cat dog bird")
    assert VocabRichnessCheck().score("cat dog bird", ctx) == 1.0
    assert TokenMixEntropyCheck().score("cat dog bird", ctx) == 1.0
    assert RareTokenRatioCheck().score("cat dog bird", ctx) == 1.0
    swctx = _ctx("the and of", "the and of")
    assert StopwordTokenRatioCheck().score("the and of", swctx) == 1.0
    assert AvgTokenFreqCheck().score("cat dog bird", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "hello world foo"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "123 456 789"},
    ])
    result = evaluate(corpus, dimensions=("alpha_token_ratio", "vocab_richness"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["alpha_token_ratio"].losses == 1
    assert dims["vocab_richness"].losses == 1