"""Tests for tokenmix dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.tokenmix import (
    CodeRatioCheck,
    DigitRatioCheck,
    TokenOverlapCheck,
    UniqueTokenRatioCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_code_ratio_identical():
    text = "here:\n```python\nprint(1)\n```\ndone"
    ctx = _ctx(text, text)
    assert CodeRatioCheck().score(text, ctx) == 1.0


def test_code_ratio_differs():
    ctx = _ctx("```python\nx=1\n```", "no code at all")
    assert CodeRatioCheck().score("no code at all", ctx) < 1.0


def test_token_overlap():
    ctx = _ctx("a b c d", "a b c d")
    assert TokenOverlapCheck().score("a b c d", ctx) == 1.0
    assert TokenOverlapCheck().score("x y z", ctx) == 0.0


def test_unique_token_ratio():
    ctx = _ctx("a a a b", "a a a b")
    assert UniqueTokenRatioCheck().score("a a a b", ctx) == 1.0


def test_digit_ratio():
    ctx = _ctx("order 12345 total 99", "order 12345 total 99")
    assert DigitRatioCheck().score("order 12345 total 99", ctx) == 1.0
    assert DigitRatioCheck().score("no digits here", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "alpha beta gamma 123",
         "candidate_output": "alpha beta gamma 123"},
        {"id": "b", "prompt": "p", "baseline_output": "alpha beta gamma 123",
         "candidate_output": "zzz yyy xxx www"},
    ])
    result = evaluate(corpus, dimensions=("token_overlap", "digit_ratio"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["token_overlap"].losses == 1
    assert dims["digit_ratio"].losses == 1
