"""Tests for entropy and repetition dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.entropy import EntropyCheck, RepetitionCheck, repetition_ratio, shannon_entropy
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_shannon_entropy_empty():
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_constant_is_zero():
    assert shannon_entropy("aaaaaa") == 0.0


def test_shannon_entropy_varied_is_higher():
    assert shannon_entropy("abcdefgh") > shannon_entropy("aaaaabbb")


def test_repetition_ratio():
    assert repetition_ratio("") == 0.0
    assert repetition_ratio("aaaaaa") == 1.0
    assert repetition_ratio("abcabc") == 0.0


def test_entropy_check_identical():
    ctx = _ctx("hello world foo bar", "hello world foo bar")
    assert EntropyCheck().score("hello world foo bar", ctx) == 1.0


def test_entropy_check_loop_scores_low():
    ctx = _ctx("a varied and interesting sentence with many words", "aaaaaaaaaaaaaaaaaaaa")
    assert EntropyCheck().score("aaaaaaaaaaaaaaaaaaaa", ctx) < 0.5


def test_repetition_check_loop():
    ctx = _ctx("normal text", "xxxxxxxxxx")
    assert RepetitionCheck().score("xxxxxxxxxx", ctx) == 0.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p",
         "baseline_output": "diverse words here now today",
         "candidate_output": "diverse words here now today"},
        {"id": "b", "prompt": "p",
         "baseline_output": "diverse words here now today",
         "candidate_output": "aaaaaaaaaaaaaaaaaaaa"},
    ])
    result = evaluate(corpus, dimensions=("entropy", "repetition"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["repetition"].losses == 1
    assert dims["entropy"].losses == 1
