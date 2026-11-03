"""Tests for similarity_extra dimensions (trigram cosine, Jaro-Winkler)."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.similarity_extra import (
    JaroWinklerCheck,
    TrigramSimilarityCheck,
    jaro_winkler,
    trigram_cosine,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_trigram_identical():
    assert trigram_cosine("hello world", "hello world") == 1.0


def test_trigram_disjoint_low():
    assert trigram_cosine("aaaa", "zzzz") < 0.1


def test_jaro_winkler_identical():
    assert jaro_winkler("martha", "martha") == 1.0


def test_jaro_winkler_known_value():
    # classic example: jaro(martha, marth) ~ 0.944, winkler ~ 0.961
    assert 0.93 < jaro_winkler("martha", "marth") < 0.98


def test_jaro_winkler_disjoint():
    assert jaro_winkler("abc", "xyz") == 0.0


def test_checks_identical():
    ctx = _ctx("the quick brown fox", "the quick brown fox")
    assert TrigramSimilarityCheck().score("the quick brown fox", ctx) == 1.0
    assert JaroWinklerCheck().score("the quick brown fox", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "hello world foo"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "zzzz yyyy xxxx"},
    ])
    result = evaluate(corpus, dimensions=("trigram", "jaro_winkler"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["trigram"].losses == 1
    assert dims["jaro_winkler"].losses == 1
