"""Tests for qa and numeric dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.numeric import MeanAbsoluteErrorCheck, NumberDriftCheck, NumberPreservedCheck
from llmdiff.checks.qa import (
    AnswersPromptCheck,
    CitationCheck,
    LanguageMatchCheck,
    NumberConsistencyCheck,
    extract_numbers,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str, prompt: str = "p") -> CheckContext:
    case = Case(id="t", prompt=prompt, baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_extract_numbers():
    assert extract_numbers("price is 12.5 and 3k units") == [12.5, 3000.0]
    assert extract_numbers("no numbers") == []


def test_answers_prompt():
    ctx = _ctx("x", "The refund policy allows returns", prompt="what is the refund policy?")
    score = AnswersPromptCheck().score("The refund policy allows returns", ctx)
    assert 0.0 < score <= 1.0


def test_number_consistency():
    ctx = _ctx("total 100 and 5 items", "total 100 and 5 items")
    assert NumberConsistencyCheck().score("total 100 and 5 items", ctx) == 1.0
    assert NumberConsistencyCheck().score("total 200 and 5 items", ctx) < 1.0


def test_citation():
    ctx = _ctx("see [1]", "see [1] and [2]")
    assert CitationCheck().score("see [1] and [2]", ctx) == 1.0
    assert CitationCheck().score("no citation", ctx) == 0.0


def test_language_match_latin():
    ctx = _ctx("hello world", "bonjour le monde")
    assert LanguageMatchCheck().score("bonjour le monde", ctx) == 1.0


def test_language_match_cjk():
    ctx = _ctx("你好世界", "hello world")
    assert LanguageMatchCheck().score("hello world", ctx) == 0.0


def test_numbers_preserved():
    ctx = _ctx("a=1 b=2", "a=1 b=2")
    assert NumberPreservedCheck().score("a=1 b=2", ctx) == 1.0
    assert NumberPreservedCheck().score("a=1", ctx) < 1.0


def test_number_drift():
    ctx = _ctx("value 100", "value 100")
    assert NumberDriftCheck().score("value 100", ctx) == 1.0
    assert NumberDriftCheck().score("value 200", ctx) < 1.0


def test_number_mae():
    ctx = _ctx("1 2 3", "1 2 3")
    assert MeanAbsoluteErrorCheck().score("1 2 3", ctx) == 1.0
    assert MeanAbsoluteErrorCheck().score("9 9 9", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "sum", "baseline_output": "The total is 42 items",
         "candidate_output": "The total is 42 items"},
        {"id": "b", "prompt": "sum", "baseline_output": "The total is 42 items",
         "candidate_output": "The total is 99 items"},
    ])
    result = evaluate(corpus, dimensions=("number_consistency", "numbers_preserved"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["number_consistency"].losses == 1
    assert dims["numbers_preserved"].losses == 1
