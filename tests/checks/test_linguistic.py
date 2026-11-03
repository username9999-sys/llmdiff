"""Tests for linguistic dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.linguistic import (
    AcronymCountCheck,
    AvgSentenceLengthCheck,
    CapitalizedWordRatioCheck,
    ClosingPunctMatchCheck,
    ColonCountCheck,
    ConsonantRatioCheck,
    DashCountCheck,
    EllipsisCountCheck,
    ExclamationCountCheck,
    FirstWordMatchCheck,
    HyphenatedWordCountCheck,
    LastWordMatchCheck,
    NumberWordRatioCheck,
    OpeningPunctMatchCheck,
    ParenthesisBalanceCheck,
    QuestionCountCheck,
    QuoteStyleMatchCheck,
    SemicolonCountCheck,
    VowelRatioCheck,
    WhitespaceRatioCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_vowel_ratio():
    ctx = _ctx("aeiou", "aeiou")
    assert VowelRatioCheck().score("aeiou", ctx) == 1.0
    assert VowelRatioCheck().score("bcdfg", ctx) < 1.0


def test_consonant_ratio():
    ctx = _ctx("bcdfg", "bcdfg")
    assert ConsonantRatioCheck().score("bcdfg", ctx) == 1.0


def test_avg_sentence_length():
    ctx = _ctx("One two. Three four.", "One two. Three four.")
    assert AvgSentenceLengthCheck().score("One two. Three four.", ctx) == 1.0
    assert AvgSentenceLengthCheck().score("One.", ctx) < 1.0


def test_question_count():
    ctx = _ctx("what? why?", "what? why?")
    assert QuestionCountCheck().score("what? why?", ctx) == 1.0
    assert QuestionCountCheck().score("what?", ctx) < 1.0


def test_exclamation_count():
    ctx = _ctx("wow! yay!", "wow! yay!")
    assert ExclamationCountCheck().score("wow! yay!", ctx) == 1.0
    assert ExclamationCountCheck().score("wow!", ctx) < 1.0


def test_ellipsis_count():
    ctx = _ctx("wait... ok", "wait... ok")
    assert EllipsisCountCheck().score("wait... ok", ctx) == 1.0
    assert EllipsisCountCheck().score("wait… ok", ctx) == 1.0
    assert EllipsisCountCheck().score("wait ok", ctx) < 1.0


def test_semicolon_count():
    ctx = _ctx("a; b;", "a; b;")
    assert SemicolonCountCheck().score("a; b;", ctx) == 1.0
    assert SemicolonCountCheck().score("a b", ctx) < 1.0


def test_colon_count():
    ctx = _ctx("a: b:", "a: b:")
    assert ColonCountCheck().score("a: b:", ctx) == 1.0
    assert ColonCountCheck().score("a b", ctx) < 1.0


def test_dash_count():
    ctx = _ctx("a - b — c", "a - b — c")
    assert DashCountCheck().score("a - b — c", ctx) == 1.0
    assert DashCountCheck().score("a b c", ctx) < 1.0


def test_paren_balance():
    ctx = _ctx("(a) (b)", "(a) (b)")
    assert ParenthesisBalanceCheck().score("(a) (b)", ctx) == 1.0
    assert ParenthesisBalanceCheck().score("(a (b)", ctx) < 1.0


def test_quote_style():
    ctx = _ctx('"hello"', '"hello"')
    assert QuoteStyleMatchCheck().score('"hello"', ctx) == 1.0
    assert QuoteStyleMatchCheck().score("“hello”", ctx) < 1.0


def test_number_word_ratio():
    ctx = _ctx("buy 2 apples", "buy 2 apples")
    assert NumberWordRatioCheck().score("buy 2 apples", ctx) == 1.0
    assert NumberWordRatioCheck().score("buy apples", ctx) < 1.0


def test_acronym_count():
    ctx = _ctx("NASA and FBI", "NASA and FBI")
    assert AcronymCountCheck().score("NASA and FBI", ctx) == 1.0
    assert AcronymCountCheck().score("NASA", ctx) < 1.0


def test_hyphenated_words():
    ctx = _ctx("well-known fact", "well-known fact")
    assert HyphenatedWordCountCheck().score("well-known fact", ctx) == 1.0
    assert HyphenatedWordCountCheck().score("well known fact", ctx) < 1.0


def test_cap_word_ratio():
    ctx = _ctx("Hello World", "Hello World")
    assert CapitalizedWordRatioCheck().score("Hello World", ctx) == 1.0
    assert CapitalizedWordRatioCheck().score("hello world", ctx) < 1.0


def test_first_word():
    ctx = _ctx("Hello there", "Hello there")
    assert FirstWordMatchCheck().score("Hello there", ctx) == 1.0
    assert FirstWordMatchCheck().score("Hi there", ctx) == 0.0


def test_last_word():
    ctx = _ctx("see you later", "see you later")
    assert LastWordMatchCheck().score("see you later", ctx) == 1.0
    assert LastWordMatchCheck().score("see you now", ctx) == 0.0


def test_opening_punct():
    ctx = _ctx('"quoted"', '"quoted"')
    assert OpeningPunctMatchCheck().score('"quoted"', ctx) == 1.0
    assert OpeningPunctMatchCheck().score("quoted", ctx) == 0.0


def test_closing_punct():
    ctx = _ctx("done.", "done.")
    assert ClosingPunctMatchCheck().score("done.", ctx) == 1.0
    assert ClosingPunctMatchCheck().score("done", ctx) == 0.0


def test_whitespace_ratio():
    ctx = _ctx("a b  c", "a b  c")
    assert WhitespaceRatioCheck().score("a b  c", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "Hello world.",
         "candidate_output": "Hello world."},
        {"id": "b", "prompt": "p", "baseline_output": "Hello world.",
         "candidate_output": "Hi there"},
    ])
    result = evaluate(corpus, dimensions=("first_word", "question_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["first_word"].losses == 1
    assert dims["question_count"].losses == 0
