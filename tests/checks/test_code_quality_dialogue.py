"""Tests for code_quality and dialogue dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code_quality import (
    IndentationCheck,
    LongLineRatioCheck,
    MaxLineLengthCheck,
    TodoMarkerCheck,
    TrailingCodeWhitespaceCheck,
)
from llmdiff.checks.dialogue import (
    BulletAnswerCheck,
    GreetingCheck,
    QuestionAnswerCheck,
    TurnMarkerCheck,
    WordCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str, prompt: str = "p") -> CheckContext:
    case = Case(id="t", prompt=prompt, baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_indent_width():
    ctx = _ctx("```python\ndef f():\n    return 1\n```", "```python\ndef f():\n  return 1\n```")
    assert IndentationCheck().score("```python\ndef f():\n  return 1\n```", ctx) < 1.0
    assert IndentationCheck().score("```python\ndef f():\n    return 1\n```", ctx) == 1.0


def test_max_line_length():
    ctx = _ctx("short", "x" * 500)
    assert MaxLineLengthCheck().score("x" * 500, ctx) < 1.0


def test_long_line_ratio():
    ctx = _ctx("a\nb", "a\n" + "x" * 200)
    assert LongLineRatioCheck().score("a\n" + "x" * 200, ctx) < 1.0


def test_todo_markers_fewer_is_fine():
    ctx = _ctx("# TODO: fix\nx=1", "x=1")
    assert TodoMarkerCheck().score("x=1", ctx) == 1.0
    assert TodoMarkerCheck().score("# TODO a\n# TODO b", ctx) < 1.0


def test_code_trailing_ws():
    ctx = _ctx("```python\nx=1\n```", "```python\nx=1 \n```")
    assert TrailingCodeWhitespaceCheck().score("```python\nx=1 \n```", ctx) < 1.0


def test_answers_question():
    ctx = _ctx("x", "42", prompt="what is 6*7?")
    assert QuestionAnswerCheck().score("42", ctx) == 1.0
    assert QuestionAnswerCheck().score("why?", ctx) == 0.0


def test_no_turn_markers():
    ctx = _ctx("x", "x")
    assert TurnMarkerCheck().score("User: hi", ctx) == 0.0
    assert TurnMarkerCheck().score("hello", ctx) == 1.0


def test_greeting_match():
    ctx = _ctx("hi there", "hello!", prompt="hello")
    assert GreetingCheck().score("hello!", ctx) == 1.0
    assert GreetingCheck().score("42", ctx) == 0.0


def test_bullet_answer():
    ctx = _ctx("- a\n- b", "- a\n- b", prompt="list the items")
    assert BulletAnswerCheck().score("- a\n- b", ctx) == 1.0
    assert BulletAnswerCheck().score("just text", ctx) == 0.0


def test_word_count():
    ctx = _ctx("one two three", "one two three")
    assert WordCountCheck().score("one two three", ctx) == 1.0
    assert WordCountCheck().score("one", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "what is 2+2?", "baseline_output": "4",
         "candidate_output": "4"},
        {"id": "b", "prompt": "what is 2+2?", "baseline_output": "4",
         "candidate_output": "I don't know?"},
    ])
    result = evaluate(corpus, dimensions=("answers_question", "word_count", "indent_width"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["answers_question"].losses == 1
    assert dims["word_count"].losses == 1
