"""Tests for formatting dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.formatting import (
    BlankLineRunCheck,
    EndsWithNewlineCheck,
    LeadingBlankLinesCheck,
    LineCountCheck,
    TabCharCheck,
    TrailingBlankLinesCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_leading_blank():
    ctx = _ctx("\n\nhello", "\n\nhello")
    assert LeadingBlankLinesCheck().score("\n\nhello", ctx) == 1.0
    assert LeadingBlankLinesCheck().score("hello", ctx) < 1.0


def test_trailing_blank():
    ctx = _ctx("hello\n\n", "hello\n\n")
    assert TrailingBlankLinesCheck().score("hello\n\n", ctx) == 1.0


def test_ends_newline():
    ctx = _ctx("hello\n", "hello\n")
    assert EndsWithNewlineCheck().score("hello\n", ctx) == 1.0
    assert EndsWithNewlineCheck().score("hello", ctx) == 0.0


def test_tabs():
    ctx = _ctx("a\tb", "a b")
    assert TabCharCheck().score("a b", ctx) == 0.0
    assert TabCharCheck().score("a\tb", ctx) == 1.0


def test_blank_run():
    ctx = _ctx("a\n\n\nb", "a\n\n\nb")
    assert BlankLineRunCheck().score("a\n\n\nb", ctx) == 1.0
    assert BlankLineRunCheck().score("a\nb", ctx) < 1.0


def test_line_count():
    ctx = _ctx("a\nb\nc", "a\nb\nc")
    assert LineCountCheck().score("a\nb\nc", ctx) == 1.0
    assert LineCountCheck().score("a", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "line1\nline2\n",
         "candidate_output": "line1\nline2\n"},
        {"id": "b", "prompt": "p", "baseline_output": "line1\nline2\n",
         "candidate_output": "line1"},
    ])
    result = evaluate(corpus, dimensions=("line_count", "ends_newline"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["line_count"].losses == 1
    assert dims["ends_newline"].losses == 1
