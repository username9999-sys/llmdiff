"""Tests for format_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.format_deep import (
    AngleBalanceCheck,
    AsteriskBalanceCheck,
    BacktickBalanceCheck,
    BraceBalanceDeepCheck,
    BracketBalanceCheck,
    BracketSpacingCheck,
    ColonSpacingCheck,
    CommaSpacingCheck,
    DollarBalanceCheck,
    DoubleSpaceCountCheck,
    IndentConsistencyCheck,
    IndentWidthMatchCheck,
    MixedIndentCheck,
    PipeBalanceCheck,
    QuotePairBalanceCheck,
    SpaceAfterPunctCountCheck,
    SpaceBeforePunctCountCheck,
    SquareBalanceCheck,
    TrailingWhitespaceLinesCheck,
    UnderscoreBalanceCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_indent_width():
    ctx = _ctx("def f():\n    return 1", "def f():\n    return 1")
    assert IndentWidthMatchCheck().score("def f():\n    return 1", ctx) == 1.0
    assert IndentWidthMatchCheck().score("def f():\n  return 1", ctx) < 1.0


def test_indent_consistency():
    ctx = _ctx("a\n  b\n  c", "a\n  b\n  c")
    assert IndentConsistencyCheck().score("a\n  b\n  c", ctx) == 1.0


def test_mixed_indent():
    ctx = _ctx("a\n  b", "a\n  b")
    assert MixedIndentCheck().score("a\n  b", ctx) == 1.0
    mixed = "a\n  b\n\tc"
    mctx = _ctx(mixed, mixed)
    assert MixedIndentCheck().score(mixed, mctx) == 1.0
    assert MixedIndentCheck().score("a\n  b", mctx) == 0.0


def test_trailing_ws_lines():
    ctx = _ctx("a\nb", "a\nb")
    assert TrailingWhitespaceLinesCheck().score("a \nb", ctx) < 1.0
    assert TrailingWhitespaceLinesCheck().score("a\nb", ctx) == 1.0


def test_double_space_count():
    ctx = _ctx("a  b", "a  b")
    assert DoubleSpaceCountCheck().score("a  b", ctx) == 1.0
    assert DoubleSpaceCountCheck().score("a b", ctx) < 1.0


def test_space_before_after_punct():
    ctx = _ctx("a , b", "a , b")
    assert SpaceBeforePunctCountCheck().score("a , b", ctx) == 1.0
    assert SpaceAfterPunctCountCheck().score("a,b", ctx) < 1.0


def test_bracket_spacing():
    ctx = _ctx("f( x )", "f( x )")
    assert BracketSpacingCheck().score("f( x )", ctx) == 1.0


def test_comma_colon_spacing():
    ctx = _ctx("a, b: c", "a, b: c")
    assert CommaSpacingCheck().score("a, b", ctx) == 1.0
    assert ColonSpacingCheck().score("a: b", ctx) == 1.0


def test_quote_balance():
    ctx = _ctx('"a"', '"a"')
    assert QuotePairBalanceCheck().score('"a"', ctx) == 1.0
    unbalanced = '"a'
    uctx = _ctx(unbalanced, unbalanced)
    assert QuotePairBalanceCheck().score(unbalanced, uctx) == 1.0
    assert QuotePairBalanceCheck().score('"a"', uctx) == 0.0


def test_bracket_balances():
    ctx = _ctx("(a) [b] {c} <d>", "(a) [b] {c} <d>")
    assert BracketBalanceCheck().score("(a)", ctx) == 1.0
    assert SquareBalanceCheck().score("[b]", ctx) == 1.0
    assert BraceBalanceDeepCheck().score("{c}", ctx) == 1.0
    assert AngleBalanceCheck().score("<d>", ctx) == 1.0


def test_symbol_balances():
    ctx = _ctx("`a` *b* _c_ |d| $e$", "`a` *b* _c_ |d| $e$")
    assert BacktickBalanceCheck().score("`a`", ctx) == 1.0
    assert AsteriskBalanceCheck().score("*b*", ctx) == 1.0
    assert UnderscoreBalanceCheck().score("_c_", ctx) == 1.0
    assert PipeBalanceCheck().score("|d|", ctx) == 1.0
    assert DollarBalanceCheck().score("$e$", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "f( a, b )",
         "candidate_output": "f( a, b )"},
        {"id": "b", "prompt": "p", "baseline_output": "f( a, b )",
         "candidate_output": "f(a,b)"},
    ])
    result = evaluate(corpus, dimensions=("comma_spacing", "paren_balance"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["comma_spacing"].losses == 1
    assert dims["paren_balance"].losses == 0
