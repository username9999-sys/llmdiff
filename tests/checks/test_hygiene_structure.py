"""Tests for hygiene and structure_md dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.hygiene import (
    CapitalizationCheck,
    DoubleSpaceCheck,
    PunctuationCheck,
    QuoteBalanceCheck,
    TrailingWhitespaceCheck,
    WhitespaceCheck,
)
from llmdiff.checks.structure_md import (
    AvgWordLengthCheck,
    FenceLanguageCheck,
    HeadingHierarchyCheck,
    NestingDepthCheck,
    ParagraphCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_punctuation_match():
    ctx = _ctx("Hello, world!", "Hello, world!")
    assert PunctuationCheck().score("Hello, world!", ctx) == 1.0
    assert PunctuationCheck().score("Hello world", ctx) < 1.0


def test_capitalization():
    ctx = _ctx("Hello World", "hello world")
    assert CapitalizationCheck().score("hello world", ctx) < 1.0
    assert CapitalizationCheck().score("Hello World", ctx) == 1.0


def test_whitespace():
    ctx = _ctx("a b", "a  b")
    assert WhitespaceCheck().score("a  b", ctx) < 1.0


def test_quote_balance():
    ctx = _ctx('"x"', '"x"')
    assert QuoteBalanceCheck().score('"x"', ctx) == 1.0
    assert QuoteBalanceCheck().score('"x', ctx) == 0.0


def test_trailing_ws():
    ctx = _ctx("clean", "clean")
    assert TrailingWhitespaceCheck().score("clean", ctx) == 1.0
    assert TrailingWhitespaceCheck().score("dirty \n", ctx) < 1.0


def test_double_space():
    ctx = _ctx("a b", "a b")
    assert DoubleSpaceCheck().score("a b", ctx) == 1.0
    assert DoubleSpaceCheck().score("a  b", ctx) < 1.0


def test_heading_hierarchy():
    ctx = _ctx("# A\n## B", "# A\n### C")
    assert HeadingHierarchyCheck().score("# A\n### C", ctx) < 1.0
    assert HeadingHierarchyCheck().score("# A\n## B", ctx) == 1.0


def test_fence_language():
    ctx = _ctx("```python\nx=1\n```", "```python\nx=1\n```")
    assert FenceLanguageCheck().score("```python\nx=1\n```", ctx) == 1.0
    assert FenceLanguageCheck().score("```js\nx=1\n```", ctx) < 1.0


def test_nesting_depth():
    ctx = _ctx("- a\n  - b", "- a")
    assert NestingDepthCheck().score("- a", ctx) < 1.0
    assert NestingDepthCheck().score("- a\n  - b", ctx) == 1.0


def test_paragraphs():
    ctx = _ctx("one\n\ntwo", "one\n\ntwo")
    assert ParagraphCountCheck().score("one\n\ntwo", ctx) == 1.0


def test_avg_word_len():
    ctx = _ctx("cat dog", "cat dog")
    assert AvgWordLengthCheck().score("cat dog", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "Hello, World!",
         "candidate_output": "Hello, World!"},
        {"id": "b", "prompt": "p", "baseline_output": "Hello, World!",
         "candidate_output": "hello world"},
    ])
    result = evaluate(corpus, dimensions=("punctuation", "capitalization", "heading_hierarchy"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["punctuation"].losses == 1
    assert dims["capitalization"].losses == 1
