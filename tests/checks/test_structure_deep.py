"""Tests for structure_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.structure_deep import (
    BlockOrderSignatureCheck,
    CodeFenceBalanceCheck,
    DuplicateSectionCheck,
    FirstLineMatchCheck,
    HeadingLevelSequenceCheck,
    HeadingSkipCountCheck,
    IndentLevelDiversityCheck,
    LastLineMatchCheck,
    LineOrderSignatureCheck,
    ListItemIndentConsistencyCheck,
    ListNestingDepthCheck,
    MaxHeadingLevelCheck,
    MinHeadingLevelCheck,
    MonotonicNumberCheck,
    ParagraphOrderSignatureCheck,
    QuoteNestingDepthCheck,
    SectionOrderJaccardCheck,
    SentenceOrderSignatureCheck,
    TableColumnConsistencyCheck,
    WordOrderSignatureCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

DOC = (
    "# Title\n\nIntro para.\n\n## Section A\n\n- item 1\n- item 2\n\n"
    "## Section B\n\n```\ncode\n```\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_heading_levels():
    ctx = _ctx(DOC, DOC)
    assert MaxHeadingLevelCheck().score(DOC, ctx) == 1.0
    assert MinHeadingLevelCheck().score(DOC, ctx) == 1.0
    assert HeadingLevelSequenceCheck().score(DOC, ctx) == 1.0


def test_heading_skips():
    ctx = _ctx("# A\n### B", "# A\n### B")
    assert HeadingSkipCountCheck().score("# A\n### B", ctx) == 1.0
    assert HeadingSkipCountCheck().score("# A\n## B", ctx) < 1.0


def test_list_nesting():
    ctx = _ctx("- a\n  - b\n    - c", "- a\n  - b\n    - c")
    assert ListNestingDepthCheck().score("- a\n  - b\n    - c", ctx) == 1.0
    assert ListItemIndentConsistencyCheck().score("- a\n  - b\n    - c", ctx) == 1.0


def test_fence_balance():
    ctx = _ctx("```\nx\n```", "```\nx\n```")
    assert CodeFenceBalanceCheck().score("```\nx\n```", ctx) == 1.0
    assert CodeFenceBalanceCheck().score("```\nx", ctx) == 0.0


def test_quote_nesting():
    ctx = _ctx("> a\n>> b", "> a\n>> b")
    assert QuoteNestingDepthCheck().score("> a\n>> b", ctx) == 1.0


def test_table_consistency():
    tbl = "| a | b |\n|---|---|\n| 1 | 2 |"
    ctx = _ctx(tbl, tbl)
    assert TableColumnConsistencyCheck().score(tbl, ctx) == 1.0


def test_order_signatures():
    ctx = _ctx(DOC, DOC)
    assert ParagraphOrderSignatureCheck().score(DOC, ctx) == 1.0
    assert SentenceOrderSignatureCheck().score(DOC, ctx) == 1.0
    assert BlockOrderSignatureCheck().score(DOC, ctx) == 1.0
    assert LineOrderSignatureCheck().score(DOC, ctx) > 0.9
    assert WordOrderSignatureCheck().score(DOC, ctx) > 0.9


def test_first_last_line():
    ctx = _ctx(DOC, DOC)
    assert FirstLineMatchCheck().score(DOC, ctx) == 1.0
    assert LastLineMatchCheck().score(DOC, ctx) == 1.0
    assert FirstLineMatchCheck().score("other\ntext", ctx) == 0.0


def test_section_jaccard_and_dups():
    ctx = _ctx(DOC, DOC)
    assert SectionOrderJaccardCheck().score(DOC, ctx) == 1.0
    dup = "# A\n## B\n# A"
    dctx = _ctx(dup, dup)
    assert DuplicateSectionCheck().score(dup, dctx) == 1.0
    assert DuplicateSectionCheck().score(DOC, dctx) < 1.0


def test_monotonic_numbers():
    ctx = _ctx("1. a\n2. b\n3. c", "1. a\n2. b\n3. c")
    assert MonotonicNumberCheck().score("1. a\n2. b\n3. c", ctx) == 1.0
    assert MonotonicNumberCheck().score("3. a\n1. b", ctx) == 0.0


def test_indent_diversity():
    ctx = _ctx("a\n  b\n    c", "a\n  b\n    c")
    assert IndentLevelDiversityCheck().score("a\n  b\n    c", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": DOC,
         "candidate_output": DOC},
        {"id": "b", "prompt": "p", "baseline_output": DOC,
         "candidate_output": "# Other\n\ntext"},
    ])
    result = evaluate(corpus, dimensions=("section_jaccard", "max_heading_level"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["section_jaccard"].losses == 1
    assert dims["max_heading_level"].losses == 1
