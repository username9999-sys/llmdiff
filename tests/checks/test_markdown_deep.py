"""Tests for markdown_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.markdown_deep import (
    MdAdmonitionCountCheck,
    MdAlertCountCheck,
    MdBlockquoteNestCountCheck,
    MdCodeFenceLangCountCheck,
    MdCollapsedRefCountCheck,
    MdFootnoteDefCountCheck,
    MdFootnoteUseCountCheck,
    MdHeadingATXCountCheck,
    MdHeadingSetextCountCheck,
    MdHrThematicCountCheck,
    MdHtmlBlockCountCheck,
    MdImageMdCountCheck,
    MdInlineCodeCountCheck,
    MdLinkAutolinkCountCheck,
    MdRefLinkDefCountCheck,
    MdRefLinkUseCountCheck,
    MdTablePipeCountCheck,
    MdTableSeparatorRowCountCheck,
    MdTaskDoneCountCheck,
    MdTaskTodoCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

MD = (
    "# Title\n\n"
    "Setext\n======\n\n"
    "- [x] done\n- [ ] todo\n\n"
    "::: warning\ncareful\n:::\n\n"
    "> [!NOTE]\nalert\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
    "text[^1]\n\n[^1]: note\n\n"
    "[full][ref] and [collapsed][]\n\n[ref]: https://x.com\n[collapsed]: https://y.com\n\n"
    "```python\nx = `inline`\n```\n\n"
    "<https://auto.link> and <a@b.com>\n\n"
    "![img](i.png)\n\n<div>html</div>\n\n---\n\n>> nested quote\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_ref_links():
    ctx = _ctx(MD, MD)
    assert MdRefLinkDefCountCheck().score(MD, ctx) == 1.0
    assert MdRefLinkUseCountCheck().score(MD, ctx) == 1.0
    assert MdCollapsedRefCountCheck().score(MD, ctx) == 1.0


def test_tasks_admonitions_alerts():
    ctx = _ctx(MD, MD)
    assert MdTaskDoneCountCheck().score(MD, ctx) == 1.0
    assert MdTaskTodoCountCheck().score(MD, ctx) == 1.0
    assert MdAdmonitionCountCheck().score(MD, ctx) == 1.0
    assert MdAlertCountCheck().score(MD, ctx) == 1.0


def test_tables_footnotes():
    ctx = _ctx(MD, MD)
    assert MdTablePipeCountCheck().score(MD, ctx) == 1.0
    assert MdTableSeparatorRowCountCheck().score(MD, ctx) == 1.0
    assert MdFootnoteUseCountCheck().score(MD, ctx) == 1.0
    assert MdFootnoteDefCountCheck().score(MD, ctx) == 1.0


def test_headings_code_links():
    ctx = _ctx(MD, MD)
    assert MdHeadingATXCountCheck().score(MD, ctx) == 1.0
    assert MdHeadingSetextCountCheck().score(MD, ctx) == 1.0
    assert MdCodeFenceLangCountCheck().score(MD, ctx) == 1.0
    assert MdInlineCodeCountCheck().score(MD, ctx) == 1.0
    assert MdLinkAutolinkCountCheck().score(MD, ctx) == 1.0
    assert MdImageMdCountCheck().score(MD, ctx) == 1.0


def test_misc_blocks():
    ctx = _ctx(MD, MD)
    assert MdHtmlBlockCountCheck().score(MD, ctx) == 1.0
    assert MdHrThematicCountCheck().score(MD, ctx) == 1.0
    assert MdBlockquoteNestCountCheck().score(MD, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": MD,
         "candidate_output": MD},
        {"id": "b", "prompt": "p", "baseline_output": MD,
         "candidate_output": "# Plain\n\ntext"},
    ])
    result = evaluate(corpus, dimensions=("md_atx_count", "md_task_done_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["md_atx_count"].losses == 0
    assert dims["md_task_done_count"].losses == 1
