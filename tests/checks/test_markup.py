"""Tests for markup dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.markup import (
    BlockquoteCountCheck,
    BoldCountCheck,
    CodeBlockCountMatchCheck,
    DefinitionListCountCheck,
    FootnoteCountCheck,
    HeadingCountCheck,
    HeadingLevelDistributionCheck,
    HrCountCheck,
    HtmlTagCountCheck,
    ImageCountCheck,
    InlineCodeCountCheck,
    ItalicCountCheck,
    LatexBlockCountCheck,
    LatexInlineCountCheck,
    LinkCountMatchCheck,
    OrderedListCountCheck,
    TableColumnCountCheck,
    TableRowCountCheck,
    TaskListItemCountCheck,
    UnorderedListCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_heading_count():
    ctx = _ctx("# A\n## B", "# A\n## B")
    assert HeadingCountCheck().score("# A\n## B", ctx) == 1.0
    assert HeadingCountCheck().score("# A", ctx) < 1.0


def test_heading_levels():
    ctx = _ctx("# A\n## B\n### C", "# A\n## B\n### C")
    assert HeadingLevelDistributionCheck().score("# A\n## B\n### C", ctx) == 1.0


def test_bold_count():
    ctx = _ctx("**bold** text", "**bold** text")
    assert BoldCountCheck().score("**bold** text", ctx) == 1.0
    assert BoldCountCheck().score("plain text", ctx) < 1.0


def test_italic_count():
    ctx = _ctx("*italic* text", "*italic* text")
    assert ItalicCountCheck().score("*italic* text", ctx) == 1.0
    assert ItalicCountCheck().score("plain text", ctx) < 1.0


def test_inline_code():
    ctx = _ctx("use `x` here", "use `x` here")
    assert InlineCodeCountCheck().score("use `x` here", ctx) == 1.0
    assert InlineCodeCountCheck().score("use x here", ctx) < 1.0


def test_code_block_count():
    ctx = _ctx("```\na\n```", "```\na\n```")
    assert CodeBlockCountMatchCheck().score("```\na\n```", ctx) == 1.0
    assert CodeBlockCountMatchCheck().score("no code", ctx) < 1.0


def test_link_count():
    ctx = _ctx("[a](http://x) [b](http://y)", "[a](http://x) [b](http://y)")
    assert LinkCountMatchCheck().score("[a](http://x) [b](http://y)", ctx) == 1.0
    assert LinkCountMatchCheck().score("[a](http://x)", ctx) < 1.0


def test_image_count():
    ctx = _ctx("![alt](img.png)", "![alt](img.png)")
    assert ImageCountCheck().score("![alt](img.png)", ctx) == 1.0
    assert ImageCountCheck().score("no image", ctx) < 1.0


def test_blockquote_count():
    ctx = _ctx("> quote", "> quote")
    assert BlockquoteCountCheck().score("> quote", ctx) == 1.0
    assert BlockquoteCountCheck().score("no quote", ctx) < 1.0


def test_hr_count():
    ctx = _ctx("---", "---")
    assert HrCountCheck().score("---", ctx) == 1.0
    assert HrCountCheck().score("text", ctx) < 1.0


def test_ordered_list():
    ctx = _ctx("1. a\n2. b", "1. a\n2. b")
    assert OrderedListCountCheck().score("1. a\n2. b", ctx) == 1.0
    assert OrderedListCountCheck().score("1. a", ctx) < 1.0


def test_unordered_list():
    ctx = _ctx("- a\n- b", "- a\n- b")
    assert UnorderedListCountCheck().score("- a\n- b", ctx) == 1.0
    assert UnorderedListCountCheck().score("- a", ctx) < 1.0


def test_task_list():
    ctx = _ctx("- [x] done\n- [ ] todo", "- [x] done\n- [ ] todo")
    assert TaskListItemCountCheck().score("- [x] done\n- [ ] todo", ctx) == 1.0
    assert TaskListItemCountCheck().score("- [x] done", ctx) < 1.0


def test_table_rows():
    tbl = "| a | b |\n|---|---|\n| 1 | 2 |"
    ctx = _ctx(tbl, tbl)
    assert TableRowCountCheck().score(tbl, ctx) == 1.0


def test_table_cols():
    tbl = "| a | b | c |\n|---|---|---|"
    ctx = _ctx(tbl, tbl)
    assert TableColumnCountCheck().score(tbl, ctx) == 1.0
    assert TableColumnCountCheck().score("| a | b |\n|---|---|", ctx) < 1.0


def test_html_tags():
    ctx = _ctx("<b>x</b>", "<b>x</b>")
    assert HtmlTagCountCheck().score("<b>x</b>", ctx) == 1.0
    assert HtmlTagCountCheck().score("x", ctx) < 1.0


def test_latex_inline():
    ctx = _ctx("$x^2$ math", "$x^2$ math")
    assert LatexInlineCountCheck().score("$x^2$ math", ctx) == 1.0
    assert LatexInlineCountCheck().score("x math", ctx) < 1.0


def test_latex_block():
    ctx = _ctx("$$x^2$$", "$$x^2$$")
    assert LatexBlockCountCheck().score("$$x^2$$", ctx) == 1.0


def test_footnote():
    ctx = _ctx("text[^1]", "text[^1]")
    assert FootnoteCountCheck().score("text[^1]", ctx) == 1.0
    assert FootnoteCountCheck().score("text", ctx) < 1.0


def test_deflist():
    ctx = _ctx("term\n: definition", "term\n: definition")
    assert DefinitionListCountCheck().score("term\n: definition", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "# Title\n**bold**",
         "candidate_output": "# Title\n**bold**"},
        {"id": "b", "prompt": "p", "baseline_output": "# Title\n**bold**",
         "candidate_output": "plain text"},
    ])
    result = evaluate(corpus, dimensions=("heading_count", "bold_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["heading_count"].losses == 1
    assert dims["bold_count"].losses == 1
