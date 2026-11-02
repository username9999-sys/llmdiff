"""Tests for format/structure dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.format import (
    CsvShapeCheck,
    HtmlTagBalanceCheck,
    LatexBalanceCheck,
    LinkCountCheck,
    MarkdownHeadingsCheck,
    MarkdownListsCheck,
    MarkdownTablesCheck,
    XmlValidCheck,
    YamlValidCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_headings_match():
    ctx = _ctx("# A\n## B", "# A\n## B")
    assert MarkdownHeadingsCheck().score("# A\n## B", ctx) == 1.0
    assert MarkdownHeadingsCheck().score("# A", ctx) < 1.0


def test_lists_match():
    ctx = _ctx("- a\n- b", "- a\n- b")
    assert MarkdownListsCheck().score("- a\n- b", ctx) == 1.0


def test_tables_presence():
    ctx = _ctx("| a | b |\n|---|---|", "| a | b |\n|---|---|")
    assert MarkdownTablesCheck().score("| a | b |\n|---|---|", ctx) == 1.0
    assert MarkdownTablesCheck().score("no table", ctx) == 0.0


def test_links():
    ctx = _ctx("see https://a.com", "see https://a.com and https://b.com")
    assert LinkCountCheck().score("see https://a.com", ctx) == 1.0
    assert LinkCountCheck().score("see https://a.com and https://b.com", ctx) < 1.0


def test_latex_balance():
    ctx = _ctx("$x$", "$x$")
    assert LatexBalanceCheck().score("$x$", ctx) == 1.0
    assert LatexBalanceCheck().score("$x", ctx) == 0.0
    assert LatexBalanceCheck().score("\\$ escaped", ctx) == 1.0


def test_xml_valid():
    ctx = _ctx("<a/>", "<a><b/></a>")
    assert XmlValidCheck().score("<a><b/></a>", ctx) == 1.0
    assert XmlValidCheck().score("<a><b></a>", ctx) == 0.0
    assert XmlValidCheck().score("not xml", ctx) == 0.0


def test_html_balance():
    ctx = _ctx("<div></div>", "<div><p>hi</p></div>")
    assert HtmlTagBalanceCheck().score("<div><p>hi</p></div>", ctx) == 1.0
    assert HtmlTagBalanceCheck().score("<div><p>hi</div>", ctx) < 1.0


def test_yaml_valid():
    ctx = _ctx("a: 1", "a: 1\nb: 2")
    assert YamlValidCheck().score("a: 1\nb: 2", ctx) == 1.0


def test_csv_shape():
    ctx = _ctx("a,b,c\n1,2,3", "a,b,c\n1,2,3")
    assert CsvShapeCheck().score("a,b,c\n1,2,3", ctx) == 1.0
    assert CsvShapeCheck().score("a,b\n1,2", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "# Title\n- x\n- y",
         "candidate_output": "# Title\n- x\n- y"},
        {"id": "b", "prompt": "p", "baseline_output": "# Title\n- x\n- y",
         "candidate_output": "no headings no list"},
    ])
    result = evaluate(corpus, dimensions=("md_headings", "md_lists"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["md_headings"].losses == 1
    assert dims["md_lists"].losses == 1
