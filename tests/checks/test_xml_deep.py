"""Tests for xml_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.xml_deep import (
    HtmlBodyPresenceCheck,
    HtmlDoctypePresenceCheck,
    HtmlHeadPresenceCheck,
    HtmlMetaCountCheck,
    HtmlScriptCountCheck,
    HtmlSemanticTagCountCheck,
    HtmlStyleCountCheck,
    XmlAttrCountCheck,
    XmlAttrNameJaccardCheck,
    XmlCdataCountCheck,
    XmlCommentCountCheck,
    XmlEntityCountCheck,
    XmlMaxDepthCheck,
    XmlParseOkCheck,
    XmlPiCountCheck,
    XmlRootTagMatchCheck,
    XmlSelfClosingCountCheck,
    XmlTagSetJaccardCheck,
    XmlTextContentLenCheck,
    XmlVoidTagCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

XML = (
    '<?xml version="1.0"?>\n'
    "<!-- comment -->\n"
    '<root a="1" b="2">\n'
    "  <child>&amp; &lt;</child>\n"
    "  <empty/>\n"
    "  <![CDATA[raw <data>]]>\n"
    "</root>\n"
)
HTML = (
    "<!DOCTYPE html>\n"
    '<html><head><meta charset="utf-8"><style>x{}</style></head>'
    "<body><header>h</header><script>s()</script><br></body></html>\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_xml_parse_root_tags():
    ctx = _ctx(XML, XML)
    assert XmlParseOkCheck().score(XML, ctx) == 1.0
    assert XmlRootTagMatchCheck().score(XML, ctx) == 1.0
    assert XmlRootTagMatchCheck().score("<other/>", ctx) == 0.0
    assert XmlTagSetJaccardCheck().score(XML, ctx) == 1.0
    assert XmlMaxDepthCheck().score(XML, ctx) == 1.0


def test_xml_attrs_text_entities():
    ctx = _ctx(XML, XML)
    assert XmlAttrCountCheck().score(XML, ctx) == 1.0
    assert XmlAttrNameJaccardCheck().score(XML, ctx) == 1.0
    assert XmlTextContentLenCheck().score(XML, ctx) == 1.0
    assert XmlEntityCountCheck().score(XML, ctx) == 1.0
    assert XmlCdataCountCheck().score(XML, ctx) == 1.0
    assert XmlCommentCountCheck().score(XML, ctx) == 1.0
    assert XmlPiCountCheck().score(XML, ctx) == 1.0
    assert XmlSelfClosingCountCheck().score(XML, ctx) == 1.0
    vctx = _ctx(HTML, HTML)
    assert XmlVoidTagCountCheck().score(HTML, vctx) == 1.0
    assert XmlVoidTagCountCheck().score(HTML, ctx) < 1.0


def test_html_structure():
    ctx = _ctx(HTML, HTML)
    assert HtmlDoctypePresenceCheck().score(HTML, ctx) == 1.0
    assert HtmlHeadPresenceCheck().score(HTML, ctx) == 1.0
    assert HtmlBodyPresenceCheck().score(HTML, ctx) == 1.0
    assert HtmlScriptCountCheck().score(HTML, ctx) == 1.0
    assert HtmlStyleCountCheck().score(HTML, ctx) == 1.0
    assert HtmlMetaCountCheck().score(HTML, ctx) == 1.0
    assert HtmlSemanticTagCountCheck().score(HTML, ctx) == 1.0
    assert HtmlDoctypePresenceCheck().score("<html></html>", ctx) == 0.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": XML,
         "candidate_output": XML},
        {"id": "b", "prompt": "p", "baseline_output": XML,
         "candidate_output": "<other/>"},
    ])
    result = evaluate(corpus, dimensions=("xml_root_tag", "xml_tag_jaccard"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["xml_root_tag"].losses == 1
    assert dims["xml_tag_jaccard"].losses == 1
