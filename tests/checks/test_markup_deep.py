"""Tests for markup_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.markup_deep import (
    AbbrTagCountCheck,
    DetailsBlockCountCheck,
    DisplayMathCountCheck,
    FootnoteDefCountCheck,
    FootnoteRefCountCheck,
    HtmlAttrCountCheck,
    HtmlClassCountCheck,
    HtmlIdCountCheck,
    IframeEmbedCountCheck,
    ImgAltCountCheck,
    InlineMathCountCheck,
    KbdKeyCountCheck,
    LinkRelCountCheck,
    LinkTargetCountCheck,
    MarkHighlightCountCheck,
    StrikethroughCountCheck,
    SubscriptCountCheck,
    SummaryTagCountCheck,
    SuperscriptCountCheck,
    VideoEmbedCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_strike_sub_sup_mark():
    ctx = _ctx("~~del~~", "~~del~~")
    assert StrikethroughCountCheck().score("~~del~~", ctx) == 1.0
    assert StrikethroughCountCheck().score("plain", ctx) < 1.0
    sctx = _ctx("H~2~O", "H~2~O")
    assert SubscriptCountCheck().score("H~2~O", sctx) == 1.0
    pctx = _ctx("x^2^", "x^2^")
    assert SuperscriptCountCheck().score("x^2^", pctx) == 1.0
    mctx = _ctx("==hl==", "==hl==")
    assert MarkHighlightCountCheck().score("==hl==", mctx) == 1.0


def test_kbd_math():
    kctx = _ctx("<kbd>Ctrl</kbd>", "<kbd>Ctrl</kbd>")
    assert KbdKeyCountCheck().score("<kbd>Ctrl</kbd>", kctx) == 1.0
    ictx = _ctx("$x^2$", "$x^2$")
    assert InlineMathCountCheck().score("$x^2$", ictx) == 1.0
    dctx = _ctx("$$x^2$$", "$$x^2$$")
    assert DisplayMathCountCheck().score("$$x^2$$", dctx) == 1.0


def test_footnotes():
    ctx = _ctx("text[^1]\n\n[^1]: note", "text[^1]\n\n[^1]: note")
    assert FootnoteRefCountCheck().score("text[^1]", ctx) == 1.0
    assert FootnoteDefCountCheck().score("[^1]: note", ctx) == 1.0


def test_html_tags():
    actx = _ctx("<abbr title='x'>A</abbr>", "<abbr title='x'>A</abbr>")
    assert AbbrTagCountCheck().score("<abbr title='x'>A</abbr>", actx) == 1.0
    html = "<details><summary>S</summary>body</details>"
    dctx = _ctx(html, html)
    assert DetailsBlockCountCheck().score("<details>", dctx) == 1.0
    assert SummaryTagCountCheck().score("<summary>S</summary>", dctx) == 1.0


def test_html_attrs():
    html = '<div class="a b" id="x">t</div>'
    ctx = _ctx(html, html)
    assert HtmlAttrCountCheck().score(html, ctx) == 1.0
    assert HtmlClassCountCheck().score('<div class="a b">t</div>', ctx) == 1.0
    assert HtmlIdCountCheck().score('<div id="x">t</div>', ctx) == 1.0


def test_img_link_attrs():
    ictx = _ctx('<img src="a.png" alt="pic">', '<img src="a.png" alt="pic">')
    assert ImgAltCountCheck().score('<img src="a.png" alt="pic">', ictx) == 1.0
    assert ImgAltCountCheck().score('<img src="a.png">', ictx) < 1.0
    rctx = _ctx('<a href="x" rel="noopener">l</a>', '<a href="x" rel="noopener">l</a>')
    assert LinkRelCountCheck().score('<a href="x" rel="noopener">l</a>', rctx) == 1.0
    tctx = _ctx('<a href="x" target="_blank">l</a>', '<a href="x" target="_blank">l</a>')
    assert LinkTargetCountCheck().score('<a href="x" target="_blank">l</a>', tctx) == 1.0


def test_embeds():
    yt = '<iframe src="https://youtube.com/embed/x"></iframe>'
    vctx = _ctx(yt, yt)
    assert VideoEmbedCountCheck().score(yt, vctx) == 1.0
    other = '<iframe src="https://x.com"></iframe>'
    assert IframeEmbedCountCheck().score(other, vctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "~~old~~ ==new==",
         "candidate_output": "~~old~~ ==new=="},
        {"id": "b", "prompt": "p", "baseline_output": "~~old~~ ==new==",
         "candidate_output": "plain text"},
    ])
    result = evaluate(corpus, dimensions=("strike_count", "mark_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["strike_count"].losses == 1
    assert dims["mark_count"].losses == 1
