"""Tests for text_patterns_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.text_patterns_deep import (
    BoldPhraseCountCheck,
    BulletPointRatioCheck,
    BuzzwordCountCheck,
    ColonListIntroCountCheck,
    EmDashUsageCountCheck,
    FillerPhraseCountCheck,
    FirstPersonPluralCountCheck,
    HedgePhraseCountCheck,
    IntensifierStackCountCheck,
    ItalicPhraseCountCheck,
    LinkInTextCountCheck,
    NominalizationCountCheck,
    NumberedListRatioCheck,
    ParentheticalAsideCountCheck,
    PassiveVoiceDeepCountCheck,
    QuoteBlockCountCheck,
    RhetoricalQuestionCountCheck,
    SemicolonUsageCountCheck,
    SuperlativeCountCheck,
    WeaselWordCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_filler_hedge_buzzword():
    ctx = _ctx("basically it is simple", "basically it is simple")
    assert FillerPhraseCountCheck().score("basically it is simple", ctx) == 1.0
    assert FillerPhraseCountCheck().score("plain text", ctx) < 1.0
    hctx = _ctx("maybe perhaps", "maybe perhaps")
    assert HedgePhraseCountCheck().score("maybe perhaps", hctx) == 1.0
    bctx = _ctx("leverage synergy", "leverage synergy")
    assert BuzzwordCountCheck().score("leverage synergy", bctx) == 1.0


def test_superlative_weasel():
    ctx = _ctx("the best thing", "the best thing")
    assert SuperlativeCountCheck().score("the best thing", ctx) == 1.0
    wctx = _ctx("up to 10 items", "up to 10 items")
    assert WeaselWordCountCheck().score("up to 10 items", wctx) == 1.0


def test_we_count_passive_nominal():
    ctx = _ctx("we think our plan", "we think our plan")
    assert FirstPersonPluralCountCheck().score("we think our plan", ctx) == 1.0
    pctx = _ctx("the ball was thrown", "the ball was thrown")
    assert PassiveVoiceDeepCountCheck().score("the ball was thrown", pctx) == 1.0
    nctx = _ctx("the implementation", "the implementation")
    assert NominalizationCountCheck().score("the implementation", nctx) == 1.0


def test_intensifier_stack_rhetorical():
    ctx = _ctx("very really good", "very really good")
    assert IntensifierStackCountCheck().score("very really good", ctx) == 1.0
    rctx = _ctx("what could go wrong?", "what could go wrong?")
    assert RhetoricalQuestionCountCheck().score("what could go wrong?", rctx) == 1.0


def test_punct_usage():
    ctx = _ctx("a — b; c", "a — b; c")
    assert EmDashUsageCountCheck().score("a — b; c", ctx) == 1.0
    assert SemicolonUsageCountCheck().score("a — b; c", ctx) == 1.0
    pctx = _ctx("this (a long aside) here", "this (a long aside) here")
    assert ParentheticalAsideCountCheck().score("this (a long aside) here", pctx) == 1.0
    cctx = _ctx("items:\n- a", "items:\n- a")
    assert ColonListIntroCountCheck().score("items:\n- a", cctx) == 1.0


def test_list_ratios():
    ctx = _ctx("- a\n- b\ntext", "- a\n- b\ntext")
    assert BulletPointRatioCheck().score("- a\n- b\ntext", ctx) == 1.0
    nctx = _ctx("1. a\n2. b\ntext", "1. a\n2. b\ntext")
    assert NumberedListRatioCheck().score("1. a\n2. b\ntext", nctx) == 1.0


def test_quote_link_bold_italic():
    ctx = _ctx("> quoted", "> quoted")
    assert QuoteBlockCountCheck().score("> quoted", ctx) == 1.0
    lctx = _ctx("[a](http://x)", "[a](http://x)")
    assert LinkInTextCountCheck().score("[a](http://x)", lctx) == 1.0
    bctx = _ctx("**bold**", "**bold**")
    assert BoldPhraseCountCheck().score("**bold**", bctx) == 1.0
    ictx = _ctx("*italic*", "*italic*")
    assert ItalicPhraseCountCheck().score("*italic*", ictx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "basically the best",
         "candidate_output": "basically the best"},
        {"id": "b", "prompt": "p", "baseline_output": "basically the best",
         "candidate_output": "plain"},
    ])
    result = evaluate(corpus, dimensions=("filler_count", "superlative_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["filler_count"].losses == 1
    assert dims["superlative_count"].losses == 1
