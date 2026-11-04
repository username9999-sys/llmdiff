"""Tests for patterns dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.patterns import (
    AllCapsLineCountCheck,
    BoilerplateScoreCheck,
    BulletHeadingCountCheck,
    ColonIntroCountCheck,
    CtaCountCheck,
    DisclaimerCountCheck,
    EllipsisLineCountCheck,
    EmDashCountCheck,
    HashtagCountCheck,
    ListiclePatternCountCheck,
    MentionCountCheck,
    MostCommonPhraseCheck,
    NumberedHeadingCountCheck,
    ParentheticalCountCheck,
    QuoteAttributionCountCheck,
    RepeatedLineCountCheck,
    RepeatedPhraseCountCheck,
    SemicolonPhraseCountCheck,
    SentenceStarterDiversityCheck,
    TransitionWordRatioCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_repeated_phrase_count():
    ctx = _ctx("the cat the cat the cat", "the cat the cat the cat")
    assert RepeatedPhraseCountCheck().score("the cat the cat the cat", ctx) == 1.0
    assert RepeatedPhraseCountCheck().score("a b c d e", ctx) < 1.0


def test_top_phrase():
    ctx = _ctx("a b a b a b", "a b a b a b")
    assert MostCommonPhraseCheck().score("a b a b a b", ctx) == 1.0


def test_starter_diversity():
    ctx = _ctx("The cat. A dog. The bird.", "The cat. A dog. The bird.")
    assert SentenceStarterDiversityCheck().score("The cat. A dog. The bird.", ctx) == 1.0


def test_transition_ratio():
    ctx = _ctx("First, x. However, y.", "First, x. However, y.")
    assert TransitionWordRatioCheck().score("First, x. However, y.", ctx) == 1.0


def test_listicle_count():
    ctx = _ctx("5 ways to code", "5 ways to code")
    assert ListiclePatternCountCheck().score("5 ways to code", ctx) == 1.0
    assert ListiclePatternCountCheck().score("no listicle", ctx) < 1.0


def test_numbered_bullet_headings():
    ctx = _ctx("## 1. Intro", "## 1. Intro")
    assert NumberedHeadingCountCheck().score("## 1. Intro", ctx) == 1.0
    bctx = _ctx("## - item", "## - item")
    assert BulletHeadingCountCheck().score("## - item", bctx) == 1.0


def test_cta_count():
    ctx = _ctx("Click here now", "Click here now")
    assert CtaCountCheck().score("Click here now", ctx) == 1.0
    assert CtaCountCheck().score("nothing", ctx) < 1.0


def test_disclaimer_count():
    ctx = _ctx("Disclaimer: not advice", "Disclaimer: not advice")
    assert DisclaimerCountCheck().score("Disclaimer: not advice", ctx) == 1.0


def test_parenthetical_count():
    ctx = _ctx("this (a note) here", "this (a note) here")
    assert ParentheticalCountCheck().score("this (a note) here", ctx) == 1.0


def test_emdash_count():
    ctx = _ctx("a — b — c", "a — b — c")
    assert EmDashCountCheck().score("a — b — c", ctx) == 1.0


def test_semicolon_phrase_count():
    ctx = _ctx("one; two. three", "one; two. three")
    assert SemicolonPhraseCountCheck().score("one; two. three", ctx) == 1.0


def test_colon_intro_count():
    ctx = _ctx("items:\n- a", "items:\n- a")
    assert ColonIntroCountCheck().score("items:\n- a", ctx) == 1.0


def test_quote_attribution_count():
    ctx = _ctx('"hi" — John', '"hi" — John')
    assert QuoteAttributionCountCheck().score('"hi" — John', ctx) == 1.0


def test_hashtag_mention_count():
    ctx = _ctx("#news @bob", "#news @bob")
    assert HashtagCountCheck().score("#news #tech", ctx) < 1.0
    assert HashtagCountCheck().score("#news", ctx) == 1.0
    assert MentionCountCheck().score("@bob", ctx) == 1.0


def test_allcaps_line_count():
    ctx = _ctx("normal\nIMPORTANT NOTICE", "normal\nIMPORTANT NOTICE")
    assert AllCapsLineCountCheck().score("normal\nIMPORTANT NOTICE", ctx) == 1.0
    assert AllCapsLineCountCheck().score("normal\nnormal", ctx) < 1.0


def test_ellipsis_line_count():
    ctx = _ctx("wait...\ndone", "wait...\ndone")
    assert EllipsisLineCountCheck().score("wait...\ndone", ctx) == 1.0


def test_repeated_line_count():
    ctx = _ctx("same\nsame\nother", "same\nsame\nother")
    assert RepeatedLineCountCheck().score("same\nsame\nother", ctx) == 1.0


def test_boilerplate_ratio():
    ctx = _ctx("In today's fast-paced world, x", "In today's fast-paced world, x")
    assert BoilerplateScoreCheck().score("In today's fast-paced world, x", ctx) == 1.0
    assert BoilerplateScoreCheck().score("plain text only", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "Click here to learn more",
         "candidate_output": "Click here to learn more"},
        {"id": "b", "prompt": "p", "baseline_output": "Click here to learn more",
         "candidate_output": "plain answer"},
    ])
    result = evaluate(corpus, dimensions=("cta_count", "boilerplate_ratio"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["cta_count"].losses == 1
    assert dims["boilerplate_ratio"].losses == 0
