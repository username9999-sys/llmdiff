"""Tests for sentiment_tone dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.sentiment_tone import (
    CapsWordCountCheck,
    CertaintyWordRatioCheck,
    ContractionRatioCheck,
    EllipsisToneCountCheck,
    EmojiSentimentScoreCheck,
    EnthusiasmMarkerCountCheck,
    ExclamationRatioCheck,
    FirstPersonRatioCheck,
    FormalityRatioCheck,
    HedgingWordRatioCheck,
    IntensifierRatioCheck,
    NegationCountCheck,
    NegativeWordRatioCheck,
    PolarityBalanceCheck,
    PolitenessMarkerCountCheck,
    PositiveWordRatioCheck,
    SarcasmMarkerCountCheck,
    SecondPersonRatioCheck,
    ThirdPersonRatioCheck,
    UrgencyMarkerCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_positive_ratio():
    ctx = _ctx("this is great and awesome", "this is great and awesome")
    assert PositiveWordRatioCheck().score("this is great and awesome", ctx) == 1.0
    assert PositiveWordRatioCheck().score("this is bad", ctx) < 1.0


def test_negative_ratio():
    ctx = _ctx("terrible awful", "terrible awful")
    assert NegativeWordRatioCheck().score("terrible awful", ctx) == 1.0


def test_polarity_balance():
    ctx = _ctx("good bad", "good bad")
    assert PolarityBalanceCheck().score("good bad", ctx) == 1.0


def test_enthusiasm_count():
    ctx = _ctx("wow!!!", "wow!!!")
    assert EnthusiasmMarkerCountCheck().score("wow!!!", ctx) == 1.0
    assert EnthusiasmMarkerCountCheck().score("ok", ctx) < 1.0


def test_hedging_ratio():
    ctx = _ctx("maybe perhaps", "maybe perhaps")
    assert HedgingWordRatioCheck().score("maybe perhaps", ctx) == 1.0


def test_certainty_ratio():
    ctx = _ctx("definitely always", "definitely always")
    assert CertaintyWordRatioCheck().score("definitely always", ctx) == 1.0


def test_formality_ratio():
    ctx = _ctx("therefore however", "therefore however")
    assert FormalityRatioCheck().score("therefore however", ctx) == 1.0


def test_contraction_ratio():
    ctx = _ctx("I don't know", "I don't know")
    assert ContractionRatioCheck().score("I don't know", ctx) == 1.0
    assert ContractionRatioCheck().score("I do not know", ctx) < 1.0


def test_person_ratios():
    fctx = _ctx("I me my", "I me my")
    assert FirstPersonRatioCheck().score("I me my", fctx) == 1.0
    assert FirstPersonRatioCheck().score("you he", fctx) < 1.0
    sctx = _ctx("you your", "you your")
    assert SecondPersonRatioCheck().score("you your", sctx) == 1.0
    tctx = _ctx("he she it", "he she it")
    assert ThirdPersonRatioCheck().score("he she it", tctx) == 1.0


def test_exclamation_ratio():
    ctx = _ctx("wow! yay!", "wow! yay!")
    assert ExclamationRatioCheck().score("wow! yay!", ctx) == 1.0


def test_ellipsis_tone():
    ctx = _ctx("wait...", "wait...")
    assert EllipsisToneCountCheck().score("wait...", ctx) == 1.0
    assert EllipsisToneCountCheck().score("wait", ctx) < 1.0


def test_caps_words():
    ctx = _ctx("STOP NOW", "STOP NOW")
    assert CapsWordCountCheck().score("STOP NOW", ctx) == 1.0
    assert CapsWordCountCheck().score("stop now", ctx) < 1.0


def test_emoji_sentiment():
    ctx = _ctx("😀 😀", "😀 😀")
    assert EmojiSentimentScoreCheck().score("😀 😀", ctx) == 1.0
    assert EmojiSentimentScoreCheck().score("👎", ctx) < 1.0


def test_intensifier_ratio():
    ctx = _ctx("very really", "very really")
    assert IntensifierRatioCheck().score("very really", ctx) == 1.0


def test_negation_count():
    ctx = _ctx("not never", "not never")
    assert NegationCountCheck().score("not never", ctx) == 1.0
    assert NegationCountCheck().score("yes ok", ctx) < 1.0


def test_sarcasm_markers():
    ctx = _ctx("yeah right", "yeah right")
    assert SarcasmMarkerCountCheck().score("yeah right", ctx) == 1.0
    assert SarcasmMarkerCountCheck().score("sincere answer", ctx) < 1.0


def test_politeness_count():
    ctx = _ctx("please thanks", "please thanks")
    assert PolitenessMarkerCountCheck().score("please thanks", ctx) == 1.0


def test_urgency_count():
    ctx = _ctx("urgent asap", "urgent asap")
    assert UrgencyMarkerCountCheck().score("urgent asap", ctx) == 1.0
    assert UrgencyMarkerCountCheck().score("later maybe", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "this is great!",
         "candidate_output": "this is great!"},
        {"id": "b", "prompt": "p", "baseline_output": "this is great!",
         "candidate_output": "this is terrible"},
    ])
    result = evaluate(corpus, dimensions=("positive_ratio", "negative_ratio"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["positive_ratio"].losses == 1
    assert dims["negative_ratio"].losses == 1
