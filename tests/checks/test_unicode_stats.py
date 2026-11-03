"""Tests for unicode_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.unicode_stats import (
    ArabicScriptCountCheck,
    AsciiRatioCheck,
    BidiControlCountCheck,
    CjkCharCountCheck,
    CodepointRangeDiversityCheck,
    CombiningMarkCountCheck,
    CyrillicScriptCountCheck,
    DecimalDigitUnicodeCountCheck,
    EmojiCodepointCountCheck,
    GraphemeEstimateCheck,
    LatinScriptRatioCheck,
    NfcNfdRatioCheck,
    NonAsciiCountCheck,
    NormalizedFormMatchCheck,
    PrivateUseCountCheck,
    ScriptDiversityCheck,
    SurrogatePairCountCheck,
    TitlecaseUnicodeCountCheck,
    UppercaseUnicodeCountCheck,
    ZeroWidthCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_ascii_ratio():
    ctx = _ctx("hello", "hello")
    assert AsciiRatioCheck().score("hello", ctx) == 1.0
    assert AsciiRatioCheck().score("héllo", ctx) < 1.0


def test_nonascii_count():
    ctx = _ctx("abc", "abc")
    assert NonAsciiCountCheck().score("abé", ctx) < 1.0
    assert NonAsciiCountCheck().score("abc", ctx) == 1.0


def test_latin_ratio():
    ctx = _ctx("hello world", "hello world")
    assert LatinScriptRatioCheck().score("hello world", ctx) == 1.0


def test_cjk_count():
    ctx = _ctx("你好", "你好")
    assert CjkCharCountCheck().score("你好", ctx) == 1.0
    assert CjkCharCountCheck().score("hello", ctx) < 1.0


def test_arabic_count():
    ctx = _ctx("مرحبا", "مرحبا")
    assert ArabicScriptCountCheck().score("مرحبا", ctx) == 1.0


def test_cyrillic_count():
    ctx = _ctx("Привет", "Привет")
    assert CyrillicScriptCountCheck().score("Привет", ctx) == 1.0


def test_emoji_codepoints():
    ctx = _ctx("hi 😀 🎉", "hi 😀 🎉")
    assert EmojiCodepointCountCheck().score("hi 😀 🎉", ctx) == 1.0
    assert EmojiCodepointCountCheck().score("hi", ctx) < 1.0


def test_combining_marks():
    ctx = _ctx("é", "é")
    assert CombiningMarkCountCheck().score("é", ctx) == 1.0


def test_nfc_match():
    ctx = _ctx("café", "café")
    assert NormalizedFormMatchCheck().score("café", ctx) == 1.0
    assert NormalizedFormMatchCheck().score("cafe", ctx) == 0.0


def test_nfc_nfd_ratio():
    ctx = _ctx("café", "café")
    assert NfcNfdRatioCheck().score("café", ctx) == 1.0


def test_cp_block_diversity():
    ctx = _ctx("abc", "abc")
    assert CodepointRangeDiversityCheck().score("abc", ctx) == 1.0


def test_private_use_count():
    ctx = _ctx("abc", "abc")
    assert PrivateUseCountCheck().score("abc", ctx) == 1.0


def test_zero_width_count():
    ctx = _ctx("a​b", "a​b")
    assert ZeroWidthCountCheck().score("a​b", ctx) == 1.0
    assert ZeroWidthCountCheck().score("ab", ctx) < 1.0


def test_bidi_control_count():
    ctx = _ctx("abc", "abc")
    assert BidiControlCountCheck().score("abc", ctx) == 1.0


def test_astral_count():
    ctx = _ctx("😀", "😀")
    assert SurrogatePairCountCheck().score("😀", ctx) == 1.0
    assert SurrogatePairCountCheck().score("a", ctx) < 1.0


def test_grapheme_est():
    ctx = _ctx("hello", "hello")
    assert GraphemeEstimateCheck().score("hello", ctx) == 1.0


def test_script_diversity():
    ctx = _ctx("hello", "hello")
    assert ScriptDiversityCheck().score("hello", ctx) == 1.0


def test_unicode_upper_count():
    ctx = _ctx("École", "École")
    assert UppercaseUnicodeCountCheck().score("École", ctx) == 1.0


def test_unicode_title_count():
    ctx = _ctx("abc", "abc")
    assert TitlecaseUnicodeCountCheck().score("abc", ctx) == 1.0


def test_unicode_digit_count():
    ctx = _ctx("abc", "abc")
    assert DecimalDigitUnicodeCountCheck().score("abc", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world",
         "candidate_output": "hello world"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world",
         "candidate_output": "你好世界"},
    ])
    result = evaluate(corpus, dimensions=("ascii_ratio", "cjk_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["ascii_ratio"].losses == 1
    assert dims["cjk_count"].losses == 1
