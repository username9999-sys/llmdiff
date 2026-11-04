"""Tests for language_dims dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.language_dims import (
    ArabicCountCheck,
    AvgWordLengthLangCheck,
    CharNgramLangScoreCheck,
    CommonWordCoverageCheck,
    DetectedLanguageMatchCheck,
    DevanagariCountCheck,
    DiacriticRatioCheck,
    DigitScriptCountCheck,
    FunctionWordRatioCheck,
    GreekCountCheck,
    HangulCountCheck,
    HebrewCountCheck,
    HiraganaCountCheck,
    KatakanaCountCheck,
    LatinExtendedCountCheck,
    RtlPresenceCheck,
    ScriptMatchCheck,
    StopwordRatioCheck,
    ThaiCountCheck,
    VowelRatioLangCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_lang_detect():
    ctx = _ctx("the cat is on the mat", "the cat is on the mat")
    assert DetectedLanguageMatchCheck().score("the dog is in the house", ctx) == 1.0
    assert DetectedLanguageMatchCheck().score("yang ini dan itu", ctx) == 0.0


def test_script_match():
    ctx = _ctx("hello world", "hello world")
    assert ScriptMatchCheck().score("foo bar", ctx) == 1.0
    assert ScriptMatchCheck().score("你好世界", ctx) == 0.0


def test_rtl_presence():
    ctx = _ctx("hello", "hello")
    assert RtlPresenceCheck().score("hello", ctx) == 1.0
    assert RtlPresenceCheck().score("مرحبا", ctx) == 0.0


def test_stopword_ratio():
    ctx = _ctx("the and of to", "the and of to")
    assert StopwordRatioCheck().score("the and of to", ctx) == 1.0


def test_diacritic_ratio():
    ctx = _ctx("café naïve", "café naïve")
    assert DiacriticRatioCheck().score("café naïve", ctx) == 1.0
    assert DiacriticRatioCheck().score("cafe naive", ctx) < 1.0


def test_vowel_ratio_lang():
    ctx = _ctx("aeiou bcdfg", "aeiou bcdfg")
    assert VowelRatioLangCheck().score("aeiou bcdfg", ctx) == 1.0


def test_common_word_cov():
    ctx = _ctx("the a an of to", "the a an of to")
    assert CommonWordCoverageCheck().score("the a an of to", ctx) == 1.0


def test_function_word_ratio():
    ctx = _ctx("the and of to in", "the and of to in")
    assert FunctionWordRatioCheck().score("the and of to in", ctx) == 1.0


def test_avg_word_len_lang():
    ctx = _ctx("aa bb cc", "aa bb cc")
    assert AvgWordLengthLangCheck().score("aa bb cc", ctx) == 1.0


def test_char_trigram_lang():
    ctx = _ctx("hello world", "hello world")
    assert CharNgramLangScoreCheck().score("hello world", ctx) == 1.0
    assert CharNgramLangScoreCheck().score("hello there", ctx) < 1.0


def test_digit_script_count():
    ctx = _ctx("abc", "abc")
    assert DigitScriptCountCheck().score("abc", ctx) == 1.0


def test_script_counts():
    assert LatinExtendedCountCheck().score("hello", _ctx("hello", "hello")) == 1.0
    assert GreekCountCheck().score("αβγ", _ctx("αβγ", "αβγ")) == 1.0
    assert HebrewCountCheck().score("אבג", _ctx("אבג", "אבג")) == 1.0
    assert ArabicCountCheck().score("ابج", _ctx("ابج", "ابج")) == 1.0
    assert DevanagariCountCheck().score("अआइ", _ctx("अआइ", "अआइ")) == 1.0
    assert ThaiCountCheck().score("กขค", _ctx("กขค", "กขค")) == 1.0
    assert HangulCountCheck().score("가나다", _ctx("가나다", "가나다")) == 1.0
    assert HiraganaCountCheck().score("あいう", _ctx("あいう", "あいう")) == 1.0
    assert KatakanaCountCheck().score("アイウ", _ctx("アイウ", "アイウ")) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "the cat is on the mat",
         "candidate_output": "the cat is on the mat"},
        {"id": "b", "prompt": "p", "baseline_output": "the cat is on the mat",
         "candidate_output": "yang ini dan itu semua"},
    ])
    result = evaluate(corpus, dimensions=("lang_detect", "script_match"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["lang_detect"].losses == 1
    assert dims["script_match"].losses == 0
