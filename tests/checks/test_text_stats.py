"""Tests for text_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.text_stats import (
    AlphaCharRatioCheck,
    ByteLengthUtf8Check,
    CharCountCheck,
    DigitCharRatioCheck,
    HapaxRatioCheck,
    LexicalDiversityCheck,
    LowerCharRatioCheck,
    NonSpaceCharCountCheck,
    ParagraphCountMatchCheck,
    PunctCharRatioCheck,
    SentenceCountCheck,
    SpaceCharRatioCheck,
    SyllableEstimateCheck,
    TypeTokenRatioCheck,
    UniqueCharRatioCheck,
    UniqueWordRatioCheck,
    UpperCharRatioCheck,
    WordLengthMaxCheck,
    WordLengthMeanCheck,
    WordLengthStdCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_char_count():
    ctx = _ctx("hello", "hello")
    assert CharCountCheck().score("hello", ctx) == 1.0
    assert CharCountCheck().score("hi", ctx) < 1.0


def test_nonspace_char_count():
    ctx = _ctx("a b", "a b")
    assert NonSpaceCharCountCheck().score("a b", ctx) == 1.0


def test_byte_len_utf8():
    ctx = _ctx("héllo", "héllo")
    assert ByteLengthUtf8Check().score("héllo", ctx) == 1.0
    assert ByteLengthUtf8Check().score("hello", ctx) < 1.0


def test_unique_char_ratio():
    ctx = _ctx("aabbcc", "aabbcc")
    assert UniqueCharRatioCheck().score("aabbcc", ctx) == 1.0


def test_digit_char_ratio():
    ctx = _ctx("a1b2", "a1b2")
    assert DigitCharRatioCheck().score("a1b2", ctx) == 1.0
    assert DigitCharRatioCheck().score("ab", ctx) < 1.0


def test_alpha_char_ratio():
    ctx = _ctx("abc123", "abc123")
    assert AlphaCharRatioCheck().score("abc123", ctx) == 1.0


def test_space_char_ratio():
    ctx = _ctx("a b  c", "a b  c")
    assert SpaceCharRatioCheck().score("a b  c", ctx) == 1.0


def test_punct_char_ratio():
    ctx = _ctx("a, b!", "a, b!")
    assert PunctCharRatioCheck().score("a, b!", ctx) == 1.0


def test_upper_char_ratio():
    ctx = _ctx("AbC", "AbC")
    assert UpperCharRatioCheck().score("AbC", ctx) == 1.0
    assert UpperCharRatioCheck().score("abc", ctx) < 1.0


def test_lower_char_ratio():
    ctx = _ctx("aBc", "aBc")
    assert LowerCharRatioCheck().score("aBc", ctx) == 1.0


def test_word_len_std():
    ctx = _ctx("aa bbbb cc", "aa bbbb cc")
    assert WordLengthStdCheck().score("aa bbbb cc", ctx) == 1.0


def test_word_len_mean():
    ctx = _ctx("aa bb cc", "aa bb cc")
    assert WordLengthMeanCheck().score("aa bb cc", ctx) == 1.0


def test_word_len_max():
    ctx = _ctx("aa bbbbb cc", "aa bbbbb cc")
    assert WordLengthMaxCheck().score("aa bbbbb cc", ctx) == 1.0
    assert WordLengthMaxCheck().score("aa bb cc", ctx) < 1.0


def test_unique_word_ratio():
    ctx = _ctx("cat dog cat", "cat dog cat")
    assert UniqueWordRatioCheck().score("cat dog cat", ctx) == 1.0


def test_lexical_diversity():
    text = " ".join(f"w{i}" for i in range(60))
    ctx = _ctx(text, text)
    assert LexicalDiversityCheck().score(text, ctx) == 1.0


def test_type_token_ratio():
    ctx = _ctx("cat dog bird", "cat dog bird")
    assert TypeTokenRatioCheck().score("cat dog bird", ctx) == 1.0


def test_hapax_ratio():
    ctx = _ctx("cat dog cat", "cat dog cat")
    assert HapaxRatioCheck().score("cat dog cat", ctx) == 1.0


def test_sentence_count():
    ctx = _ctx("One. Two! Three?", "One. Two! Three?")
    assert SentenceCountCheck().score("One. Two! Three?", ctx) == 1.0
    assert SentenceCountCheck().score("One.", ctx) < 1.0


def test_paragraph_count():
    ctx = _ctx("para1\n\npara2", "para1\n\npara2")
    assert ParagraphCountMatchCheck().score("para1\n\npara2", ctx) == 1.0
    assert ParagraphCountMatchCheck().score("para1", ctx) < 1.0


def test_syllable_est():
    ctx = _ctx("hello world", "hello world")
    assert SyllableEstimateCheck().score("hello world", ctx) == 1.0
    assert SyllableEstimateCheck().score("hi", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world",
         "candidate_output": "hello world"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world",
         "candidate_output": "hi"},
    ])
    result = evaluate(corpus, dimensions=("char_count", "sentence_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["char_count"].losses == 1
    assert dims["sentence_count"].losses == 0
