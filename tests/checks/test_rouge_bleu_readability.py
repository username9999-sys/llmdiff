"""Tests for rouge, bleu, readability, and instructions dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.bleu import BleuCheck, bleu_score, brevity_penalty
from llmdiff.checks.instructions import (
    FormatFollowCheck,
    JsonModeCheck,
    LengthInstructionCheck,
    NoApologyCheck,
)
from llmdiff.checks.readability import (
    FleschKincaidCheck,
    SentenceStatsCheck,
    SyllableCountCheck,
    count_syllables,
    flesch_kincaid_grade,
)
from llmdiff.checks.rouge import Rouge1Check, Rouge2Check, RougeLCheck, rouge_l_f1
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str, prompt: str = "p") -> CheckContext:
    case = Case(id="t", prompt=prompt, baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_rouge_identical():
    ctx = _ctx("the quick brown fox", "the quick brown fox")
    assert Rouge1Check().score("the quick brown fox", ctx) == 1.0
    assert Rouge2Check().score("the quick brown fox", ctx) == 1.0
    assert RougeLCheck().score("the quick brown fox", ctx) == 1.0


def test_rouge_partial():
    ctx = _ctx("the quick brown fox jumps", "the quick brown fox")
    assert 0.0 < Rouge1Check().score("the quick brown fox", ctx) < 1.0
    assert rouge_l_f1("a b c d", "a b x d") < 1.0


def test_bleu_identical_is_one():
    assert bleu_score("a b c d", "a b c d") == 1.0


def test_brevity_penalty():
    assert brevity_penalty(10, 5) == 1.0
    assert brevity_penalty(0, 5) == 0.0
    assert 0.0 < brevity_penalty(5, 10) < 1.0


def test_bleu_check_partial():
    ctx = _ctx("one two three four five", "one two three")
    assert 0.0 < BleuCheck().score("one two three", ctx) < 1.0


def test_syllables():
    assert count_syllables("hello") == 2
    assert count_syllables("a") == 1
    assert count_syllables("") == 0


def test_flesch_kincaid():
    simple = flesch_kincaid_grade("The cat sat. It was happy.")
    hard = flesch_kincaid_grade(
        "The epistemological ramifications of poststructuralist hermeneutics "
        "necessitate comprehensive reconceptualization of interdisciplinary methodologies."
    )
    assert hard > simple


def test_readability_check_identical():
    text = "The cat sat on the mat. It was happy."
    ctx = _ctx(text, text)
    assert FleschKincaidCheck().score(text, ctx) == 1.0


def test_sentence_stats():
    ctx = _ctx("One. Two. Three.", "One. Two. Three.")
    assert SentenceStatsCheck().score("One. Two. Three.", ctx) == 1.0
    assert SyllableCountCheck().score("One. Two. Three.", ctx) == 1.0


def test_format_follow_json():
    ctx = _ctx('{"a": 1}', '{"a": 1}', prompt="return as JSON")
    assert FormatFollowCheck().score('{"a": 1}', ctx) == 1.0
    assert FormatFollowCheck().score("not json", ctx) == 0.0


def test_json_mode():
    ctx = _ctx("x", '{"ok": true}', prompt="give me JSON output")
    assert JsonModeCheck().score('{"ok": true}', ctx) == 1.0
    assert JsonModeCheck().score("nope", ctx) == 0.0
    assert JsonModeCheck().score("anything", _ctx("x", "anything", prompt="plain text")) == 1.0


def test_length_instruction():
    ctx = _ctx("x", "one two three four five six", prompt="write at most 5 words")
    assert LengthInstructionCheck().score("one two three four five six", ctx) < 1.0
    assert LengthInstructionCheck().score("one two three", ctx) == 1.0


def test_no_apology():
    ctx = _ctx("x", "x")
    assert NoApologyCheck().score("Here is the answer.", ctx) == 1.0
    assert NoApologyCheck().score("I'm sorry, but no.", ctx) == 0.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "the quick brown fox jumps over",
         "candidate_output": "the quick brown fox jumps over"},
        {"id": "b", "prompt": "p", "baseline_output": "the quick brown fox jumps over",
         "candidate_output": "zzz yyy xxx"},
    ])
    result = evaluate(corpus, dimensions=("rouge1", "rougeL", "bleu"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["rouge1"].losses == 1
    assert dims["rougeL"].losses == 1
    assert dims["bleu"].losses == 1
