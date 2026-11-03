"""Tests for readability_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.readability_stats import (
    AdverbEstimateCheck,
    ArticleRatioCheck,
    AutomatedReadabilityCheck,
    AvgSyllablesPerWordCheck,
    ColemanLiauCheck,
    ConjunctionRatioCheck,
    ExclamationSentenceRatioCheck,
    FleschReadingEaseCheck,
    FogIndexCheck,
    GunningFogWordsCheck,
    LongSentenceRatioCheck,
    PassiveVoiceEstimateCheck,
    PolysyllableRatioCheck,
    PrepositionRatioCheck,
    PronounRatioCheck,
    QuestionSentenceRatioCheck,
    SentenceLengthVarianceCheck,
    ShortSentenceRatioCheck,
    SmogIndexCheck,
    WordLengthVarianceCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


SIMPLE = "The cat sat. The dog ran. It was fun."
COMPLEX = (
    "The epistemological ramifications of poststructuralist "
    "hermeneutics necessitate comprehensive reconceptualization."
)


def test_flesch_ease():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert FleschReadingEaseCheck().score(SIMPLE, ctx) == 1.0


def test_fog_index():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert FogIndexCheck().score(SIMPLE, ctx) == 1.0


def test_smog_index():
    text = " ".join([SIMPLE] * 3)
    ctx = _ctx(text, text)
    assert SmogIndexCheck().score(text, ctx) == 1.0


def test_coleman_liau():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert ColemanLiauCheck().score(SIMPLE, ctx) == 1.0


def test_ari():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert AutomatedReadabilityCheck().score(SIMPLE, ctx) == 1.0


def test_fog_words():
    ctx = _ctx(COMPLEX, COMPLEX)
    assert GunningFogWordsCheck().score(COMPLEX, ctx) == 1.0
    assert GunningFogWordsCheck().score(SIMPLE, ctx) < 1.0


def test_avg_syllables():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert AvgSyllablesPerWordCheck().score(SIMPLE, ctx) == 1.0


def test_polysyllable_ratio():
    ctx = _ctx(COMPLEX, COMPLEX)
    assert PolysyllableRatioCheck().score(COMPLEX, ctx) == 1.0


def test_sent_len_variance():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert SentenceLengthVarianceCheck().score(SIMPLE, ctx) == 1.0


def test_word_len_variance():
    ctx = _ctx(SIMPLE, SIMPLE)
    assert WordLengthVarianceCheck().score(SIMPLE, ctx) == 1.0


def test_passive_ratio():
    ctx = _ctx("The ball was thrown by John.", "The ball was thrown by John.")
    assert PassiveVoiceEstimateCheck().score("The ball was thrown by John.", ctx) == 1.0
    assert PassiveVoiceEstimateCheck().score("John threw the ball.", ctx) < 1.0


def test_preposition_ratio():
    ctx = _ctx("the cat on the mat", "the cat on the mat")
    assert PrepositionRatioCheck().score("the cat on the mat", ctx) == 1.0


def test_conjunction_ratio():
    ctx = _ctx("cats and dogs", "cats and dogs")
    assert ConjunctionRatioCheck().score("cats and dogs", ctx) == 1.0


def test_pronoun_ratio():
    ctx = _ctx("I saw him", "I saw him")
    assert PronounRatioCheck().score("I saw him", ctx) == 1.0


def test_article_ratio():
    ctx = _ctx("a cat and the dog", "a cat and the dog")
    assert ArticleRatioCheck().score("a cat and the dog", ctx) == 1.0


def test_adverb_ratio():
    ctx = _ctx("she runs quickly", "she runs quickly")
    assert AdverbEstimateCheck().score("she runs quickly", ctx) == 1.0


def test_question_ratio():
    ctx = _ctx("What is this? Yes.", "What is this? Yes.")
    assert QuestionSentenceRatioCheck().score("What is this? Yes.", ctx) == 1.0


def test_exclamation_ratio():
    ctx = _ctx("Wow! Great.", "Wow! Great.")
    assert ExclamationSentenceRatioCheck().score("Wow! Great.", ctx) == 1.0


def test_long_sent_ratio():
    long_sent = "This is a very long sentence with many many words in it indeed yes."
    ctx = _ctx(long_sent, long_sent)
    assert LongSentenceRatioCheck().score(long_sent, ctx) == 1.0


def test_short_sent_ratio():
    ctx = _ctx("Hi there. Bye now.", "Hi there. Bye now.")
    assert ShortSentenceRatioCheck().score("Hi there. Bye now.", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": SIMPLE,
         "candidate_output": SIMPLE},
        {"id": "b", "prompt": "p", "baseline_output": SIMPLE,
         "candidate_output": COMPLEX},
    ])
    result = evaluate(corpus, dimensions=("fog_word_count", "avg_syllables"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["fog_word_count"].losses == 1
    assert dims["avg_syllables"].losses == 1
