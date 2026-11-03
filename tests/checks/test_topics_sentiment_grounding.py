"""Tests for topics, sentiment, and grounding dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.grounding import CitationSupportCheck, EntityConsistencyCheck, extract_entities
from llmdiff.checks.sentiment import SentimentCheck, ToneMatchCheck, sentiment_polarity
from llmdiff.checks.topics import KeywordCoverageCheck, TopicDriftCheck, extract_keywords
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_extract_keywords_skips_stopwords():
    kws = extract_keywords("The the the cat sat on the mat mat")
    assert "the" not in kws
    assert "cat" in kws and "mat" in kws


def test_keyword_coverage_identical():
    ctx = _ctx("Paris is the capital of France", "Paris is the capital of France")
    assert KeywordCoverageCheck().score("Paris is the capital of France", ctx) == 1.0


def test_keyword_coverage_partial():
    ctx = _ctx("Paris is the capital of France", "London is big")
    assert KeywordCoverageCheck().score("London is big", ctx) < 1.0


def test_topic_drift():
    ctx = _ctx("Python programming language tutorial", "Python programming language tutorial")
    assert TopicDriftCheck().score("Python programming language tutorial", ctx) == 1.0
    assert TopicDriftCheck().score("cooking recipes for dinner", ctx) < 1.0


def test_sentiment_polarity():
    assert sentiment_polarity("I love this, it is great and wonderful") > 0
    assert sentiment_polarity("this is terrible and broken, I hate it") < 0
    assert sentiment_polarity("the table is wooden") == 0.0


def test_sentiment_check_same_sign():
    ctx = _ctx("I love this great product", "this is wonderful and amazing")
    assert SentimentCheck().score("this is wonderful and amazing", ctx) == 1.0
    assert SentimentCheck().score("terrible broken garbage", ctx) == 0.0


def test_tone_match():
    ctx = _ctx("great wonderful", "great wonderful")
    assert ToneMatchCheck().score("great wonderful", ctx) == 1.0


def test_extract_entities():
    ents = extract_entities("Barack Obama visited New York City")
    assert "Barack Obama" in ents
    assert "New York City" in ents


def test_entity_consistency():
    ctx = _ctx("Barack Obama visited Paris", "Barack Obama visited Berlin")
    assert 0.0 < EntityConsistencyCheck().score("Barack Obama visited Berlin", ctx) < 1.0
    assert EntityConsistencyCheck().score("Barack Obama visited Paris", ctx) == 1.0


def test_citation_support():
    ctx = _ctx("see [1] and [2]", "see [1] and [2]")
    assert CitationSupportCheck().score("see [1] and [2]", ctx) == 1.0
    assert CitationSupportCheck().score("no citations", ctx) == 0.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "Paris is the capital of France",
         "candidate_output": "Paris is the capital of France"},
        {"id": "b", "prompt": "p", "baseline_output": "Paris is the capital of France",
         "candidate_output": "Tokyo is big and crowded"},
    ])
    result = evaluate(corpus, dimensions=("keyword_coverage", "topic_drift", "sentiment"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["keyword_coverage"].losses == 1
    assert dims["topic_drift"].losses == 1
