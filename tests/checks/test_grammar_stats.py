"""Tests for grammar_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.grammar_stats import (
    ArticleACountCheck,
    ArticleAnCountCheck,
    ArticleTheCountCheck,
    AuxVerbCountCheck,
    ConditionalMarkerCountCheck,
    ConjunctionCountCheck,
    ExclamationGrammarCountCheck,
    FutureMarkerCountCheck,
    GerundCountCheck,
    InfinitiveCountCheck,
    ModalVerbCountCheck,
    NegationGrammarCountCheck,
    PassiveAuxCountCheck,
    PastTenseVerbCountCheck,
    PrepositionCountCheck,
    PresentTenseVerbCountCheck,
    PronounCountCheck,
    QuestionWordCountCheck,
    RelativePronounCountCheck,
    SubordinateClauseCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_tense_counts():
    ctx = _ctx("he walked and talked", "he walked and talked")
    assert PastTenseVerbCountCheck().score("he walked and talked", ctx) == 1.0
    pctx = _ctx("she runs fast", "she runs fast")
    assert PresentTenseVerbCountCheck().score("she runs fast", pctx) == 1.0
    fctx = _ctx("we will go", "we will go")
    assert FutureMarkerCountCheck().score("we will go", fctx) == 1.0


def test_modal_gerund_infinitive():
    ctx = _ctx("you can try", "you can try")
    assert ModalVerbCountCheck().score("you can try", ctx) == 1.0
    gctx = _ctx("running is fun", "running is fun")
    assert GerundCountCheck().score("running is fun", gctx) == 1.0
    ictx = _ctx("to be or not", "to be or not")
    assert InfinitiveCountCheck().score("to be or not", ictx) == 1.0


def test_passive_conditional_subordinate():
    ctx = _ctx("the cake was eaten", "the cake was eaten")
    assert PassiveAuxCountCheck().score("the cake was eaten", ctx) == 1.0
    cctx = _ctx("if you want", "if you want")
    assert ConditionalMarkerCountCheck().score("if you want", cctx) == 1.0
    sctx = _ctx("because it rained", "because it rained")
    assert SubordinateClauseCountCheck().score("because it rained", sctx) == 1.0


def test_rel_pronoun_articles():
    ctx = _ctx("the man who came", "the man who came")
    assert RelativePronounCountCheck().score("the man who came", ctx) == 1.0
    assert ArticleTheCountCheck().score("the cat", ctx) == 1.0
    actx = _ctx("a dog", "a dog")
    assert ArticleACountCheck().score("a dog", actx) == 1.0
    anctx = _ctx("an apple", "an apple")
    assert ArticleAnCountCheck().score("an apple", anctx) == 1.0


def test_prep_conj_pronoun_aux():
    ctx = _ctx("in the box on top", "in the box on top")
    assert PrepositionCountCheck().score("in the box on top", ctx) == 1.0
    cctx = _ctx("cats and dogs", "cats and dogs")
    assert ConjunctionCountCheck().score("cats and dogs", cctx) == 1.0
    pctx = _ctx("I saw him", "I saw him")
    assert PronounCountCheck().score("I saw him", pctx) == 1.0
    auxctx = _ctx("she is running", "she is running")
    assert AuxVerbCountCheck().score("she is running", auxctx) == 1.0


def test_negation_qword_exclamation():
    ctx = _ctx("not never", "not never")
    assert NegationGrammarCountCheck().score("not never", ctx) == 1.0
    qctx = _ctx("what is this?", "what is this?")
    assert QuestionWordCountCheck().score("what is this?", qctx) == 1.0
    ectx = _ctx("wow!", "wow!")
    assert ExclamationGrammarCountCheck().score("wow!", ectx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "he walked and the dog ran",
         "candidate_output": "he walked and the dog ran"},
        {"id": "b", "prompt": "p", "baseline_output": "he walked and the dog ran",
         "candidate_output": "nothing here"},
    ])
    result = evaluate(corpus, dimensions=("past_verb_count", "article_the_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["past_verb_count"].losses == 1
    assert dims["article_the_count"].losses == 1
