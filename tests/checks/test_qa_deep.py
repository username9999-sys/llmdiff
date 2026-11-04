"""Tests for qa_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.qa_deep import (
    AnswerLengthAdequacyCheck,
    AnswersYesNoCheck,
    BoldClaimCountCheck,
    CaveatCountCheck,
    CitationBracketCountCheck,
    ComparisonMarkerCountCheck,
    ConfidenceMarkerRatioCheck,
    ContradictionMarkerCountCheck,
    DefinitionPresenceCheck,
    DirectAnswerFirstCheck,
    ExampleCountCheck,
    FollowUpSuggestionCountCheck,
    ListAnswerCoverageCheck,
    ProsConsBalanceCheck,
    RefusalCountCheck,
    RepetitionInAnswerCheck,
    SourceMentionCountCheck,
    StepMarkerCountCheck,
    SummaryPresenceCheck,
    UncertaintyMarkerRatioCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str, prompt: str = "p") -> CheckContext:
    case = Case(id="t", prompt=prompt, baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_yesno_answer():
    ctx = _ctx("Yes, it works.", "Yes, it works.", prompt="Is it working?")
    assert AnswersYesNoCheck().score("Yes, it works.", ctx) == 1.0
    assert AnswersYesNoCheck().score("Maybe later.", ctx) == 0.0


def test_refusal_count():
    ctx = _ctx("I can't do that.", "I can't do that.")
    assert RefusalCountCheck().score("I can't do that.", ctx) == 1.0
    no_refusal = _ctx("Yes, I can help with that.", "Yes, I can help with that.")
    assert RefusalCountCheck().score("I cannot help with this.", no_refusal) < 1.0


def test_caveat_count():
    ctx = _ctx("Yes. However, note this.", "Yes. However, note this.")
    assert CaveatCountCheck().score("Yes. However, note this.", ctx) == 1.0


def test_confidence_ratio():
    ctx = _ctx("definitely yes", "definitely yes")
    assert ConfidenceMarkerRatioCheck().score("definitely yes", ctx) == 1.0


def test_citation_source_count():
    ctx = _ctx("as shown [1]", "as shown [1]")
    assert CitationBracketCountCheck().score("as shown [1] and [2]", ctx) < 1.0
    assert CitationBracketCountCheck().score("as shown [1]", ctx) == 1.0
    sctx = _ctx("according to X", "according to X")
    assert SourceMentionCountCheck().score("according to X", sctx) == 1.0


def test_contradiction_count():
    ctx = _ctx("that contradicts", "that contradicts")
    assert ContradictionMarkerCountCheck().score("that contradicts", ctx) == 1.0


def test_repetition():
    ctx = _ctx("Same sentence. Same sentence.", "Same sentence. Same sentence.")
    assert RepetitionInAnswerCheck().score("Same sentence. Same sentence.", ctx) == 1.0
    assert RepetitionInAnswerCheck().score("One. Two.", ctx) < 1.0


def test_adequacy():
    ctx = _ctx("a long enough answer here", "a long enough answer here")
    assert AnswerLengthAdequacyCheck().score("a long enough answer here", ctx) == 1.0
    assert AnswerLengthAdequacyCheck().score("short", ctx) == 0.0


def test_direct_answer_first():
    ctx = _ctx("Sure! Here is the answer.", "Sure! Here is the answer.")
    assert DirectAnswerFirstCheck().score("Sure! Here is the answer.", ctx) == 1.0
    plain = _ctx("The answer is 42.", "The answer is 42.")
    assert DirectAnswerFirstCheck().score("The answer is 42.", plain) == 1.0
    assert DirectAnswerFirstCheck().score("Sure! The answer is 42.", plain) == 0.0


def test_list_coverage():
    ctx = _ctx("- a\n- b\n- c", "- a\n- b\n- c", prompt="give me 3 items")
    assert ListAnswerCoverageCheck().score("- a\n- b\n- c", ctx) == 1.0
    assert ListAnswerCoverageCheck().score("- a", ctx) < 1.0


def test_comparison_count():
    ctx = _ctx("A vs B", "A vs B")
    assert ComparisonMarkerCountCheck().score("A versus B", ctx) == 1.0


def test_pros_cons_balance():
    ctx = _ctx("pros and cons", "pros and cons")
    assert ProsConsBalanceCheck().score("pros and cons", ctx) == 1.0


def test_step_count():
    ctx = _ctx("Step 1: a\nStep 2: b", "Step 1: a\nStep 2: b")
    assert StepMarkerCountCheck().score("Step 1: a\nStep 2: b", ctx) == 1.0


def test_example_count():
    ctx = _ctx("for example, x", "for example, x")
    assert ExampleCountCheck().score("for example, x", ctx) == 1.0


def test_definition_summary_presence():
    ctx = _ctx("X refers to Y", "X refers to Y")
    assert DefinitionPresenceCheck().score("X refers to Y", ctx) == 1.0
    sctx = _ctx("In summary, done", "In summary, done")
    assert SummaryPresenceCheck().score("In summary, done", sctx) == 1.0


def test_followup_count():
    ctx = _ctx("Would you like more?", "Would you like more?")
    assert FollowUpSuggestionCountCheck().score("Would you like more?", ctx) == 1.0


def test_uncertainty_ratio():
    ctx = _ctx("maybe yes", "maybe yes")
    assert UncertaintyMarkerRatioCheck().score("maybe yes", ctx) == 1.0


def test_bold_claim_count():
    ctx = _ctx("always true", "always true")
    assert BoldClaimCountCheck().score("always true", ctx) == 1.0
    assert BoldClaimCountCheck().score("sometimes true", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "Is it working?", "baseline_output": "Yes, it works fine here.",
         "candidate_output": "Yes, it works fine here."},
        {"id": "b", "prompt": "Is it working?", "baseline_output": "Yes, it works fine here.",
         "candidate_output": "I am not sure about that."},
    ])
    result = evaluate(corpus, dimensions=("yesno_answer", "uncertainty_ratio"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["yesno_answer"].losses == 1
    assert dims["uncertainty_ratio"].losses == 1
