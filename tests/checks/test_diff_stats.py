"""Tests for diff_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.diff_stats import (
    AddedLineCountCheck,
    AddedWordCountCheck,
    AddRemoveRatioCheck,
    AvgAddedLineLenCheck,
    AvgRemovedLineLenCheck,
    ChangeDensityCheck,
    ChangedLineCountCheck,
    ContextLineRatioCheck,
    DiffHeaderCountCheck,
    DiffHunkCountDeepCheck,
    DiffLineCountCheck,
    DiffSimilarityRatioCheck,
    FilePairCountCheck,
    LongestAddedLineCheck,
    LongestRemovedLineCheck,
    PlusMinusBalanceCheck,
    RemovedLineCountCheck,
    RemovedWordCountCheck,
    UnifiedDiffLineCountCheck,
    WordChurnRatioCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

DIFF = (
    "--- a/file.py\n"
    "+++ b/file.py\n"
    "@@ -1,3 +1,4 @@\n"
    " context\n"
    "-old line\n"
    "+new line\n"
    "+added line\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_added_removed_count():
    ctx = _ctx(DIFF, DIFF)
    assert AddedLineCountCheck().score(DIFF, ctx) == 1.0
    assert RemovedLineCountCheck().score(DIFF, ctx) == 1.0
    assert AddedLineCountCheck().score("+ only one\n", ctx) < 1.0


def test_changed_count():
    ctx = _ctx("! changed\n", "! changed\n")
    assert ChangedLineCountCheck().score("! changed\n", ctx) == 1.0


def test_add_remove_ratio():
    ctx = _ctx(DIFF, DIFF)
    assert AddRemoveRatioCheck().score(DIFF, ctx) == 1.0


def test_hunk_and_line_counts():
    ctx = _ctx(DIFF, DIFF)
    assert DiffHunkCountDeepCheck().score(DIFF, ctx) == 1.0
    assert DiffLineCountCheck().score(DIFF, ctx) == 1.0
    assert UnifiedDiffLineCountCheck().score(DIFF, ctx) == 1.0


def test_context_ratio():
    ctx = _ctx(DIFF, DIFF)
    assert ContextLineRatioCheck().score(DIFF, ctx) == 1.0


def test_diff_similarity():
    ctx = _ctx(DIFF, DIFF)
    assert DiffSimilarityRatioCheck().score(DIFF, ctx) == 1.0
    assert DiffSimilarityRatioCheck().score("totally different", ctx) < 1.0


def test_longest_avg_lines():
    ctx = _ctx(DIFF, DIFF)
    assert LongestAddedLineCheck().score(DIFF, ctx) == 1.0
    assert LongestRemovedLineCheck().score(DIFF, ctx) == 1.0
    assert AvgAddedLineLenCheck().score(DIFF, ctx) == 1.0
    assert AvgRemovedLineLenCheck().score(DIFF, ctx) == 1.0


def test_word_counts():
    ctx = _ctx(DIFF, DIFF)
    assert AddedWordCountCheck().score(DIFF, ctx) == 1.0
    assert RemovedWordCountCheck().score(DIFF, ctx) == 1.0
    assert WordChurnRatioCheck().score(DIFF, ctx) == 1.0


def test_balance_header_filepair():
    ctx = _ctx(DIFF, DIFF)
    assert PlusMinusBalanceCheck().score(DIFF, ctx) == 1.0
    assert DiffHeaderCountCheck().score(DIFF, ctx) == 1.0
    git_diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    gctx = _ctx(git_diff, git_diff)
    assert FilePairCountCheck().score(git_diff, gctx) == 1.0


def test_change_density():
    ctx = _ctx(DIFF, DIFF)
    assert ChangeDensityCheck().score(DIFF, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": DIFF,
         "candidate_output": DIFF},
        {"id": "b", "prompt": "p", "baseline_output": DIFF,
         "candidate_output": "+ totally\n+ different\n+ lines\n"},
    ])
    result = evaluate(corpus, dimensions=("added_lines", "diff_hunk_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["added_lines"].losses == 1
    assert dims["diff_hunk_count"].losses == 1
