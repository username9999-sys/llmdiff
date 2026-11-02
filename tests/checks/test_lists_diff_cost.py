"""Tests for lists, diffview, cost, and composite dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.composite import WeightedCompositeCheck, composite_from_names
from llmdiff.checks.cost import CostCheck, LatencyCheck, TokenCountCheck
from llmdiff.checks.diffview import DiffHunkCountCheck, EditDistanceCheck, LineChangeRatioCheck
from llmdiff.checks.lists import (
    ListItemCountCheck,
    ListItemOverlapCheck,
    ListOrderCheck,
    TableShapeCheck,
    extract_list_items,
    extract_table_rows,
)
from llmdiff.checks.text import ExactMatchCheck, SimilarityCheck
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str, metadata: dict | None = None) -> CheckContext:
    case = Case(
        id="t", prompt="p", baseline_output=baseline, candidate_output=candidate,
        metadata=metadata or {},
    )
    return CheckContext(case=case, metadata=case.metadata)


def test_extract_list_items():
    assert extract_list_items("- a\n- b\n1. c") == ["a", "b", "c"]


def test_extract_table_rows_skips_separator():
    rows = extract_table_rows("| a | b |\n|---|---|\n| 1 | 2 |")
    assert rows == [["a", "b"], ["1", "2"]]


def test_list_count():
    ctx = _ctx("- a\n- b", "- a\n- b")
    assert ListItemCountCheck().score("- a\n- b", ctx) == 1.0
    assert ListItemCountCheck().score("- a", ctx) < 1.0


def test_list_overlap():
    ctx = _ctx("- a\n- b", "- a\n- c")
    assert 0.0 < ListItemOverlapCheck().score("- a\n- c", ctx) < 1.0


def test_list_order():
    ctx = _ctx("- a\n- b\n- c", "- c\n- b\n- a")
    assert ListOrderCheck().score("- c\n- b\n- a", ctx) == 0.0
    assert ListOrderCheck().score("- a\n- b\n- c", ctx) == 1.0


def test_table_shape():
    ctx = _ctx("| a | b |\n|---|---|\n| 1 | 2 |", "| a | b |\n|---|---|\n| 1 | 2 |")
    assert TableShapeCheck().score("| a | b |\n|---|---|\n| 1 | 2 |", ctx) == 1.0


def test_edit_distance():
    ctx = _ctx("hello world", "hello world")
    assert EditDistanceCheck().score("hello world", ctx) == 1.0
    assert EditDistanceCheck().score("xyz", ctx) < 1.0


def test_diff_hunks_zero_when_identical():
    ctx = _ctx("a\nb\nc", "a\nb\nc")
    assert DiffHunkCountCheck().score("a\nb\nc", ctx) == 1.0


def test_line_survival():
    ctx = _ctx("keep\nthis\nline", "keep\nthis\nline")
    assert LineChangeRatioCheck().score("keep\nthis\nline", ctx) == 1.0
    assert LineChangeRatioCheck().score("keep\nthis", ctx) < 1.0


def test_latency_metadata():
    ctx = _ctx("x", "x", {"baseline_ms": 100.0, "candidate_ms": 50.0})
    assert LatencyCheck().score("x", ctx) == 1.0
    ctx2 = _ctx("x", "x", {"baseline_ms": 100.0, "candidate_ms": 500.0})
    assert LatencyCheck().score("x", ctx2) < 1.0
    assert LatencyCheck().score("x", _ctx("x", "x")) == 1.0


def test_tokens_metadata():
    ctx = _ctx("x", "x", {"baseline_tokens_out": 100, "candidate_tokens_out": 50})
    assert TokenCountCheck().score("x", ctx) == 1.0
    ctx2 = _ctx("x", "x", {"baseline_tokens_out": 100, "candidate_tokens_out": 300})
    assert TokenCountCheck().score("x", ctx2) < 1.0


def test_cost_metadata():
    ctx = _ctx("x", "x", {"baseline_cost_usd": 0.01, "candidate_cost_usd": 0.02})
    assert CostCheck().score("x", ctx) < 1.0
    assert CostCheck().score("x", _ctx("x", "x")) == 1.0


def test_composite_weighted():
    comp = WeightedCompositeCheck("combo", [(ExactMatchCheck(), 1.0), (SimilarityCheck(), 1.0)])
    ctx = _ctx("abc", "abc")
    assert comp.score("abc", ctx) == 1.0
    ctx2 = _ctx("abc", "abd")
    assert 0.0 < comp.score("abd", ctx2) < 1.0


def test_composite_from_names():
    registry = {"exact": ExactMatchCheck(), "similarity": SimilarityCheck()}
    comp = composite_from_names("c", registry, {"exact": 2.0, "similarity": 1.0})
    ctx = _ctx("abc", "abc")
    assert comp.score("abc", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "- one\n- two\n- three",
         "candidate_output": "- one\n- two\n- three"},
        {"id": "b", "prompt": "p", "baseline_output": "- one\n- two\n- three",
         "candidate_output": "- one"},
    ])
    result = evaluate(corpus, dimensions=("list_count", "list_overlap", "edit_distance"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["list_count"].losses == 1
    assert dims["list_overlap"].losses == 1
    assert dims["edit_distance"].losses == 1
