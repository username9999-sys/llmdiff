"""Tests for e2e.harness — full offline pipeline."""

from __future__ import annotations

import pytest

from llmdiff.e2e.harness import chat_once, run_e2e
from llmdiff.e2e.mock_llm_server import MockLLMServer


def test_chat_once():
    with MockLLMServer({"ping": "pong"}) as srv:
        assert chat_once(srv.base_url, "ping") == "pong"


def test_run_e2e_identical_outputs_tie():
    prompts = ["a", "b", "c"]
    res = run_e2e(
        prompts,
        responses={"a": "1", "b": "2", "c": "3"},
        candidate_responses={"a": "1", "b": "2", "c": "3"},
    )
    assert res.n_cases == 3
    assert len(res.captured) == 3
    m = res.result.matrix
    assert m.losses == 0
    assert m.wins == 0  # identical -> all ties on exact


def test_run_e2e_detects_regression():
    prompts = ["a", "b", "c", "d"]
    res = run_e2e(
        prompts,
        responses={"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"},
        candidate_responses={"a": "alpha", "b": "WRONG", "c": "WRONG", "d": "delta"},
        dimensions=("exact", "similarity"),
    )
    m = res.result.matrix
    assert m.losses > 0
    assert m.get("exact").losses == 2


def test_run_e2e_empty_prompts_raises():
    with pytest.raises(ValueError):
        run_e2e([])
