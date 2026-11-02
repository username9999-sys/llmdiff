"""End-to-end scenarios: full pipeline runs with expected verdict outcomes.

Each scenario runs run_e2e() with scripted baseline/candidate responses and
asserts what the verdict matrix should report. All offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmdiff.e2e.harness import run_e2e

__all__ = ["E2EScenario", "E2E_SCENARIOS", "run_scenario_e2e", "run_all_e2e"]


@dataclass(frozen=True, slots=True)
class E2EScenario:
    """One e2e scenario: prompts, responses, and the expected outcome."""

    name: str
    prompts: tuple[str, ...]
    baseline: dict[str, str]
    candidate: dict[str, str]
    dimensions: tuple[str, ...] = ("exact", "similarity", "content")
    expect_losses_on: str = "exact"
    min_losses: int = 1


E2E_SCENARIOS: dict[str, E2EScenario] = {
    "identical": E2EScenario(
        name="identical",
        prompts=("p1", "p2", "p3"),
        baseline={"p1": "alpha", "p2": "beta", "p3": "gamma"},
        candidate={"p1": "alpha", "p2": "beta", "p3": "gamma"},
        min_losses=0,
    ),
    "reworded": E2EScenario(
        name="reworded",
        prompts=("p1", "p2", "p3", "p4"),
        baseline={"p1": "alpha", "p2": "beta", "p3": "gamma", "p4": "delta"},
        candidate={"p1": "alpha!", "p2": "beta?", "p3": "gamma.", "p4": "delta"},
        min_losses=3,
    ),
    "field_drop": E2EScenario(
        name="field_drop",
        prompts=("j1", "j2", "j3"),
        baseline={"j1": '{"a": 1, "b": 2}', "j2": '{"x": 9, "y": 8}', "j3": '{"k": 1}'},
        candidate={"j1": '{"a": 1}', "j2": '{"x": 9}', "j3": "{}"},
        dimensions=("structure", "json_valid"),
        expect_losses_on="structure",
        min_losses=3,
    ),
    "refusal": E2EScenario(
        name="refusal",
        prompts=("r1", "r2"),
        baseline={"r1": "here is the answer", "r2": "another answer"},
        candidate={"r1": "I'm sorry, I cannot help with that.", "r2": "I cannot assist."},
        dimensions=("content", "exact"),
        expect_losses_on="content",
        min_losses=2,
    ),
    "truncation": E2EScenario(
        name="truncation",
        prompts=("t1", "t2"),
        baseline={"t1": "a long and complete answer about the topic", "t2": "another full answer here"},
        candidate={"t1": "a long and...", "t2": "another..."},
        dimensions=("length", "content", "similarity"),
        expect_losses_on="length",
        min_losses=2,
    ),
}


def run_scenario_e2e(name: str):
    """Run one named e2e scenario. Returns (scenario, E2EResult)."""
    if name not in E2E_SCENARIOS:
        raise ValueError(f"unknown e2e scenario {name!r}; choose from {sorted(E2E_SCENARIOS)}")
    sc = E2E_SCENARIOS[name]
    result = run_e2e(
        list(sc.prompts),
        responses=sc.baseline,
        candidate_responses=sc.candidate,
        dimensions=sc.dimensions,
    )
    return sc, result


def run_all_e2e() -> list:
    """Run every e2e scenario."""
    return [run_scenario_e2e(name) for name in sorted(E2E_SCENARIOS)]
