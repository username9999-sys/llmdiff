"""Run benchmark scenarios through the DiffEngine with timing and memory."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

from llmdiff import Corpus, evaluate
from llmdiff.benchmark.scenarios import SCENARIOS, build_scenario

__all__ = ["ScenarioResult", "run_scenario", "run_all_scenarios"]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Outcome of running one benchmark scenario."""

    name: str
    n_cases: int
    elapsed_s: float
    peak_memory_bytes: int
    verdict: str
    wins: int
    ties: int
    losses: int
    regressed_dimensions: tuple[str, ...]
    detected: bool
    dimensions: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def run_scenario(
    name: str,
    *,
    dimensions: tuple[str, ...] | None = None,
    measure_memory: bool = True,
) -> ScenarioResult:
    """Run one named scenario and return timing/memory/verdict metrics."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    records = build_scenario(name)
    corpus = Corpus.from_records(records)

    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    result = evaluate(corpus, dimensions=dimensions)
    elapsed = time.perf_counter() - start
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak = 0

    matrix = result.matrix
    detected = matrix.losses > 0
    return ScenarioResult(
        name=name,
        n_cases=len(corpus),
        elapsed_s=elapsed,
        peak_memory_bytes=peak,
        verdict=matrix.verdict,
        wins=matrix.wins,
        ties=matrix.ties,
        losses=matrix.losses,
        regressed_dimensions=matrix.regressed_dimensions,
        detected=detected,
        dimensions=tuple(d.dimension for d in matrix.dimensions),
    )


def run_all_scenarios(*, dimensions: tuple[str, ...] | None = None) -> list[ScenarioResult]:
    """Run every predefined scenario and return their results."""
    return [run_scenario(name, dimensions=dimensions) for name in sorted(SCENARIOS)]
