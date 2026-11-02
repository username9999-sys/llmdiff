"""E2E performance tests: evaluate a big corpus under time/memory budgets."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from llmdiff import Corpus, evaluate
from llmdiff.e2e.fixtures import big_corpus_records

__all__ = ["E2EPerfResult", "perf_run", "E2E_BUDGETS"]

# Budgets for the e2e perf test: 10k cases in < 60s, peak < 1 GB.
E2E_BUDGETS = {"max_seconds": 60.0, "max_memory_bytes": 1024 * 1024 * 1024}


@dataclass(frozen=True, slots=True)
class E2EPerfResult:
    """Outcome of an e2e performance run."""

    n_cases: int
    n_dimensions: int
    elapsed_s: float
    peak_memory_bytes: int
    cases_per_second: float
    within_time_budget: bool
    within_memory_budget: bool


def perf_run(
    n_cases: int = 10_000,
    *,
    dimensions: tuple[str, ...] | None = None,
    seed: int = 0,
    budgets: dict[str, float] | None = None,
) -> E2EPerfResult:
    """Evaluate a big generated corpus and check the e2e perf budgets."""
    merged = {**E2E_BUDGETS, **(budgets or {})}
    corpus = Corpus.from_records(big_corpus_records(n_cases, seed=seed))

    tracemalloc.start()
    start = time.perf_counter()
    result = evaluate(corpus, dimensions=dimensions)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cps = n_cases / elapsed if elapsed > 0 else 0.0
    return E2EPerfResult(
        n_cases=n_cases,
        n_dimensions=len(result.matrix.dimensions),
        elapsed_s=elapsed,
        peak_memory_bytes=peak,
        cases_per_second=cps,
        within_time_budget=elapsed <= merged["max_seconds"],
        within_memory_budget=peak <= merged["max_memory_bytes"],
    )
