"""Performance tests: evaluate large corpora under time and memory budgets."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from llmdiff import Corpus, evaluate
from llmdiff.benchmark.corpus_gen import gen_corpus

__all__ = ["PerfResult", "perf_evaluate", "PERF_BUDGETS"]


@dataclass(frozen=True, slots=True)
class PerfResult:
    """Outcome of a performance evaluation."""

    n_cases: int
    n_dimensions: int
    elapsed_s: float
    peak_memory_bytes: int
    cases_per_second: float
    within_time_budget: bool
    within_memory_budget: bool


# Default budgets: 10k cases in < 30s, peak < 512 MB.
PERF_BUDGETS = {"max_seconds": 30.0, "max_memory_bytes": 512 * 1024 * 1024}


def perf_evaluate(
    n_cases: int = 10_000,
    *,
    kind: str = "summarization",
    dimensions: tuple[str, ...] | None = None,
    seed: int = 0,
    budgets: dict[str, float] | None = None,
) -> PerfResult:
    """Evaluate a generated corpus of ``n_cases`` and check perf budgets.

    Args:
        n_cases: Corpus size to generate.
        kind: Generator kind (see benchmark.corpus_gen.GENERATORS).
        dimensions: Dimensions to evaluate; None = all built-ins.
        seed: Generator seed (deterministic).
        budgets: Override PERF_BUDGETS keys.
    """
    merged = {**PERF_BUDGETS, **(budgets or {})}
    corpus = Corpus.from_records(gen_corpus(kind, n_cases, seed=seed))

    tracemalloc.start()
    start = time.perf_counter()
    result = evaluate(corpus, dimensions=dimensions)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_dims = len(result.matrix.dimensions)
    cps = n_cases / elapsed if elapsed > 0 else 0.0
    return PerfResult(
        n_cases=n_cases,
        n_dimensions=n_dims,
        elapsed_s=elapsed,
        peak_memory_bytes=peak,
        cases_per_second=cps,
        within_time_budget=elapsed <= merged["max_seconds"],
        within_memory_budget=peak <= merged["max_memory_bytes"],
    )
