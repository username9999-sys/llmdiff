"""The diff engine: evaluate a corpus against a set of comparison dimensions.

The engine is deliberately thin. It owns exactly three responsibilities:

1. call each check twice, once per side, so both scores come from one definition;
2. turn the pair of scores into a verdict;
3. aggregate the verdicts into the matrix.

Everything else - what a dimension means, how a report looks, how a corpus is
replayed - lives in its own module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from llmdiff.core.case import Case
from llmdiff.core.checks import Check, CheckContext, CheckResult, Verdict, compare_scores
from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import CheckError
from llmdiff.core.matrix import VerdictMatrix
from llmdiff.core.registry import CheckRegistry, default_registry
from llmdiff.core.result import CaseDiff, DiffResult

__all__ = ["DiffConfig", "DiffEngine", "evaluate"]


@dataclass(frozen=True, slots=True)
class DiffConfig:
    """Settings controlling one evaluation run.

    Attributes:
        dimensions: Names of the dimensions to evaluate. ``None`` evaluates all
            registered dimensions.
        tolerance: Absolute score difference below which a comparison counts as
            a tie rather than a win or a loss.
        fail_fast: When true, the first check that raises aborts the run.
            When false, the failing check is skipped and the run continues.
    """

    dimensions: tuple[str, ...] | None = None
    tolerance: float = 0.0
    fail_fast: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this config."""
        return {
            "dimensions": list(self.dimensions) if self.dimensions is not None else None,
            "tolerance": self.tolerance,
            "fail_fast": self.fail_fast,
        }


class DiffEngine:
    """Evaluates a corpus against a registry of comparison dimensions."""

    __slots__ = ("_registry", "_config")

    def __init__(
        self,
        registry: CheckRegistry | None = None,
        config: DiffConfig | None = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._config = config if config is not None else DiffConfig()

    @property
    def registry(self) -> CheckRegistry:
        """The registry this engine draws dimensions from."""
        return self._registry

    @property
    def config(self) -> DiffConfig:
        """The default configuration for this engine."""
        return self._config

    def run(self, corpus: Corpus, *, config: DiffConfig | None = None) -> DiffResult:
        """Evaluate every case in ``corpus``.

        Args:
            corpus: The cases to evaluate.
            config: Overrides the engine's default configuration for this run.

        Returns:
            The :class:`DiffResult` holding the matrix and per-case diffs.

        Raises:
            CheckError: If a check raises and ``fail_fast`` is enabled, or if a
                requested dimension is not registered.
        """
        active = config if config is not None else self._config
        checks = self._registry.select(active.dimensions)

        all_results: list[CheckResult] = []
        case_diffs: list[CaseDiff] = []

        for case in corpus:
            context = CheckContext(case=case, metadata=case.metadata)
            case_results: list[CheckResult] = []
            for check in checks:
                result = self._evaluate(check, case, context, active)
                if result is None:
                    continue
                case_results.append(result)
                all_results.append(result)
            case_diffs.append(CaseDiff(case_id=case.id, results=tuple(case_results)))

        matrix = VerdictMatrix.from_results(all_results, case_ids=[c.id for c in corpus])
        return DiffResult(matrix=matrix, cases=tuple(case_diffs), config=active)

    def _evaluate(
        self,
        check: Check,
        case: Case,
        context: CheckContext,
        config: DiffConfig,
    ) -> CheckResult | None:
        try:
            baseline_score = self._score(check, case.baseline_output, context)
            candidate_score = self._score(check, case.candidate_output, context)
        except Exception as exc:
            if config.fail_fast:
                raise CheckError(f"check {check.name!r} failed on case {case.id!r}: {exc}") from exc
            return None

        verdict = compare_scores(baseline_score, candidate_score, tolerance=config.tolerance)
        return CheckResult(
            dimension=check.name,
            verdict=verdict,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            detail=self._describe(check, verdict, baseline_score, candidate_score),
            metadata={"case_id": case.id},
        )

    @staticmethod
    def _score(check: Check, output: str, context: CheckContext) -> float:
        value = float(check.score(output, context))
        if value != value:  # NaN
            raise CheckError(f"check {check.name!r} returned NaN")
        return max(0.0, min(1.0, value))

    @staticmethod
    def _describe(
        check: Check,
        verdict: Verdict,
        baseline_score: float,
        candidate_score: float,
    ) -> str:
        delta = candidate_score - baseline_score
        if verdict is Verdict.TIE:
            return f"{check.name}: unchanged ({baseline_score:.3f})"
        direction = "improved" if verdict is Verdict.WIN else "regressed"
        return (
            f"{check.name}: {direction} by {abs(delta):.3f} "
            f"({baseline_score:.3f} -> {candidate_score:.3f})"
        )


def evaluate(
    corpus: Corpus,
    *,
    dimensions: Iterable[str] | None = None,
    tolerance: float = 0.0,
    registry: CheckRegistry | None = None,
) -> DiffResult:
    """Evaluate a corpus with a one-call convenience wrapper.

    Args:
        corpus: The cases to evaluate.
        dimensions: Names of the dimensions to evaluate. ``None`` evaluates all.
        tolerance: Absolute score difference treated as a tie.
        registry: Registry to draw dimensions from. Defaults to the built-ins.

    Returns:
        The :class:`DiffResult` for this evaluation.
    """
    config = DiffConfig(
        dimensions=tuple(dimensions) if dimensions is not None else None,
        tolerance=tolerance,
    )
    return DiffEngine(registry=registry, config=config).run(corpus)
