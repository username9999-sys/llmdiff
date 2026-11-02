"""Aggregation of per-case check results into the verdict matrix.

The verdict matrix is the product's signature output: for every comparison
dimension it reports how many cases the candidate won, tied or lost, plus the
mean score on each side so a reader can see the size of the shift, not just
its direction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llmdiff.core.checks import CheckResult, Verdict

__all__ = ["DimensionSummary", "VerdictMatrix"]


@dataclass(frozen=True, slots=True)
class DimensionSummary:
    """Win/tie/loss counts and mean scores for one comparison dimension.

    Attributes:
        dimension: Name of the comparison dimension.
        wins: Cases where the candidate scored strictly higher.
        ties: Cases where the scores were equal within tolerance.
        losses: Cases where the candidate scored strictly lower.
        baseline_mean: Mean baseline score across all evaluated cases.
        candidate_mean: Mean candidate score across all evaluated cases.
        details: Per-case results, keyed by case id.
    """

    dimension: str
    wins: int = 0
    ties: int = 0
    losses: int = 0
    baseline_mean: float = 0.0
    candidate_mean: float = 0.0
    details: Mapping[str, CheckResult] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of cases evaluated for this dimension."""
        return self.wins + self.ties + self.losses

    @property
    def delta(self) -> float:
        """Difference between the mean candidate and baseline scores."""
        return self.candidate_mean - self.baseline_mean

    @property
    def win_rate(self) -> float:
        """Fraction of cases the candidate won, between 0.0 and 1.0."""
        return self.wins / self.total if self.total else 0.0

    @property
    def loss_rate(self) -> float:
        """Fraction of cases the candidate lost, between 0.0 and 1.0."""
        return self.losses / self.total if self.total else 0.0

    @property
    def verdict(self) -> str:
        """Dominant verdict for this dimension: ``win``, ``tie`` or ``loss``."""
        return Verdict.from_counts(self.wins, self.ties, self.losses)

    @property
    def is_regression(self) -> bool:
        """True when losses outnumber wins for this dimension."""
        return self.losses > self.wins

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this summary."""
        return {
            "dimension": self.dimension,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "total": self.total,
            "win_rate": round(self.win_rate, 6),
            "loss_rate": round(self.loss_rate, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "delta": round(self.delta, 6),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class VerdictMatrix:
    """The full verdict matrix across every evaluated dimension.

    Attributes:
        dimensions: Per-dimension summaries, ordered by first appearance.
        total_cases: Number of cases in the evaluated corpus.
        case_ids: Ids of the evaluated cases, in corpus order.
    """

    dimensions: tuple[DimensionSummary, ...] = ()
    total_cases: int = 0
    case_ids: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.dimensions)

    def __len__(self) -> int:
        return len(self.dimensions)

    def __getitem__(self, index: int) -> DimensionSummary:
        return self.dimensions[index]

    @property
    def wins(self) -> int:
        """Total wins across all dimensions."""
        return sum(summary.wins for summary in self.dimensions)

    @property
    def ties(self) -> int:
        """Total ties across all dimensions."""
        return sum(summary.ties for summary in self.dimensions)

    @property
    def losses(self) -> int:
        """Total losses across all dimensions."""
        return sum(summary.losses for summary in self.dimensions)

    @property
    def verdict(self) -> str:
        """Dominant verdict across the whole matrix."""
        return Verdict.from_counts(self.wins, self.ties, self.losses)

    @property
    def regression_rate(self) -> float:
        """Fraction of all comparisons that regressed."""
        total = self.wins + self.ties + self.losses
        return self.losses / total if total else 0.0

    @property
    def regressed_dimensions(self) -> tuple[str, ...]:
        """Names of dimensions where losses outnumber wins."""
        return tuple(summary.dimension for summary in self.dimensions if summary.is_regression)

    def get(self, dimension: str) -> DimensionSummary | None:
        """Return the summary for ``dimension``, or ``None`` when absent."""
        for summary in self.dimensions:
            if summary.dimension == dimension:
                return summary
        return None

    def worst(self, count: int = 1) -> tuple[DimensionSummary, ...]:
        """Return the ``count`` dimensions with the most losses.

        Ties are broken by lower mean candidate score, then by name, so the
        ordering is deterministic.
        """
        ordered = sorted(self.dimensions, key=lambda s: (-s.losses, s.candidate_mean, s.dimension))
        return tuple(ordered[:count])

    def best(self, count: int = 1) -> tuple[DimensionSummary, ...]:
        """Return the ``count`` dimensions with the most wins."""
        ordered = sorted(self.dimensions, key=lambda s: (-s.wins, -s.candidate_mean, s.dimension))
        return tuple(ordered[:count])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this matrix."""
        return {
            "total_cases": self.total_cases,
            "verdict": self.verdict,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "regression_rate": round(self.regression_rate, 6),
            "regressed_dimensions": list(self.regressed_dimensions),
            "dimensions": [summary.to_dict() for summary in self.dimensions],
        }

    def to_rows(self) -> list[list[Any]]:
        """Return the matrix as rows, ready for tabular rendering."""
        return [
            [
                summary.dimension,
                summary.wins,
                summary.ties,
                summary.losses,
                round(summary.baseline_mean, 4),
                round(summary.candidate_mean, 4),
                round(summary.delta, 4),
            ]
            for summary in self.dimensions
        ]

    @classmethod
    def from_results(
        cls,
        results: Iterable[CheckResult],
        *,
        case_ids: Sequence[str] = (),
    ) -> VerdictMatrix:
        """Build a matrix from a flat stream of per-case check results.

        Args:
            results: Every check result produced by the engine.
            case_ids: Ids of the evaluated cases, in corpus order.

        Returns:
            The aggregated verdict matrix.
        """
        buckets: dict[str, list[CheckResult]] = {}
        for result in results:
            buckets.setdefault(result.dimension, []).append(result)

        summaries: list[DimensionSummary] = []
        for dimension, bucket in buckets.items():
            wins = sum(1 for r in bucket if r.verdict is Verdict.WIN)
            ties = sum(1 for r in bucket if r.verdict is Verdict.TIE)
            losses = sum(1 for r in bucket if r.verdict is Verdict.LOSS)
            baseline_mean = sum(r.baseline_score for r in bucket) / len(bucket)
            candidate_mean = sum(r.candidate_score for r in bucket) / len(bucket)
            details = {r.metadata.get("case_id", str(index)): r for index, r in enumerate(bucket)}
            summaries.append(
                DimensionSummary(
                    dimension=dimension,
                    wins=wins,
                    ties=ties,
                    losses=losses,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    details=details,
                )
            )

        total_cases = len(case_ids) or max((len(s.details) for s in summaries), default=0)
        return cls(
            dimensions=tuple(summaries),
            total_cases=total_cases,
            case_ids=tuple(case_ids),
        )
