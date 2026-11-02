"""The comparison contract shared by every check dimension.

A *check* scores a single output on a 0.0-1.0 scale. The engine evaluates the
same check against the baseline output and the candidate output, then derives a
verdict:

- candidate score > baseline score  -> ``win``
- candidate score < baseline score  -> ``loss``
- equal (within tolerance)         -> ``tie``

Scoring both sides with one definition is what makes the verdict matrix
trustworthy: a dimension can never be gamed by scoring the candidate against a
different yardstick than the baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from llmdiff.core.case import Case

__all__ = [
    "Verdict",
    "CheckContext",
    "CheckResult",
    "Check",
    "compare_scores",
    "DEFAULT_TOLERANCE",
]

DEFAULT_TOLERANCE = 0.0


class Verdict(str, Enum):
    """Outcome of comparing a candidate output against a baseline output."""

    WIN = "win"
    TIE = "tie"
    LOSS = "loss"

    @property
    def is_regression(self) -> bool:
        """True when this verdict represents a regression."""
        return self is Verdict.LOSS

    @property
    def is_improvement(self) -> bool:
        """True when this verdict represents an improvement."""
        return self is Verdict.WIN

    @classmethod
    def from_counts(cls, wins: int, ties: int, losses: int) -> str:
        """Return the dominant verdict for a set of counts.

        Returns:
            ``"win"``, ``"tie"`` or ``"loss"``. A 1-1 split is reported as
            ``"tie"`` because neither side dominated.
        """
        if wins > losses:
            return cls.WIN.value
        if losses > wins:
            return cls.LOSS.value
        return cls.TIE.value


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check may need besides the output being scored.

    Attributes:
        case: The case the output belongs to.
        metadata: Case metadata, exposed for convenience.
    """

    case: Case
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """The prompt that produced the output."""
        return self.case.prompt

    @property
    def baseline_output(self) -> str:
        """The pre-change output for this case."""
        return self.case.baseline_output

    @property
    def candidate_output(self) -> str:
        """The post-change output for this case."""
        return self.case.candidate_output


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one check on one case.

    Attributes:
        dimension: Name of the check dimension.
        verdict: Whether the candidate improved, regressed or tied.
        baseline_score: Score assigned to the baseline output.
        candidate_score: Score assigned to the candidate output.
        detail: Human-readable explanation of the verdict.
        metadata: Extra structured detail, e.g. the fields that differ.
    """

    dimension: str
    verdict: Verdict
    baseline_score: float
    candidate_score: float
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        """Signed difference between candidate and baseline score."""
        return self.candidate_score - self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this result."""
        return {
            "dimension": self.dimension,
            "verdict": self.verdict.value,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "delta": self.delta,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class Check(Protocol):
    """Protocol every comparison dimension implements.

    A check scores one output. The engine is responsible for calling it twice
    (baseline and candidate) and deriving the verdict.
    """

    @property
    def name(self) -> str:
        """Stable identifier of this dimension."""
        ...

    def score(self, output: str, context: CheckContext) -> float:
        """Score ``output`` between 0.0 and 1.0, where higher is better."""
        ...


def compare_scores(
    baseline_score: float,
    candidate_score: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Verdict:
    """Derive a verdict from two scores.

    Args:
        baseline_score: Score of the baseline output.
        candidate_score: Score of the candidate output.
        tolerance: Absolute difference below which the scores count as equal.

    Returns:
        The :class:`Verdict` implied by the two scores.
    """
    if candidate_score > baseline_score + tolerance:
        return Verdict.WIN
    if candidate_score < baseline_score - tolerance:
        return Verdict.LOSS
    return Verdict.TIE
