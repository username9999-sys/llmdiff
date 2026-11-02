"""Per-case and whole-run result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmdiff.core.checks import CheckResult, Verdict
from llmdiff.core.matrix import VerdictMatrix

__all__ = ["CaseDiff", "DiffResult"]


@dataclass(frozen=True, slots=True)
class CaseDiff:
    """Comparison outcome for one case: one CheckResult per dimension."""

    case_id: str
    results: tuple[CheckResult, ...] = ()

    @property
    def wins(self) -> int:
        """Dimensions the candidate won."""
        return sum(1 for r in self.results if r.verdict is Verdict.WIN)

    @property
    def ties(self) -> int:
        """Dimensions that tied."""
        return sum(1 for r in self.results if r.verdict is Verdict.TIE)

    @property
    def losses(self) -> int:
        """Dimensions the candidate lost."""
        return sum(1 for r in self.results if r.verdict is Verdict.LOSS)

    @property
    def verdict(self) -> str:
        """Dominant verdict for this case."""
        return Verdict.from_counts(self.wins, self.ties, self.losses)

    @property
    def is_regression(self) -> bool:
        """True when losses outnumber wins."""
        return self.losses > self.wins

    def regressions(self) -> tuple[CheckResult, ...]:
        """Results that represent a regression."""
        return tuple(r for r in self.results if r.verdict is Verdict.LOSS)

    def improvements(self) -> tuple[CheckResult, ...]:
        """Results that represent an improvement."""
        return tuple(r for r in self.results if r.verdict is Verdict.WIN)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of this case diff."""
        return {
            "case_id": self.case_id,
            "verdict": self.verdict,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Complete outcome of evaluating a corpus.

    Attributes:
        matrix: Aggregated verdict matrix.
        cases: Per-case diffs, in corpus order.
        config: Configuration the run used.
    """

    matrix: VerdictMatrix
    cases: tuple[CaseDiff, ...] = ()
    config: Any = field(default=None)

    @property
    def regressions(self) -> tuple[CaseDiff, ...]:
        """Cases that regressed on more dimensions than they improved."""
        return tuple(case for case in self.cases if case.is_regression)

    @property
    def improvements(self) -> tuple[CaseDiff, ...]:
        """Cases that improved on more dimensions than they regressed."""
        return tuple(case for case in self.cases if case.wins > case.losses)

    def case(self, case_id: str) -> CaseDiff | None:
        """Return the diff for ``case_id``, or None when absent."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of this result."""
        config = self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        return {
            "matrix": self.matrix.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "config": config,
        }
