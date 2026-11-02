"""Regression gate evaluation for the CLI.

A gate decides whether a :class:`llmdiff.core.DiffResult` (or a serialised
result loaded from disk) passes a set of thresholds.  On failure it raises
:class:`llmdiff.core.RegressionGateError` carrying human-readable failure
descriptions; the CLI catches it and exits with code 1.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llmdiff.core import DiffResult
from llmdiff.core.errors import FormatError, RegressionGateError

__all__ = [
    "GateThresholds",
    "evaluate_gate",
    "gate_from_result",
    "load_result_json",
    "result_from_dict",
]


class GateThresholds:
    """Thresholds a result is checked against.

    Attributes:
        max_loss_rate: Maximum allowed fraction of all comparisons that are
            losses (matrix-level ``regression_rate``).
        max_regressed_dims: Maximum number of dimensions whose own verdict is
            a regression (losses > wins).
        min_win_rate: Minimum required win rate across all comparisons.
    """

    __slots__ = ("max_loss_rate", "max_regressed_dims", "min_win_rate")

    def __init__(
        self,
        *,
        max_loss_rate: float | None = None,
        max_regressed_dims: int | None = None,
        min_win_rate: float | None = None,
    ) -> None:
        # None means "not set" - the corresponding check is skipped. The
        # explicit ``gate`` command applies its own defaults via with_defaults().
        self.max_loss_rate = None if max_loss_rate is None else float(max_loss_rate)
        self.max_regressed_dims = None if max_regressed_dims is None else int(max_regressed_dims)
        self.min_win_rate = None if min_win_rate is None else float(min_win_rate)

    @property
    def enabled(self) -> bool:
        """True when at least one threshold is set."""
        return (
            self.max_loss_rate is not None
            or self.max_regressed_dims is not None
            or self.min_win_rate is not None
        )

    def with_defaults(self) -> GateThresholds:
        """Return a copy with unset thresholds replaced by the gate defaults."""
        return GateThresholds(
            max_loss_rate=0.05 if self.max_loss_rate is None else self.max_loss_rate,
            max_regressed_dims=0 if self.max_regressed_dims is None else self.max_regressed_dims,
            min_win_rate=0.0 if self.min_win_rate is None else self.min_win_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_loss_rate": self.max_loss_rate,
            "max_regressed_dims": self.max_regressed_dims,
            "min_win_rate": self.min_win_rate,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GateThresholds:
        return cls(
            max_loss_rate=data.get("max_loss_rate"),
            max_regressed_dims=data.get("max_regressed_dims"),
            min_win_rate=data.get("min_win_rate"),
        )


def evaluate_gate(result: DiffResult, thresholds: GateThresholds) -> list[str]:
    """Return the list of gate failures for ``result`` (empty means pass)."""
    failures: list[str] = []
    matrix = result.matrix
    total = matrix.wins + matrix.ties + matrix.losses
    loss_rate = matrix.regression_rate
    win_rate = (matrix.wins / total) if total else 0.0
    if thresholds.max_loss_rate is not None and loss_rate > thresholds.max_loss_rate:
        failures.append(
            f"loss rate {loss_rate:.4f} exceeds max_loss_rate {thresholds.max_loss_rate:.4f} "
            f"({matrix.losses}/{total} comparisons lost)"
        )
    regressed = matrix.regressed_dimensions
    if thresholds.max_regressed_dims is not None and len(regressed) > thresholds.max_regressed_dims:
        failures.append(
            f"{len(regressed)} regressed dimension(s) {list(regressed)} exceed "
            f"max_regressed_dims {thresholds.max_regressed_dims}"
        )
        for case in result.regressions:
            dims = ", ".join(r.dimension for r in case.regressions())
            failures.append(f"case {case.case_id!r} regressed on: {dims}")
    if thresholds.min_win_rate is not None and win_rate < thresholds.min_win_rate:
        failures.append(
            f"win rate {win_rate:.4f} below min_win_rate {thresholds.min_win_rate:.4f} "
            f"({matrix.wins}/{total} comparisons won)"
        )
    return failures


def gate_from_result(result: DiffResult, thresholds: GateThresholds) -> None:
    """Raise :class:`RegressionGateError` when ``result`` fails the gate."""
    failures = evaluate_gate(result, thresholds)
    if failures:
        raise RegressionGateError(
            f"regression gate failed with {len(failures)} violation(s)", failures=failures
        )


def result_from_dict(data: Mapping[str, Any]) -> DiffResult:
    """Rebuild a :class:`DiffResult` from its ``to_dict()`` representation.

    Only the aggregate matrix is fully reconstructed; per-case results are
    rebuilt from the ``cases`` list when present.  This is enough for the
    ``gate`` command, which needs matrix-level statistics and per-case
    regression flags.
    """
    from llmdiff.core.checks import CheckResult, Verdict
    from llmdiff.core.matrix import DimensionSummary, VerdictMatrix
    from llmdiff.core.result import CaseDiff

    matrix_data = data.get("matrix") or {}
    summaries = []
    for s in matrix_data.get("dimensions", []):
        details = {
            cid: CheckResult(
                dimension=r.get("dimension", s.get("dimension", "")),
                verdict=Verdict(r.get("verdict", "tie")),
                baseline_score=float(r.get("baseline_score", 0.0)),
                candidate_score=float(r.get("candidate_score", 0.0)),
                detail=r.get("detail", ""),
                metadata=r.get("metadata") or {},
            )
            for cid, r in (s.get("details") or {}).items()
        }
        summaries.append(
            DimensionSummary(
                dimension=s.get("dimension", ""),
                wins=int(s.get("wins", 0)),
                ties=int(s.get("ties", 0)),
                losses=int(s.get("losses", 0)),
                baseline_mean=float(s.get("baseline_mean", 0.0)),
                candidate_mean=float(s.get("candidate_mean", 0.0)),
                details=details,
            )
        )
    matrix = VerdictMatrix(
        dimensions=tuple(summaries),
        total_cases=int(matrix_data.get("total_cases", 0)),
        case_ids=tuple(matrix_data.get("case_ids", [])),
    )
    cases = []
    for c in data.get("cases", []):
        results = tuple(
            CheckResult(
                dimension=r.get("dimension", ""),
                verdict=Verdict(r.get("verdict", "tie")),
                baseline_score=float(r.get("baseline_score", 0.0)),
                candidate_score=float(r.get("candidate_score", 0.0)),
                detail=r.get("detail", ""),
                metadata=r.get("metadata") or {},
            )
            for r in c.get("results", [])
        )
        cases.append(CaseDiff(case_id=c.get("case_id", ""), results=results))
    return DiffResult(matrix=matrix, cases=tuple(cases), config=data.get("config"))


def load_result_json(path: str | Path) -> DiffResult:
    """Load a JSON-serialised DiffResult from ``path`` (``"-"`` = stdin)."""
    if str(path) == "-":
        import sys

        text = sys.stdin.read()
        origin = "<stdin>"
    else:
        target = Path(path)
        if not target.exists():
            raise FormatError(f"result file not found: {target}")
        text = target.read_text(encoding="utf-8")
        origin = str(target)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatError(f"{origin}: invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or "matrix" not in data:
        raise FormatError(f"{origin}: not a llmdiff result JSON (missing 'matrix')")
    return result_from_dict(data)
