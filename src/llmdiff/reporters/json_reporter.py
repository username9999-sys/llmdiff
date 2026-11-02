"""JSON report renderer.

Serialises a :class:`DiffResult` to JSON with an explicit schema version so
consumers can detect format changes. Supports pretty (indented) and compact
output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, config_to_dict

__all__ = ["JsonReporter", "SCHEMA_VERSION"]

#: Version of the JSON report schema emitted by :class:`JsonReporter`.
SCHEMA_VERSION = "1.0.0"


class JsonReporter(BaseReporter):
    """Render a :class:`DiffResult` as a JSON document."""

    name = "json"

    def __init__(self, *, pretty: bool = True, include_generated_at: bool = True) -> None:
        self.pretty = pretty
        self.include_generated_at = include_generated_at

    def render(self, result: DiffResult) -> str:
        """Render ``result`` to a JSON string."""
        payload = self.to_payload(result)
        if self.pretty:
            return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)

    def to_payload(self, result: DiffResult) -> dict[str, Any]:
        """Build the JSON-serialisable payload for ``result``."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tool": "llmdiff",
            "matrix": result.matrix.to_dict(),
            "cases": [case.to_dict() for case in result.cases],
            "config": config_to_dict(result.config),
        }
        if self.include_generated_at:
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    @staticmethod
    def from_json(text: str) -> DiffResult:
        """Rebuild a :class:`DiffResult` from JSON produced by :meth:`render`.

        This is the inverse of :meth:`render` for the parts that matter for
        trend comparison (matrix + cases). The config is restored as a plain
        dict because the original config object type is not recoverable.

        Raises:
            ValueError: If ``text`` is not valid JSON or lacks required keys.
        """
        from llmdiff.core.checks import CheckResult, Verdict
        from llmdiff.core.matrix import DimensionSummary, VerdictMatrix
        from llmdiff.core.result import CaseDiff

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON report: {exc}") from exc
        if not isinstance(payload, dict) or "matrix" not in payload or "cases" not in payload:
            raise ValueError("JSON report missing required keys 'matrix'/'cases'")

        matrix_data = payload["matrix"]
        dimensions = tuple(
            DimensionSummary(
                dimension=d["dimension"],
                wins=d.get("wins", 0),
                ties=d.get("ties", 0),
                losses=d.get("losses", 0),
                baseline_mean=d.get("baseline_mean", 0.0),
                candidate_mean=d.get("candidate_mean", 0.0),
            )
            for d in matrix_data.get("dimensions", [])
        )
        matrix = VerdictMatrix(
            dimensions=dimensions,
            total_cases=matrix_data.get("total_cases", len(payload.get("cases", []))),
            case_ids=tuple(c.get("case_id", "") for c in payload.get("cases", [])),
        )
        cases = tuple(
            CaseDiff(
                case_id=c["case_id"],
                results=tuple(
                    CheckResult(
                        dimension=r["dimension"],
                        verdict=Verdict(r["verdict"]),
                        baseline_score=r.get("baseline_score", 0.0),
                        candidate_score=r.get("candidate_score", 0.0),
                        detail=r.get("detail", ""),
                        metadata=r.get("metadata", {}),
                    )
                    for r in c.get("results", [])
                ),
            )
            for c in payload.get("cases", [])
        )
        return DiffResult(matrix=matrix, cases=cases, config=payload.get("config"))
