"""SARIF 2.1.0 report renderer for GitHub code scanning.

Emits a SARIF log with one ``result`` per regressed (case, dimension) pair.
Severity mapping: dimensions whose name suggests safety/correctness
(``exact``, ``json_valid``, ``structure``, ``content``) map to ``error``;
everything else maps to ``warning``. All strings are JSON-encoded by
:mod:`json`, so no manual escaping is needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, config_to_dict

__all__ = ["SarifReporter", "SARIF_VERSION", "ERROR_DIMENSIONS"]

#: SARIF spec version emitted by this reporter.
SARIF_VERSION = "2.1.0"

#: Dimensions treated as hard errors (correctness/safety) rather than warnings.
ERROR_DIMENSIONS = frozenset({"exact", "json_valid", "structure", "content"})

#: Tool driver metadata.
_TOOL_DRIVER = {
    "name": "llmdiff",
    "informationUri": "https://github.com/username9999-sys/llmdiff",
    "version": "0.1.0",
    "rules": [],
}


class SarifReporter(BaseReporter):
    """Render a :class:`DiffResult` as a SARIF 2.1.0 JSON document."""

    name = "sarif"

    def __init__(self, *, pretty: bool = True) -> None:
        self.pretty = pretty

    def render(self, result: DiffResult) -> str:
        """Render ``result`` to a SARIF JSON string."""
        payload = self.to_payload(result)
        if self.pretty:
            return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)

    def to_payload(self, result: DiffResult) -> dict[str, Any]:
        """Build the SARIF payload for ``result``."""
        rules = self._rules(result)
        results = self._results(result)
        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": {"driver": {**_TOOL_DRIVER, "rules": rules}},
                    "results": results,
                    "properties": {
                        "total_cases": result.matrix.total_cases,
                        "verdict": result.matrix.verdict,
                        "wins": result.matrix.wins,
                        "ties": result.matrix.ties,
                        "losses": result.matrix.losses,
                        "regression_rate": round(result.matrix.regression_rate, 6),
                        "config": config_to_dict(result.config),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ],
        }

    def _rules(self, result: DiffResult) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        seen: set[str] = set()
        for summary in result.matrix.dimensions:
            if summary.dimension in seen:
                continue
            seen.add(summary.dimension)
            level = "error" if summary.dimension in ERROR_DIMENSIONS else "warning"
            rules.append(
                {
                    "id": f"llmdiff/{summary.dimension}",
                    "name": f"{summary.dimension} regression",
                    "shortDescription": {
                        "text": f"Candidate regressed on dimension '{summary.dimension}'"
                    },
                    "fullDescription": {
                        "text": (
                            f"The candidate output scored lower than the baseline on the "
                            f"'{summary.dimension}' dimension "
                            f"(baseline mean {summary.baseline_mean:.4f}, candidate mean "
                            f"{summary.candidate_mean:.4f})."
                        )
                    },
                    "defaultConfiguration": {"level": level},
                    "properties": {"dimension": summary.dimension, "verdict": summary.verdict},
                }
            )
        return rules

    def _results(self, result: DiffResult) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for case in result.cases:
            for check in case.results:
                if check.verdict.value != "loss":
                    continue
                level = "error" if check.dimension in ERROR_DIMENSIONS else "warning"
                results.append(
                    {
                        "ruleId": f"llmdiff/{check.dimension}",
                        "level": level,
                        "message": {
                            "text": (
                                f"Case '{case.case_id}' regressed on '{check.dimension}': "
                                f"{check.baseline_score:.4f} -> {check.candidate_score:.4f} "
                                f"(delta {check.delta:+.4f}). {check.detail}"
                            )
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": f"case://{case.case_id}"},
                                    "region": {"snippet": {"text": check.detail or ""}},
                                }
                            }
                        ],
                        "properties": {
                            "case_id": case.case_id,
                            "dimension": check.dimension,
                            "baseline_score": check.baseline_score,
                            "candidate_score": check.candidate_score,
                            "delta": check.delta,
                            "metadata": dict(check.metadata),
                        },
                    }
                )
        return results
