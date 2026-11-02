"""Trend report renderer: compare two runs.

Compares a current :class:`DiffResult` against a previous one (loaded from a
JSON report file produced by :class:`~llmdiff.reporters.json_reporter.JsonReporter`)
and renders how the win/loss rates per dimension moved between the two runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, format_delta, format_pct
from llmdiff.reporters.json_reporter import JsonReporter

__all__ = ["TrendReporter", "DimensionTrend", "load_previous_result"]


def load_previous_result(path: str | Path) -> DiffResult:
    """Load a previous :class:`DiffResult` from a JSON report file.

    Accepts either a full JSON report (as produced by ``JsonReporter.render``)
    or a bare ``DiffResult.to_dict()`` payload.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not valid JSON or lacks the required keys.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"previous result file not found: {target}")
    text = target.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {target}")
    if "matrix" in payload and "cases" in payload:
        return JsonReporter.from_json(text)
    # Bare DiffResult.to_dict() shape: same keys, so the same path works.
    return JsonReporter.from_json(text)


class DimensionTrend:
    """Movement of one dimension between two runs."""

    __slots__ = ("dimension", "prev_win_rate", "curr_win_rate", "prev_loss_rate",
                 "curr_loss_rate", "prev_verdict", "curr_verdict", "delta_win_rate")

    def __init__(
        self,
        dimension: str,
        prev_win_rate: float,
        curr_win_rate: float,
        prev_loss_rate: float,
        curr_loss_rate: float,
        prev_verdict: str,
        curr_verdict: str,
    ) -> None:
        self.dimension = dimension
        self.prev_win_rate = prev_win_rate
        self.curr_win_rate = curr_win_rate
        self.prev_loss_rate = prev_loss_rate
        self.curr_loss_rate = curr_loss_rate
        self.prev_verdict = prev_verdict
        self.curr_verdict = curr_verdict
        self.delta_win_rate = curr_win_rate - prev_win_rate

    @property
    def direction(self) -> str:
        """``improved``, ``regressed`` or ``stable`` based on win-rate movement."""
        if self.delta_win_rate > 1e-9:
            return "improved"
        if self.delta_win_rate < -1e-9:
            return "regressed"
        return "stable"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form of this trend."""
        return {
            "dimension": self.dimension,
            "prev_win_rate": round(self.prev_win_rate, 6),
            "curr_win_rate": round(self.curr_win_rate, 6),
            "prev_loss_rate": round(self.prev_loss_rate, 6),
            "curr_loss_rate": round(self.curr_loss_rate, 6),
            "delta_win_rate": round(self.delta_win_rate, 6),
            "prev_verdict": self.prev_verdict,
            "curr_verdict": self.curr_verdict,
            "direction": self.direction,
        }


class TrendReporter(BaseReporter):
    """Render a trend report comparing a current run to a previous one."""

    name = "trend"

    def __init__(self, *, title: str = "llmdiff trend") -> None:
        self.title = title

    def render(self, result: DiffResult, previous: DiffResult) -> str:
        """Render a markdown trend report for ``result`` vs ``previous``.

        Note: unlike other reporters this takes two results; the registry's
        ``render`` protocol passes only one, so call this method directly or
        use :meth:`render_with_file`.
        """
        trends = self.compare(result, previous)
        lines = [
            f"# {self.title}",
            "",
            self._overall_line(result, previous),
            "",
            "| Dimension | Prev verdict | Curr verdict | Prev win% | Curr win% | "
            "Δ win% | Prev loss% | Curr loss% | Trend |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for trend in trends:
            arrow = {"improved": "📈", "regressed": "📉", "stable": "➖"}[trend.direction]
            row = (
                f"| `{trend.dimension}` | {trend.prev_verdict} | {trend.curr_verdict} | "
                f"{format_pct(trend.prev_win_rate, digits=0)} | "
                f"{format_pct(trend.curr_win_rate, digits=0)} | "
                f"{format_delta(trend.delta_win_rate * 100, digits=1, sign=True)}pp | "
                f"{format_pct(trend.prev_loss_rate, digits=0)} | "
                f"{format_pct(trend.curr_loss_rate, digits=0)} | "
                f"{arrow} {trend.direction} |"
            )
            lines.append(row)
        regressed = [t for t in trends if t.direction == "regressed"]
        lines.append("")
        if regressed:
            lines.append(
                "**Regressed since previous run:** "
                + ", ".join(f"`{t.dimension}`" for t in regressed)
            )
        else:
            lines.append("**No dimension regressed since the previous run.**")
        return "\n".join(lines) + "\n"

    def render_with_file(self, result: DiffResult, previous_path: str | Path) -> str:
        """Load the previous result from ``previous_path`` and render the trend."""
        return self.render(result, load_previous_result(previous_path))

    def compare(self, current: DiffResult, previous: DiffResult) -> list[DimensionTrend]:
        """Compute per-dimension trends between ``current`` and ``previous``."""
        trends: list[DimensionTrend] = []
        curr_dims = {s.dimension: s for s in current.matrix.dimensions}
        prev_dims = {s.dimension: s for s in previous.matrix.dimensions}
        for dimension in sorted(set(curr_dims) | set(prev_dims)):
            curr = curr_dims.get(dimension)
            prev = prev_dims.get(dimension)
            trends.append(
                DimensionTrend(
                    dimension=dimension,
                    prev_win_rate=prev.win_rate if prev else 0.0,
                    curr_win_rate=curr.win_rate if curr else 0.0,
                    prev_loss_rate=prev.loss_rate if prev else 0.0,
                    curr_loss_rate=curr.loss_rate if curr else 0.0,
                    prev_verdict=prev.verdict if prev else "n/a",
                    curr_verdict=curr.verdict if curr else "n/a",
                )
            )
        return trends

    def render_json(self, result: DiffResult, previous: DiffResult) -> str:
        """Render the trend comparison as JSON."""
        payload = {
            "current": {
                "verdict": result.matrix.verdict,
                "wins": result.matrix.wins,
                "ties": result.matrix.ties,
                "losses": result.matrix.losses,
                "regression_rate": round(result.matrix.regression_rate, 6),
            },
            "previous": {
                "verdict": previous.matrix.verdict,
                "wins": previous.matrix.wins,
                "ties": previous.matrix.ties,
                "losses": previous.matrix.losses,
                "regression_rate": round(previous.matrix.regression_rate, 6),
            },
            "dimensions": [t.to_dict() for t in self.compare(result, previous)],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _overall_line(current: DiffResult, previous: DiffResult) -> str:
        c, p = current.matrix, previous.matrix
        delta_pp = format_delta((c.regression_rate - p.regression_rate) * 100, digits=1)
        return (
            f"Overall: **{c.verdict}** ({c.wins}W/{c.ties}T/{c.losses}L, "
            f"{format_pct(c.regression_rate)} regressions) vs previous **{p.verdict}** "
            f"({p.wins}W/{p.ties}T/{p.losses}L, {format_pct(p.regression_rate)}) — "
            f"regression-rate delta {delta_pp}pp."
        )
