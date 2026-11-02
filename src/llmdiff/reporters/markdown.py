"""Full markdown report renderer.

Produces a GitHub-flavoured markdown document with a verdict header, the
win/tie/loss matrix table, per-dimension sections, a regressed-cases section
with per-check details, and a config footer. Also exposes
:meth:`MarkdownReporter.render_summary` for a one-paragraph digest suitable
for chat tools.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llmdiff.core.result import CaseDiff, DiffResult
from llmdiff.reporters.base import (
    BaseReporter,
    config_to_dict,
    escape_md_cell,
    format_delta,
    format_pct,
    format_score,
    verdict_emoji,
    verdict_label,
)

__all__ = ["MarkdownReporter"]


class MarkdownReporter(BaseReporter):
    """Render a :class:`DiffResult` as a full markdown report."""

    name = "markdown"

    def __init__(self, *, title: str = "llmdiff report", show_config: bool = True) -> None:
        self.title = title
        self.show_config = show_config

    # ------------------------------------------------------------------ API

    def render(self, result: DiffResult) -> str:
        """Render the full markdown report for ``result``."""
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(self._header(result))
        lines.append("")
        lines.append(self._matrix_section(result))
        lines.append("")
        lines.extend(self._dimension_sections(result))
        lines.append(self._regressions_section(result))
        lines.append("")
        lines.extend(self._improvements_section(result))
        if self.show_config:
            lines.append("")
            lines.extend(self._config_footer(result))
        return "\n".join(lines).rstrip() + "\n"

    def render_summary(self, result: DiffResult) -> str:
        """Render a one-paragraph plain-text/markdown summary of ``result``."""
        matrix = result.matrix
        verdict = matrix.verdict
        emoji = verdict_emoji(verdict)
        parts = [
            f"{emoji} **{verdict_label(verdict)}** across {matrix.total_cases} case(s): ",
            f"{matrix.wins} win(s), {matrix.ties} tie(s), {matrix.losses} loss(es)",
            f" ({format_pct(matrix.regression_rate)} regression rate).",
        ]
        if matrix.regressed_dimensions:
            dims = ", ".join(f"`{d}`" for d in matrix.regressed_dimensions)
            parts.append(f" Regressed dimensions: {dims}.")
        worst = matrix.worst(1)
        if worst and worst[0].losses:
            w = worst[0]
            parts.append(
                f" Worst dimension: `{w.dimension}` "
                f"({w.losses} loss(es), delta {format_delta(w.delta)})."
            )
        return "".join(parts)

    # ------------------------------------------------------------- sections

    def _header(self, result: DiffResult) -> str:
        matrix = result.matrix
        emoji = verdict_emoji(matrix.verdict)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"> Verdict: {emoji} **{matrix.verdict.upper()}** · "
            f"{matrix.wins}W / {matrix.ties}T / {matrix.losses}L · "
            f"{matrix.total_cases} case(s) · {len(matrix.dimensions)} dimension(s) · "
            f"generated {stamp}"
        )

    def _matrix_section(self, result: DiffResult) -> str:
        matrix = result.matrix
        lines = [
            "## Verdict matrix",
            "",
            "| Dimension | Verdict | Wins | Ties | Losses | Baseline mean | "
            "Candidate mean | Delta | Win rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in matrix.dimensions:
            row = (
                f"| {escape_md_cell(summary.dimension)} | "
                f"{verdict_emoji(summary.verdict)} {summary.verdict} | "
                f"{summary.wins} | {summary.ties} | {summary.losses} | "
                f"{format_score(summary.baseline_mean)} | "
                f"{format_score(summary.candidate_mean)} | "
                f"{format_delta(summary.delta)} | "
                f"{format_pct(summary.win_rate)} |"
            )
            lines.append(row)
        lines.append("")
        lines.append(
            f"**Totals:** {matrix.wins} wins · {matrix.ties} ties · {matrix.losses} losses · "
            f"regression rate {format_pct(matrix.regression_rate)}"
        )
        if matrix.regressed_dimensions:
            lines.append("")
            lines.append(
                "**Regressed dimensions:** "
                + ", ".join(f"`{d}`" for d in matrix.regressed_dimensions)
            )
        return "\n".join(lines)

    def _dimension_sections(self, result: DiffResult) -> list[str]:
        lines: list[str] = ["## Dimensions", ""]
        for summary in result.matrix.dimensions:
            emoji = verdict_emoji(summary.verdict)
            lines.append(
                f"### {emoji} `{summary.dimension}` — {summary.verdict} "
                f"({summary.wins}W/{summary.ties}T/{summary.losses}L)"
            )
            lines.append("")
            lines.append(
                f"- baseline mean: {format_score(summary.baseline_mean)} · "
                f"candidate mean: {format_score(summary.candidate_mean)} · "
                f"delta: {format_delta(summary.delta)}"
            )
            lines.append(f"- win rate: {format_pct(summary.win_rate)} · "
                         f"loss rate: {format_pct(summary.loss_rate)}")
            per_case = self._dimension_case_table(result, summary.dimension)
            if per_case:
                lines.append("")
                lines.extend(per_case)
            lines.append("")
        return lines

    def _dimension_case_table(self, result: DiffResult, dimension: str) -> list[str]:
        rows = [
            "| Case | Verdict | Baseline | Candidate | Delta | Detail |",
            "|---|---|---:|---:|---:|---|",
        ]
        found = False
        for case in result.cases:
            for check in case.results:
                if check.dimension != dimension:
                    continue
                found = True
                row = (
                    f"| {escape_md_cell(case.case_id)} | "
                    f"{verdict_emoji(check.verdict.value)} {check.verdict.value} | "
                    f"{format_score(check.baseline_score)} | "
                    f"{format_score(check.candidate_score)} | "
                    f"{format_delta(check.delta)} | "
                    f"{escape_md_cell(check.detail)} |"
                )
                rows.append(row)
        return rows if found else []

    def _regressions_section(self, result: DiffResult) -> str:
        lines = ["## Regressed cases", ""]
        regressed = result.regressions
        if not regressed:
            lines.append("_No regressed cases. 🎉_")
            return "\n".join(lines)
        lines.append(f"{len(regressed)} case(s) regressed (losses outnumber wins):")
        lines.append("")
        for case in regressed:
            lines.extend(self._case_block(case, heading="###"))
        return "\n".join(lines)

    def _improvements_section(self, result: DiffResult) -> list[str]:
        lines = ["## Improved cases", ""]
        improved = result.improvements
        if not improved:
            lines.append("_No improved cases._")
            return lines
        for case in improved:
            lines.extend(self._case_block(case, heading="###"))
        return lines

    def _case_block(self, case: CaseDiff, *, heading: str) -> list[str]:
        emoji = verdict_emoji(case.verdict)
        lines = [
            f"{heading} {emoji} `{escape_md_cell(case.case_id)}` — "
            f"{case.verdict} ({case.wins}W/{case.ties}T/{case.losses}L)",
            "",
            "| Dimension | Verdict | Baseline | Candidate | Delta | Detail |",
            "|---|---|---:|---:|---:|---|",
        ]
        for check in case.results:
            row = (
                f"| {escape_md_cell(check.dimension)} | "
                f"{verdict_emoji(check.verdict.value)} {check.verdict.value} | "
                f"{format_score(check.baseline_score)} | "
                f"{format_score(check.candidate_score)} | "
                f"{format_delta(check.delta)} | "
                f"{escape_md_cell(check.detail)} |"
            )
            lines.append(row)
        lines.append("")
        return lines

    def _config_footer(self, result: DiffResult) -> list[str]:
        lines = ["## Configuration", "", "```json", self._config_json(result), "```"]
        return lines

    @staticmethod
    def _config_json(result: DiffResult) -> str:
        import json

        payload: dict[str, Any] = {
            "total_cases": result.matrix.total_cases,
            "verdict": result.matrix.verdict,
            "config": config_to_dict(result.config),
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
