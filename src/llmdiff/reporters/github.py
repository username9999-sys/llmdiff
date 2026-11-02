"""GitHub PR comment / status-check renderer.

Produces a markdown body suitable for a pull-request comment, using
collapsible ``<details>`` blocks, plus a conclusion string for the commit
status API (``success`` / ``failure`` / ``neutral``).
"""

from __future__ import annotations

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import (
    BaseReporter,
    escape_md_cell,
    format_delta,
    format_pct,
    format_score,
    verdict_emoji,
    verdict_label,
)

__all__ = ["GithubReporter", "CONCLUSION_SUCCESS", "CONCLUSION_FAILURE", "CONCLUSION_NEUTRAL"]

CONCLUSION_SUCCESS = "success"
CONCLUSION_FAILURE = "failure"
CONCLUSION_NEUTRAL = "neutral"


class GithubReporter(BaseReporter):
    """Render a :class:`DiffResult` as a GitHub PR comment body."""

    name = "github"

    def __init__(self, *, fail_on_loss: bool = True, context: str = "llmdiff") -> None:
        self.fail_on_loss = fail_on_loss
        self.context = context

    def render(self, result: DiffResult) -> str:
        """Render the PR comment markdown for ``result``."""
        matrix = result.matrix
        emoji = verdict_emoji(matrix.verdict)
        lines: list[str] = [
            f"## {emoji} llmdiff: {verdict_label(matrix.verdict)}",
            "",
            f"**{matrix.wins}** wins · **{matrix.ties}** ties · **{matrix.losses}** losses "
            f"across **{matrix.total_cases}** case(s) "
            f"(regression rate {format_pct(matrix.regression_rate)}).",
            "",
            "<details><summary>Verdict matrix</summary>",
            "",
            "| Dimension | Verdict | W | T | L | Baseline | Candidate | Delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in matrix.dimensions:
            row = (
                f"| {escape_md_cell(summary.dimension)} | "
                f"{verdict_emoji(summary.verdict)} {summary.verdict} | "
                f"{summary.wins} | {summary.ties} | {summary.losses} | "
                f"{format_score(summary.baseline_mean)} | "
                f"{format_score(summary.candidate_mean)} | "
                f"{format_delta(summary.delta)} |"
            )
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        lines.append("")
        regressed = result.regressions
        if regressed:
            lines.append(f"<details><summary>❌ Regressed cases ({len(regressed)})</summary>")
            lines.append("")
            for case in regressed:
                lines.append(f"### `{escape_md_cell(case.case_id)}` — {case.verdict}")
                lines.append("")
                lines.append("| Dimension | Baseline | Candidate | Delta | Detail |")
                lines.append("|---|---:|---:|---:|---|")
                for check in case.results:
                    if check.verdict.value != "loss":
                        continue
                    row = (
                        f"| {escape_md_cell(check.dimension)} | "
                        f"{format_score(check.baseline_score)} | "
                        f"{format_score(check.candidate_score)} | "
                        f"{format_delta(check.delta)} | "
                        f"{escape_md_cell(check.detail)} |"
                    )
                    lines.append(row)
                lines.append("")
            lines.append("</details>")
            lines.append("")
        improved = result.improvements
        if improved:
            lines.append(f"<details><summary>✅ Improved cases ({len(improved)})</summary>")
            lines.append("")
            for case in improved:
                lines.append(
                    f"- `{escape_md_cell(case.case_id)}` "
                    f"({case.wins}W/{case.ties}T/{case.losses}L)"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")
        lines.append(
            "<details><summary>All cases</summary>"
        )
        lines.append("")
        lines.append("| Case | Verdict | W | T | L |")
        lines.append("|---|---|---:|---:|---:|")
        for case in result.cases:
            row = (
                f"| {escape_md_cell(case.case_id)} | "
                f"{verdict_emoji(case.verdict)} {case.verdict} | "
                f"{case.wins} | {case.ties} | {case.losses} |"
            )
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        return "\n".join(lines).rstrip() + "\n"

    def conclusion(self, result: DiffResult) -> str:
        """Return the commit-status conclusion for ``result``.

        ``failure`` when any case regressed (and ``fail_on_loss`` is set),
        ``success`` when there are wins and no losses, otherwise ``neutral``.
        """
        matrix = result.matrix
        if self.fail_on_loss and matrix.losses > 0:
            return CONCLUSION_FAILURE
        if matrix.wins > 0 and matrix.losses == 0:
            return CONCLUSION_SUCCESS
        return CONCLUSION_NEUTRAL

    def status_payload(self, result: DiffResult, *, description_max: int = 140) -> dict[str, str]:
        """Return a payload suitable for the GitHub ``POST /statuses`` API."""
        conclusion = self.conclusion(result)
        matrix = result.matrix
        description = (
            f"{matrix.wins}W/{matrix.ties}T/{matrix.losses}L "
            f"({format_pct(matrix.regression_rate)} regressions)"
        )
        if len(description) > description_max:
            description = description[: description_max - 1] + "…"
        return {"state": conclusion, "context": self.context, "description": description}
