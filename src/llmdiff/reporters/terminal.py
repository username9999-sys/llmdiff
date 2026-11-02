"""ANSI-coloured terminal report renderer.

Renders the verdict matrix and regressed cases as aligned plain-text tables
with optional ANSI colours. Colour is disabled automatically when stdout is
not a TTY, when the ``NO_COLOR`` environment variable is set (any value), or
when ``force_color``/``force_plain`` is passed explicitly. No external
dependencies.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, format_delta, format_pct, format_score

__all__ = ["TerminalReporter", "Palette"]

_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}
_VERDICT_COLOR = {"win": "green", "tie": "yellow", "loss": "red"}


class Palette:
    """Tiny ANSI colour helper that can be switched off entirely."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(_CODES.get(style, "") for style in styles)
        if not prefix:
            return text
        return f"{prefix}{text}{_RESET}"

    def verdict(self, verdict: str, text: str) -> str:
        color = _VERDICT_COLOR.get(verdict)
        if color is None:
            return text
        return self.paint(text, color, "bold")


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("CLICOLOR") == "0":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class TerminalReporter(BaseReporter):
    """Render a :class:`DiffResult` for the terminal."""

    name = "terminal"

    def __init__(
        self,
        *,
        color: bool | None = None,
        stream: TextIO | None = None,
        width: int = 100,
        show_cases: bool = True,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        if color is None:
            color = _supports_color(self._stream)
        self.palette = Palette(color)
        self.width = max(40, width)
        self.show_cases = show_cases

    # ------------------------------------------------------------------ API

    def render(self, result: DiffResult) -> str:
        """Render ``result`` as an ANSI/plain terminal report string."""
        p = self.palette
        lines: list[str] = []
        matrix = result.matrix
        title = p.paint("llmdiff report", "bold", "cyan")
        lines.append(title)
        lines.append("=" * min(self.width, 72))
        verdict_line = p.verdict(
            matrix.verdict,
            f"VERDICT: {matrix.verdict.upper()}  "
            f"{matrix.wins}W / {matrix.ties}T / {matrix.losses}L  "
            f"({matrix.total_cases} cases, {len(matrix.dimensions)} dimensions, "
            f"regression rate {format_pct(matrix.regression_rate)})",
        )
        lines.append(verdict_line)
        lines.append("")
        lines.extend(self._matrix_table(result))
        if matrix.regressed_dimensions:
            lines.append("")
            lines.append(
                p.paint("Regressed dimensions: ", "bold")
                + ", ".join(matrix.regressed_dimensions)
            )
        if self.show_cases:
            lines.append("")
            lines.extend(self._regressed_cases(result))
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------- sections

    def _matrix_table(self, result: DiffResult) -> list[str]:
        p = self.palette
        headers = ("Dimension", "Verdict", "W", "T", "L", "Base", "Cand", "Delta", "Win%")
        rows: list[list[str]] = []
        for summary in result.matrix.dimensions:
            rows.append(
                [
                    summary.dimension,
                    p.verdict(summary.verdict, summary.verdict.upper()),
                    str(summary.wins),
                    str(summary.ties),
                    str(summary.losses),
                    format_score(summary.baseline_mean),
                    format_score(summary.candidate_mean),
                    format_delta(summary.delta),
                    format_pct(summary.win_rate, digits=0),
                ]
            )
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                # visible length ignores ANSI escapes
                visible = _strip_ansi(cell)
                widths[i] = max(widths[i], len(visible))
        lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
        lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                visible = _strip_ansi(cell)
                pad = widths[i] - len(visible)
                cells.append(cell + " " * pad)
            lines.append("  ".join(cells))
        return lines

    def _regressed_cases(self, result: DiffResult) -> list[str]:
        p = self.palette
        lines = [p.paint("Regressed cases:", "bold")]
        regressed = result.regressions
        if not regressed:
            lines.append(p.paint("  none", "green"))
            return lines
        for case in regressed:
            lines.append(
                f"  {p.verdict(case.verdict, case.case_id)} "
                f"({case.wins}W/{case.ties}T/{case.losses}L)"
            )
            for check in case.regressions():
                lines.append(
                    f"    {p.paint(check.dimension, 'magenta')}: "
                    f"{format_score(check.baseline_score)} -> "
                    f"{format_score(check.candidate_score)} "
                    f"({format_delta(check.delta)}) {check.detail}"
                )
        return lines


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so column widths measure visible text."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\033" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and text[j] not in "mGKHF":
                j += 1
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
