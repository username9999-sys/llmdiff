"""Code-quality hygiene dimensions for code-containing outputs."""

from __future__ import annotations

import re
from collections import Counter

from llmdiff.checks.code import extract_fenced_blocks
from llmdiff.core.checks import CheckContext

__all__ = [
    "IndentationCheck",
    "MaxLineLengthCheck",
    "TodoMarkerCheck",
    "LongLineRatioCheck",
    "TrailingCodeWhitespaceCheck",
]

_MAX = 200_000
_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)


def _code_lines(text: str) -> list[str]:
    return [ln for _, code in extract_fenced_blocks(text) for ln in code.splitlines()]


class IndentationCheck:
    """1.0 when the dominant indent width matches the baseline."""

    @property
    def name(self) -> str:
        return "indent_width"

    @staticmethod
    def _dominant(lines: list[str]) -> int:
        widths = Counter()
        for ln in lines:
            m = re.match(r"^( +)\S", ln)
            if m:
                widths[len(m.group(1))] += 1
        if not widths:
            return 0
        return widths.most_common(1)[0][0]

    def score(self, output: str, context: CheckContext) -> float:
        base = self._dominant(_code_lines(context.baseline_output))
        cand = self._dominant(_code_lines(output))
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class MaxLineLengthCheck:
    """1.0 when the max line length is within the baseline's limit (+10%)."""

    @property
    def name(self) -> str:
        return "max_line_length"

    def score(self, output: str, context: CheckContext) -> float:
        base = max((len(ln) for ln in context.baseline_output[:_MAX].splitlines()), default=0)
        cand = max((len(ln) for ln in output[:_MAX].splitlines()), default=0)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        limit = base * 1.1
        if cand <= limit:
            return 1.0
        return max(0.0, 1.0 - (cand - limit) / max(1, limit))


class LongLineRatioCheck:
    """1.0 when the fraction of lines over 100 chars matches the baseline."""

    @property
    def name(self) -> str:
        return "long_line_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = text[:_MAX].splitlines()
            if not lines:
                return 0.0
            return sum(1 for ln in lines if len(ln) > 100) / len(lines)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        return max(0.0, 1.0 - abs(cand - base))


class TodoMarkerCheck:
    """1.0 when TODO/FIXME marker count is <= baseline (fewer is fine)."""

    @property
    def name(self) -> str:
        return "todo_markers"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_TODO_RE.findall(context.baseline_output[:_MAX]))
        cand = len(_TODO_RE.findall(output[:_MAX]))
        if cand <= base:
            return 1.0
        return max(0.0, 1.0 - (cand - base) / max(1, base))


class TrailingCodeWhitespaceCheck:
    """1.0 when no code line has trailing whitespace."""

    @property
    def name(self) -> str:
        return "code_trailing_ws"

    def score(self, output: str, context: CheckContext) -> float:
        lines = _code_lines(output)
        if not lines:
            return 1.0
        bad = sum(1 for ln in lines if ln != ln.rstrip())
        return max(0.0, 1.0 - bad / len(lines))
