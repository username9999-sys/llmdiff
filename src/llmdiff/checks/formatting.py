"""Whitespace/formatting hygiene dimensions for plain text outputs."""

from __future__ import annotations

from llmdiff.core.checks import CheckContext

__all__ = [
    "LeadingBlankLinesCheck",
    "TrailingBlankLinesCheck",
    "EndsWithNewlineCheck",
    "TabCharCheck",
    "BlankLineRunCheck",
    "LineCountCheck",
]

_MAX = 200_000


class LeadingBlankLinesCheck:
    """1.0 when the number of leading blank lines matches the baseline."""

    @property
    def name(self) -> str:
        return "leading_blank"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            n = 0
            for ln in text[:_MAX].splitlines():
                if ln.strip() == "":
                    n += 1
                else:
                    break
            return n

        base = count(context.baseline_output)
        cand = count(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class TrailingBlankLinesCheck:
    """1.0 when the number of trailing blank lines matches the baseline."""

    @property
    def name(self) -> str:
        return "trailing_blank"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            lines = text[:_MAX].splitlines()
            n = 0
            for ln in reversed(lines):
                if ln.strip() == "":
                    n += 1
                else:
                    break
            return n

        base = count(context.baseline_output)
        cand = count(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class EndsWithNewlineCheck:
    """1.0 when the output ends with a newline iff the baseline does."""

    @property
    def name(self) -> str:
        return "ends_newline"

    def score(self, output: str, context: CheckContext) -> float:
        base = context.baseline_output[:_MAX].endswith("\n")
        cand = output[:_MAX].endswith("\n")
        return 1.0 if base == cand else 0.0


class TabCharCheck:
    """1.0 when tab characters appear iff the baseline has them."""

    @property
    def name(self) -> str:
        return "tabs"

    def score(self, output: str, context: CheckContext) -> float:
        base = "\t" in context.baseline_output[:_MAX]
        cand = "\t" in output[:_MAX]
        return 1.0 if base == cand else 0.0


class BlankLineRunCheck:
    """1.0 when the max run of consecutive blank lines matches the baseline."""

    @property
    def name(self) -> str:
        return "blank_run"

    def score(self, output: str, context: CheckContext) -> float:
        def max_run(text: str) -> int:
            best = run = 0
            for ln in text[:_MAX].splitlines():
                if ln.strip() == "":
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            return best

        base = max_run(context.baseline_output)
        cand = max_run(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class LineCountCheck:
    """1.0 when line count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "line_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(context.baseline_output[:_MAX].splitlines())
        cand = len(output[:_MAX].splitlines())
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)
