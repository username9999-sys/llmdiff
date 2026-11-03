"""More text-quality dimensions: punctuation, capitalization, whitespace, quotes."""

from __future__ import annotations

import re
import string

from llmdiff.core.checks import CheckContext

__all__ = [
    "PunctuationCheck",
    "CapitalizationCheck",
    "WhitespaceCheck",
    "QuoteBalanceCheck",
    "TrailingWhitespaceCheck",
    "DoubleSpaceCheck",
]

_MAX = 200_000


class PunctuationCheck:
    """1.0 when punctuation count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "punctuation"

    def score(self, output: str, context: CheckContext) -> float:
        base = sum(1 for c in context.baseline_output[:_MAX] if c in string.punctuation)
        cand = sum(1 for c in output[:_MAX] if c in string.punctuation)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class CapitalizationCheck:
    """1.0 when the fraction of capitalized words matches the baseline."""

    @property
    def name(self) -> str:
        return "capitalization"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            words = re.findall(r"[A-Za-z]+", text[:_MAX])
            if not words:
                return 0.0
            return sum(1 for w in words if w[0].isupper()) / len(words)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        return max(0.0, 1.0 - abs(cand - base))


class WhitespaceCheck:
    """1.0 when whitespace char count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "whitespace"

    def score(self, output: str, context: CheckContext) -> float:
        base = sum(1 for c in context.baseline_output[:_MAX] if c.isspace())
        cand = sum(1 for c in output[:_MAX] if c.isspace())
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class QuoteBalanceCheck:
    """1.0 when double-quote count is even (balanced), else 0."""

    @property
    def name(self) -> str:
        return "quote_balance"

    def score(self, output: str, context: CheckContext) -> float:
        return 1.0 if output[:_MAX].count('"') % 2 == 0 else 0.0


class TrailingWhitespaceCheck:
    """1.0 when no line has trailing whitespace."""

    @property
    def name(self) -> str:
        return "trailing_ws"

    def score(self, output: str, context: CheckContext) -> float:
        lines = output[:_MAX].splitlines()
        if not lines:
            return 1.0
        bad = sum(1 for ln in lines if ln != ln.rstrip())
        return max(0.0, 1.0 - bad / len(lines))


class DoubleSpaceCheck:
    """1.0 when no double spaces appear mid-line."""

    @property
    def name(self) -> str:
        return "double_space"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        hits = len(re.findall(r"\S  +\S", sample))
        if hits == 0:
            return 1.0
        return max(0.0, 1.0 - hits / 50.0)
