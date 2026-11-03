"""Code-mix and token-overlap dimensions."""

from __future__ import annotations

from llmdiff.checks.code import extract_fenced_blocks
from llmdiff.checks.ngram import tokenize
from llmdiff.core.checks import CheckContext

__all__ = ["CodeRatioCheck", "TokenOverlapCheck", "UniqueTokenRatioCheck", "DigitRatioCheck"]

_MAX = 200_000


class CodeRatioCheck:
    """1.0 when the fraction of output inside fenced code blocks matches baseline."""

    @property
    def name(self) -> str:
        return "code_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            sample = text[:_MAX]
            code_len = sum(len(code) for _, code in extract_fenced_blocks(sample))
            return code_len / max(1, len(sample))

        base = ratio(context.baseline_output)
        cand = ratio(output)
        return max(0.0, 1.0 - abs(cand - base))


class TokenOverlapCheck:
    """Jaccard overlap of the full token sets vs baseline."""

    @property
    def name(self) -> str:
        return "token_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        base = set(tokenize(context.baseline_output))
        cand = set(tokenize(output))
        if not base and not cand:
            return 1.0
        union = base | cand
        if not union:
            return 1.0
        return len(base & cand) / len(union)


class UniqueTokenRatioCheck:
    """1.0 when unique-token ratio (unique/total) matches the baseline."""

    @property
    def name(self) -> str:
        return "unique_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            toks = tokenize(text)
            return len(set(toks)) / len(toks) if toks else 1.0

        base = ratio(context.baseline_output)
        cand = ratio(output)
        return max(0.0, 1.0 - abs(cand - base))


class DigitRatioCheck:
    """1.0 when the fraction of digit characters matches the baseline."""

    @property
    def name(self) -> str:
        return "digit_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            sample = text[:_MAX]
            if not sample:
                return 0.0
            return sum(1 for c in sample if c.isdigit()) / len(sample)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
