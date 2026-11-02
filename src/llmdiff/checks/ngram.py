"""N-gram overlap dimensions: token-level precision/recall/F1 vs the baseline.

These dimensions catch a candidate that keeps the right words but in the wrong
amount or order. They are pure stdlib and fully deterministic.
"""

from __future__ import annotations

import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "TokenNgramCheck",
    "NgramPrecisionCheck",
    "NgramRecallCheck",
    "NgramF1Check",
    "tokenize",
    "ngrams",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MAX_TOKENS = 100_000


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization, capped at ``_MAX_TOKENS`` tokens."""
    return _TOKEN_RE.findall(text.lower())[:_MAX_TOKENS]


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Return the multiset of n-grams of size ``n`` from ``tokens``."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class _BaseNgramCheck:
    """Shared scoring logic for n-gram overlap checks."""

    n: int = 1

    def _overlap(self, output: str, context: CheckContext) -> tuple[Counter, Counter]:
        base = ngrams(tokenize(context.baseline_output), self.n)
        cand = ngrams(tokenize(output), self.n)
        return base, cand


class TokenNgramCheck(_BaseNgramCheck):
    """Unigram (token) Jaccard similarity against the baseline."""

    n = 1

    @property
    def name(self) -> str:
        return "token_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = self._overlap(output, context)
        if not base and not cand:
            return 1.0
        union = base | cand
        if not union:
            return 1.0
        inter = base & cand
        return sum(inter.values()) / sum(union.values())


class NgramPrecisionCheck(_BaseNgramCheck):
    """Fraction of candidate n-grams that also appear in the baseline."""

    n = 2

    @property
    def name(self) -> str:
        return "ngram_precision"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = self._overlap(output, context)
        total = sum(cand.values())
        if total == 0:
            return 1.0 if sum(base.values()) == 0 else 0.0
        inter = base & cand
        return sum(inter.values()) / total


class NgramRecallCheck(_BaseNgramCheck):
    """Fraction of baseline n-grams that also appear in the candidate."""

    n = 2

    @property
    def name(self) -> str:
        return "ngram_recall"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = self._overlap(output, context)
        total = sum(base.values())
        if total == 0:
            return 1.0 if sum(cand.values()) == 0 else 0.0
        inter = base & cand
        return sum(inter.values()) / total


class NgramF1Check(_BaseNgramCheck):
    """Harmonic mean of n-gram precision and recall against the baseline."""

    n = 3

    @property
    def name(self) -> str:
        return "ngram_f1"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = self._overlap(output, context)
        b_total = sum(base.values())
        c_total = sum(cand.values())
        if b_total == 0 and c_total == 0:
            return 1.0
        inter = base & cand
        i_total = sum(inter.values())
        if i_total == 0:
            return 0.0
        precision = i_total / c_total if c_total else 0.0
        recall = i_total / b_total if b_total else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
