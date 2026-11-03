"""ROUGE-style overlap dimensions (ROUGE-1/2/L), stdlib only."""

from __future__ import annotations

import difflib
from collections import Counter

from llmdiff.checks.ngram import tokenize
from llmdiff.core.checks import CheckContext

__all__ = ["Rouge1Check", "Rouge2Check", "RougeLCheck", "rouge_l_f1"]


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1 based on LCS length between token sequences."""
    a = tokenize(reference)
    b = tokenize(candidate)
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    if len(a) * len(b) > 4_000_000:
        return difflib.SequenceMatcher(None, a, b).ratio()
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    prec = lcs / len(b) if b else 0.0
    rec = lcs / len(a) if a else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


class _BaseRouge:
    n: int = 1

    def _score_n(self, output: str, context: CheckContext) -> float:
        base = _ngram_counts(tokenize(context.baseline_output), self.n)
        cand = _ngram_counts(tokenize(output), self.n)
        if not base and not cand:
            return 1.0
        if not base or not cand:
            return 0.0
        overlap = sum((base & cand).values())
        return overlap / sum(base.values())


class Rouge1Check(_BaseRouge):
    """ROUGE-1 recall of baseline unigrams present in the output."""

    n = 1

    @property
    def name(self) -> str:
        return "rouge1"

    def score(self, output: str, context: CheckContext) -> float:
        return self._score_n(output, context)


class Rouge2Check(_BaseRouge):
    """ROUGE-2 recall of baseline bigrams present in the output."""

    n = 2

    @property
    def name(self) -> str:
        return "rouge2"

    def score(self, output: str, context: CheckContext) -> float:
        return self._score_n(output, context)


class RougeLCheck:
    """ROUGE-L F1 between output and baseline."""

    @property
    def name(self) -> str:
        return "rougeL"

    def score(self, output: str, context: CheckContext) -> float:
        return rouge_l_f1(context.baseline_output, output)
