"""BLEU-style n-gram precision with brevity penalty, stdlib only."""

from __future__ import annotations

import math
from collections import Counter

from llmdiff.checks.ngram import tokenize
from llmdiff.core.checks import CheckContext

__all__ = ["BleuCheck", "bleu_score", "brevity_penalty"]


def brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """BP = 1 if c > r, else exp(1 - r/c)."""
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - reference_len / candidate_len)


def bleu_score(reference: str, candidate: str, *, max_n: int = 4) -> float:
    """BLEU with uniform weights over 1..max_n n-grams (smoothed)."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not cand_tokens:
        return 0.0
    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        cand_ngrams = Counter(
            tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)
        )
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((ref_ngrams & cand_ngrams).values())
        precisions.append(overlap / sum(cand_ngrams.values()))
    log_sum = sum(math.log(max(p, 1e-12)) for p in precisions)
    geo_mean = math.exp(log_sum / max_n)
    return brevity_penalty(len(cand_tokens), len(ref_tokens)) * geo_mean


class BleuCheck:
    """BLEU score of the output against the baseline reference."""

    def __init__(self, max_n: int = 4) -> None:
        self._max_n = max_n

    @property
    def name(self) -> str:
        return "bleu"

    def score(self, output: str, context: CheckContext) -> float:
        return bleu_score(context.baseline_output, output, max_n=self._max_n)
