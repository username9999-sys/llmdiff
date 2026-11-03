"""Character-level similarity dimensions: trigram cosine and Jaro-Winkler."""

from __future__ import annotations

import math
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = ["TrigramSimilarityCheck", "JaroWinklerCheck", "trigram_cosine", "jaro_winkler"]

_MAX = 100_000


def _trigrams(text: str) -> Counter:
    sample = text[:_MAX]
    padded = f"  {sample}  "
    return Counter(padded[i : i + 3] for i in range(len(padded) - 2))


def trigram_cosine(a: str, b: str) -> float:
    """Cosine similarity over character trigram count vectors."""
    va, vb = _trigrams(a), _trigrams(b)
    if not va and not vb:
        return 1.0
    if not va or not vb:
        return 0.0
    dot = sum(v * vb.get(k, 0) for k, v in va.items())
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity in [0, 1] (prefix bonus up to 4 chars)."""
    sa, sb = a[:_MAX], b[:_MAX]
    if sa == sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    la, lb = len(sa), len(sb)
    match_dist = max(0, max(la, lb) // 2 - 1)
    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    for i in range(la):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, lb)
        for j in range(lo, hi):
            if not b_matches[j] and sa[i] == sb[j]:
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(la):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if sa[i] != sb[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    jaro = (matches / la + matches / lb + (matches - transpositions) / matches) / 3.0
    prefix = 0
    for x, y in zip(sa, sb):
        if x == y and prefix < 4:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1.0 - jaro)


class TrigramSimilarityCheck:
    """Cosine similarity over character trigrams vs the baseline."""

    @property
    def name(self) -> str:
        return "trigram"

    def score(self, output: str, context: CheckContext) -> float:
        return trigram_cosine(context.baseline_output, output)


class JaroWinklerCheck:
    """Jaro-Winkler similarity vs the baseline."""

    @property
    def name(self) -> str:
        return "jaro_winkler"

    def score(self, output: str, context: CheckContext) -> float:
        return jaro_winkler(context.baseline_output, output)
