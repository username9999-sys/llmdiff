"""Entropy and repetition dimensions: detect degenerate or looped output."""

from __future__ import annotations

import math
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = ["EntropyCheck", "RepetitionCheck", "shannon_entropy", "repetition_ratio"]

_MAX_CHARS = 200_000


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per char) of ``text``, capped for huge inputs."""
    sample = text[:_MAX_CHARS]
    if not sample:
        return 0.0
    counts = Counter(sample)
    n = len(sample)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def repetition_ratio(text: str) -> float:
    """Fraction of characters that belong to a repeated run (>= 3 same char)."""
    sample = text[:_MAX_CHARS]
    if not sample:
        return 0.0
    repeated = 0
    run_char = sample[0]
    run_len = 1
    for ch in sample[1:]:
        if ch == run_char:
            run_len += 1
        else:
            if run_len >= 3:
                repeated += run_len
            run_char = ch
            run_len = 1
    if run_len >= 3:
        repeated += run_len
    return repeated / len(sample)


class EntropyCheck:
    """Score based on how far the output's entropy is from the baseline's.

    A candidate that collapses into a loop ("aaaa...") or a constant has much
    lower entropy than a healthy baseline, so this scores 1.0 when entropies
    match and decays as they diverge.
    """

    @property
    def name(self) -> str:
        return "entropy"

    def score(self, output: str, context: CheckContext) -> float:
        base_e = shannon_entropy(context.baseline_output)
        cand_e = shannon_entropy(output)
        if base_e == 0.0:
            return 1.0 if cand_e == 0.0 else max(0.0, 1.0 - cand_e / 8.0)
        diff = abs(cand_e - base_e)
        return max(0.0, 1.0 - diff / max(base_e, 1e-9))


class RepetitionCheck:
    """1.0 minus the repetition ratio: penalizes looped/degenerate output."""

    @property
    def name(self) -> str:
        return "repetition"

    def score(self, output: str, context: CheckContext) -> float:
        return 1.0 - repetition_ratio(output)
