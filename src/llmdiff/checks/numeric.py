"""Numeric dimensions: number extraction, tolerance comparison, drift."""

from __future__ import annotations

from llmdiff.checks.qa import extract_numbers
from llmdiff.core.checks import CheckContext

__all__ = ["NumberPreservedCheck", "NumberDriftCheck", "MeanAbsoluteErrorCheck"]


class NumberPreservedCheck:
    """1.0 when all baseline numbers appear in the output (exact match)."""

    @property
    def name(self) -> str:
        return "numbers_preserved"

    def score(self, output: str, context: CheckContext) -> float:
        base = extract_numbers(context.baseline_output)
        if not base:
            return 1.0
        cand = set(extract_numbers(output))
        preserved = sum(1 for b in base if b in cand)
        return preserved / len(base)


class NumberDriftCheck:
    """1.0 minus normalized mean absolute drift of matched numbers."""

    @property
    def name(self) -> str:
        return "number_drift"

    def score(self, output: str, context: CheckContext) -> float:
        base = extract_numbers(context.baseline_output)
        cand = extract_numbers(output)
        if not base:
            return 1.0
        if not cand:
            return 0.0
        drifts = []
        for b in base:
            nearest = min(cand, key=lambda c: abs(c - b))
            denom = max(abs(b), 1e-9)
            drifts.append(abs(nearest - b) / denom)
        mae = sum(drifts) / len(drifts)
        return max(0.0, 1.0 - mae)


class MeanAbsoluteErrorCheck:
    """Inverse normalized MAE between the sorted number multisets."""

    @property
    def name(self) -> str:
        return "number_mae"

    def score(self, output: str, context: CheckContext) -> float:
        base = sorted(extract_numbers(context.baseline_output))
        cand = sorted(extract_numbers(output))
        if not base and not cand:
            return 1.0
        if not base or not cand:
            return 0.0
        n = max(len(base), len(cand))
        b = base + [0.0] * (n - len(base))
        c = cand + [0.0] * (n - len(cand))
        scale = max(1.0, max(abs(x) for x in b))
        mae = sum(abs(x - y) for x, y in zip(b, c, strict=True)) / n / scale
        return max(0.0, 1.0 - mae)
