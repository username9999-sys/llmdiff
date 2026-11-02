"""Metadata-driven cost dimensions: latency, tokens, and USD cost."""

from __future__ import annotations

from llmdiff.core.checks import CheckContext

__all__ = ["LatencyCheck", "TokenCountCheck", "CostCheck"]


def _meta_float(context: CheckContext, key: str) -> float | None:
    value = context.metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LatencyCheck:
    """Score from ``baseline_ms``/``candidate_ms`` metadata.

    1.0 when the candidate is at least as fast as the baseline; decays with
    slowdown ratio. Neutral (1.0) when metadata is absent.
    """

    @property
    def name(self) -> str:
        return "latency"

    def score(self, output: str, context: CheckContext) -> float:
        base = _meta_float(context, "baseline_ms")
        cand = _meta_float(context, "candidate_ms")
        if base is None or cand is None or base <= 0:
            return 1.0
        ratio = cand / base
        if ratio <= 1.0:
            return 1.0
        return max(0.0, 1.0 - (ratio - 1.0) / 4.0)


class TokenCountCheck:
    """Score from token-count metadata; 1.0 when candidate uses no more tokens."""

    @property
    def name(self) -> str:
        return "tokens"

    def score(self, output: str, context: CheckContext) -> float:
        base = _meta_float(context, "baseline_tokens_out") or 0.0
        cand = _meta_float(context, "candidate_tokens_out") or 0.0
        base_in = _meta_float(context, "baseline_tokens_in") or 0.0
        cand_in = _meta_float(context, "candidate_tokens_in") or 0.0
        base_total = base + base_in
        cand_total = cand + cand_in
        if base_total <= 0 or cand_total <= 0:
            return 1.0
        if cand_total <= base_total:
            return 1.0
        return max(0.0, 1.0 - (cand_total - base_total) / base_total)


class CostCheck:
    """Score from ``baseline_cost_usd``/``candidate_cost_usd`` metadata."""

    @property
    def name(self) -> str:
        return "cost"

    def score(self, output: str, context: CheckContext) -> float:
        base = _meta_float(context, "baseline_cost_usd")
        cand = _meta_float(context, "candidate_cost_usd")
        if base is None or cand is None or base <= 0:
            return 1.0
        if cand <= base:
            return 1.0
        return max(0.0, 1.0 - (cand - base) / base)
