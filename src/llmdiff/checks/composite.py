"""Composite dimension: weighted combination of other checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from llmdiff.core.checks import Check, CheckContext

__all__ = ["WeightedCompositeCheck", "composite_from_names"]


class WeightedCompositeCheck:
    """Combine several checks into one score with weights.

    The composite score is the weighted mean of the child scores, so it stays
    in [0, 1]. Weights are normalized internally.
    """

    def __init__(self, name: str, children: Sequence[tuple[Check, float]]) -> None:
        if not children:
            raise ValueError("composite check needs at least one child")
        self._name = name
        self._children = tuple(children)
        total = sum(w for _, w in children)
        if total <= 0:
            raise ValueError("composite weights must sum to a positive value")
        self._weights = tuple(w / total for _, w in children)

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> tuple[Check, ...]:
        return tuple(c for c, _ in self._children)

    def score(self, output: str, context: CheckContext) -> float:
        total = 0.0
        for (check, _), weight in zip(self._children, self._weights, strict=True):
            total += weight * check.score(output, context)
        return max(0.0, min(1.0, total))


def composite_from_names(
    name: str,
    registry: Mapping[str, Check],
    weights: Mapping[str, float],
) -> WeightedCompositeCheck:
    """Build a composite from a registry by dimension names and weights."""
    children = [(registry[dim], weights.get(dim, 1.0)) for dim in weights]
    return WeightedCompositeCheck(name, children)
