"""Hallucination / grounding heuristics: citation support and entity consistency."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = ["EntityConsistencyCheck", "CitationSupportCheck", "extract_entities"]

_MAX = 200_000
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_CITE_RE = re.compile(r"(\[\d+\]|https?://[^\s)\]]+|\(source:|doi:)", re.IGNORECASE)


def extract_entities(text: str) -> set[str]:
    """Capitalized multi-word sequences as a crude named-entity proxy."""
    return {m.group(0) for m in _ENTITY_RE.finditer(text[:_MAX])}


class EntityConsistencyCheck:
    """Fraction of baseline named entities preserved in the output."""

    @property
    def name(self) -> str:
        return "entity_consistency"

    def score(self, output: str, context: CheckContext) -> float:
        base = extract_entities(context.baseline_output)
        if not base:
            return 1.0
        cand = extract_entities(output)
        preserved = sum(1 for e in base if e in cand)
        return preserved / len(base)


class CitationSupportCheck:
    """1.0 when the output has citations iff the baseline does, else proportional.

    A candidate that drops all citations when the baseline had them scores 0.
    """

    @property
    def name(self) -> str:
        return "citation_support"

    def score(self, output: str, context: CheckContext) -> float:
        base_cites = len(_CITE_RE.findall(context.baseline_output[:_MAX]))
        cand_cites = len(_CITE_RE.findall(output[:_MAX]))
        if base_cites == 0:
            return 1.0 if cand_cites == 0 else max(0.0, 1.0 - cand_cites / 10.0)
        if cand_cites == 0:
            return 0.0
        return min(1.0, cand_cites / base_cites)
