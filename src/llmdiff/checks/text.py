"""Text-level comparison dimensions."""

from __future__ import annotations

import difflib
import re

from llmdiff.core.checks import CheckContext

__all__ = ["ExactMatchCheck", "SimilarityCheck", "LengthCheck", "ContentCheck"]

_REFUSAL_PATTERNS = (
    re.compile(r"^(i'?m sorry|i cannot|i can't|i'm unable|as an ai)", re.IGNORECASE),
)
_TRUNCATION_SUFFIXES = ("...", "\u2026")
_TERMINAL_CHARS = set("!?.\"')]}")


class ExactMatchCheck:
    """1.0 when the output is byte-identical to the baseline, else 0.0."""

    @property
    def name(self) -> str:
        return "exact"

    def score(self, output: str, context: CheckContext) -> float:
        return 1.0 if output == context.baseline_output else 0.0


class SimilarityCheck:
    """difflib sequence ratio between the output and the baseline."""

    @property
    def name(self) -> str:
        return "similarity"

    def score(self, output: str, context: CheckContext) -> float:
        base = context.baseline_output
        if not base:
            return 1.0 if not output else 0.0
        return difflib.SequenceMatcher(None, base, output).ratio()


class LengthCheck:
    """Full score while length stays within [0.5x, 2x] of the baseline."""

    @property
    def name(self) -> str:
        return "length"

    def score(self, output: str, context: CheckContext) -> float:
        base_len = max(1, len(context.baseline_output))
        ratio = len(output) / base_len
        if 0.5 <= ratio <= 2.0:
            return 1.0
        if ratio < 0.5:
            return max(0.0, ratio / 0.5)
        return max(0.0, 1.0 - (ratio - 2.0) / 2.0)


class ContentCheck:
    """Heuristic content quality: non-empty, not a refusal, not truncated."""

    @property
    def name(self) -> str:
        return "content"

    def score(self, output: str, context: CheckContext) -> float:
        stripped = output.strip()
        if not stripped:
            return 0.0
        passed = 0
        if not any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS):
            passed += 1
        if not stripped.endswith(_TRUNCATION_SUFFIXES):
            passed += 1
        if stripped[-1].isalnum() or stripped[-1] in _TERMINAL_CHARS:
            passed += 1
        return passed / 3.0
