"""More structural dimensions: headings hierarchy, code fence language, nesting depth."""

from __future__ import annotations

import re

from llmdiff.checks.code import extract_fenced_blocks
from llmdiff.core.checks import CheckContext

__all__ = [
    "HeadingHierarchyCheck",
    "FenceLanguageCheck",
    "NestingDepthCheck",
    "ParagraphCountCheck",
    "AvgWordLengthCheck",
]

_MAX = 200_000
_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)
_PARA_RE = re.compile(r"\n\s*\n")
_WORD_RE = re.compile(r"[A-Za-z']+")


class HeadingHierarchyCheck:
    """1.0 when heading levels never skip (no jump from # to ###)."""

    @property
    def name(self) -> str:
        return "heading_hierarchy"

    def score(self, output: str, context: CheckContext) -> float:
        levels = [len(m.group(1)) for m in _HEADING_RE.finditer(output[:_MAX])]
        if len(levels) < 2:
            return 1.0
        skips = sum(1 for a, b in zip(levels, levels[1:], strict=False) if b > a + 1)
        if skips == 0:
            return 1.0
        return max(0.0, 1.0 - skips / len(levels))


class FenceLanguageCheck:
    """1.0 when fenced code block languages match the baseline set."""

    @property
    def name(self) -> str:
        return "fence_language"

    def score(self, output: str, context: CheckContext) -> float:
        base = {lang for lang, _ in extract_fenced_blocks(context.baseline_output)}
        cand = {lang for lang, _ in extract_fenced_blocks(output)}
        if not base:
            return 1.0 if not cand else 0.0
        union = base | cand
        if not union:
            return 1.0
        return len(base & cand) / len(union)


class NestingDepthCheck:
    """1.0 when max markdown list nesting depth matches the baseline."""

    @property
    def name(self) -> str:
        return "nesting_depth"

    @staticmethod
    def _max_depth(text: str) -> int:
        best = 0
        for line in text[:_MAX].splitlines():
            m = re.match(r"^(\s*)[-*+]\s", line)
            if m:
                best = max(best, len(m.group(1)) // 2 + 1)
        return best

    def score(self, output: str, context: CheckContext) -> float:
        base = self._max_depth(context.baseline_output)
        cand = self._max_depth(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class ParagraphCountCheck:
    """1.0 when paragraph (blank-line separated) count matches the baseline."""

    @property
    def name(self) -> str:
        return "paragraphs"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_PARA_RE.findall(context.baseline_output[:_MAX])) + 1
        cand = len(_PARA_RE.findall(output[:_MAX])) + 1
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class AvgWordLengthCheck:
    """1.0 when average word length is close to the baseline's."""

    @property
    def name(self) -> str:
        return "avg_word_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            words = _WORD_RE.findall(text[:_MAX])
            return sum(len(w) for w in words) / len(words) if words else 0.0

        base = avg(context.baseline_output)
        cand = avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
