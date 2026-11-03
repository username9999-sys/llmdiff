"""Readability dimensions: Flesch-Kincaid grade, syllable estimate, sentence stats."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "FleschKincaidCheck",
    "SyllableCountCheck",
    "SentenceStatsCheck",
    "count_syllables",
    "flesch_kincaid_grade",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")


def count_syllables(word: str) -> int:
    """Heuristic syllable count for an English word (min 1)."""
    w = word.lower().strip("'")
    if not w:
        return 0
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and len(w) > 2:
        w = w[:-1]
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level; higher = harder to read."""
    sample = text[:_MAX]
    words = _WORD_RE.findall(sample)
    sentences = max(1, len(_SENT_RE.findall(sample)))
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59


class FleschKincaidCheck:
    """1.0 when the output's reading grade is close to the baseline's."""

    @property
    def name(self) -> str:
        return "readability"

    def score(self, output: str, context: CheckContext) -> float:
        base = flesch_kincaid_grade(context.baseline_output)
        cand = flesch_kincaid_grade(output)
        diff = abs(cand - base)
        return max(0.0, 1.0 - diff / 10.0)


class SyllableCountCheck:
    """1.0 when total syllable count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "syllables"

    def score(self, output: str, context: CheckContext) -> float:
        base = sum(count_syllables(w) for w in _WORD_RE.findall(context.baseline_output[:_MAX]))
        cand = sum(count_syllables(w) for w in _WORD_RE.findall(output[:_MAX]))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class SentenceStatsCheck:
    """1.0 when sentence count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "sentences"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_SENT_RE.findall(context.baseline_output[:_MAX]))
        cand = len(_SENT_RE.findall(output[:_MAX]))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)
