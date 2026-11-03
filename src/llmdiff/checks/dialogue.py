"""Dialogue and conversational-structure dimensions."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "QuestionAnswerCheck",
    "TurnMarkerCheck",
    "GreetingCheck",
    "BulletAnswerCheck",
    "WordCountCheck",
]

_MAX = 200_000
_TURN_RE = re.compile(r"^(?:user|assistant|system|human|ai)\s*:", re.IGNORECASE | re.MULTILINE)
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", re.IGNORECASE
)


class QuestionAnswerCheck:
    """1.0 when the output answers a question (does not end with '?')."""

    @property
    def name(self) -> str:
        return "answers_question"

    def score(self, output: str, context: CheckContext) -> float:
        if "?" not in context.prompt[:_MAX]:
            return 1.0  # not a question -> neutral
        sample = output.strip()[:_MAX]
        if not sample:
            return 0.0
        return 0.0 if sample.endswith("?") else 1.0


class TurnMarkerCheck:
    """1.0 when the output does not leak chat turn markers (User:/Assistant:)."""

    @property
    def name(self) -> str:
        return "no_turn_markers"

    def score(self, output: str, context: CheckContext) -> float:
        return 0.0 if _TURN_RE.search(output[:_MAX]) else 1.0


class GreetingCheck:
    """1.0 when the output greets iff the prompt greets."""

    @property
    def name(self) -> str:
        return "greeting_match"

    def score(self, output: str, context: CheckContext) -> float:
        prompt_greet = bool(_GREETING_RE.match(context.prompt.strip()[:_MAX]))
        out_greet = bool(_GREETING_RE.match(output.strip()[:_MAX]))
        return 1.0 if prompt_greet == out_greet else 0.0


class BulletAnswerCheck:
    """1.0 when a list-style prompt gets a list-style answer."""

    @property
    def name(self) -> str:
        return "bullet_answer"

    def score(self, output: str, context: CheckContext) -> float:
        prompt = context.prompt[:_MAX].lower()
        if not re.search(r"\blist\b|\bbullets?\b|\bitems?\b", prompt):
            return 1.0
        sample = output[:_MAX]
        has_list = bool(re.search(r"^\s*[-*•]|\n\s*[-*•]|\d+\.", sample, re.MULTILINE))
        return 1.0 if has_list else 0.0


class WordCountCheck:
    """1.0 when word count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "word_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(context.baseline_output[:_MAX].split())
        cand = len(output[:_MAX].split())
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)
