"""Instruction-following dimensions: does the output follow the prompt's format asks."""

from __future__ import annotations

import json
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "FormatFollowCheck",
    "JsonModeCheck",
    "LengthInstructionCheck",
    "NoApologyCheck",
]

_MAX = 200_000


class FormatFollowCheck:
    """1.0 when output format matches what the prompt asked for.

    Detects prompt asks like "as JSON", "as a list", "in one sentence",
    "no explanation", and checks the output accordingly.
    """

    @property
    def name(self) -> str:
        return "format_follow"

    def score(self, output: str, context: CheckContext) -> float:
        prompt = context.prompt[:_MAX].lower()
        sample = output[:_MAX]
        checks: list[bool] = []
        if "json" in prompt:
            try:
                json.loads(sample.strip())
                checks.append(True)
            except (json.JSONDecodeError, ValueError):
                checks.append(False)
        if re.search(r"\b(as a|in a) list\b|bullet", prompt):
            checks.append(
                bool(re.search(r"^\s*[-*•]|\n\s*[-*•]|\d+\.", sample, re.MULTILINE))
            )
        if re.search(r"one sentence|single sentence", prompt):
            sentences = len(re.findall(r"[.!?]+(?:\s|$)", sample))
            checks.append(sentences <= 1)
        if re.search(r"no explanation|just the|only the", prompt):
            checks.append(len(sample.strip().splitlines()) <= 3)
        if not checks:
            return 1.0
        return sum(checks) / len(checks)


class JsonModeCheck:
    """1.0 when the prompt asked for JSON and the output is valid JSON."""

    @property
    def name(self) -> str:
        return "json_mode"

    def score(self, output: str, context: CheckContext) -> float:
        if "json" not in context.prompt[:_MAX].lower():
            return 1.0  # not requested -> neutral
        try:
            json.loads(output.strip()[:_MAX])
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0


class LengthInstructionCheck:
    """Honors 'at most N words' / 'around N words' instructions in the prompt."""

    @property
    def name(self) -> str:
        return "length_instruction"

    _AT_MOST = re.compile(r"at most (\d+) words")
    _AROUND = re.compile(r"(?:around|about|~)\s*(\d+) words")

    def score(self, output: str, context: CheckContext) -> float:
        prompt = context.prompt[:_MAX].lower()
        words = len(output.split())
        m = self._AT_MOST.search(prompt)
        if m:
            limit = int(m.group(1))
            if words <= limit:
                return 1.0
            return max(0.0, 1.0 - (words - limit) / max(1, limit))
        m = self._AROUND.search(prompt)
        if m:
            target = int(m.group(1))
            if target == 0:
                return 1.0
            return max(0.0, 1.0 - abs(words - target) / target)
        return 1.0


class NoApologyCheck:
    """1.0 when the output does not start with an apology/refusal opener."""

    @property
    def name(self) -> str:
        return "no_apology"

    _OPENERS = ("i'm sorry", "i am sorry", "sorry,", "i cannot", "i can't", "as an ai")

    def score(self, output: str, context: CheckContext) -> float:
        stripped = output.strip()[:_MAX].lower()
        return 0.0 if stripped.startswith(self._OPENERS) else 1.0
