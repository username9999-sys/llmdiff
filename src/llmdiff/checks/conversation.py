"""Conversation-structure dimensions: multi-turn markers, roles, and flow."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "RoleMarkerCountCheck",
    "UserTurnCountCheck",
    "AssistantTurnCountCheck",
    "SystemTurnCountCheck",
    "TurnAlternationCheck",
    "CodeBlockPerTurnCheck",
    "QuestionPerTurnCheck",
    "AcknowledgmentCountCheck",
    "ClarificationCountCheck",
    "FollowUpQuestionCountCheck",
    "SummarySectionCountCheck",
    "BulletTurnCountCheck",
    "NumberedTurnCountCheck",
    "QuoteTurnCountCheck",
    "MarkdownTurnCountCheck",
    "EmptyTurnCheck",
    "LongTurnRatioCheck",
    "ShortTurnRatioCheck",
    "TurnCountMatchCheck",
    "ConversationDepthCheck",
]

_MAX = 200_000
_ROLE_RE = re.compile(r"^(?:user|assistant|system|human|ai|bot)\s*:", re.IGNORECASE | re.MULTILINE)
_USER_RE = re.compile(r"^(?:user|human)\s*:", re.IGNORECASE | re.MULTILINE)
_ASST_RE = re.compile(r"^(?:assistant|ai|bot)\s*:", re.IGNORECASE | re.MULTILINE)
_SYS_RE = re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)
_QUOTE_RE = re.compile(r"^>\s?\S", re.MULTILINE)
_MD_RE = re.compile(r"^#{1,6}\s|\*\*[^*]+\*\*|`[^`]+`", re.MULTILINE)
_ACK_RE = re.compile(r"\b(?:got it|understood|sure|okay|ok|thanks|thank you)\b", re.IGNORECASE)
_CLARIFY_RE = re.compile(
    r"\b(?:could you clarify|what do you mean|can you explain|to be clear)\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"\b(?:what about|how about|any other|also,|additionally)\b.*\?",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"^#{1,4}\s+.*(?:summary|conclusion|in short|tl;dr)",
    re.MULTILINE | re.IGNORECASE,
)


def _s(text: str) -> str:
    return text[:_MAX]


def _turns(text: str) -> list[str]:
    parts = _ROLE_RE.split(_s(text))
    return [p for p in parts if p.strip()]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class RoleMarkerCountCheck:
    @property
    def name(self) -> str:
        return "role_marker_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ROLE_RE.findall(_s(context.baseline_output))),
            len(_ROLE_RE.findall(_s(output))),
        )


class UserTurnCountCheck:
    @property
    def name(self) -> str:
        return "user_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_USER_RE.findall(_s(context.baseline_output))),
            len(_USER_RE.findall(_s(output))),
        )


class AssistantTurnCountCheck:
    @property
    def name(self) -> str:
        return "assistant_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ASST_RE.findall(_s(context.baseline_output))),
            len(_ASST_RE.findall(_s(output))),
        )


class SystemTurnCountCheck:
    @property
    def name(self) -> str:
        return "system_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SYS_RE.findall(_s(context.baseline_output))),
            len(_SYS_RE.findall(_s(output))),
        )


class TurnAlternationCheck:
    """1.0 when the user/assistant alternation pattern matches the baseline."""

    @property
    def name(self) -> str:
        return "turn_alternation"

    def score(self, output: str, context: CheckContext) -> float:
        def pattern(text: str) -> list[str]:
            out = []
            for m in re.finditer(
                r"^(user|human|assistant|ai|bot|system)\s*:",
                _s(text),
                re.IGNORECASE | re.MULTILINE,
            ):
                role = m.group(1).lower()
                if role in ("user", "human"):
                    out.append("u")
                elif role in ("assistant", "ai", "bot"):
                    out.append("a")
                else:
                    out.append("s")
            return out

        base, cand = pattern(context.baseline_output), pattern(output)
        if not base and not cand:
            return 1.0
        if len(base) != len(cand):
            return max(0.0, 1.0 - abs(len(base) - len(cand)) / max(len(base), len(cand)))
        match = sum(1 for a, b in zip(base, cand, strict=False) if a == b)
        return match / len(base)


class CodeBlockPerTurnCheck:
    @property
    def name(self) -> str:
        return "codeblock_per_turn"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            turns = _turns(text)
            if not turns:
                return 0.0
            with_code = sum(1 for t in turns if "```" in t)
            return with_code / len(turns)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class QuestionPerTurnCheck:
    @property
    def name(self) -> str:
        return "question_per_turn"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            turns = _turns(text)
            if not turns:
                return 0.0
            with_q = sum(1 for t in turns if "?" in t)
            return with_q / len(turns)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AcknowledgmentCountCheck:
    @property
    def name(self) -> str:
        return "ack_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ACK_RE.findall(_s(context.baseline_output))),
            len(_ACK_RE.findall(_s(output))),
        )


class ClarificationCountCheck:
    @property
    def name(self) -> str:
        return "clarify_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CLARIFY_RE.findall(_s(context.baseline_output))),
            len(_CLARIFY_RE.findall(_s(output))),
        )


class FollowUpQuestionCountCheck:
    @property
    def name(self) -> str:
        return "followup_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FOLLOWUP_RE.findall(_s(context.baseline_output))),
            len(_FOLLOWUP_RE.findall(_s(output))),
        )


class SummarySectionCountCheck:
    @property
    def name(self) -> str:
        return "summary_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUMMARY_RE.findall(_s(context.baseline_output))),
            len(_SUMMARY_RE.findall(_s(output))),
        )


class BulletTurnCountCheck:
    @property
    def name(self) -> str:
        return "bullet_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _turns(text) if _BULLET_RE.search(t))

        return _count_match(count(context.baseline_output), count(output))


class NumberedTurnCountCheck:
    @property
    def name(self) -> str:
        return "numbered_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _turns(text) if _NUMBERED_RE.search(t))

        return _count_match(count(context.baseline_output), count(output))


class QuoteTurnCountCheck:
    @property
    def name(self) -> str:
        return "quote_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _turns(text) if _QUOTE_RE.search(t))

        return _count_match(count(context.baseline_output), count(output))


class MarkdownTurnCountCheck:
    @property
    def name(self) -> str:
        return "md_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _turns(text) if _MD_RE.search(t))

        return _count_match(count(context.baseline_output), count(output))


class EmptyTurnCheck:
    """1.0 when the count of empty turns matches the baseline."""

    @property
    def name(self) -> str:
        return "empty_turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _turns(text) if not t.strip())

        return _count_match(count(context.baseline_output), count(output))


class LongTurnRatioCheck:
    """Ratio of turns longer than 500 chars."""

    @property
    def name(self) -> str:
        return "long_turn_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            turns = _turns(text)
            if not turns:
                return 0.0
            return sum(1 for t in turns if len(t) > 500) / len(turns)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ShortTurnRatioCheck:
    """Ratio of turns shorter than 20 chars."""

    @property
    def name(self) -> str:
        return "short_turn_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            turns = _turns(text)
            if not turns:
                return 0.0
            return sum(1 for t in turns if len(t.strip()) < 20) / len(turns)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class TurnCountMatchCheck:
    @property
    def name(self) -> str:
        return "turn_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_turns(context.baseline_output)), len(_turns(output)))


class ConversationDepthCheck:
    """1.0 when the number of role-marker lines matches the baseline."""

    @property
    def name(self) -> str:
        return "conv_depth"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ROLE_RE.findall(_s(context.baseline_output))),
            len(_ROLE_RE.findall(_s(output))),
        )
