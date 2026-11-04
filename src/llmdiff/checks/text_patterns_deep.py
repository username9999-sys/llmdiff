"""Deep text-pattern dimensions: clichés, hedges, filler, and rhetorical markers."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "FillerPhraseCountCheck",
    "HedgePhraseCountCheck",
    "BuzzwordCountCheck",
    "SuperlativeCountCheck",
    "FirstPersonPluralCountCheck",
    "PassiveVoiceDeepCountCheck",
    "NominalizationCountCheck",
    "WeaselWordCountCheck",
    "IntensifierStackCountCheck",
    "RhetoricalQuestionCountCheck",
    "EmDashUsageCountCheck",
    "SemicolonUsageCountCheck",
    "ParentheticalAsideCountCheck",
    "ColonListIntroCountCheck",
    "BulletPointRatioCheck",
    "NumberedListRatioCheck",
    "QuoteBlockCountCheck",
    "LinkInTextCountCheck",
    "BoldPhraseCountCheck",
    "ItalicPhraseCountCheck",
]

_MAX = 200_000

_FILLERS = re.compile(
    r"\b(?:basically|actually|literally|simply|just|really|very|quite|rather|"
    r"honestly|to be honest|in my opinion|IMO|IMHO)\b",
    re.IGNORECASE,
)
_HEDGES = re.compile(
    r"\b(?:maybe|perhaps|might|could|possibly|probably|likely|seems|appears|"
    r"sort of|kind of|somewhat|generally|typically|usually|often)\b",
    re.IGNORECASE,
)
_BUZZWORDS = re.compile(
    r"\b(?:leverage|synergy|paradigm|holistic|robust|scalable|innovative|"
    r"disruptive|game-?changer|cutting-?edge|best-?in-?class|world-?class|"
    r"next-?gen|state-?of-?the-?art|empower|deliverable|bandwidth|deep dive)\b",
    re.IGNORECASE,
)
_SUPERLATIVES = re.compile(
    r"\b(?:best|worst|greatest|biggest|smallest|fastest|slowest|most|least|"
    r"largest|highest|lowest|ultimate|perfect|amazing|incredible)\b",
    re.IGNORECASE,
)
_WEASELS = re.compile(
    r"\b(?:up to|as much as|many|several|some|various|numerous|few|"
    r"reportedly|allegedly|apparently|supposedly|claimed)\b",
    re.IGNORECASE,
)
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am)\s+\w+(?:ed|en)\b",
    re.IGNORECASE,
)
_NOMINAL = re.compile(r"\b\w+(?:tion|ment|ance|ence|ity|ness|ism|ship)\b", re.IGNORECASE)
_INTENSIFIER_STACK = re.compile(
    r"\b(?:very|really|extremely|incredibly|highly|absolutely|totally)\s+"
    r"(?:very|really|extremely|incredibly|highly|absolutely|totally)\b",
    re.IGNORECASE,
)
_RHET_Q = re.compile(r"\b(?:who|what|when|where|why|how)\s+\w+[^.?!]*\?")
_EMDASH = re.compile(r"—")
_SEMICOLON = re.compile(r";")
_PAREN = re.compile(r"\([^)]{5,120}\)")
_COLON_INTRO = re.compile(r":\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_QUOTE_BLOCK = re.compile(r"^>\s?\S", re.MULTILINE)
_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
_BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
_ITALIC = re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)")


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class FillerPhraseCountCheck:
    @property
    def name(self) -> str:
        return "filler_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FILLERS.findall(_s(context.baseline_output))),
            len(_FILLERS.findall(_s(output))),
        )


class HedgePhraseCountCheck:
    @property
    def name(self) -> str:
        return "hedge_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HEDGES.findall(_s(context.baseline_output))),
            len(_HEDGES.findall(_s(output))),
        )


class BuzzwordCountCheck:
    @property
    def name(self) -> str:
        return "buzzword_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BUZZWORDS.findall(_s(context.baseline_output))),
            len(_BUZZWORDS.findall(_s(output))),
        )


class SuperlativeCountCheck:
    @property
    def name(self) -> str:
        return "superlative_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUPERLATIVES.findall(_s(context.baseline_output))),
            len(_SUPERLATIVES.findall(_s(output))),
        )


class FirstPersonPluralCountCheck:
    @property
    def name(self) -> str:
        return "we_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\b(?:we|us|our|ours)\b", re.IGNORECASE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class PassiveVoiceDeepCountCheck:
    @property
    def name(self) -> str:
        return "passive_deep_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PASSIVE.findall(_s(context.baseline_output))),
            len(_PASSIVE.findall(_s(output))),
        )


class NominalizationCountCheck:
    @property
    def name(self) -> str:
        return "nominalization_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NOMINAL.findall(_s(context.baseline_output))),
            len(_NOMINAL.findall(_s(output))),
        )


class WeaselWordCountCheck:
    @property
    def name(self) -> str:
        return "weasel_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_WEASELS.findall(_s(context.baseline_output))),
            len(_WEASELS.findall(_s(output))),
        )


class IntensifierStackCountCheck:
    @property
    def name(self) -> str:
        return "intensifier_stack_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INTENSIFIER_STACK.findall(_s(context.baseline_output))),
            len(_INTENSIFIER_STACK.findall(_s(output))),
        )


class RhetoricalQuestionCountCheck:
    @property
    def name(self) -> str:
        return "rhetorical_q_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RHET_Q.findall(_s(context.baseline_output))),
            len(_RHET_Q.findall(_s(output))),
        )


class EmDashUsageCountCheck:
    @property
    def name(self) -> str:
        return "emdash_usage"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EMDASH.findall(_s(context.baseline_output))),
            len(_EMDASH.findall(_s(output))),
        )


class SemicolonUsageCountCheck:
    @property
    def name(self) -> str:
        return "semicolon_usage"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SEMICOLON.findall(_s(context.baseline_output))),
            len(_SEMICOLON.findall(_s(output))),
        )


class ParentheticalAsideCountCheck:
    @property
    def name(self) -> str:
        return "parenthetical_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PAREN.findall(_s(context.baseline_output))),
            len(_PAREN.findall(_s(output))),
        )


class ColonListIntroCountCheck:
    @property
    def name(self) -> str:
        return "colon_intro_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COLON_INTRO.findall(_s(context.baseline_output))),
            len(_COLON_INTRO.findall(_s(output))),
        )


class BulletPointRatioCheck:
    @property
    def name(self) -> str:
        return "bullet_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = [ln for ln in _s(text).splitlines() if ln.strip()]
            return len(_BULLET.findall(_s(text))) / len(lines) if lines else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NumberedListRatioCheck:
    @property
    def name(self) -> str:
        return "numbered_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = [ln for ln in _s(text).splitlines() if ln.strip()]
            return len(_NUMBERED.findall(_s(text))) / len(lines) if lines else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class QuoteBlockCountCheck:
    @property
    def name(self) -> str:
        return "quote_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTE_BLOCK.findall(_s(context.baseline_output))),
            len(_QUOTE_BLOCK.findall(_s(output))),
        )


class LinkInTextCountCheck:
    @property
    def name(self) -> str:
        return "link_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LINK.findall(_s(context.baseline_output))),
            len(_LINK.findall(_s(output))),
        )


class BoldPhraseCountCheck:
    @property
    def name(self) -> str:
        return "bold_phrase_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOLD.findall(_s(context.baseline_output))),
            len(_BOLD.findall(_s(output))),
        )


class ItalicPhraseCountCheck:
    @property
    def name(self) -> str:
        return "italic_phrase_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ITALIC.findall(_s(context.baseline_output))),
            len(_ITALIC.findall(_s(output))),
        )
