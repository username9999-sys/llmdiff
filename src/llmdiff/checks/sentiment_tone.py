"""Sentiment and tone dimensions: polarity lexicons, emotion markers, formality."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "PositiveWordRatioCheck",
    "NegativeWordRatioCheck",
    "PolarityBalanceCheck",
    "EnthusiasmMarkerCountCheck",
    "HedgingWordRatioCheck",
    "CertaintyWordRatioCheck",
    "FormalityRatioCheck",
    "ContractionRatioCheck",
    "FirstPersonRatioCheck",
    "SecondPersonRatioCheck",
    "ThirdPersonRatioCheck",
    "ExclamationRatioCheck",
    "EllipsisToneCountCheck",
    "CapsWordCountCheck",
    "EmojiSentimentScoreCheck",
    "IntensifierRatioCheck",
    "NegationCountCheck",
    "SarcasmMarkerCountCheck",
    "PolitenessMarkerCountCheck",
    "UrgencyMarkerCountCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")

_POSITIVE = {
    "good", "great", "awesome", "excellent", "amazing", "wonderful", "fantastic",
    "love", "like", "happy", "glad", "best", "better", "nice", "perfect", "thanks",
    "thank", "yes", "cool", "brilliant", "beautiful", "enjoy", "win", "success",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "angry", "worst",
    "worse", "poor", "fail", "failure", "broken", "wrong", "no", "never",
    "annoying", "ugly", "disappointed", "sorry", "unfortunately", "problem", "error",
}
_HEDGING = {
    "maybe", "perhaps", "might", "could", "possibly", "probably", "seems",
    "appears", "sort of", "kind of", "i think", "i guess", "arguably", "likely",
}
_CERTAINTY = {
    "definitely", "certainly", "absolutely", "clearly", "obviously", "always",
    "never", "undoubtedly", "indeed", "of course", "surely", "must", "will",
}
_FORMAL = {
    "therefore", "however", "furthermore", "moreover", "consequently", "thus",
    "hence", "nevertheless", "regarding", "concerning", "pursuant", "utilize",
    "demonstrate", "require", "additional", "numerous", "subsequent",
}
_INTENSIFIERS = {
    "very", "really", "extremely", "absolutely", "totally", "completely",
    "highly", "incredibly", "super", "so", "too", "quite", "rather",
}
_NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot",
}
_SARCASM_MARKERS = re.compile(
    r"\b(?:yeah right|sure thing|as if|oh great|wonderful|totally)\b",
    re.IGNORECASE,
)
_POLITE = {"please", "thank", "thanks", "sorry", "excuse", "kindly", "appreciate", "regards"}
_URGENT = {"urgent", "asap", "immediately", "now", "hurry", "quickly", "deadline", "emergency"}
_POS_EMOJI = {
    "😀", "😃", "😄", "😁", "😊", "🙂", "😍", "🥰", "😘", "👍",
    "🎉", "❤️", "🔥", "✨", "😂", "🤣",
}
_NEG_EMOJI = {"😞", "😔", "😢", "😭", "😡", "😠", "👎", "💔", "😩", "😰", "😱", "🙁", "☹️"}


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(_s(text))]


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class PositiveWordRatioCheck:
    @property
    def name(self) -> str:
        return "positive_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _POSITIVE) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NegativeWordRatioCheck:
    @property
    def name(self) -> str:
        return "negative_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _NEGATIVE) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PolarityBalanceCheck:
    """pos/(pos+neg) balance compared to baseline."""

    @property
    def name(self) -> str:
        return "polarity_balance"

    def score(self, output: str, context: CheckContext) -> float:
        def balance(text: str) -> float:
            ws = _words(text)
            p = sum(1 for w in ws if w in _POSITIVE)
            n = sum(1 for w in ws if w in _NEGATIVE)
            if p + n == 0:
                return 0.5
            return p / (p + n)

        return _ratio_match(balance(context.baseline_output), balance(output))


class EnthusiasmMarkerCountCheck:
    @property
    def name(self) -> str:
        return "enthusiasm_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return s.count("!") + sum(1 for c in s if c in _POS_EMOJI)

        return _count_match(count(context.baseline_output), count(output))


class HedgingWordRatioCheck:
    @property
    def name(self) -> str:
        return "hedging_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _HEDGING) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CertaintyWordRatioCheck:
    @property
    def name(self) -> str:
        return "certainty_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _CERTAINTY) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class FormalityRatioCheck:
    @property
    def name(self) -> str:
        return "formality_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _FORMAL) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ContractionRatioCheck:
    @property
    def name(self) -> str:
        return "contraction_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _WORD_RE.findall(_s(text))
            return sum(1 for w in ws if "'" in w) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class FirstPersonRatioCheck:
    @property
    def name(self) -> str:
        return "first_person_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            first = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
            return sum(1 for w in ws if w in first) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SecondPersonRatioCheck:
    @property
    def name(self) -> str:
        return "second_person_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in {"you", "your", "yours"}) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ThirdPersonRatioCheck:
    @property
    def name(self) -> str:
        return "third_person_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            third = {"he", "she", "it", "they", "him", "her", "them", "his", "their"}
            return sum(1 for w in ws if w in third) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ExclamationRatioCheck:
    @property
    def name(self) -> str:
        return "exclamation_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return s.count("!") / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class EllipsisToneCountCheck:
    @property
    def name(self) -> str:
        return "ellipsis_tone_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return s.count("…") + len(re.findall(r"\.\.\.", s))

        return _count_match(count(context.baseline_output), count(output))


class CapsWordCountCheck:
    """Count of fully-uppercase words (length >= 2), a shouting signal."""

    @property
    def name(self) -> str:
        return "caps_word_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(
                1 for w in _WORD_RE.findall(_s(text)) if len(w) >= 2 and w.isupper()
            )

        return _count_match(count(context.baseline_output), count(output))


class EmojiSentimentScoreCheck:
    """(pos - neg) / total emoji, compared to baseline."""

    @property
    def name(self) -> str:
        return "emoji_sentiment"

    def score(self, output: str, context: CheckContext) -> float:
        def score(text: str) -> float:
            s = _s(text)
            pos = sum(1 for c in s if c in _POS_EMOJI)
            neg = sum(1 for c in s if c in _NEG_EMOJI)
            total = pos + neg
            if total == 0:
                return 0.0
            return (pos - neg) / total

        return _ratio_match(score(context.baseline_output), score(output))


class IntensifierRatioCheck:
    @property
    def name(self) -> str:
        return "intensifier_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if w in _INTENSIFIERS) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NegationCountCheck:
    @property
    def name(self) -> str:
        return "negation_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for w in _words(text) if w in _NEGATIONS)

        return _count_match(count(context.baseline_output), count(output))


class SarcasmMarkerCountCheck:
    @property
    def name(self) -> str:
        return "sarcasm_markers"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len(_SARCASM_MARKERS.findall(_s(text)))

        return _count_match(count(context.baseline_output), count(output))


class PolitenessMarkerCountCheck:
    @property
    def name(self) -> str:
        return "politeness_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for w in _words(text) if w in _POLITE)

        return _count_match(count(context.baseline_output), count(output))


class UrgencyMarkerCountCheck:
    @property
    def name(self) -> str:
        return "urgency_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for w in _words(text) if w in _URGENT)

        return _count_match(count(context.baseline_output), count(output))
