"""Readability dimensions: grade levels, sentence/word complexity, and pacing."""

from __future__ import annotations

import math
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "FleschReadingEaseCheck",
    "FogIndexCheck",
    "SmogIndexCheck",
    "ColemanLiauCheck",
    "AutomatedReadabilityCheck",
    "GunningFogWordsCheck",
    "AvgSyllablesPerWordCheck",
    "PolysyllableRatioCheck",
    "SentenceLengthVarianceCheck",
    "WordLengthVarianceCheck",
    "PassiveVoiceEstimateCheck",
    "PrepositionRatioCheck",
    "ConjunctionRatioCheck",
    "PronounRatioCheck",
    "ArticleRatioCheck",
    "AdverbEstimateCheck",
    "QuestionSentenceRatioCheck",
    "ExclamationSentenceRatioCheck",
    "LongSentenceRatioCheck",
    "ShortSentenceRatioCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")
_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am)\s+\w+ed\b|\bby\s+\w+",
    re.IGNORECASE,
)
_PREP_RE = re.compile(
    r"\b(?:in|on|at|to|for|with|of|by|from|about|into|over|after|under|between|through)\b",
    re.IGNORECASE,
)
_CONJ_RE = re.compile(
    r"\b(?:and|but|or|nor|so|yet|because|although|while|if|then|than)\b",
    re.IGNORECASE,
)
_PRON_RE = re.compile(
    r"\b(?:i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|its|our|their)\b",
    re.IGNORECASE,
)
_ART_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_ADVERB_RE = re.compile(r"\b\w+ly\b", re.IGNORECASE)


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text))


def _sents(text: str) -> list[str]:
    return [s for s in _SENT_RE.findall(_s(text)) if s.strip()]


def _syllables(word: str) -> int:
    groups = _VOWEL_RE.findall(word.lower())
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class FleschReadingEaseCheck:
    """Flesch Reading Ease score compared to baseline."""

    @property
    def name(self) -> str:
        return "flesch_ease"

    def score(self, output: str, context: CheckContext) -> float:
        def fre(text: str) -> float:
            ws = _words(text)
            ss = _sents(text)
            if not ws or not ss:
                return 0.0
            syl = sum(_syllables(w) for w in ws)
            return 206.835 - 1.015 * (len(ws) / len(ss)) - 84.6 * (syl / len(ws))

        base, cand = fre(context.baseline_output), fre(output)
        return _ratio_match(base, cand, tol=2.0)


class FogIndexCheck:
    @property
    def name(self) -> str:
        return "fog_index"

    def score(self, output: str, context: CheckContext) -> float:
        def fog(text: str) -> float:
            ws = _words(text)
            ss = _sents(text)
            if not ws or not ss:
                return 0.0
            hard = sum(1 for w in ws if _syllables(w) >= 3)
            return 0.4 * (len(ws) / len(ss) + 100 * hard / len(ws))

        base, cand = fog(context.baseline_output), fog(output)
        return _ratio_match(base, cand, tol=0.5)


class SmogIndexCheck:
    @property
    def name(self) -> str:
        return "smog_index"

    def score(self, output: str, context: CheckContext) -> float:
        def smog(text: str) -> float:
            ss = _sents(text)
            ws = _words(text)
            if not ss or not ws or len(ss) < 3:
                return 0.0
            hard = sum(1 for w in ws if _syllables(w) >= 3)
            return 1.043 * math.sqrt(hard * 30 / len(ss)) + 3.1291

        base, cand = smog(context.baseline_output), smog(output)
        return _ratio_match(base, cand, tol=0.5)


class ColemanLiauCheck:
    @property
    def name(self) -> str:
        return "coleman_liau"

    def score(self, output: str, context: CheckContext) -> float:
        def cl(text: str) -> float:
            s = _s(text)
            ws = _words(text)
            ss = _sents(text)
            if not ws or not ss:
                return 0.0
            letters = sum(1 for c in s if c.isalpha())
            letters_per_word = 100 * letters / len(ws)
            sents_per_word = 100 * len(ss) / len(ws)
            return 0.0588 * letters_per_word - 0.296 * sents_per_word - 15.8

        base, cand = cl(context.baseline_output), cl(output)
        return _ratio_match(base, cand, tol=0.5)


class AutomatedReadabilityCheck:
    @property
    def name(self) -> str:
        return "ari_index"

    def score(self, output: str, context: CheckContext) -> float:
        def ari(text: str) -> float:
            s = _s(text)
            ws = _words(text)
            ss = _sents(text)
            if not ws or not ss:
                return 0.0
            chars = sum(1 for c in s if not c.isspace())
            return 4.71 * chars / len(ws) + 0.5 * len(ws) / len(ss) - 21.43

        base, cand = ari(context.baseline_output), ari(output)
        return _ratio_match(base, cand, tol=0.5)


class GunningFogWordsCheck:
    """Count of 'fog words' (3+ syllables) — lower matches baseline."""

    @property
    def name(self) -> str:
        return "fog_word_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for w in _words(text) if _syllables(w) >= 3)

        return _count_match(count(context.baseline_output), count(output))


class AvgSyllablesPerWordCheck:
    @property
    def name(self) -> str:
        return "avg_syllables"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            ws = _words(text)
            return sum(_syllables(w) for w in ws) / len(ws) if ws else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class PolysyllableRatioCheck:
    @property
    def name(self) -> str:
        return "polysyllable_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return sum(1 for w in ws if _syllables(w) >= 3) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SentenceLengthVarianceCheck:
    @property
    def name(self) -> str:
        return "sent_len_variance"

    def score(self, output: str, context: CheckContext) -> float:
        def var(text: str) -> float:
            lens = [len(_WORD_RE.findall(s)) for s in _sents(text)]
            if len(lens) < 2:
                return 0.0
            mean = sum(lens) / len(lens)
            return sum((x - mean) ** 2 for x in lens) / len(lens)

        base, cand = var(context.baseline_output), var(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class WordLengthVarianceCheck:
    @property
    def name(self) -> str:
        return "word_len_variance"

    def score(self, output: str, context: CheckContext) -> float:
        def var(text: str) -> float:
            lens = [len(w) for w in _words(text)]
            if len(lens) < 2:
                return 0.0
            mean = sum(lens) / len(lens)
            return sum((x - mean) ** 2 for x in lens) / len(lens)

        base, cand = var(context.baseline_output), var(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class PassiveVoiceEstimateCheck:
    @property
    def name(self) -> str:
        return "passive_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ss = _sents(text)
            if not ss:
                return 0.0
            return sum(1 for s in ss if _PASSIVE_RE.search(s)) / len(ss)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PrepositionRatioCheck:
    @property
    def name(self) -> str:
        return "preposition_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_PREP_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ConjunctionRatioCheck:
    @property
    def name(self) -> str:
        return "conjunction_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_CONJ_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PronounRatioCheck:
    @property
    def name(self) -> str:
        return "pronoun_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_PRON_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ArticleRatioCheck:
    @property
    def name(self) -> str:
        return "article_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_ART_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AdverbEstimateCheck:
    @property
    def name(self) -> str:
        return "adverb_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_ADVERB_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class QuestionSentenceRatioCheck:
    @property
    def name(self) -> str:
        return "question_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ss = _sents(text)
            if not ss:
                return 0.0
            return sum(1 for s in ss if s.strip().endswith("?")) / len(ss)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ExclamationSentenceRatioCheck:
    @property
    def name(self) -> str:
        return "exclamation_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ss = _sents(text)
            if not ss:
                return 0.0
            return sum(1 for s in ss if s.strip().endswith("!")) / len(ss)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class LongSentenceRatioCheck:
    """Ratio of sentences with more than 25 words."""

    @property
    def name(self) -> str:
        return "long_sent_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ss = _sents(text)
            if not ss:
                return 0.0
            return sum(1 for s in ss if len(_WORD_RE.findall(s)) > 25) / len(ss)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ShortSentenceRatioCheck:
    """Ratio of sentences with 5 or fewer words."""

    @property
    def name(self) -> str:
        return "short_sent_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ss = _sents(text)
            if not ss:
                return 0.0
            return sum(1 for s in ss if len(_WORD_RE.findall(s)) <= 5) / len(ss)

        return _ratio_match(ratio(context.baseline_output), ratio(output))
