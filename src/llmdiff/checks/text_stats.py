"""Text-statistics dimensions: lengths, ratios, and distribution stats."""

from __future__ import annotations

import math
import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "CharCountCheck",
    "NonSpaceCharCountCheck",
    "ByteLengthUtf8Check",
    "UniqueCharRatioCheck",
    "DigitCharRatioCheck",
    "AlphaCharRatioCheck",
    "SpaceCharRatioCheck",
    "PunctCharRatioCheck",
    "UpperCharRatioCheck",
    "LowerCharRatioCheck",
    "WordLengthStdCheck",
    "WordLengthMeanCheck",
    "WordLengthMaxCheck",
    "UniqueWordRatioCheck",
    "LexicalDiversityCheck",
    "TypeTokenRatioCheck",
    "HapaxRatioCheck",
    "SentenceCountCheck",
    "ParagraphCountMatchCheck",
    "SyllableEstimateCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text))


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class CharCountCheck:
    @property
    def name(self) -> str:
        return "char_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_s(context.baseline_output)), len(_s(output)))


class NonSpaceCharCountCheck:
    @property
    def name(self) -> str:
        return "nonspace_char_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for c in _s(text) if not c.isspace())

        return _count_match(count(context.baseline_output), count(output))


class ByteLengthUtf8Check:
    @property
    def name(self) -> str:
        return "byte_len_utf8"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_s(context.baseline_output).encode("utf-8", errors="replace")),
            len(_s(output).encode("utf-8", errors="replace")),
        )


class UniqueCharRatioCheck:
    @property
    def name(self) -> str:
        return "unique_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return len(set(s)) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class DigitCharRatioCheck:
    @property
    def name(self) -> str:
        return "digit_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if c.isdigit()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AlphaCharRatioCheck:
    @property
    def name(self) -> str:
        return "alpha_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if c.isalpha()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SpaceCharRatioCheck:
    @property
    def name(self) -> str:
        return "space_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if c.isspace()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PunctCharRatioCheck:
    @property
    def name(self) -> str:
        return "punct_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if not c.isalnum() and not c.isspace()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class UpperCharRatioCheck:
    @property
    def name(self) -> str:
        return "upper_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if c.isupper()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class LowerCharRatioCheck:
    @property
    def name(self) -> str:
        return "lower_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return sum(1 for c in s if c.islower()) / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class WordLengthStdCheck:
    """1.0 when the std-dev of word lengths matches the baseline."""

    @property
    def name(self) -> str:
        return "word_len_std"

    def score(self, output: str, context: CheckContext) -> float:
        def std(text: str) -> float:
            lens = [len(w) for w in _words(text)]
            if len(lens) < 2:
                return 0.0
            mean = sum(lens) / len(lens)
            return math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))

        base, cand = std(context.baseline_output), std(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class WordLengthMeanCheck:
    @property
    def name(self) -> str:
        return "word_len_mean"

    def score(self, output: str, context: CheckContext) -> float:
        def mean(text: str) -> float:
            lens = [len(w) for w in _words(text)]
            return sum(lens) / len(lens) if lens else 0.0

        base, cand = mean(context.baseline_output), mean(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class WordLengthMaxCheck:
    @property
    def name(self) -> str:
        return "word_len_max"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            lens = [len(w) for w in _words(text)]
            return max(lens) if lens else 0

        return _count_match(mx(context.baseline_output), mx(output))


class UniqueWordRatioCheck:
    @property
    def name(self) -> str:
        return "unique_word_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(set(w.lower() for w in ws)) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class LexicalDiversityCheck:
    """Moving-average type-token ratio (MATTR) over a 50-word window."""

    @property
    def name(self) -> str:
        return "lexical_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def mattr(text: str, window: int = 50) -> float:
            ws = [w.lower() for w in _words(text)]
            if len(ws) < window:
                return len(set(ws)) / len(ws) if ws else 0.0
            ratios = []
            for i in range(len(ws) - window + 1):
                seg = ws[i : i + window]
                ratios.append(len(set(seg)) / window)
            return sum(ratios) / len(ratios)

        return _ratio_match(mattr(context.baseline_output), mattr(output))


class TypeTokenRatioCheck:
    @property
    def name(self) -> str:
        return "type_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ttr(text: str) -> float:
            ws = [w.lower() for w in _words(text)]
            return len(set(ws)) / len(ws) if ws else 0.0

        return _ratio_match(ttr(context.baseline_output), ttr(output))


class HapaxRatioCheck:
    """Ratio of words occurring exactly once."""

    @property
    def name(self) -> str:
        return "hapax_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = [w.lower() for w in _words(text)]
            if not ws:
                return 0.0
            counts = Counter(ws)
            return sum(1 for w in ws if counts[w] == 1) / len(ws)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SentenceCountCheck:
    @property
    def name(self) -> str:
        return "sentence_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len(re.findall(r"[.!?]+(?:\s|$)", _s(text)))

        return _count_match(count(context.baseline_output), count(output))


class ParagraphCountMatchCheck:
    @property
    def name(self) -> str:
        return "paragraph_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len([p for p in _s(text).split("\n\n") if p.strip()])

        return _count_match(count(context.baseline_output), count(output))


class SyllableEstimateCheck:
    """1.0 when the estimated syllable count matches the baseline."""

    @property
    def name(self) -> str:
        return "syllable_est"

    def score(self, output: str, context: CheckContext) -> float:
        def est(text: str) -> int:
            total = 0
            for w in _words(text):
                groups = _VOWEL_GROUP_RE.findall(w)
                n = len(groups)
                if w.lower().endswith("e") and n > 1:
                    n -= 1
                total += max(1, n)
            return total

        return _count_match(est(context.baseline_output), est(output))
