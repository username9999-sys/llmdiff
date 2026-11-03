"""Linguistic surface dimensions: punctuation, word-shape, and style ratios."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "VowelRatioCheck",
    "ConsonantRatioCheck",
    "AvgSentenceLengthCheck",
    "QuestionCountCheck",
    "ExclamationCountCheck",
    "EllipsisCountCheck",
    "SemicolonCountCheck",
    "ColonCountCheck",
    "DashCountCheck",
    "ParenthesisBalanceCheck",
    "QuoteStyleMatchCheck",
    "NumberWordRatioCheck",
    "AcronymCountCheck",
    "HyphenatedWordCountCheck",
    "CapitalizedWordRatioCheck",
    "FirstWordMatchCheck",
    "LastWordMatchCheck",
    "OpeningPunctMatchCheck",
    "ClosingPunctMatchCheck",
    "WhitespaceRatioCheck",
]

_MAX = 200_000
_VOWELS = set("aeiouAEIOU")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _sample(text: str) -> str:
    return text[:_MAX]


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class VowelRatioCheck:
    """1.0 when the vowel/letter ratio matches the baseline."""

    @property
    def name(self) -> str:
        return "vowel_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            letters = [c for c in _sample(text) if c.isalpha()]
            if not letters:
                return 0.0
            return sum(1 for c in letters if c in _VOWELS) / len(letters)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ConsonantRatioCheck:
    """1.0 when the consonant/letter ratio matches the baseline."""

    @property
    def name(self) -> str:
        return "consonant_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            letters = [c for c in _sample(text) if c.isalpha()]
            if not letters:
                return 0.0
            return sum(1 for c in letters if c not in _VOWELS) / len(letters)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AvgSentenceLengthCheck:
    """1.0 when the average words-per-sentence matches the baseline."""

    @property
    def name(self) -> str:
        return "avg_sentence_length"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            sents = [s for s in _SENT_SPLIT_RE.split(_sample(text)) if s.strip()]
            if not sents:
                return 0.0
            return sum(len(s.split()) for s in sents) / len(sents)

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class QuestionCountCheck:
    """1.0 when the number of question marks matches the baseline."""

    @property
    def name(self) -> str:
        return "question_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = _sample(context.baseline_output).count("?")
        cand = _sample(output).count("?")
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class ExclamationCountCheck:
    """1.0 when the number of exclamation marks matches the baseline."""

    @property
    def name(self) -> str:
        return "exclamation_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = _sample(context.baseline_output).count("!")
        cand = _sample(output).count("!")
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class EllipsisCountCheck:
    """1.0 when the number of ellipses matches the baseline."""

    @property
    def name(self) -> str:
        return "ellipsis_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _sample(text)
            return s.count("…") + len(re.findall(r"\.\.\.", s))

        base, cand = count(context.baseline_output), count(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class SemicolonCountCheck:
    """1.0 when the number of semicolons matches the baseline."""

    @property
    def name(self) -> str:
        return "semicolon_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = _sample(context.baseline_output).count(";")
        cand = _sample(output).count(";")
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class ColonCountCheck:
    """1.0 when the number of colons matches the baseline."""

    @property
    def name(self) -> str:
        return "colon_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = _sample(context.baseline_output).count(":")
        cand = _sample(output).count(":")
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class DashCountCheck:
    """1.0 when the number of dashes (hyphen/em/en) matches the baseline."""

    @property
    def name(self) -> str:
        return "dash_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(_sample(text).count(ch) for ch in ("-", "—", "–"))

        base, cand = count(context.baseline_output), count(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class ParenthesisBalanceCheck:
    """1.0 when parenthesis balance (open minus close) matches the baseline."""

    @property
    def name(self) -> str:
        return "paren_balance"

    def score(self, output: str, context: CheckContext) -> float:
        def balance(text: str) -> int:
            s = _sample(text)
            return s.count("(") - s.count(")")

        base, cand = balance(context.baseline_output), balance(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class QuoteStyleMatchCheck:
    """1.0 when straight-vs-curly quote usage matches the baseline."""

    @property
    def name(self) -> str:
        return "quote_style"

    def score(self, output: str, context: CheckContext) -> float:
        def curly_ratio(text: str) -> float:
            s = _sample(text)
            quotes = sum(s.count(ch) for ch in ('"', "'", "“", "”", "‘", "’"))
            if quotes == 0:
                return 0.0
            curly = sum(s.count(ch) for ch in ("“", "”", "‘", "’"))
            return curly / quotes

        return _ratio_match(curly_ratio(context.baseline_output), curly_ratio(output))


class NumberWordRatioCheck:
    """1.0 when the digit-word/word ratio matches the baseline."""

    @property
    def name(self) -> str:
        return "number_word_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            words = _WORD_RE.findall(_sample(text))
            if not words:
                return 0.0
            numeric = sum(1 for w in words if any(c.isdigit() for c in w))
            return numeric / len(words)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AcronymCountCheck:
    """1.0 when the number of acronyms (2+ uppercase) matches the baseline."""

    @property
    def name(self) -> str:
        return "acronym_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_ACRONYM_RE.findall(_sample(context.baseline_output)))
        cand = len(_ACRONYM_RE.findall(_sample(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class HyphenatedWordCountCheck:
    """1.0 when the number of hyphenated words matches the baseline."""

    @property
    def name(self) -> str:
        return "hyphenated_words"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for w in _WORD_RE.findall(_sample(text)) if "-" in w)

        base, cand = count(context.baseline_output), count(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class CapitalizedWordRatioCheck:
    """1.0 when the ratio of capitalized words matches the baseline."""

    @property
    def name(self) -> str:
        return "cap_word_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            words = _WORD_RE.findall(_sample(text))
            if not words:
                return 0.0
            return sum(1 for w in words if w[0].isupper()) / len(words)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class FirstWordMatchCheck:
    """1.0 when the first word matches the baseline's first word."""

    @property
    def name(self) -> str:
        return "first_word"

    def score(self, output: str, context: CheckContext) -> float:
        def first(text: str) -> str:
            words = _WORD_RE.findall(_sample(text))
            return words[0].lower() if words else ""

        base, cand = first(context.baseline_output), first(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class LastWordMatchCheck:
    """1.0 when the last word matches the baseline's last word."""

    @property
    def name(self) -> str:
        return "last_word"

    def score(self, output: str, context: CheckContext) -> float:
        def last(text: str) -> str:
            words = _WORD_RE.findall(_sample(text))
            return words[-1].lower() if words else ""

        base, cand = last(context.baseline_output), last(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class OpeningPunctMatchCheck:
    """1.0 when the first non-space character matches the baseline's."""

    @property
    def name(self) -> str:
        return "opening_punct"

    def score(self, output: str, context: CheckContext) -> float:
        def first_char(text: str) -> str:
            s = _sample(text).lstrip()
            return s[0] if s else ""

        base, cand = first_char(context.baseline_output), first_char(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class ClosingPunctMatchCheck:
    """1.0 when the last non-space character matches the baseline's."""

    @property
    def name(self) -> str:
        return "closing_punct"

    def score(self, output: str, context: CheckContext) -> float:
        def last_char(text: str) -> str:
            s = _sample(text).rstrip()
            return s[-1] if s else ""

        base, cand = last_char(context.baseline_output), last_char(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class WhitespaceRatioCheck:
    """1.0 when the whitespace/char ratio matches the baseline."""

    @property
    def name(self) -> str:
        return "whitespace_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _sample(text)
            if not s:
                return 0.0
            return sum(1 for c in s if c.isspace()) / len(s)

        return _ratio_match(ratio(context.baseline_output), ratio(output))
