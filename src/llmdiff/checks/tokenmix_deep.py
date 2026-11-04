"""Deep token-mix dimensions: token class ratios, vocabulary stats, and mix entropy."""

from __future__ import annotations

import math
import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "AlphaTokenRatioCheck",
    "NumericTokenRatioCheck",
    "PunctTokenRatioCheck",
    "SpaceTokenRatioCheck",
    "MixedAlnumTokenRatioCheck",
    "CamelTokenRatioCheck",
    "SnakeTokenRatioCheck",
    "UpperTokenRatioCheck",
    "LowerTokenRatioCheck",
    "UnderscoreTokenRatioCheck",
    "HyphenTokenRatioCheck",
    "DotTokenRatioCheck",
    "SlashTokenRatioCheck",
    "TokenLengthStdCheck",
    "TokenLengthSkewCheck",
    "VocabRichnessCheck",
    "TokenMixEntropyCheck",
    "RareTokenRatioCheck",
    "StopwordTokenRatioCheck",
    "AvgTokenFreqCheck",
]

_MAX = 200_000
_TOKEN_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"[A-Za-z]+")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "it", "for", "on",
    "with", "as", "at", "by", "this", "that", "are", "was", "be", "not",
}


def _s(text: str) -> str:
    return text[:_MAX]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_s(text))


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _class_ratio(text: str, pred) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if pred(t)) / len(toks)


class AlphaTokenRatioCheck:
    @property
    def name(self) -> str:
        return "alpha_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: t.isalpha()),
            _class_ratio(output, lambda t: t.isalpha()),
        )


class NumericTokenRatioCheck:
    @property
    def name(self) -> str:
        return "numeric_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: t.isdigit()),
            _class_ratio(output, lambda t: t.isdigit()),
        )


class PunctTokenRatioCheck:
    @property
    def name(self) -> str:
        return "punct_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(
                context.baseline_output,
                lambda t: not t.isalnum() and not any(c.isalpha() or c.isdigit() for c in t),
            ),
            _class_ratio(
                output,
                lambda t: not t.isalnum() and not any(c.isalpha() or c.isdigit() for c in t),
            ),
        )


class SpaceTokenRatioCheck:
    @property
    def name(self) -> str:
        return "space_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            return s.count(" ") / len(s) if s else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class MixedAlnumTokenRatioCheck:
    @property
    def name(self) -> str:
        return "mixed_alnum_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def pred(t: str) -> bool:
            return any(c.isalpha() for c in t) and any(c.isdigit() for c in t)

        return _ratio_match(
            _class_ratio(context.baseline_output, pred),
            _class_ratio(output, pred),
        )


class CamelTokenRatioCheck:
    @property
    def name(self) -> str:
        return "camel_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def pred(t: str) -> bool:
            return t[:1].islower() and any(c.isupper() for c in t[1:]) and t.isalnum()

        return _ratio_match(
            _class_ratio(context.baseline_output, pred),
            _class_ratio(output, pred),
        )


class SnakeTokenRatioCheck:
    @property
    def name(self) -> str:
        return "snake_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: "_" in t and t.islower()),
            _class_ratio(output, lambda t: "_" in t and t.islower()),
        )


class UpperTokenRatioCheck:
    @property
    def name(self) -> str:
        return "upper_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: t.isupper() and len(t) > 1),
            _class_ratio(output, lambda t: t.isupper() and len(t) > 1),
        )


class LowerTokenRatioCheck:
    @property
    def name(self) -> str:
        return "lower_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: t.islower()),
            _class_ratio(output, lambda t: t.islower()),
        )


class UnderscoreTokenRatioCheck:
    @property
    def name(self) -> str:
        return "underscore_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: "_" in t),
            _class_ratio(output, lambda t: "_" in t),
        )


class HyphenTokenRatioCheck:
    @property
    def name(self) -> str:
        return "hyphen_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: "-" in t),
            _class_ratio(output, lambda t: "-" in t),
        )


class DotTokenRatioCheck:
    @property
    def name(self) -> str:
        return "dot_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: "." in t),
            _class_ratio(output, lambda t: "." in t),
        )


class SlashTokenRatioCheck:
    @property
    def name(self) -> str:
        return "slash_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        return _ratio_match(
            _class_ratio(context.baseline_output, lambda t: "/" in t),
            _class_ratio(output, lambda t: "/" in t),
        )


class TokenLengthStdCheck:
    @property
    def name(self) -> str:
        return "token_len_std"

    def score(self, output: str, context: CheckContext) -> float:
        def std(text: str) -> float:
            lens = [len(t) for t in _tokens(text)]
            if len(lens) < 2:
                return 0.0
            mean = sum(lens) / len(lens)
            return math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))

        base, cand = std(context.baseline_output), std(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class TokenLengthSkewCheck:
    @property
    def name(self) -> str:
        return "token_len_skew"

    def score(self, output: str, context: CheckContext) -> float:
        def skew(text: str) -> float:
            lens = [len(t) for t in _tokens(text)]
            if len(lens) < 3:
                return 0.0
            mean = sum(lens) / len(lens)
            var = sum((x - mean) ** 2 for x in lens) / len(lens)
            if var == 0:
                return 0.0
            sd = math.sqrt(var)
            return sum((x - mean) ** 3 for x in lens) / (len(lens) * sd**3)

        base, cand = skew(context.baseline_output), skew(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class VocabRichnessCheck:
    """Type-token ratio over word tokens."""

    @property
    def name(self) -> str:
        return "vocab_richness"

    def score(self, output: str, context: CheckContext) -> float:
        def ttr(text: str) -> float:
            ws = _WORD_RE.findall(_s(text).lower())
            return len(set(ws)) / len(ws) if ws else 0.0

        return _ratio_match(ttr(context.baseline_output), ttr(output))


class TokenMixEntropyCheck:
    """Shannon entropy over the token-class distribution."""

    @property
    def name(self) -> str:
        return "token_mix_entropy"

    def score(self, output: str, context: CheckContext) -> float:
        def entropy(text: str) -> float:
            toks = _tokens(text)
            if not toks:
                return 0.0
            classes = []
            for t in toks:
                if t.isalpha():
                    classes.append("alpha")
                elif t.isdigit():
                    classes.append("num")
                elif any(c.isalpha() or c.isdigit() for c in t):
                    classes.append("mixed")
                else:
                    classes.append("punct")
            counts = Counter(classes)
            n = len(classes)
            return -sum((c / n) * math.log2(c / n) for c in counts.values())

        base, cand = entropy(context.baseline_output), entropy(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class RareTokenRatioCheck:
    """Ratio of tokens occurring exactly once in the text."""

    @property
    def name(self) -> str:
        return "rare_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            toks = [t.lower() for t in _tokens(text)]
            if not toks:
                return 0.0
            counts = Counter(toks)
            return sum(1 for t in toks if counts[t] == 1) / len(toks)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class StopwordTokenRatioCheck:
    @property
    def name(self) -> str:
        return "stopword_token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            toks = [t.lower().strip(".,!?;:") for t in _tokens(text)]
            if not toks:
                return 0.0
            return sum(1 for t in toks if t in _STOPWORDS) / len(toks)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AvgTokenFreqCheck:
    """Average frequency of tokens within the text."""

    @property
    def name(self) -> str:
        return "avg_token_freq"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            toks = [t.lower() for t in _tokens(text)]
            if not toks:
                return 0.0
            counts = Counter(toks)
            return sum(counts[t] for t in toks) / len(toks)

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
