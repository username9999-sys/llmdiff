"""N-gram statistics dimensions: overlap, distribution, and repetition at n-gram level."""

from __future__ import annotations

import math
import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "UnigramOverlapCheck",
    "BigramOverlapCheck",
    "TrigramOverlapCheck",
    "UnigramJaccardCheck",
    "BigramJaccardCheck",
    "CharBigramOverlapCheck",
    "CharTrigramOverlapCheck",
    "MostCommonBigramMatchCheck",
    "MostCommonTrigramMatchCheck",
    "BigramEntropyCheck",
    "TrigramEntropyCheck",
    "RepeatedBigramRatioCheck",
    "RepeatedTrigramRatioCheck",
    "MaxRepeatRunCheck",
    "UniqueBigramRatioCheck",
    "UniqueTrigramRatioCheck",
    "NgramCountMatchCheck",
    "ShannonEntropyCheck",
    "ZipfSlopeCheck",
    "NgramDiversityCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[a-z0-9']+")


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text).lower())


def _ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def _chars(text: str) -> list[str]:
    return list(_s(text))


def _overlap(base: list[tuple[str, ...]], cand: list[tuple[str, ...]]) -> float:
    if not base and not cand:
        return 1.0
    if not base or not cand:
        return 0.0
    bset = set(base)
    return len(bset & set(cand)) / len(bset)


def _jaccard(base: list[tuple[str, ...]], cand: list[tuple[str, ...]]) -> float:
    if not base and not cand:
        return 1.0
    bset, cset = set(base), set(cand)
    union = bset | cset
    return len(bset & cset) / len(union) if union else 1.0


def _entropy(items: list[tuple[str, ...]]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class UnigramOverlapCheck:
    @property
    def name(self) -> str:
        return "unigram_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        return _overlap(_ngrams(_words(context.baseline_output), 1), _ngrams(_words(output), 1))


class BigramOverlapCheck:
    @property
    def name(self) -> str:
        return "bigram_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        return _overlap(_ngrams(_words(context.baseline_output), 2), _ngrams(_words(output), 2))


class TrigramOverlapCheck:
    @property
    def name(self) -> str:
        return "trigram_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        return _overlap(_ngrams(_words(context.baseline_output), 3), _ngrams(_words(output), 3))


class UnigramJaccardCheck:
    @property
    def name(self) -> str:
        return "unigram_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(_ngrams(_words(context.baseline_output), 1), _ngrams(_words(output), 1))


class BigramJaccardCheck:
    @property
    def name(self) -> str:
        return "bigram_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(_ngrams(_words(context.baseline_output), 2), _ngrams(_words(output), 2))


class CharBigramOverlapCheck:
    @property
    def name(self) -> str:
        return "char_bigram_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        return _overlap(_ngrams(_chars(context.baseline_output), 2), _ngrams(_chars(output), 2))


class CharTrigramOverlapCheck:
    @property
    def name(self) -> str:
        return "char_trigram_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        return _overlap(_ngrams(_chars(context.baseline_output), 3), _ngrams(_chars(output), 3))


class MostCommonBigramMatchCheck:
    """1.0 when the most frequent bigram matches the baseline's."""

    @property
    def name(self) -> str:
        return "top_bigram"

    def score(self, output: str, context: CheckContext) -> float:
        def top(text: str) -> tuple[str, ...] | None:
            grams = _ngrams(_words(text), 2)
            return Counter(grams).most_common(1)[0][0] if grams else None

        base, cand = top(context.baseline_output), top(output)
        if base is None and cand is None:
            return 1.0
        return 1.0 if base == cand else 0.0


class MostCommonTrigramMatchCheck:
    @property
    def name(self) -> str:
        return "top_trigram"

    def score(self, output: str, context: CheckContext) -> float:
        def top(text: str) -> tuple[str, ...] | None:
            grams = _ngrams(_words(text), 3)
            return Counter(grams).most_common(1)[0][0] if grams else None

        base, cand = top(context.baseline_output), top(output)
        if base is None and cand is None:
            return 1.0
        return 1.0 if base == cand else 0.0


class BigramEntropyCheck:
    @property
    def name(self) -> str:
        return "bigram_entropy"

    def score(self, output: str, context: CheckContext) -> float:
        base = _entropy(_ngrams(_words(context.baseline_output), 2))
        cand = _entropy(_ngrams(_words(output), 2))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class TrigramEntropyCheck:
    @property
    def name(self) -> str:
        return "trigram_entropy"

    def score(self, output: str, context: CheckContext) -> float:
        base = _entropy(_ngrams(_words(context.baseline_output), 3))
        cand = _entropy(_ngrams(_words(output), 3))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class RepeatedBigramRatioCheck:
    """Ratio of bigram occurrences that are repeats of an earlier bigram."""

    @property
    def name(self) -> str:
        return "repeated_bigram_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            grams = _ngrams(_words(text), 2)
            if not grams:
                return 0.0
            seen: set[tuple[str, ...]] = set()
            repeated = 0
            for g in grams:
                if g in seen:
                    repeated += 1
                else:
                    seen.add(g)
            return repeated / len(grams)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class RepeatedTrigramRatioCheck:
    @property
    def name(self) -> str:
        return "repeated_trigram_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            grams = _ngrams(_words(text), 3)
            if not grams:
                return 0.0
            seen: set[tuple[str, ...]] = set()
            repeated = 0
            for g in grams:
                if g in seen:
                    repeated += 1
                else:
                    seen.add(g)
            return repeated / len(grams)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class MaxRepeatRunCheck:
    """1.0 when the longest run of a repeated word matches the baseline."""

    @property
    def name(self) -> str:
        return "max_repeat_run"

    def score(self, output: str, context: CheckContext) -> float:
        def max_run(text: str) -> int:
            ws = _words(text)
            best = run = 1
            for i in range(1, len(ws)):
                if ws[i] == ws[i - 1]:
                    run += 1
                    best = max(best, run)
                else:
                    run = 1
            return best if ws else 0

        return _count_match(max_run(context.baseline_output), max_run(output))


class UniqueBigramRatioCheck:
    @property
    def name(self) -> str:
        return "unique_bigram_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            grams = _ngrams(_words(text), 2)
            return len(set(grams)) / len(grams) if grams else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class UniqueTrigramRatioCheck:
    @property
    def name(self) -> str:
        return "unique_trigram_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            grams = _ngrams(_words(text), 3)
            return len(set(grams)) / len(grams) if grams else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NgramCountMatchCheck:
    """1.0 when the total bigram count matches the baseline."""

    @property
    def name(self) -> str:
        return "bigram_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ngrams(_words(context.baseline_output), 2)),
            len(_ngrams(_words(output), 2)),
        )


class ShannonEntropyCheck:
    """Shannon entropy over characters."""

    @property
    def name(self) -> str:
        return "char_entropy"

    def score(self, output: str, context: CheckContext) -> float:
        base = _entropy([(c,) for c in _chars(context.baseline_output)])
        cand = _entropy([(c,) for c in _chars(output)])
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class ZipfSlopeCheck:
    """1.0 when the Zipf slope (log-log rank/freq) of words matches the baseline."""

    @property
    def name(self) -> str:
        return "zipf_slope"

    def score(self, output: str, context: CheckContext) -> float:
        def slope(text: str) -> float:
            counts = sorted(Counter(_words(text)).values(), reverse=True)
            if len(counts) < 2:
                return 0.0
            xs = [math.log2(i + 1) for i in range(len(counts))]
            ys = [math.log2(c) for c in counts]
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            denom = sum((x - mx) ** 2 for x in xs)
            if denom == 0:
                return 0.0
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denom

        base, cand = slope(context.baseline_output), slope(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class NgramDiversityCheck:
    """Ratio of unique bigrams to total bigrams."""

    @property
    def name(self) -> str:
        return "ngram_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def diversity(text: str) -> float:
            grams = _ngrams(_words(text), 2)
            if not grams:
                return 0.0
            return len(set(grams)) / len(grams)

        return _ratio_match(diversity(context.baseline_output), diversity(output))
