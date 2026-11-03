"""Advanced similarity metrics: edit distance variants, cosine, and set-based scores."""

from __future__ import annotations

import math
import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "LevenshteinRatioCheck",
    "DamerauLevenshteinRatioCheck",
    "HammingSimilarityCheck",
    "JaccardWordCheck",
    "DiceCoefficientCheck",
    "CosineWordVectorCheck",
    "CosineCharTrigramCheck",
    "OverlapCoefficientCheck",
    "SorensenDiceBigramCheck",
    "ContainmentRatioCheck",
    "LongestCommonSubseqRatioCheck",
    "LcsWordRatioCheck",
    "RatcliffObershelpCheck",
    "SmithWatermanRatioCheck",
    "CosineUnigramCheck",
    "EuclideanWordDistCheck",
    "ManhattanWordDistCheck",
    "HellingerWordDistCheck",
    "BhattacharyyaWordCheck",
    "KLDivergenceWordCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[a-z0-9']+")


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text).lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _damerau(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for x in b:
        cur = [0] * (len(a) + 1)
        for j, y in enumerate(a, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _lcs_str(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for x in b:
        cur = [0] * (len(a) + 1)
        for j, y in enumerate(a, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _cosine_from_counts(ca: Counter[str], cb: Counter[str]) -> float:
    dot = sum(ca[k] * cb.get(k, 0) for k in ca)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return dot / (na * nb)


def _trigrams(text: str) -> list[str]:
    s = _s(text)
    return [s[i : i + 3] for i in range(len(s) - 2)]


class LevenshteinRatioCheck:
    """1 - normalized Levenshtein distance between baseline and output."""

    @property
    def name(self) -> str:
        return "levenshtein_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        denom = max(len(a), len(b))
        if denom == 0:
            return 1.0
        return 1.0 - _levenshtein(a, b) / denom


class DamerauLevenshteinRatioCheck:
    """1 - normalized Damerau-Levenshtein distance (with transpositions)."""

    @property
    def name(self) -> str:
        return "damerau_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        denom = max(len(a), len(b))
        if denom == 0:
            return 1.0
        return 1.0 - _damerau(a, b) / denom


class HammingSimilarityCheck:
    """Fraction of positions where characters match (padded to equal length)."""

    @property
    def name(self) -> str:
        return "hamming_sim"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        n = max(len(a), len(b))
        if n == 0:
            return 1.0
        matches = sum(
            1 for i in range(n) if (a[i] if i < len(a) else "") == (b[i] if i < len(b) else "")
        )
        return matches / n


class JaccardWordCheck:
    @property
    def name(self) -> str:
        return "jaccard_words"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = set(_words(context.baseline_output)), set(_words(output))
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class DiceCoefficientCheck:
    """2*|A∩B| / (|A|+|B|) over word sets."""

    @property
    def name(self) -> str:
        return "dice_words"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = set(_words(context.baseline_output)), set(_words(output))
        denom = len(a) + len(b)
        if denom == 0:
            return 1.0
        return 2 * len(a & b) / denom


class CosineWordVectorCheck:
    @property
    def name(self) -> str:
        return "cosine_words"

    def score(self, output: str, context: CheckContext) -> float:
        return _cosine_from_counts(
            Counter(_words(context.baseline_output)), Counter(_words(output))
        )


class CosineCharTrigramCheck:
    @property
    def name(self) -> str:
        return "cosine_char_trigrams"

    def score(self, output: str, context: CheckContext) -> float:
        return _cosine_from_counts(
            Counter(_trigrams(context.baseline_output)), Counter(_trigrams(output))
        )


class OverlapCoefficientCheck:
    """|A∩B| / min(|A|,|B|) over word sets."""

    @property
    def name(self) -> str:
        return "overlap_coef"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = set(_words(context.baseline_output)), set(_words(output))
        m = min(len(a), len(b))
        if m == 0:
            return 1.0 if len(a) == len(b) else 0.0
        return len(a & b) / m


class SorensenDiceBigramCheck:
    @property
    def name(self) -> str:
        return "dice_bigrams"

    def score(self, output: str, context: CheckContext) -> float:
        def bigrams(text: str) -> set[tuple[str, str]]:
            ws = _words(text)
            return {tuple(ws[i : i + 2]) for i in range(len(ws) - 1)}

        a, b = bigrams(context.baseline_output), bigrams(output)
        denom = len(a) + len(b)
        if denom == 0:
            return 1.0
        return 2 * len(a & b) / denom


class ContainmentRatioCheck:
    """|A∩B| / |A| — how much of the baseline word set is contained in the output."""

    @property
    def name(self) -> str:
        return "containment"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = set(_words(context.baseline_output)), set(_words(output))
        if not a:
            return 1.0
        return len(a & b) / len(a)


class LongestCommonSubseqRatioCheck:
    """LCS length over characters, normalized by max length."""

    @property
    def name(self) -> str:
        return "lcs_char_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        denom = max(len(a), len(b))
        if denom == 0:
            return 1.0
        return _lcs_str(a, b) / denom


class LcsWordRatioCheck:
    """LCS length over words, normalized by max word count."""

    @property
    def name(self) -> str:
        return "lcs_word_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _words(context.baseline_output), _words(output)
        denom = max(len(a), len(b))
        if denom == 0:
            return 1.0
        return _lcs_len(a, b) / denom


class RatcliffObershelpCheck:
    """Ratcliff-Obershelp similarity: 2*M / (|A|+|B|) over chars."""

    @property
    def name(self) -> str:
        return "ratcliff_obershelp"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        denom = len(a) + len(b)
        if denom == 0:
            return 1.0
        return 2 * _lcs_str(a, b) / denom


class SmithWatermanRatioCheck:
    """Smith-Waterman local alignment score normalized by baseline length."""

    @property
    def name(self) -> str:
        return "smith_waterman"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _s(context.baseline_output), _s(output)
        if not a:
            return 1.0 if not b else 0.0
        la, lb = len(a), len(b)
        prev = [0] * (lb + 1)
        best = 0
        for i in range(1, la + 1):
            cur = [0] * (lb + 1)
            for j in range(1, lb + 1):
                match = 2 if a[i - 1] == b[j - 1] else -1
                cur[j] = max(0, prev[j - 1] + match, prev[j] - 1, cur[j - 1] - 1)
                best = max(best, cur[j])
            prev = cur
        return best / (2 * la)


class CosineUnigramCheck:
    @property
    def name(self) -> str:
        return "cosine_unigrams"

    def score(self, output: str, context: CheckContext) -> float:
        return _cosine_from_counts(
            Counter(_s(context.baseline_output).lower().split()),
            Counter(_s(output).lower().split()),
        )


class EuclideanWordDistCheck:
    """1 - normalized Euclidean distance between word-count vectors."""

    @property
    def name(self) -> str:
        return "euclidean_words"

    def score(self, output: str, context: CheckContext) -> float:
        ca = Counter(_words(context.baseline_output))
        cb = Counter(_words(output))
        keys = set(ca) | set(cb)
        dist = math.sqrt(sum((ca.get(k, 0) - cb.get(k, 0)) ** 2 for k in keys))
        norm = math.sqrt(sum(ca.get(k, 0) ** 2 + cb.get(k, 0) ** 2 for k in keys))
        if norm == 0:
            return 1.0
        return 1.0 - dist / norm


class ManhattanWordDistCheck:
    """1 - normalized Manhattan distance between word-count vectors."""

    @property
    def name(self) -> str:
        return "manhattan_words"

    def score(self, output: str, context: CheckContext) -> float:
        ca = Counter(_words(context.baseline_output))
        cb = Counter(_words(output))
        keys = set(ca) | set(cb)
        dist = sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in keys)
        norm = sum(ca.get(k, 0) + cb.get(k, 0) for k in keys)
        if norm == 0:
            return 1.0
        return 1.0 - dist / norm


class HellingerWordDistCheck:
    """1 - Hellinger distance between normalized word distributions."""

    @property
    def name(self) -> str:
        return "hellinger_words"

    def score(self, output: str, context: CheckContext) -> float:
        ca = Counter(_words(context.baseline_output))
        cb = Counter(_words(output))
        ta = sum(ca.values())
        tb = sum(cb.values())
        if ta == 0 or tb == 0:
            return 1.0 if ta == tb else 0.0
        keys = set(ca) | set(cb)
        h = math.sqrt(
            sum((math.sqrt(ca.get(k, 0) / ta) - math.sqrt(cb.get(k, 0) / tb)) ** 2 for k in keys)
        ) / math.sqrt(2)
        return 1.0 - h


class BhattacharyyaWordCheck:
    """Bhattacharyya coefficient between normalized word distributions."""

    @property
    def name(self) -> str:
        return "bhattacharyya_words"

    def score(self, output: str, context: CheckContext) -> float:
        ca = Counter(_words(context.baseline_output))
        cb = Counter(_words(output))
        ta = sum(ca.values())
        tb = sum(cb.values())
        if ta == 0 or tb == 0:
            return 1.0 if ta == tb else 0.0
        keys = set(ca) | set(cb)
        return sum(math.sqrt(ca.get(k, 0) / ta * cb.get(k, 0) / tb) for k in keys)


class KLDivergenceWordCheck:
    """1 - normalized symmetric KL divergence between word distributions."""

    @property
    def name(self) -> str:
        return "kl_words"

    def score(self, output: str, context: CheckContext) -> float:
        ca = Counter(_words(context.baseline_output))
        cb = Counter(_words(output))
        ta = sum(ca.values())
        tb = sum(cb.values())
        if ta == 0 or tb == 0:
            return 1.0 if ta == tb else 0.0
        keys = set(ca) | set(cb)
        eps = 1e-12
        kl_ab = sum(
            (ca.get(k, 0) / ta) * math.log2((ca.get(k, 0) / ta + eps) / (cb.get(k, 0) / tb + eps))
            for k in keys
        )
        kl_ba = sum(
            (cb.get(k, 0) / tb) * math.log2((cb.get(k, 0) / tb + eps) / (ca.get(k, 0) / ta + eps))
            for k in keys
        )
        sym = (kl_ab + kl_ba) / 2
        return max(0.0, 1.0 - sym / 10.0)
