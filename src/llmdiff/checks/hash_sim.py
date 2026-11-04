"""Hash-based similarity dimensions: simhash, minhash, and content fingerprints."""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "SimhashSimilarityCheck",
    "SimhashHammingCheck",
    "MinhashJaccardCheck",
    "ContentHashEqualCheck",
    "Md5PrefixMatchCheck",
    "Sha256PrefixMatchCheck",
    "RollingHashMatchCheck",
    "ShingleSetJaccardCheck",
    "WordShingleJaccardCheck",
    "CharShingleJaccardCheck",
    "MinhashSignatureMatchCheck",
    "SimhashBucketCountCheck",
    "HashPrefixLenMatchCheck",
    "FnvHashMatchCheck",
    "Crc32MatchCheck",
    "Blake2bPrefixMatchCheck",
    "HashEntropyCheck",
    "ShingleCountMatchCheck",
    "SimhashBitFlipCountCheck",
    "MinhashMinCountCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[a-z0-9']+")


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text).lower())


def _shingles(items: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(items[i : i + n]) for i in range(len(items) - n + 1)}


def _simhash(text: str, bits: int = 64) -> int:
    s = _s(text).lower()
    v = [0] * bits
    for i in range(len(s) - 2):
        h = int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= 1 << b
    return out


def _minhash(text: str, num: int = 32) -> list[int]:
    s = _s(text).lower()
    shingles = _shingles(list(s), 3)
    if not shingles:
        return [0] * num
    out = []
    for k in range(num):
        best = None
        for sh in shingles:
            h = int(hashlib.md5((str(k) + str(sh)).encode()).hexdigest(), 16)
            if best is None or h < best:
                best = h
        out.append(best or 0)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class SimhashSimilarityCheck:
    """Cosine-like similarity from simhash: 1 - hamming/64."""

    @property
    def name(self) -> str:
        return "simhash_sim"

    def score(self, output: str, context: CheckContext) -> float:
        a = _simhash(context.baseline_output)
        b = _simhash(output)
        return 1.0 - _hamming(a, b) / 64


class SimhashHammingCheck:
    """1.0 when the simhash popcount matches the baseline's."""

    @property
    def name(self) -> str:
        return "simhash_hamming"

    def score(self, output: str, context: CheckContext) -> float:
        base_h = _hamming(_simhash(context.baseline_output), 0)
        cand_h = _hamming(_simhash(output), 0)
        return 1.0 if base_h == cand_h else max(0.0, 1.0 - abs(cand_h - base_h) / 64)


class MinhashJaccardCheck:
    """Jaccard similarity of minhash signatures."""

    @property
    def name(self) -> str:
        return "minhash_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = set(_minhash(context.baseline_output))
        b = set(_minhash(output))
        return _jaccard(a, b)


class ContentHashEqualCheck:
    """1.0 when md5 of baseline and output are equal."""

    @property
    def name(self) -> str:
        return "md5_equal"

    def score(self, output: str, context: CheckContext) -> float:
        a = hashlib.md5(_s(context.baseline_output).encode()).hexdigest()
        b = hashlib.md5(_s(output).encode()).hexdigest()
        return 1.0 if a == b else 0.0


class Md5PrefixMatchCheck:
    """Fraction of matching hex chars in the md5 prefix (first 8)."""

    @property
    def name(self) -> str:
        return "md5_prefix"

    def score(self, output: str, context: CheckContext) -> float:
        a = hashlib.md5(_s(context.baseline_output).encode()).hexdigest()[:8]
        b = hashlib.md5(_s(output).encode()).hexdigest()[:8]
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 8


class Sha256PrefixMatchCheck:
    @property
    def name(self) -> str:
        return "sha256_prefix"

    def score(self, output: str, context: CheckContext) -> float:
        a = hashlib.sha256(_s(context.baseline_output).encode()).hexdigest()[:8]
        b = hashlib.sha256(_s(output).encode()).hexdigest()[:8]
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 8


class RollingHashMatchCheck:
    """1.0 when the sum of rolling hashes of lines matches the baseline."""

    @property
    def name(self) -> str:
        return "rolling_hash_sum"

    def score(self, output: str, context: CheckContext) -> float:
        def rsum(text: str) -> int:
            total = 0
            for ln in _s(text).splitlines():
                h = 0
                for ch in ln:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                total = (total + h) & 0xFFFFFFFF
            return total

        base, cand = rsum(context.baseline_output), rsum(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class ShingleSetJaccardCheck:
    """Jaccard over char 3-shingles."""

    @property
    def name(self) -> str:
        return "char3_shingle_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(
            _shingles(list(_s(context.baseline_output).lower()), 3),
            _shingles(list(_s(output).lower()), 3),
        )


class WordShingleJaccardCheck:
    """Jaccard over word 2-shingles."""

    @property
    def name(self) -> str:
        return "word2_shingle_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(
            _shingles(_words(context.baseline_output), 2),
            _shingles(_words(output), 2),
        )


class CharShingleJaccardCheck:
    """Jaccard over char 5-shingles."""

    @property
    def name(self) -> str:
        return "char5_shingle_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(
            _shingles(list(_s(context.baseline_output).lower()), 5),
            _shingles(list(_s(output).lower()), 5),
        )


class MinhashSignatureMatchCheck:
    """Fraction of matching minhash signature components."""

    @property
    def name(self) -> str:
        return "minhash_sig_match"

    def score(self, output: str, context: CheckContext) -> float:
        a = _minhash(context.baseline_output)
        b = _minhash(output)
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / len(a)


class SimhashBucketCountCheck:
    """Number of set bits in the simhash (population count)."""

    @property
    def name(self) -> str:
        return "simhash_popcount"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            bin(_simhash(context.baseline_output)).count("1"),
            bin(_simhash(output)).count("1"),
        )


class HashPrefixLenMatchCheck:
    """1.0 when the count of leading zeros in md5 hex matches the baseline's."""

    @property
    def name(self) -> str:
        return "hash_common_prefix"

    def score(self, output: str, context: CheckContext) -> float:
        def leading_zeros(text: str) -> int:
            h = hashlib.md5(_s(text).encode()).hexdigest()
            return len(h) - len(h.lstrip("0"))

        return _count_match(
            leading_zeros(context.baseline_output),
            leading_zeros(output),
        )


class FnvHashMatchCheck:
    """1.0 when the FNV-1a 64 hash matches the baseline's."""

    @property
    def name(self) -> str:
        return "fnv1a_equal"

    def score(self, output: str, context: CheckContext) -> float:
        def fnv(text: str) -> int:
            h = 0xCBF29CE484222325
            for ch in _s(text):
                h = ((h ^ ord(ch)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            return h

        return 1.0 if fnv(context.baseline_output) == fnv(output) else 0.0


class Crc32MatchCheck:
    @property
    def name(self) -> str:
        return "crc32_prefix"

    def score(self, output: str, context: CheckContext) -> float:
        a = f"{zlib.crc32(_s(context.baseline_output).encode()):08x}"
        b = f"{zlib.crc32(_s(output).encode()):08x}"
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 8


class Blake2bPrefixMatchCheck:
    @property
    def name(self) -> str:
        return "blake2b_prefix"

    def score(self, output: str, context: CheckContext) -> float:
        a = hashlib.blake2b(_s(context.baseline_output).encode(), digest_size=4).hexdigest()
        b = hashlib.blake2b(_s(output).encode(), digest_size=4).hexdigest()
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 8


class HashEntropyCheck:
    """Entropy of the md5 hex digest characters (higher = more random-looking)."""

    @property
    def name(self) -> str:
        return "hash_entropy"

    def score(self, output: str, context: CheckContext) -> float:
        import math

        def entropy(text: str) -> float:
            h = hashlib.md5(_s(text).encode()).hexdigest()
            counts = Counter(h)
            n = len(h)
            return -sum((c / n) * math.log2(c / n) for c in counts.values())

        base, cand = entropy(context.baseline_output), entropy(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class ShingleCountMatchCheck:
    """1.0 when the number of char 3-shingles matches the baseline."""

    @property
    def name(self) -> str:
        return "shingle_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_shingles(list(_s(context.baseline_output).lower()), 3)),
            len(_shingles(list(_s(output).lower()), 3)),
        )


class SimhashBitFlipCountCheck:
    """1 - normalized hamming distance between baseline and output simhash."""

    @property
    def name(self) -> str:
        return "simhash_bitflip"

    def score(self, output: str, context: CheckContext) -> float:
        h = _hamming(_simhash(context.baseline_output), _simhash(output))
        return 1.0 - h / 64


class MinhashMinCountCheck:
    """1.0 when the minimum minhash component matches the baseline's."""

    @property
    def name(self) -> str:
        return "minhash_min"

    def score(self, output: str, context: CheckContext) -> float:
        a = min(_minhash(context.baseline_output))
        b = min(_minhash(output))
        return 1.0 if a == b else max(0.0, 1.0 - abs(a - b) / max(1, a))
