"""LSH-style dimensions: banded simhash, random-projection, and bucket fingerprints."""

from __future__ import annotations

import hashlib
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "SimhashBandMatchCheck",
    "SimhashBandCountCheck",
    "RandomProjectionSignCheck",
    "RpBucketJaccardCheck",
    "CharNgramLshJaccardCheck",
    "WordNgramLshJaccardCheck",
    "MinhashLshBandMatchCheck",
    "SimhashLowBitsMatchCheck",
    "SimhashHighBitsMatchCheck",
    "HashTrickBucketCountCheck",
    "FeatureHashDimCheck",
    "SimhashCosineCheck",
    "BandSignatureEqualCheck",
    "LshCollisionProbCheck",
    "SimhashNeighborCountCheck",
    "RpDotProductSignCheck",
    "CharShingleHashSetCheck",
    "WordShingleHashSetCheck",
    "SimhashPrefixBandCheck",
    "SimhashSuffixBandCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[a-z0-9']+")
_BANDS = 8
_BAND_BITS = 8


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text).lower())


def _simhash64(text: str) -> int:
    s = _s(text).lower()
    v = [0] * 64
    for i in range(len(s) - 2):
        h = int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16)
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if v[b] > 0:
            out |= 1 << b
    return out


def _bands(h: int, bands: int = _BANDS, bits: int = _BAND_BITS) -> list[int]:
    return [(h >> (i * bits)) & ((1 << bits) - 1) for i in range(bands)]


def _rp_signs(text: str, dims: int = 16) -> list[int]:
    s = _s(text).lower()
    signs = []
    for d in range(dims):
        total = 0
        for i in range(len(s) - 2):
            h = int(hashlib.md5(f"{d}:{s[i:i+3]}".encode()).hexdigest(), 16)
            total += h
        signs.append(1 if total % 2 == 0 else 0)
    return signs


def _shingle_hashes(items: list[str], n: int) -> set[int]:
    out = set()
    for i in range(len(items) - n + 1):
        sh = tuple(items[i : i + n])
        out.add(int(hashlib.md5(str(sh).encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class SimhashBandMatchCheck:
    """Fraction of matching simhash bands between baseline and output."""

    @property
    def name(self) -> str:
        return "simhash_band_match"

    def score(self, output: str, context: CheckContext) -> float:
        a = _bands(_simhash64(context.baseline_output))
        b = _bands(_simhash64(output))
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / len(a)


class SimhashBandCountCheck:
    @property
    def name(self) -> str:
        return "simhash_band_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_bands(_simhash64(context.baseline_output))),
            len(_bands(_simhash64(output))),
        )


class RandomProjectionSignCheck:
    """Fraction of matching random-projection sign bits."""

    @property
    def name(self) -> str:
        return "rp_sign_match"

    def score(self, output: str, context: CheckContext) -> float:
        a = _rp_signs(context.baseline_output)
        b = _rp_signs(output)
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / len(a)


class RpBucketJaccardCheck:
    """Jaccard of RP bucket id sets."""

    @property
    def name(self) -> str:
        return "rp_bucket_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {tuple(_rp_signs(context.baseline_output))}
        b = {tuple(_rp_signs(output))}
        return _jaccard(a, b)


class CharNgramLshJaccardCheck:
    @property
    def name(self) -> str:
        return "char_lsh_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(
            _shingle_hashes(list(_s(context.baseline_output).lower()), 3),
            _shingle_hashes(list(_s(output).lower()), 3),
        )


class WordNgramLshJaccardCheck:
    @property
    def name(self) -> str:
        return "word_lsh_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return _jaccard(
            _shingle_hashes(_words(context.baseline_output), 2),
            _shingle_hashes(_words(output), 2),
        )


class MinhashLshBandMatchCheck:
    """Fraction of matching low-16-bit minhash band values."""

    @property
    def name(self) -> str:
        return "minhash_band_match"

    def score(self, output: str, context: CheckContext) -> float:
        def lowhash(text: str) -> list[int]:
            s = _s(text).lower()
            return [
                int(hashlib.md5(s[i : i + 4].encode()).hexdigest(), 16) & 0xFFFF
                for i in range(0, max(1, len(s) - 3), 4)
            ][:16]

        a, b = lowhash(context.baseline_output), lowhash(output)
        if not a or not b:
            return 1.0 if not a and not b else 0.0
        n = min(len(a), len(b))
        return sum(1 for i in range(n) if a[i] == b[i]) / n


class SimhashLowBitsMatchCheck:
    @property
    def name(self) -> str:
        return "simhash_low_bits"

    def score(self, output: str, context: CheckContext) -> float:
        a = _simhash64(context.baseline_output) & 0xFFFF
        b = _simhash64(output) & 0xFFFF
        same = 16 - bin(a ^ b).count("1")
        return same / 16


class SimhashHighBitsMatchCheck:
    @property
    def name(self) -> str:
        return "simhash_high_bits"

    def score(self, output: str, context: CheckContext) -> float:
        a = (_simhash64(context.baseline_output) >> 48) & 0xFFFF
        b = (_simhash64(output) >> 48) & 0xFFFF
        same = 16 - bin(a ^ b).count("1")
        return same / 16


class HashTrickBucketCountCheck:
    """Number of distinct feature-hash buckets (dim=2^10)."""

    @property
    def name(self) -> str:
        return "hashed_bucket_count"

    def score(self, output: str, context: CheckContext) -> float:
        def buckets(text: str) -> int:
            s = _s(text).lower()
            return len(
                {
                    int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16) % 1024
                    for i in range(len(s) - 2)
                }
            )

        return _count_match(buckets(context.baseline_output), buckets(output))


class FeatureHashDimCheck:
    """1.0 when the hashed feature dimension (2^12) occupancy matches."""

    @property
    def name(self) -> str:
        return "feature_hash_dim"

    def score(self, output: str, context: CheckContext) -> float:
        def occ(text: str) -> int:
            s = _s(text).lower()
            return len(
                {
                    int(hashlib.md5(s[i : i + 4].encode()).hexdigest(), 16) % 4096
                    for i in range(len(s) - 3)
                }
            )

        return _count_match(occ(context.baseline_output), occ(output))


class SimhashCosineCheck:
    """Cosine similarity of simhash bit-vectors."""

    @property
    def name(self) -> str:
        return "simhash_cosine"

    def score(self, output: str, context: CheckContext) -> float:
        a = _simhash64(context.baseline_output)
        b = _simhash64(output)
        dot = sum(1 for i in range(64) if ((a >> i) & 1) == ((b >> i) & 1))
        return dot / 64


class BandSignatureEqualCheck:
    """1.0 when the full band signature tuple is identical."""

    @property
    def name(self) -> str:
        return "band_sig_equal"

    def score(self, output: str, context: CheckContext) -> float:
        a = tuple(_bands(_simhash64(context.baseline_output)))
        b = tuple(_bands(_simhash64(output)))
        return 1.0 if a == b else 0.0


class LshCollisionProbCheck:
    """Estimated collision probability from band match count (r^b)."""

    @property
    def name(self) -> str:
        return "lsh_collision_prob"

    def score(self, output: str, context: CheckContext) -> float:
        a = _bands(_simhash64(context.baseline_output))
        b = _bands(_simhash64(output))
        matches = sum(1 for x, y in zip(a, b, strict=False) if x == y)
        r = matches / len(a)
        return r**_BANDS


class SimhashNeighborCountCheck:
    """Count of bands within hamming distance 3 of the baseline's bands."""

    @property
    def name(self) -> str:
        return "simhash_neighbors"

    def score(self, output: str, context: CheckContext) -> float:
        def neighbors(text: str) -> int:
            h = _simhash64(text)
            bands = _bands(h)
            n = 0
            for band in bands:
                for delta in range(1 << _BAND_BITS):
                    if bin(band ^ delta).count("1") <= 3:
                        n += 1
            return n

        return _count_match(neighbors(context.baseline_output), neighbors(output))


class RpDotProductSignCheck:
    """Fraction of matching signs of dot products with random vectors."""

    @property
    def name(self) -> str:
        return "rp_dot_sign"

    def score(self, output: str, context: CheckContext) -> float:
        return RandomProjectionSignCheck().score(output, context)


class CharShingleHashSetCheck:
    @property
    def name(self) -> str:
        return "char_shingle_hash_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return CharNgramLshJaccardCheck().score(output, context)


class WordShingleHashSetCheck:
    @property
    def name(self) -> str:
        return "word_shingle_hash_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        return WordNgramLshJaccardCheck().score(output, context)


class SimhashPrefixBandCheck:
    """Match of the first 4 bands only."""

    @property
    def name(self) -> str:
        return "simhash_prefix_band"

    def score(self, output: str, context: CheckContext) -> float:
        a = _bands(_simhash64(context.baseline_output))[:4]
        b = _bands(_simhash64(output))[:4]
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 4


class SimhashSuffixBandCheck:
    """Match of the last 4 bands only."""

    @property
    def name(self) -> str:
        return "simhash_suffix_band"

    def score(self, output: str, context: CheckContext) -> float:
        a = _bands(_simhash64(context.baseline_output))[-4:]
        b = _bands(_simhash64(output))[-4:]
        return sum(1 for x, y in zip(a, b, strict=False) if x == y) / 4
