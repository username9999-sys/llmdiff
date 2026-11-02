"""Deduplication of captured records by prompt.

Two levels:

* **Exact** — SHA-256 of the normalised prompt; identical prompts collapse.
* **Near-duplicate** — SimHash (64-bit) of the prompt's token multiset; two
  prompts whose hashes differ in at most ``threshold`` bits are considered
  near-duplicates.

Classes:

* :class:`ExactDeduper` — keeps the first record per prompt hash.
* :class:`SimHashDeduper` — keeps the first record per simhash "bucket";
  a new record is dropped when any kept hash is within ``threshold`` hamming
  distance.  For large corpora a bucketed index (by the first 16-bit
  prefix blocks) keeps the lookup fast enough; here we use the standard
  "check the 4 sub-blocks" trick which bounds candidates without a full scan.
* :class:`DedupPipeline` — runs exact then near-dup, exposes stats.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from llmdiff.capture.errors import DedupError

__all__ = [
    "normalize_prompt",
    "prompt_hash",
    "simhash",
    "hamming",
    "ExactDeduper",
    "SimHashDeduper",
    "DedupPipeline",
    "dedup_records",
]

Record = Mapping[str, Any]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_prompt(prompt: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation-ish tokens.

    Used before hashing so that ``"Hello,  world!"`` and ``"hello world"``
    hash identically.
    """
    if not isinstance(prompt, str):
        raise DedupError(f"prompt must be str, got {type(prompt).__name__}")
    lowered = prompt.lower()
    tokens = _TOKEN_RE.findall(lowered)
    return " ".join(tokens)


def prompt_hash(prompt: str) -> str:
    """SHA-256 hex of the normalised prompt."""
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ simhash
def simhash(prompt: str, *, bits: int = 64) -> int:
    """Compute the SimHash of a prompt's token multiset.

    Each token contributes its md5-derived ``bits``-wide vector, weighted by
    token frequency; the sign of each accumulated dimension forms the hash.
    """
    if bits <= 0 or bits > 64:
        raise DedupError(f"simhash bits must be in 1..64, got {bits}")
    tokens = _TOKEN_RE.findall(prompt.lower())
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - simhash only
        h = int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)
        for i in range(bits):
            if h & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1
    result = 0
    for i in range(bits):
        if vector[i] > 0:
            result |= 1 << i
    return result


def hamming(a: int, b: int) -> int:
    """Hamming distance between two ints."""
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ dedupers
class ExactDeduper:
    """Drop records whose normalised prompt was already seen."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.dropped = 0

    def is_duplicate(self, record: Record) -> bool:
        h = prompt_hash(str(record.get("prompt", "")))
        if h in self._seen:
            self.dropped += 1
            return True
        self._seen.add(h)
        return False

    def filter(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        return [dict(r) for r in records if not self.is_duplicate(r)]

    def __repr__(self) -> str:
        return f"ExactDeduper(seen={len(self._seen)}, dropped={self.dropped})"


class SimHashDeduper:
    """Drop near-duplicate prompts using SimHash + hamming threshold.

    Args:
        threshold: Maximum hamming distance for two prompts to count as
            near-duplicates (default 3, the classic value for 64-bit
            simhash).
        bits: SimHash width (default 64).

    Lookup uses the standard block trick: split the hash into ``blocks``
    sub-blocks; a candidate can only be within ``threshold`` bits if at least
    one sub-block matches exactly, so we index kept hashes by each sub-block
    and only compare against candidates sharing a sub-block.
    """

    def __init__(self, threshold: int = 3, *, bits: int = 64, blocks: int = 4) -> None:
        if not (0 <= threshold < bits):
            raise DedupError(f"threshold must be in [0, {bits}), got {threshold}")
        if blocks < 1 or (bits % blocks) != 0:
            raise DedupError(f"blocks must divide bits ({bits}), got {blocks}")
        self.threshold = threshold
        self.bits = bits
        self.blocks = blocks
        self._kept: list[int] = []
        self._index: dict[int, list[int]] = {}
        self.dropped = 0

    def _subblocks(self, h: int) -> list[int]:
        width = self.bits // self.blocks
        mask = (1 << width) - 1
        return [(h >> (i * width)) & mask for i in range(self.blocks)]

    def is_duplicate(self, record: Record) -> bool:
        h = simhash(str(record.get("prompt", "")), bits=self.bits)
        candidates: set[int] = set()
        for sb in self._subblocks(h):
            for idx in self._index.get(sb, ()):  # index stores positions
                candidates.add(idx)
        for idx in candidates:
            if hamming(self._kept[idx], h) <= self.threshold:
                self.dropped += 1
                return True
        pos = len(self._kept)
        self._kept.append(h)
        for sb in self._subblocks(h):
            self._index.setdefault(sb, []).append(pos)
        return False

    def filter(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        return [dict(r) for r in records if not self.is_duplicate(r)]

    def __repr__(self) -> str:
        return (
            f"SimHashDeduper(threshold={self.threshold}, bits={self.bits}, "
            f"kept={len(self._kept)}, dropped={self.dropped})"
        )


class DedupPipeline:
    """Exact dedup followed by near-dup dedup.

    Args:
        near_threshold: Hamming threshold for the simhash stage; ``None``
            disables near-dup entirely.
    """

    def __init__(self, *, near_threshold: int | None = 3, bits: int = 64) -> None:
        self.exact = ExactDeduper()
        self.near = (
            SimHashDeduper(near_threshold, bits=bits) if near_threshold is not None else None
        )

    def is_duplicate(self, record: Record) -> bool:
        if self.exact.is_duplicate(record):
            return True
        return self.near is not None and self.near.is_duplicate(record)

    def filter(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for r in records:
            if not self.is_duplicate(r):
                kept.append(dict(r))
        return kept

    @property
    def stats(self) -> dict[str, int]:
        return {
            "exact_dropped": self.exact.dropped,
            "near_dropped": self.near.dropped if self.near else 0,
        }

    def __repr__(self) -> str:
        return f"DedupPipeline({self.stats})"


def dedup_records(
    records: Iterable[Record],
    *,
    near_threshold: int | None = 3,
) -> list[dict[str, Any]]:
    """Convenience: dedup a list of record dicts, exact + near."""
    return DedupPipeline(near_threshold=near_threshold).filter(records)
