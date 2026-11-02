"""Sampling strategies for captured traffic.

Every sampler implements the :class:`Sampler` protocol::

    should_keep(record: Mapping[str, Any]) -> bool

where ``record`` is the raw dict ``{"prompt": ..., "output": ..., "metadata": ...}``
seen by :class:`~llmdiff.capture.record.TrafficRecorder` before hooks run.

Strategies provided:

* :class:`RandomSampler` — keep each record with probability ``p``.
* :class:`RateLimitSampler` — keep exactly 1 of every ``n`` records.
* :class:`ReservoirSampler` — fixed-size reservoir over a stream; once full,
  each new record replaces a random existing one with probability ``k/n``.
  Exposes :attr:`kept` so callers can drain the sample.
* :class:`StratifiedSampler` — per-tag rate limiting: each tag gets its own
  1/N counter so rare tags are not starved by frequent ones.
* :class:`HeadSampler` — keep only the first ``n`` records.
* :class:`CompositeSampler` — AND/OR composition of samplers.

All samplers are deterministic when given a seed.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from llmdiff.capture.errors import SamplingError

__all__ = [
    "Sampler",
    "AlwaysSampler",
    "NeverSampler",
    "RandomSampler",
    "RateLimitSampler",
    "ReservoirSampler",
    "StratifiedSampler",
    "HeadSampler",
    "CompositeSampler",
    "make_sampler",
]


@runtime_checkable
class Sampler(Protocol):
    """Protocol: decide whether a raw record should be kept."""

    def should_keep(self, record: Mapping[str, Any]) -> bool: ...


def _tags_of(record: Mapping[str, Any]) -> tuple[str, ...]:
    meta = record.get("metadata") or {}
    if isinstance(meta, Mapping):
        tags = meta.get("tags") or ()
        if isinstance(tags, str):
            return (tags,)
        return tuple(str(t) for t in tags)
    return ()


class AlwaysSampler:
    """Keep everything (default)."""

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        return True

    def __repr__(self) -> str:
        return "AlwaysSampler()"


class NeverSampler:
    """Drop everything (useful to disable capture without code changes)."""

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        return False

    def __repr__(self) -> str:
        return "NeverSampler()"


class RandomSampler:
    """Keep each record independently with probability ``p``.

    Args:
        p: Probability in ``(0, 1]``.  ``p=1`` keeps everything.
        seed: Seed for the internal RNG (deterministic when set).
    """

    def __init__(self, p: float = 1.0, *, seed: int | None = None) -> None:
        if not (0.0 < p <= 1.0):
            raise SamplingError(f"RandomSampler p must be in (0, 1], got {p}")
        self.p = p
        self._rng = random.Random(seed)

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        if self.p >= 1.0:
            return True
        return self._rng.random() < self.p

    def __repr__(self) -> str:
        return f"RandomSampler(p={self.p})"


class RateLimitSampler:
    """Keep exactly one record out of every ``n`` (deterministic 1/N).

    The first record is always kept, then every ``n``-th after that.
    """

    def __init__(self, n: int = 10) -> None:
        if not isinstance(n, int) or n < 1:
            raise SamplingError(f"RateLimitSampler n must be a positive int, got {n!r}")
        self.n = n
        self._count = 0

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        self._count += 1
        return (self._count - 1) % self.n == 0

    def __repr__(self) -> str:
        return f"RateLimitSampler(n={self.n})"


class ReservoirSampler:
    """Maintain a fixed-size reservoir sample over a stream of records.

    The first ``capacity`` records are kept.  Afterwards, the ``i``-th record
    (1-based over the whole stream) replaces a uniformly random kept record
    with probability ``capacity / i`` (Algorithm R).

    Use :attr:`kept` to read the current sample (list of raw records) and
    :attr:`seen` for the total number of records observed.
    """

    def __init__(self, capacity: int = 1000, *, seed: int | None = None) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise SamplingError(
                f"ReservoirSampler capacity must be a positive int, got {capacity!r}"
            )
        self.capacity = capacity
        self._rng = random.Random(seed)
        self.kept: list[dict[str, Any]] = []
        self.seen = 0

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        self.seen += 1
        item = dict(record)
        if len(self.kept) < self.capacity:
            self.kept.append(item)
            return True
        j = self._rng.randrange(self.seen)  # 0 .. seen-1
        if j < self.capacity:
            self.kept[j] = item
            return True
        return False

    def sample(self) -> list[dict[str, Any]]:
        """Return a copy of the current reservoir contents."""
        return list(self.kept)

    def __repr__(self) -> str:
        return (
            f"ReservoirSampler(capacity={self.capacity}, seen={self.seen}, "
            f"kept={len(self.kept)})"
        )


class StratifiedSampler:
    """Per-tag 1/N rate limiting.

    Each tag (from ``record["metadata"]["tags"]``) gets its own counter, so a
    flood of ``"prod"`` traffic does not starve rare ``"canary"`` traffic.
    Records with no tags fall into the ``"__untagged__"`` bucket.

    Args:
        n: Keep 1 of every ``n`` per bucket.
        default_n: Rate for untagged records (defaults to ``n``).
    """

    def __init__(self, n: int = 10, *, default_n: int | None = None) -> None:
        if not isinstance(n, int) or n < 1:
            raise SamplingError(f"StratifiedSampler n must be a positive int, got {n!r}")
        self.n = n
        self.default_n = default_n if default_n is not None else n
        if self.default_n < 1:
            raise SamplingError("default_n must be a positive int")
        self._counters: dict[str, int] = {}

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        tags = _tags_of(record)
        if not tags:
            bucket, n = "__untagged__", self.default_n
        else:
            bucket, n = tags[0], self.n
        self._counters[bucket] = self._counters.get(bucket, 0) + 1
        return (self._counters[bucket] - 1) % n == 0

    def counts(self) -> dict[str, int]:
        """Per-bucket observation counts."""
        return dict(self._counters)

    def __repr__(self) -> str:
        return f"StratifiedSampler(n={self.n}, buckets={len(self._counters)})"


class HeadSampler:
    """Keep only the first ``n`` records, drop the rest."""

    def __init__(self, n: int = 100) -> None:
        if not isinstance(n, int) or n < 0:
            raise SamplingError(f"HeadSampler n must be a non-negative int, got {n!r}")
        self.n = n
        self._count = 0

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        self._count += 1
        return self._count <= self.n

    def __repr__(self) -> str:
        return f"HeadSampler(n={self.n}, seen={self._count})"


class CompositeSampler:
    """Combine samplers with AND (``"all"``) or OR (``"any"``) semantics.

    Note: for stateful samplers (reservoir, rate-limit) AND/OR still calls
    every child's ``should_keep`` exactly once per record, so their internal
    counters stay consistent.
    """

    def __init__(self, samplers: Sequence[Sampler], *, mode: str = "all") -> None:
        if mode not in ("all", "any"):
            raise SamplingError(f"CompositeSampler mode must be 'all' or 'any', got {mode!r}")
        if not samplers:
            raise SamplingError("CompositeSampler needs at least one sampler")
        self.samplers = list(samplers)
        self.mode = mode

    def should_keep(self, record: Mapping[str, Any]) -> bool:
        results = (s.should_keep(record) for s in self.samplers)
        if self.mode == "all":
            return all(results)
        return any(results)

    def __repr__(self) -> str:
        return f"CompositeSampler(mode={self.mode!r}, children={self.samplers!r})"


def make_sampler(spec: Mapping[str, Any] | Sampler | None) -> Sampler:
    """Build a sampler from a config mapping.

    Supported shapes::

        {"type": "random", "p": 0.1, "seed": 42}
        {"type": "rate", "n": 10}
        {"type": "reservoir", "capacity": 500, "seed": 1}
        {"type": "stratified", "n": 5}
        {"type": "head", "n": 100}
        {"type": "all"|"any", "samplers": [ ... ]}
        {"type": "always"} / {"type": "never"}
    """
    if spec is None:
        return AlwaysSampler()
    if isinstance(spec, Sampler) or hasattr(spec, "should_keep"):
        return spec  # type: ignore[return-value]
    if not isinstance(spec, Mapping):
        raise SamplingError(f"sampler spec must be a mapping or Sampler, got {type(spec).__name__}")
    kind = str(spec.get("type", "always")).lower()
    if kind == "always":
        return AlwaysSampler()
    if kind == "never":
        return NeverSampler()
    if kind == "random":
        return RandomSampler(float(spec.get("p", 1.0)), seed=spec.get("seed"))
    if kind == "rate":
        return RateLimitSampler(int(spec.get("n", 10)))
    if kind == "reservoir":
        return ReservoirSampler(int(spec.get("capacity", 1000)), seed=spec.get("seed"))
    if kind == "stratified":
        return StratifiedSampler(int(spec.get("n", 10)), default_n=spec.get("default_n"))
    if kind == "head":
        return HeadSampler(int(spec.get("n", 100)))
    if kind in ("all", "any"):
        children = spec.get("samplers") or []
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            raise SamplingError("composite sampler needs a 'samplers' list")
        return CompositeSampler([make_sampler(c) for c in children], mode=kind)
    raise SamplingError(f"unknown sampler type {kind!r}")
