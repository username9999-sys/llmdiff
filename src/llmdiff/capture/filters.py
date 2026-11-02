"""Record filters for captured corpora.

A filter is any object with ``keep(record: Mapping) -> bool``.  Records are
plain dicts (the ``Case.to_dict()`` shape): ``{"id", "prompt",
"baseline_output", "candidate_output", "metadata"}``.

Provided filters:

* :class:`TagFilter` — keep records carrying any/all of the given tags.
* :class:`ModelFilter` — keep records whose ``metadata["model"]`` matches.
* :class:`PromptLengthFilter` / :class:`OutputLengthFilter` — length ranges.
* :class:`PredicateFilter` — wrap any callable.
* :class:`FilterChain` — AND-composition of filters; also ``invert`` support.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from llmdiff.capture.errors import FilterError

__all__ = [
    "RecordFilter",
    "TagFilter",
    "ModelFilter",
    "PromptLengthFilter",
    "OutputLengthFilter",
    "PredicateFilter",
    "InvertFilter",
    "FilterChain",
    "apply_filters",
]

Record = Mapping[str, Any]


@runtime_checkable
class RecordFilter(Protocol):
    """Protocol: decide whether a record should be kept."""

    def keep(self, record: Record) -> bool: ...


def _meta(record: Record) -> Mapping[str, Any]:
    meta = record.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def _tags(record: Record) -> set[str]:
    tags = _meta(record).get("tags") or ()
    if isinstance(tags, str):
        return {tags}
    return {str(t) for t in tags}


class TagFilter:
    """Keep records carrying the given tags.

    Args:
        tags: Tag names.
        match_all: Require every tag (default: any overlap is enough).
    """

    def __init__(self, tags: Iterable[str], *, match_all: bool = False) -> None:
        self.tags = {str(t) for t in tags}
        if not self.tags:
            raise FilterError("TagFilter needs at least one tag")
        self.match_all = match_all

    def keep(self, record: Record) -> bool:
        have = _tags(record)
        if self.match_all:
            return self.tags.issubset(have)
        return bool(self.tags & have)

    def __repr__(self) -> str:
        return f"TagFilter(tags={sorted(self.tags)!r}, match_all={self.match_all})"


class ModelFilter:
    """Keep records whose ``metadata["model"]`` is in ``models``."""

    def __init__(self, models: Iterable[str]) -> None:
        self.models = {str(m) for m in models}
        if not self.models:
            raise FilterError("ModelFilter needs at least one model")

    def keep(self, record: Record) -> bool:
        return str(_meta(record).get("model", "")) in self.models

    def __repr__(self) -> str:
        return f"ModelFilter(models={sorted(self.models)!r})"


class PromptLengthFilter:
    """Keep records whose prompt length (chars) is within ``[min_len, max_len]``.

    Bounds are optional; pass ``None`` for an open side.
    """

    def __init__(self, min_len: int | None = None, max_len: int | None = None) -> None:
        if min_len is not None and min_len < 0:
            raise FilterError("min_len must be >= 0")
        if max_len is not None and max_len < 0:
            raise FilterError("max_len must be >= 0")
        if min_len is not None and max_len is not None and min_len > max_len:
            raise FilterError(f"min_len {min_len} > max_len {max_len}")
        self.min_len = min_len
        self.max_len = max_len

    def keep(self, record: Record) -> bool:
        length = len(str(record.get("prompt", "")))
        if self.min_len is not None and length < self.min_len:
            return False
        return not (self.max_len is not None and length > self.max_len)

    def __repr__(self) -> str:
        return f"PromptLengthFilter(min={self.min_len}, max={self.max_len})"


class OutputLengthFilter:
    """Keep records whose candidate output length is within ``[min_len, max_len]``."""

    def __init__(self, min_len: int | None = None, max_len: int | None = None) -> None:
        if min_len is not None and min_len < 0:
            raise FilterError("min_len must be >= 0")
        if max_len is not None and max_len < 0:
            raise FilterError("max_len must be >= 0")
        if min_len is not None and max_len is not None and min_len > max_len:
            raise FilterError(f"min_len {min_len} > max_len {max_len}")
        self.min_len = min_len
        self.max_len = max_len

    def keep(self, record: Record) -> bool:
        length = len(str(record.get("candidate_output", "")))
        if self.min_len is not None and length < self.min_len:
            return False
        return not (self.max_len is not None and length > self.max_len)

    def __repr__(self) -> str:
        return f"OutputLengthFilter(min={self.min_len}, max={self.max_len})"


class PredicateFilter:
    """Wrap an arbitrary ``callable(record) -> bool`` as a filter."""

    def __init__(self, predicate: Callable[[Record], bool], *, name: str = "predicate") -> None:
        if not callable(predicate):
            raise FilterError("PredicateFilter needs a callable")
        self.predicate = predicate
        self.name = name

    def keep(self, record: Record) -> bool:
        return bool(self.predicate(record))

    def __repr__(self) -> str:
        return f"PredicateFilter(name={self.name!r})"


class InvertFilter:
    """Negate another filter."""

    def __init__(self, inner: RecordFilter) -> None:
        self.inner = inner

    def keep(self, record: Record) -> bool:
        return not self.inner.keep(record)

    def __repr__(self) -> str:
        return f"InvertFilter({self.inner!r})"


class FilterChain:
    """AND-composition: a record is kept only if every filter keeps it.

    Args:
        filters: Filters applied in order.  Short-circuits on the first
            rejection.
    """

    def __init__(self, filters: Iterable[RecordFilter] = ()) -> None:
        self.filters: list[RecordFilter] = list(filters)
        for f in self.filters:
            if not hasattr(f, "keep") or not callable(f.keep):
                raise FilterError(f"{f!r} does not implement the RecordFilter protocol")

    def keep(self, record: Record) -> bool:
        return all(f.keep(record) for f in self.filters)

    def filter(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        """Return the kept records as a list of plain dicts."""
        return [dict(r) for r in records if self.keep(r)]

    def __repr__(self) -> str:
        return f"FilterChain({self.filters!r})"


def apply_filters(
    records: Iterable[Record],
    filters: RecordFilter | Sequence[RecordFilter] | None,
) -> list[dict[str, Any]]:
    """Apply one filter or a sequence (AND-ed) to ``records``."""
    if filters is None:
        return [dict(r) for r in records]
    if isinstance(filters, Sequence) and not hasattr(filters, "keep"):
        chain = FilterChain(filters)
    else:
        chain = filters if isinstance(filters, FilterChain) else FilterChain([filters])  # type: ignore[arg-type]
    return chain.filter(records)
