"""The :class:`Corpus` container and its JSON Lines serialisation.

The corpus is what makes ``llmdiff`` different from a hand-written golden
dataset: it is captured from what users actually sent, so it reflects the real
distribution of prompts instead of the distribution a developer imagined.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llmdiff.core.case import Case
from llmdiff.core.errors import CorpusError, FormatError

__all__ = ["Corpus", "CorpusStats"]


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregate counts describing a corpus."""

    total: int
    with_tags: int
    models: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of these stats."""
        return {"total": self.total, "with_tags": self.with_tags, "models": list(self.models)}


class Corpus(Sequence[Case]):
    """An ordered, immutable collection of :class:`Case` objects.

    ``Corpus`` implements :class:`~collections.abc.Sequence`, so it supports
    ``len()``, indexing and iteration in the usual way.
    """

    __slots__ = ("_cases",)

    def __init__(self, cases: Iterable[Case] = ()) -> None:
        materialised = tuple(cases)
        seen: set[str] = set()
        for case in materialised:
            if not isinstance(case, Case):
                raise CorpusError(f"corpus contains a non-Case object: {type(case).__name__}")
            if case.id in seen:
                raise CorpusError(f"duplicate case id in corpus: {case.id!r}")
            seen.add(case.id)
        self._cases = materialised

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, index: int | slice) -> Case | tuple[Case, ...]:
        return self._cases[index]

    def __iter__(self) -> Iterator[Case]:
        return iter(self._cases)

    def __repr__(self) -> str:
        return f"Corpus({len(self._cases)} cases)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Corpus):
            return NotImplemented
        return self._cases == other._cases

    def __hash__(self) -> int:
        return hash(self._cases)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> Corpus:
        """Build a corpus from an iterable of mappings.

        Raises:
            CorpusError: If any record is malformed or ids collide.
        """
        return cls(Case.from_dict(record) for record in records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Corpus:
        """Load a corpus from a JSON Lines file.

        Blank lines are skipped. Each remaining line must be a JSON object.

        Raises:
            FormatError: If a line is not valid JSON.
            CorpusError: If a line is not a valid case.
        """
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise FormatError(f"cannot read corpus file {source}: {exc}") from exc
        return cls.from_jsonl_text(text, origin=str(source))

    @classmethod
    def from_jsonl_text(cls, text: str, *, origin: str = "<string>") -> Corpus:
        """Load a corpus from JSON Lines text.

        Raises:
            FormatError: If a line is not valid JSON.
            CorpusError: If a line is not a valid case.
        """
        cases: list[Case] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{origin}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, Mapping):
                raise FormatError(f"{origin}:{line_no}: expected a JSON object")
            try:
                cases.append(Case.from_dict(payload))
            except CorpusError as exc:
                raise FormatError(f"{origin}:{line_no}: {exc}") from exc
        return cls(cases)

    def to_jsonl(self, path: str | Path) -> Path:
        """Write this corpus to a JSON Lines file and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for case in self._cases:
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return target

    def to_jsonl_text(self) -> str:
        """Serialise this corpus to JSON Lines text."""
        return "".join(
            json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for case in self._cases
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return this corpus as a list of JSON-serialisable mappings."""
        return [case.to_dict() for case in self._cases]

    def get(self, case_id: str) -> Case | None:
        """Return the case with the given id, or ``None`` when absent."""
        for case in self._cases:
            if case.id == case_id:
                return case
        return None

    def filter_by_tags(self, tags: Iterable[str], *, match_all: bool = False) -> Corpus:
        """Return the subset of cases carrying the given tags.

        Args:
            tags: Tags to filter on.
            match_all: When true a case must carry every tag; otherwise any
                overlap is enough.
        """
        wanted = {str(tag) for tag in tags}
        if not wanted:
            return self
        if match_all:
            return Corpus(case for case in self._cases if wanted.issubset(set(case.tags)))
        return Corpus(case for case in self._cases if wanted & set(case.tags))

    def filter_by_model(self, model: str) -> Corpus:
        """Return the subset of cases whose recorded model matches."""
        return Corpus(case for case in self._cases if case.model == model)

    def merge_candidates(self, other: Corpus) -> Corpus:
        """Return a corpus whose candidate outputs come from ``other``.

        Cases are matched by id. Baseline outputs and prompts are kept from
        ``self``; candidate outputs and metadata are taken from ``other``.

        Raises:
            CorpusError: If ``other`` contains an id absent from ``self``.
        """
        lookup = {case.id: case for case in other}
        merged: list[Case] = []
        for case in self._cases:
            source = lookup.get(case.id)
            if source is None:
                raise CorpusError(f"candidate corpus has no case with id {case.id!r}")
            merged.append(
                replace(
                    case,
                    candidate_output=source.candidate_output,
                    metadata=dict(source.metadata),
                )
            )
        return Corpus(merged)

    def stats(self) -> CorpusStats:
        """Return aggregate counts describing this corpus."""
        models = sorted({case.model for case in self._cases if case.model})
        return CorpusStats(
            total=len(self._cases),
            with_tags=sum(1 for case in self._cases if case.tags),
            models=tuple(models),
        )

    def sample(self, count: int, *, seed: int = 0) -> Corpus:
        """Return a deterministic random sample of at most ``count`` cases.

        Args:
            count: Maximum number of cases to return.
            seed: Seed for the random generator, so samples are reproducible.
        """
        if count <= 0:
            return Corpus(())
        if count >= len(self._cases):
            return self
        import random

        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(self._cases)), count))
        return Corpus(self._cases[index] for index in chosen)
