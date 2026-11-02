"""The :class:`Case` unit of comparison.

A *case* is one prompt together with the output produced by the baseline
(system under test before the change) and the output produced by the candidate
(system under test after the change).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from llmdiff.core.errors import CorpusError

__all__ = ["Case"]

_REQUIRED_FIELDS = ("id", "prompt", "baseline_output", "candidate_output")


@dataclass(frozen=True, slots=True)
class Case:
    """A single prompt compared across two system versions.

    Attributes:
        id: Stable identifier, unique within a corpus.
        prompt: The input that was sent to the model.
        baseline_output: Output produced by the pre-change system.
        candidate_output: Output produced by the post-change system.
        metadata: Free-form annotations such as ``model``, ``tags`` or
            ``timestamp`` (ISO-8601, e.g. ``2026-11-01T12:00:00Z``).
    """

    id: str
    prompt: str
    baseline_output: str
    candidate_output: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CorpusError("case id must be a non-empty string")
        for name in ("prompt", "baseline_output", "candidate_output"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise CorpusError(
                    f"case {self.id!r}: {name} must be a string, got {type(value).__name__}"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Case:
        """Build a case from a mapping, validating required fields.

        Raises:
            CorpusError: If a required field is missing or has the wrong type.
        """
        if not isinstance(data, Mapping):
            raise CorpusError(f"case must be a mapping, got {type(data).__name__}")
        missing = [name for name in _REQUIRED_FIELDS if name not in data]
        if missing:
            raise CorpusError(f"case missing required field(s): {', '.join(missing)}")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise CorpusError(f"case {data.get('id')!r}: metadata must be a mapping")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            baseline_output=str(data["baseline_output"]),
            candidate_output=str(data["candidate_output"]),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this case."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "baseline_output": self.baseline_output,
            "candidate_output": self.candidate_output,
            "metadata": dict(self.metadata),
        }

    @property
    def tags(self) -> tuple[str, ...]:
        """Tags attached to this case, always as a tuple of strings."""
        raw = self.metadata.get("tags") or ()
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(tag) for tag in raw)

    @property
    def model(self) -> str | None:
        """The model that produced the candidate output, when recorded."""
        value = self.metadata.get("model")
        return str(value) if value is not None else None

    def with_candidate(self, output: str, **metadata: Any) -> Case:
        """Return a copy of this case with a different candidate output."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, candidate_output=output, metadata=merged)

    def with_baseline(self, output: str, **metadata: Any) -> Case:
        """Return a copy of this case with a different baseline output."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, baseline_output=output, metadata=merged)
