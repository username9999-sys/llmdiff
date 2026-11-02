"""Thread-safe in-memory recorder for captured LLM traffic.

:class:`TrafficRecorder` is the heart of the capture subpackage: middleware,
proxies and SDK wrappers all funnel exchanges into a recorder, which turns
each ``(prompt, output)`` pair into a :class:`llmdiff.core.case.Case` and can
persist the accumulated cases as a JSONL corpus via
:class:`llmdiff.core.corpus.Corpus`.

Features:

* thread-safe appends (``threading.Lock``);
* automatic id generation (sequential with a configurable prefix, or uuid4);
* optional sampling (record only 1/N via a pluggable sampler);
* tagging;
* redaction hooks — callables applied to prompt/output before storing;
* ``save(path)`` / ``load(path)`` round-trip through ``Corpus`` JSONL.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llmdiff.capture.errors import RecordError
from llmdiff.core.case import Case
from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import CorpusError, FormatError

__all__ = [
    "RedactionHook",
    "SamplerLike",
    "TrafficRecorder",
    "Record",
]


@runtime_checkable
class SamplerLike(Protocol):
    """Minimal sampler protocol used by the recorder (see sampling.py)."""

    def should_keep(self, record: Mapping[str, Any]) -> bool: ...


RedactionHook = Callable[[str], str]
"""A hook transforms a prompt or output string before it is stored."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TrafficRecorder:
    """Collects captured LLM exchanges and persists them as a corpus.

    Args:
        id_prefix: Prefix for auto-generated sequential ids (e.g. ``"rec"``
            produces ``rec-000001``).
        id_strategy: ``"sequential"`` (default) or ``"uuid"``.
        sampler: Optional object with ``should_keep(record_dict) -> bool``.
            When it returns ``False`` the exchange is dropped (not stored).
        redaction_hooks: Iterable of callables applied to the prompt and to
            the output before storing.  Applied in order.  A
            :class:`~llmdiff.capture.redact.RedactionPipeline` can be adapted
            with ``pipeline.redact``.
        tags: Tags attached to every record by default.
        metadata: Extra metadata merged into every record.
        auto_timestamp: When true (default) a ``timestamp`` (ISO-8601 UTC) is
            added to each record's metadata unless already present.
    """

    def __init__(
        self,
        *,
        id_prefix: str = "rec",
        id_strategy: str = "sequential",
        sampler: SamplerLike | None = None,
        redaction_hooks: Iterable[RedactionHook] | None = None,
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        auto_timestamp: bool = True,
    ) -> None:
        if id_strategy not in ("sequential", "uuid"):
            raise RecordError(f"unknown id_strategy {id_strategy!r}; use 'sequential' or 'uuid'")
        if not isinstance(id_prefix, str) or not id_prefix:
            raise RecordError("id_prefix must be a non-empty string")
        self.id_prefix = id_prefix
        self.id_strategy = id_strategy
        self.sampler = sampler
        self.redaction_hooks: list[RedactionHook] = list(redaction_hooks or [])
        for hook in self.redaction_hooks:
            if not callable(hook):
                raise RecordError("redaction hooks must be callable")
        self.default_tags: tuple[str, ...] = tuple(str(t) for t in (tags or ()))
        self.default_metadata: dict[str, Any] = dict(metadata or {})
        self.auto_timestamp = auto_timestamp
        self._lock = threading.Lock()
        self._cases: list[Case] = []
        self._counter = 0
        self._dropped = 0

    # ------------------------------------------------------------- ids
    def _next_id(self) -> str:
        if self.id_strategy == "uuid":
            return f"{self.id_prefix}-{uuid.uuid4().hex[:12]}"
        self._counter += 1
        return f"{self.id_prefix}-{self._counter:06d}"

    # ------------------------------------------------------------- hooks
    def _apply_hooks(self, text: str) -> str:
        for hook in self.redaction_hooks:
            text = hook(text)
        return text

    # ------------------------------------------------------------- API
    def record(
        self,
        prompt: str,
        output: str,
        *,
        baseline_output: str | None = None,
        case_id: str | None = None,
        tags: Iterable[str] | None = None,
        **metadata: Any,
    ) -> Case | None:
        """Record one exchange.

        Args:
            prompt: The prompt sent to the model (redacted before storing).
            output: The model's output (redacted before storing).  Stored as
                the *candidate* output.
            baseline_output: Optional baseline output; defaults to ``""``.
            case_id: Explicit id; auto-generated when omitted.
            tags: Extra tags merged with the recorder's default tags.
            **metadata: Extra metadata merged over the recorder defaults.

        Returns:
            The stored :class:`Case`, or ``None`` when the sampler dropped
            the record.
        """
        if not isinstance(prompt, str) or not isinstance(output, str):
            raise RecordError("prompt and output must be strings")
        merged_meta: dict[str, Any] = dict(self.default_metadata)
        merged_meta.update(metadata)
        merged_tags = list(self.default_tags) + [str(t) for t in (tags or ())]
        if merged_tags:
            merged_meta["tags"] = merged_tags
        if self.auto_timestamp and "timestamp" not in merged_meta:
            merged_meta["timestamp"] = _utcnow_iso()

        raw_record = {
            "prompt": prompt,
            "output": output,
            "metadata": merged_meta,
        }
        if self.sampler is not None:
            try:
                keep = self.sampler.should_keep(raw_record)
            except Exception as exc:  # sampler bugs must not break capture
                raise RecordError(f"sampler raised: {exc}") from exc
            if not keep:
                with self._lock:
                    self._dropped += 1
                return None

        clean_prompt = self._apply_hooks(prompt)
        clean_output = self._apply_hooks(output)
        clean_baseline = self._apply_hooks(baseline_output) if baseline_output is not None else ""

        with self._lock:
            case_id = case_id or self._next_id()
            if any(c.id == case_id for c in self._cases):
                raise RecordError(f"duplicate case id {case_id!r}")
            case = Case(
                id=case_id,
                prompt=clean_prompt,
                baseline_output=clean_baseline,
                candidate_output=clean_output,
                metadata=merged_meta,
            )
            self._cases.append(case)
            return case

    def add_case(self, case: Case) -> Case:
        """Add an already-built :class:`Case` (hooks are NOT applied)."""
        if not isinstance(case, Case):
            raise RecordError(f"expected Case, got {type(case).__name__}")
        with self._lock:
            if any(c.id == case.id for c in self._cases):
                raise RecordError(f"duplicate case id {case.id!r}")
            self._cases.append(case)
            return case

    # ------------------------------------------------------------- views
    @property
    def cases(self) -> tuple[Case, ...]:
        """A snapshot tuple of all recorded cases."""
        with self._lock:
            return tuple(self._cases)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cases)

    @property
    def dropped_count(self) -> int:
        """How many records the sampler rejected."""
        with self._lock:
            return self._dropped

    def corpus(self) -> Corpus:
        """Return the recorded cases as an immutable :class:`Corpus`."""
        return Corpus(self.cases)

    def clear(self) -> None:
        """Drop all recorded cases (and reset the sequential counter)."""
        with self._lock:
            self._cases.clear()
            self._counter = 0
            self._dropped = 0

    # ------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        """Write all recorded cases to ``path`` as JSONL via ``Corpus``.

        The write is atomic: data goes to a temp file in the same directory
        and is then renamed over the target.
        """
        target = Path(path)
        corpus = self.corpus()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        try:
            corpus.to_jsonl(tmp)
            tmp.replace(target)
        except OSError as exc:
            raise RecordError(f"cannot save recorder output to {target}: {exc}") from exc
        return target

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> TrafficRecorder:
        """Load a JSONL corpus written by :meth:`save` into a new recorder.

        Extra keyword arguments are forwarded to the constructor.
        """
        source = Path(path)
        try:
            corpus = Corpus.from_jsonl(source)
        except (FormatError, CorpusError) as exc:
            raise RecordError(f"cannot load recorder file {source}: {exc}") from exc
        recorder = cls(**kwargs)
        for case in corpus:
            recorder.add_case(case)
        with recorder._lock:
            recorder._counter = len(recorder._cases)
        return recorder

    def save_records(self, path: str | Path) -> Path:
        """Alias of :meth:`save` writing raw record dicts (same JSONL shape)."""
        return self.save(path)

    def to_jsonl_text(self) -> str:
        """Serialise recorded cases to JSONL text."""
        return self.corpus().to_jsonl_text()

    @classmethod
    def from_jsonl_text(cls, text: str, **kwargs: Any) -> TrafficRecorder:
        """Load a recorder from JSONL text."""
        try:
            corpus = Corpus.from_jsonl_text(text)
        except (FormatError, CorpusError) as exc:
            raise RecordError(f"cannot parse JSONL: {exc}") from exc
        recorder = cls(**kwargs)
        for case in corpus:
            recorder.add_case(case)
        with recorder._lock:
            recorder._counter = len(recorder._cases)
        return recorder

    # ------------------------------------------------------------- misc
    def stats(self) -> dict[str, Any]:
        """Small summary dict: counts, tags, models, dropped."""
        cases = self.cases
        tags: set[str] = set()
        models: set[str] = set()
        for case in cases:
            tags.update(case.tags)
            if case.model:
                models.add(case.model)
        return {
            "recorded": len(cases),
            "dropped": self.dropped_count,
            "tags": sorted(tags),
            "models": sorted(models),
            "first_id": cases[0].id if cases else None,
            "last_id": cases[-1].id if cases else None,
        }

    def __repr__(self) -> str:
        return (
            f"TrafficRecorder(recorded={len(self)}, dropped={self.dropped_count}, "
            f"id_strategy={self.id_strategy!r})"
        )


# Backwards-friendly alias: a "record" is just the dict shape record() sees.
Record = dict[str, Any]


def records_to_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise raw record mappings to JSONL text (one JSON object per line)."""
    lines = []
    for record in records:
        lines.append(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")
