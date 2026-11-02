"""Session management for capture runs.

A :class:`CaptureSession` groups the records of one capture run: it owns a
recorder, stamps every record with a ``session_id`` in metadata, rotates the
output file per session (``<stem>.<session_id>.jsonl`` by default) and can
produce a small summary of what was captured.

Typical use::

    from llmdiff.capture import CaptureSession

    with CaptureSession(out_dir="captures") as session:
        session.recorder.record("hi", "hello", model="gpt-4o")
    # writes captures/<session_id>.jsonl and prints a summary
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmdiff.capture.errors import SessionError
from llmdiff.capture.record import TrafficRecorder

__all__ = ["CaptureSession", "SessionSummary"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SessionSummary:
    """Lightweight stats about a finished session."""

    def __init__(
        self,
        session_id: str,
        started_at: str,
        ended_at: str,
        recorded: int,
        dropped: int,
        output_path: str | None,
        models: Iterable[str],
        tags: Iterable[str],
    ) -> None:
        self.session_id = session_id
        self.started_at = started_at
        self.ended_at = ended_at
        self.recorded = recorded
        self.dropped = dropped
        self.output_path = output_path
        self.models = tuple(sorted(set(models)))
        self.tags = tuple(sorted(set(tags)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "recorded": self.recorded,
            "dropped": self.dropped,
            "output_path": self.output_path,
            "models": list(self.models),
            "tags": list(self.tags),
        }

    def __repr__(self) -> str:
        return (
            f"SessionSummary(id={self.session_id!r}, recorded={self.recorded}, "
            f"dropped={self.dropped}, out={self.output_path!r})"
        )


class CaptureSession:
    """A capture run: recorder + session id + per-session output file.

    Args:
        out_dir: Directory where per-session JSONL files are written on
            :meth:`end` / context-manager exit.  Created when missing.
        session_id: Explicit session id; a uuid4-based one is generated when
            omitted.
        file_pattern: Name pattern for the output file; may contain
            ``{session_id}``, ``{date}`` and ``{stem}`` placeholders.
        recorder: Existing recorder to wrap; a fresh one is created when
            omitted.  When supplied, the session id is still injected into
            every record's metadata.
        tags: Extra tags added to every record (in addition to the session
            tag ``session:<id>``).
        **recorder_kwargs: Forwarded to :class:`TrafficRecorder` when a new
            recorder is created.
    """

    DEFAULT_PATTERN = "{stem}.{session_id}.jsonl"

    def __init__(
        self,
        out_dir: str | Path = "captures",
        *,
        session_id: str | None = None,
        file_pattern: str = DEFAULT_PATTERN,
        recorder: TrafficRecorder | None = None,
        tags: Iterable[str] = (),
        stem: str = "traffic",
        **recorder_kwargs: Any,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self.file_pattern = file_pattern
        self.stem = stem
        self._extra_tags = tuple(tags)
        self._started_at = _utcnow_iso()
        self._ended_at: str | None = None
        self._output_path: Path | None = None
        if recorder is None:
            recorder = TrafficRecorder(**recorder_kwargs)
        if not isinstance(recorder, TrafficRecorder):
            raise SessionError("recorder must be a TrafficRecorder")
        self.recorder = recorder
        self._closed = False

    # ------------------------------------------------------------- context
    def __enter__(self) -> CaptureSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end()

    # ------------------------------------------------------------- API
    @property
    def session_tag(self) -> str:
        return f"session:{self.session_id}"

    def record(self, prompt: str, output: str, **metadata: Any) -> Any:
        """Record one exchange tagged with this session's id."""
        if self._closed:
            raise SessionError("session is ended; start a new CaptureSession")
        meta = dict(metadata)
        meta["session_id"] = self.session_id
        tags = [self.session_tag, *self._extra_tags]
        existing = meta.pop("tags", None)
        if existing:
            if isinstance(existing, str):
                tags.append(existing)
            else:
                tags.extend(str(t) for t in existing)
        return self.recorder.record(prompt, output, tags=tags, **meta)

    def output_path(self) -> Path:
        """Resolve the per-session output file path (without writing)."""
        name = self.file_pattern.format(
            session_id=self.session_id,
            stem=self.stem,
            date=datetime.now(timezone.utc).strftime("%Y%m%d"),
        )
        return self.out_dir / name

    def end(self, *, save: bool = True) -> SessionSummary:
        """End the session: optionally save the recorder and return a summary."""
        if self._closed:
            raise SessionError("session already ended")
        self._closed = True
        self._ended_at = _utcnow_iso()
        if save:
            self._output_path = self.recorder.save(self.output_path())
        stats = self.recorder.stats()
        return SessionSummary(
            session_id=self.session_id,
            started_at=self._started_at,
            ended_at=self._ended_at,
            recorded=stats["recorded"],
            dropped=stats["dropped"],
            output_path=str(self._output_path) if self._output_path else None,
            models=stats["models"],
            tags=stats["tags"],
        )

    @property
    def is_active(self) -> bool:
        return not self._closed

    def summary(self) -> dict[str, Any]:
        """Current stats without ending the session."""
        stats = self.recorder.stats()
        return {
            "session_id": self.session_id,
            "started_at": self._started_at,
            "active": self.is_active,
            **stats,
        }

    def __repr__(self) -> str:
        state = "active" if self.is_active else "ended"
        return f"CaptureSession(id={self.session_id!r}, {state}, recorded={len(self.recorder)})"


def list_sessions(out_dir: str | Path) -> list[dict[str, Any]]:
    """Scan ``out_dir`` for per-session JSONL files and summarise each.

    Returns a list of dicts with ``session_id`` (parsed from the filename or
    from record metadata), ``path``, ``recorded`` and ``models``.
    """
    directory = Path(out_dir)
    if not directory.exists():
        raise SessionError(f"session directory does not exist: {directory}")
    from llmdiff.core.corpus import Corpus
    from llmdiff.core.errors import CorpusError, FormatError

    sessions: list[dict[str, Any]] = []
    for file in sorted(directory.glob("*.jsonl")):
        try:
            corpus = Corpus.from_jsonl(file)
        except (FormatError, CorpusError):
            continue
        session_ids = {
            str(c.metadata.get("session_id"))
            for c in corpus
            if c.metadata.get("session_id")
        }
        models = sorted({c.model for c in corpus if c.model})
        sessions.append(
            {
                "path": str(file),
                "session_id": sorted(session_ids)[0] if session_ids else file.stem,
                "recorded": len(corpus),
                "models": models,
            }
        )
    return sessions
