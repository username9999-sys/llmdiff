"""Corpus loading and serialisation helpers for the CLI.

All formats are handled with the stdlib only:

* ``.jsonl`` / ``.json``-lines - one JSON object per line (the native format);
* ``.json`` - a JSON array of case records;
* ``.csv`` - a flat table with the columns
  ``id,prompt,baseline_output,candidate_output,metadata`` (metadata is a
  JSON-encoded string in the CSV cell).

Records are validated through :meth:`llmdiff.core.Case.from_dict` /
:meth:`llmdiff.core.Corpus.from_records`, so malformed input raises
:class:`llmdiff.core.CorpusError` which the CLI maps to exit code 2.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llmdiff.core import Case, Corpus
from llmdiff.core.errors import FormatError

__all__ = [
    "load_corpus",
    "save_corpus",
    "records_to_jsonl",
    "records_from_jsonl",
    "records_to_csv",
    "records_from_csv",
    "records_to_json_array",
    "records_from_json_array",
    "detect_format",
    "CSV_COLUMNS",
]

CSV_COLUMNS = ("id", "prompt", "baseline_output", "candidate_output", "metadata")


def detect_format(path: str | Path) -> str:
    """Return ``"jsonl"``, ``"json"`` or ``"csv"`` based on the file suffix.

    Raises:
        FormatError: when the suffix is not recognised.
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return "jsonl"
    if suffix == "json":
        return "json"
    if suffix == "csv":
        return "csv"
    raise FormatError(f"cannot detect format of {path!r} (expected .jsonl, .json or .csv)")


def records_from_jsonl(text: str, *, origin: str = "<jsonl>") -> list[dict[str, Any]]:
    """Parse JSON-lines text into a list of record dicts."""
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise FormatError(f"{origin}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise FormatError(f"{origin}:{lineno}: expected a JSON object per line")
        records.append(record)
    return records


def records_to_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise records to JSON-lines text (one object per line)."""
    lines = [json.dumps(dict(record), ensure_ascii=False, default=str) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def records_from_json_array(text: str, *, origin: str = "<json>") -> list[dict[str, Any]]:
    """Parse a JSON array of records."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatError(f"{origin}: invalid JSON: {exc.msg}") from exc
    if isinstance(data, dict) and "cases" in data and isinstance(data["cases"], list):
        data = data["cases"]
    if not isinstance(data, list):
        raise FormatError(f"{origin}: expected a JSON array of case records")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise FormatError(f"{origin}: element {index} is not a JSON object")
    return list(data)


def records_to_json_array(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise records as a pretty-printed JSON array."""
    return json.dumps(list(records), indent=2, ensure_ascii=False, default=str) + "\n"


def records_from_csv(text: str, *, origin: str = "<csv>") -> list[dict[str, Any]]:
    """Parse CSV text into case records.

    The ``metadata`` column, when present, must contain a JSON object.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise FormatError(f"{origin}: empty CSV")
    missing = [c for c in CSV_COLUMNS[:4] if c not in reader.fieldnames]
    if missing:
        raise FormatError(f"{origin}: CSV is missing required column(s): {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    for lineno, row in enumerate(reader, start=2):
        metadata_raw = row.get("metadata") or ""
        if metadata_raw.strip():
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{origin}:{lineno}: metadata is not valid JSON: {exc.msg}") from exc
            if not isinstance(metadata, dict):
                raise FormatError(f"{origin}:{lineno}: metadata must be a JSON object")
        else:
            metadata = {}
        records.append(
            {
                "id": row.get("id") or "",
                "prompt": row.get("prompt") or "",
                "baseline_output": row.get("baseline_output") or "",
                "candidate_output": row.get("candidate_output") or "",
                "metadata": metadata,
            }
        )
    return records


def records_to_csv(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise records to CSV text with a header row."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        metadata = record.get("metadata") or {}
        writer.writerow(
            {
                "id": record.get("id", ""),
                "prompt": record.get("prompt", ""),
                "baseline_output": record.get("baseline_output", ""),
                "candidate_output": record.get("candidate_output", ""),
                "metadata": json.dumps(metadata, ensure_ascii=False, default=str)
                if metadata
                else "",
            }
        )
    return buffer.getvalue()


def _records_from_text(text: str, fmt: str, origin: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        return records_from_jsonl(text, origin=origin)
    if fmt == "json":
        return records_from_json_array(text, origin=origin)
    if fmt == "csv":
        return records_from_csv(text, origin=origin)
    raise FormatError(f"unsupported format {fmt!r}")


def _records_to_text(records: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "jsonl":
        return records_to_jsonl(records)
    if fmt == "json":
        return records_to_json_array(records)
    if fmt == "csv":
        return records_to_csv(records)
    raise FormatError(f"unsupported format {fmt!r}")


def load_corpus(path: str | Path, *, fmt: str | None = None) -> Corpus:
    """Load a corpus from ``path``.

    Args:
        path: File to read (``"-"`` reads stdin).
        fmt: Force a format (``jsonl``/``json``/``csv``); auto-detected from the
            suffix when ``None``.  Stdin defaults to ``jsonl``.

    Raises:
        FormatError: on unreadable or malformed input.
        CorpusError: on semantically invalid records.
    """
    if str(path) == "-":
        import sys

        text = sys.stdin.read()
        resolved_fmt = fmt or "jsonl"
        origin = "<stdin>"
    else:
        target = Path(path)
        if not target.exists():
            raise FormatError(f"corpus file not found: {target}")
        text = target.read_text(encoding="utf-8")
        resolved_fmt = fmt or detect_format(target)
        origin = str(target)
    records = _records_from_text(text, resolved_fmt, origin)
    return Corpus.from_records(records)


def save_corpus(corpus: Corpus, path: str | Path, *, fmt: str | None = None) -> Path:
    """Write ``corpus`` to ``path`` in the requested (or detected) format.

    ``path == "-"`` writes to stdout.  Returns the resolved path.
    """
    records = corpus.to_records()
    if str(path) == "-":
        import sys

        resolved_fmt = fmt or "jsonl"
        sys.stdout.write(_records_to_text(records, resolved_fmt))
        return Path("-")
    target = Path(path)
    resolved_fmt = fmt or detect_format(target)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_records_to_text(records, resolved_fmt), encoding="utf-8")
    return target


def cases_to_records(cases: Iterable[Case]) -> list[dict[str, Any]]:
    """Convert :class:`Case` objects to plain record dicts."""
    return [case.to_dict() for case in cases]
