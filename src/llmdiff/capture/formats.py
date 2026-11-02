"""Corpus import/export beyond JSONL.

Supported formats:

* **CSV** — import/export.  Columns: ``id, prompt, baseline_output,
  candidate_output, metadata_json, tags`` (metadata serialised as JSON in one
  column; tags as ``;``-joined).  Uses stdlib :mod:`csv` with proper quoting.
* **JSON array** — import/export a single JSON array of case objects.
* **Parquet** — export only, and only when ``pandas`` (and a parquet engine)
  are importable; guarded with try/except ImportError.
* **SQLite** — export (and import) using stdlib :mod:`sqlite3`; one table
  ``cases`` with the same columns as CSV plus a ``metadata_json`` column.

All functions validate strictly and raise
:class:`~llmdiff.capture.errors.FormatExportError` with a helpful message on
malformed input.  Everything round-trips through
:class:`llmdiff.core.corpus.Corpus`.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llmdiff.capture.errors import FormatExportError
from llmdiff.core.case import Case
from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import CorpusError, FormatError

__all__ = [
    "CSV_COLUMNS",
    "records_to_csv_text",
    "csv_text_to_records",
    "export_csv",
    "import_csv",
    "export_json_array",
    "import_json_array",
    "export_parquet",
    "export_sqlite",
    "import_sqlite",
    "export_corpus",
    "import_corpus",
    "detect_format",
]

CSV_COLUMNS = ("id", "prompt", "baseline_output", "candidate_output", "metadata_json", "tags")


# --------------------------------------------------------------- helpers
def _case_to_row(case: Case) -> dict[str, str]:
    meta = dict(case.metadata)
    tags = meta.pop("tags", None)
    if tags is None:
        tags = case.tags
    tag_str = tags if isinstance(tags, str) else ";".join(str(t) for t in tags)
    return {
        "id": case.id,
        "prompt": case.prompt,
        "baseline_output": case.baseline_output,
        "candidate_output": case.candidate_output,
        "metadata_json": json.dumps(meta, ensure_ascii=False, sort_keys=True),
        "tags": tag_str,
    }


def _row_to_case(row: Mapping[str, Any], *, origin: str, line: int) -> Case:
    try:
        case_id = str(row["id"])
        prompt = str(row.get("prompt", ""))
        baseline = str(row.get("baseline_output", ""))
        candidate = str(row.get("candidate_output", ""))
    except KeyError as exc:
        raise FormatExportError(f"{origin}:{line}: missing column {exc}") from exc
    meta_raw = row.get("metadata_json") or "{}"
    try:
        metadata = json.loads(meta_raw)
    except json.JSONDecodeError as exc:
        raise FormatExportError(
            f"{origin}:{line}: metadata_json is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(metadata, dict):
        raise FormatExportError(f"{origin}:{line}: metadata_json must be a JSON object")
    tags_raw = row.get("tags") or ""
    tags = [t for t in (t.strip() for t in str(tags_raw).split(";")) if t]
    if tags:
        metadata["tags"] = tags
    try:
        return Case(
            id=case_id,
            prompt=prompt,
            baseline_output=baseline,
            candidate_output=candidate,
            metadata=metadata,
        )
    except CorpusError as exc:
        raise FormatExportError(f"{origin}:{line}: {exc}") from exc


def _corpus_from_records(records: Iterable[Mapping[str, Any]], *, origin: str) -> Corpus:
    cases: list[Case] = []
    for i, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise FormatExportError(f"{origin}: record #{i} is not an object")
        try:
            cases.append(Case.from_dict(record))
        except CorpusError as exc:
            raise FormatExportError(f"{origin}: record #{i}: {exc}") from exc
    try:
        return Corpus(cases)
    except CorpusError as exc:
        raise FormatExportError(f"{origin}: {exc}") from exc


# --------------------------------------------------------------- CSV
def records_to_csv_text(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise case records (``Case.to_dict()`` shape) to CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        case = Case.from_dict(record) if not isinstance(record, Case) else record
        writer.writerow(_case_to_row(case))
    return buffer.getvalue()


def csv_text_to_records(text: str, *, origin: str = "<csv>") -> list[dict[str, Any]]:
    """Parse CSV text into a list of case dicts (validated)."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise FormatExportError(f"{origin}: empty CSV")
    missing = {"id", "prompt"} - set(reader.fieldnames)
    if missing:
        raise FormatExportError(f"{origin}: missing required columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for line, row in enumerate(reader, start=2):  # header is line 1
        case = _row_to_case(row, origin=origin, line=line)
        records.append(case.to_dict())
    return records


def export_csv(corpus: Corpus, path: str | Path) -> Path:
    """Write a corpus to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(records_to_csv_text(corpus.to_records()), encoding="utf-8")
    return target


def import_csv(path: str | Path) -> Corpus:
    """Load a corpus from a CSV file written by :func:`export_csv`."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise FormatExportError(f"cannot read CSV {source}: {exc}") from exc
    return _corpus_from_records(csv_text_to_records(text, origin=str(source)), origin=str(source))


# --------------------------------------------------------------- JSON array
def export_json_array(corpus: Corpus, path: str | Path, *, indent: int | None = 2) -> Path:
    """Write a corpus as a single JSON array."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(corpus.to_records(), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return target


def import_json_array(path: str | Path) -> Corpus:
    """Load a corpus from a JSON array file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FormatExportError(f"cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FormatExportError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise FormatExportError(f"{source}: expected a JSON array of case objects")
    return _corpus_from_records(payload, origin=str(source))


# --------------------------------------------------------------- Parquet
def export_parquet(corpus: Corpus, path: str | Path) -> Path:
    """Export a corpus to Parquet.  Requires pandas + pyarrow/fastparquet.

    Raises:
        FormatExportError: If pandas or a parquet engine is not installed.
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FormatExportError(
            "Parquet export requires pandas (and pyarrow or fastparquet); "
            "install them or use export_csv/export_json_array instead"
        ) from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [_case_to_row(case) for case in corpus]
    frame = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    try:
        frame.to_parquet(target, index=False)
    except ImportError as exc:
        raise FormatExportError(
            f"Parquet export needs a parquet engine (pyarrow or fastparquet): {exc}"
        ) from exc
    except Exception as exc:  # engine-specific errors
        raise FormatExportError(f"parquet write to {target} failed: {exc}") from exc
    return target


# --------------------------------------------------------------- SQLite
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    baseline_output TEXT NOT NULL DEFAULT '',
    candidate_output TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT ''
)
"""


def export_sqlite(corpus: Corpus, path: str | Path) -> Path:
    """Write a corpus to a SQLite database (table ``cases``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    try:
        conn.execute("DROP TABLE IF EXISTS cases")
        conn.execute(_SQLITE_SCHEMA)
        conn.executemany(
            "INSERT INTO cases (id, prompt, baseline_output, candidate_output, "
            "metadata_json, tags) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    case.id,
                    case.prompt,
                    case.baseline_output,
                    case.candidate_output,
                    json.dumps(
                        {k: v for k, v in case.metadata.items() if k != "tags"},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    ";".join(case.tags),
                )
                for case in corpus
            ],
        )
        conn.commit()
    except sqlite3.Error as exc:
        raise FormatExportError(f"sqlite write to {target} failed: {exc}") from exc
    finally:
        conn.close()
    return target


def import_sqlite(path: str | Path) -> Corpus:
    """Load a corpus from a SQLite database written by :func:`export_sqlite`."""
    source = Path(path)
    if not source.exists():
        raise FormatExportError(f"sqlite file not found: {source}")
    conn = sqlite3.connect(str(source))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, prompt, baseline_output, candidate_output, metadata_json, tags FROM cases"
        ).fetchall()
    except sqlite3.Error as exc:
        raise FormatExportError(f"cannot read cases from {source}: {exc}") from exc
    finally:
        conn.close()
    cases: list[Case] = []
    for line, row in enumerate(rows, start=1):
        cases.append(
            _row_to_case(dict(row), origin=str(source), line=line)
        )
    try:
        return Corpus(cases)
    except CorpusError as exc:
        raise FormatExportError(f"{source}: {exc}") from exc


# --------------------------------------------------------------- dispatch
def detect_format(path: str | Path) -> str:
    """Guess a corpus file's format from its extension/content.

    Returns one of ``"jsonl"``, ``"json"``, ``"csv"``, ``"sqlite"``,
    ``"parquet"``.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    if suffix == ".parquet":
        return "parquet"
    # sniff content
    try:
        head = source.read_bytes()[:512]
    except OSError as exc:
        raise FormatExportError(f"cannot read {source}: {exc}") from exc
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:15] == b"SQLite format 3":
        return "sqlite"
    stripped = head.lstrip()
    if stripped.startswith(b"[") or stripped.startswith(b"{"):
        # a JSON object could be JSONL too; check for newline-separated objects
        text = head.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1 and all(ln.strip().startswith("{") for ln in lines):
            return "jsonl"
        return "json"
    return "csv"


def export_corpus(corpus: Corpus, path: str | Path, *, fmt: str | None = None) -> Path:
    """Export a corpus to ``path``; format inferred from extension or ``fmt``."""
    target = Path(path)
    kind = (fmt or detect_format(target)).lower()
    if kind == "jsonl":
        return corpus.to_jsonl(target)
    if kind == "csv":
        return export_csv(corpus, target)
    if kind == "json":
        return export_json_array(corpus, target)
    if kind == "sqlite":
        return export_sqlite(corpus, target)
    if kind == "parquet":
        return export_parquet(corpus, target)
    raise FormatExportError(f"unknown export format {kind!r}")


def import_corpus(path: str | Path, *, fmt: str | None = None) -> Corpus:
    """Import a corpus from ``path``; format inferred from extension or ``fmt``."""
    source = Path(path)
    kind = (fmt or detect_format(source)).lower()
    if kind == "jsonl":
        try:
            return Corpus.from_jsonl(source)
        except (FormatError, CorpusError) as exc:
            raise FormatExportError(f"cannot import {source}: {exc}") from exc
    if kind == "csv":
        return import_csv(source)
    if kind == "json":
        return import_json_array(source)
    if kind == "sqlite":
        return import_sqlite(source)
    raise FormatExportError(
        f"import for format {kind!r} is not supported "
        "(parquet import needs pandas; use export only)"
    )
