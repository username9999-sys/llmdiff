"""Tests for llmdiff.capture.formats — round-trips, all offline."""

from __future__ import annotations

import json
import sqlite3

import pytest

from llmdiff.capture.errors import FormatExportError
from llmdiff.capture.formats import (
    csv_text_to_records,
    detect_format,
    export_corpus,
    export_csv,
    export_json_array,
    export_sqlite,
    import_corpus,
    import_csv,
    import_json_array,
    import_sqlite,
    records_to_csv_text,
)
from llmdiff.core.corpus import Corpus


def _corpus() -> Corpus:
    return Corpus.from_records(
        [
            {
                "id": "a",
                "prompt": "hello, world",
                "baseline_output": "b1",
                "candidate_output": "c1",
                "metadata": {"model": "m1", "tags": ["x", "y"], "note": "comma, quote \" here"},
            },
            {
                "id": "b",
                "prompt": "line1\nline2",
                "baseline_output": "",
                "candidate_output": "c2",
                "metadata": {"model": "m2"},
            },
        ]
    )


def test_csv_roundtrip(tmp_path):
    corpus = _corpus()
    path = export_csv(corpus, tmp_path / "c.csv")
    loaded = import_csv(path)
    assert loaded == corpus


def test_csv_text_roundtrip_handles_commas_quotes_newlines():
    corpus = _corpus()
    text = records_to_csv_text(corpus.to_records())
    records = csv_text_to_records(text)
    assert Corpus.from_records(records) == corpus


def test_csv_missing_column_rejected():
    with pytest.raises(FormatExportError):
        csv_text_to_records("prompt,candidate_output\nhi,yo\n")


def test_csv_bad_metadata_json_rejected():
    bad = "id,prompt,baseline_output,candidate_output,metadata_json,tags\nx,p,,o,{nope,\n"
    with pytest.raises(FormatExportError):
        csv_text_to_records(bad)


def test_json_array_roundtrip(tmp_path):
    corpus = _corpus()
    path = export_json_array(corpus, tmp_path / "c.json")
    loaded = import_json_array(path)
    assert loaded == corpus
    # file really is a JSON array
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 2


def test_import_json_array_rejects_object(tmp_path):
    p = tmp_path / "o.json"
    p.write_text('{"id": "a"}', encoding="utf-8")
    with pytest.raises(FormatExportError):
        import_json_array(p)


def test_sqlite_roundtrip(tmp_path):
    corpus = _corpus()
    path = export_sqlite(corpus, tmp_path / "c.db")
    loaded = import_sqlite(path)
    assert loaded == corpus
    # verify the table shape directly
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT id, prompt FROM cases ORDER BY id").fetchall()
        assert rows == [("a", "hello, world"), ("b", "line1\nline2")]
    finally:
        conn.close()


def test_import_sqlite_missing_file(tmp_path):
    with pytest.raises(FormatExportError):
        import_sqlite(tmp_path / "nope.db")


def test_export_corpus_dispatch_jsonl(tmp_path):
    corpus = _corpus()
    path = export_corpus(corpus, tmp_path / "c.jsonl")
    assert Corpus.from_jsonl(path) == corpus


def test_import_corpus_dispatch(tmp_path):
    corpus = _corpus()
    csv_path = export_csv(corpus, tmp_path / "c.csv")
    json_path = export_json_array(corpus, tmp_path / "c.json")
    db_path = export_sqlite(corpus, tmp_path / "c.db")
    assert import_corpus(csv_path) == corpus
    assert import_corpus(json_path) == corpus
    assert import_corpus(db_path) == corpus


def test_detect_format(tmp_path):
    jl = tmp_path / "a.jsonl"
    jl.write_text(
        '{"id":"x","prompt":"p","baseline_output":"","candidate_output":"o","metadata":{}}\n',
        encoding="utf-8",
    )
    assert detect_format(jl) == "jsonl"
    js = tmp_path / "a.json"
    js.write_text("[]", encoding="utf-8")
    assert detect_format(js) == "json"
    csv_p = tmp_path / "a.csv"
    csv_p.write_text("id,prompt\n", encoding="utf-8")
    assert detect_format(csv_p) == "csv"
    db = tmp_path / "a.db"
    export_sqlite(_corpus(), db)
    assert detect_format(db) == "sqlite"


def test_export_corpus_unknown_format(tmp_path):
    with pytest.raises(FormatExportError):
        export_corpus(_corpus(), tmp_path / "c.weird", fmt="weird")


def test_import_corpus_parquet_unsupported_message(tmp_path):
    p = tmp_path / "c.parquet"
    p.write_bytes(b"PAR1" + b"\x00" * 100)
    with pytest.raises(FormatExportError):
        import_corpus(p)


def test_parquet_export_guarded(tmp_path):
    from llmdiff.capture.formats import export_parquet

    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401

        path = export_parquet(_corpus(), tmp_path / "c.parquet")
        assert path.exists()
    except ImportError:
        with pytest.raises(FormatExportError):
            export_parquet(_corpus(), tmp_path / "c.parquet")
