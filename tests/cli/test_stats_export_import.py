"""Tests for ``stats``, ``export`` and ``import``."""

from __future__ import annotations

import json

from llmdiff.cli import main
from llmdiff.cli.corpus_io import load_corpus


def test_stats_prints_total(example_corpus_path, capsys):
    code = main(["stats", str(example_corpus_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "5 cases" in out
    assert "tags:" in out
    assert "models:" in out
    assert "histogram" in out


def test_stats_missing_file_exit_2(tmp_path, capsys):
    code = main(["stats", str(tmp_path / "nope.jsonl")])
    assert code == 2


def test_export_jsonl_to_csv_roundtrip(example_corpus_path, tmp_path, capsys):
    csv_path = tmp_path / "out.csv"
    code = main(["export", str(example_corpus_path), "--to", "csv", "--out", str(csv_path)])
    assert code == 0
    assert csv_path.exists()
    back = tmp_path / "back.jsonl"
    code = main(["import", str(csv_path), "--out", str(back)])
    assert code == 0
    original = load_corpus(example_corpus_path)
    imported = load_corpus(back)
    assert [c.id for c in original] == [c.id for c in imported]
    assert [c.prompt for c in original] == [c.prompt for c in imported]
    assert [c.baseline_output for c in original] == [c.baseline_output for c in imported]
    assert [c.candidate_output for c in original] == [c.candidate_output for c in imported]


def test_export_jsonl_to_json_array(example_corpus_path, tmp_path):
    out = tmp_path / "arr.json"
    code = main(["export", str(example_corpus_path), "--to", "json", "--out", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 5


def test_export_to_stdout(example_corpus_path, capsys):
    code = main(["export", str(example_corpus_path), "--to", "csv", "--out", "-"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("id,prompt,baseline_output,candidate_output,metadata")


def test_export_bad_input_exit_2(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    code = main(["export", str(bad), "--to", "csv", "--out", "-"])
    assert code == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_import_refuses_overwrite(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({
        "id": "x", "prompt": "p",
        "baseline_output": "a", "candidate_output": "b",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "corpus.jsonl"
    assert main(["import", str(src), "--out", str(out)]) == 0
    code = main(["import", str(src), "--out", str(out)])
    assert code == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["import", str(src), "--out", str(out), "--force"]) == 0


def test_import_csv_with_metadata(tmp_path, capsys):
    import csv as csv_mod
    import io

    buffer = io.StringIO()
    writer = csv_mod.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "prompt", "baseline_output", "candidate_output", "metadata"])
    writer.writerow(["c1", "hi", "a", "b", json.dumps({"model": "m1", "tags": ["t1"]})])
    src = tmp_path / "in.csv"
    src.write_text(buffer.getvalue(), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    code = main(["import", str(src), "--out", str(out)])
    assert code == 0
    corpus = load_corpus(out)
    assert corpus[0].metadata["model"] == "m1"
    assert corpus[0].metadata["tags"] == ["t1"]


def test_import_csv_bad_metadata_exit_2(tmp_path, capsys):
    csv_text = "id,prompt,baseline_output,candidate_output,metadata\nc1,hi,a,b,not-json\n"
    src = tmp_path / "in.csv"
    src.write_text(csv_text, encoding="utf-8")
    code = main(["import", str(src), "--out", "-"])
    assert code == 2
    assert "metadata is not valid JSON" in capsys.readouterr().err
