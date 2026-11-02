"""Tests for ``llmdiff compare``."""

from __future__ import annotations

import json

from llmdiff.cli import main


def test_compare_fixture_exit_0(example_corpus_path, capsys):
    code = main(["compare", str(example_corpus_path), "--max-loss-rate", "1.0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "verdict matrix" in out
    assert "gate: PASS" in out


def test_compare_regressed_corpus_gate_fails(regressed_corpus_path, capsys):
    code = main(["compare", str(regressed_corpus_path), "--max-loss-rate", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert "GATE FAIL" in out


def test_compare_json_output_file(example_corpus_path, tmp_path, capsys):
    out_file = tmp_path / "result.json"
    code = main([
        "compare", str(example_corpus_path),
        "--format", "json", "--output", str(out_file),
        "--max-loss-rate", "1.0",
    ])
    assert code == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "matrix" in data and "cases" in data
    assert data["matrix"]["total_cases"] == 5


def test_compare_dimensions_subset(example_corpus_path, capsys):
    code = main([
        "compare", str(example_corpus_path),
        "--dimensions", "exact", "--max-loss-rate", "1.0",
    ])
    out = capsys.readouterr().out
    assert code == 0
    header = out.split("== per-case summary ==")[0]
    assert "exact" in header
    assert "similarity" not in header


def test_compare_tags_filter(example_corpus_path, capsys):
    code = main([
        "compare", str(example_corpus_path),
        "--tags", "json", "--max-loss-rate", "1.0",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "json-extract" in out
    assert "summarize-article" not in out.split("== per-case summary ==")[1]


def test_compare_tags_no_match_exit_2(example_corpus_path, capsys):
    code = main(["compare", str(example_corpus_path), "--tags", "does-not-exist"])
    assert code == 2
    assert "no cases match" in capsys.readouterr().err


def test_compare_missing_file_exit_2(tmp_path, capsys):
    code = main(["compare", str(tmp_path / "nope.jsonl")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_compare_unknown_dimension_exit_2(example_corpus_path, capsys):
    code = main(["compare", str(example_corpus_path), "--dimensions", "bogus-dim"])
    assert code == 2


def test_compare_markdown_format(example_corpus_path, capsys):
    code = main([
        "compare", str(example_corpus_path),
        "--format", "md", "--max-loss-rate", "1.0",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "# llmdiff report" in out
    assert "| dimension |" in out


def test_compare_csv_and_junit_and_sarif_and_html(example_corpus_path, capsys):
    for fmt, marker in (
        ("csv", "case_id,dimension"),
        ("junit", "<testsuites"),
        ("sarif", '"version": "2.1.0"'),
        ("html", "<!doctype html>"),
    ):
        code = main([
            "compare", str(example_corpus_path),
            "--format", fmt, "--max-loss-rate", "1.0",
        ])
        out = capsys.readouterr().out
        assert code == 0, fmt
        assert marker in out, fmt
