"""Shared fixtures for the CLI test suite.

All tests are offline: they call ``llmdiff.cli.main([...])`` directly with
``tmp_path`` fixtures and assert on exit codes and captured stdout/stderr.
"""

from __future__ import annotations

import json

import pytest

from llmdiff.cli.corpus_io import save_corpus
from llmdiff.cli.examples import demo_records, example_records, regression_records
from llmdiff.core import Corpus


@pytest.fixture()
def example_corpus_path(tmp_path):
    """Path to a 5-case example corpus written as JSONL."""
    path = tmp_path / "corpus.jsonl"
    save_corpus(Corpus.from_records(example_records()), path)
    return path


@pytest.fixture()
def demo_corpus_path(tmp_path):
    """Path to the 12-case demo corpus written as JSONL."""
    path = tmp_path / "demo.jsonl"
    save_corpus(Corpus.from_records(demo_records()), path)
    return path


@pytest.fixture()
def regressed_corpus_path(tmp_path):
    """Path to a corpus whose candidate clearly regresses."""
    path = tmp_path / "regressed.jsonl"
    save_corpus(Corpus.from_records(regression_records()), path)
    return path


@pytest.fixture()
def result_json_path(tmp_path, demo_corpus_path):
    """Path to a saved DiffResult JSON produced by ``compare --format json``."""
    from llmdiff.cli import main

    out = tmp_path / "result.json"
    code = main(["compare", str(demo_corpus_path), "--format", "json", "--output", str(out),
                 "--max-loss-rate", "1.0"])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "matrix" in data
    return out
