"""Tests for ``llmdiff init`` and ``llmdiff gate``."""

from __future__ import annotations

import json

from llmdiff.cli import main
from llmdiff.cli.corpus_io import load_corpus


def test_init_writes_parseable_jsonl(tmp_path, capsys):
    out = tmp_path / "corpus.jsonl"
    code = main(["init", "--out", str(out)])
    assert code == 0
    corpus = load_corpus(out)
    assert len(corpus) == 5
    assert "wrote 5 example cases" in capsys.readouterr().out


def test_init_refuses_overwrite_without_force(tmp_path, capsys):
    out = tmp_path / "corpus.jsonl"
    assert main(["init", "--out", str(out)]) == 0
    code = main(["init", "--out", str(out)])
    assert code == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["init", "--out", str(out), "--force"]) == 0


def test_init_stdout(tmp_path, capsys):
    code = main(["init", "--out", "-"])
    out = capsys.readouterr().out
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        assert {"id", "prompt", "baseline_output", "candidate_output"} <= set(record)


def test_gate_pass(result_json_path, capsys):
    code = main(["gate", str(result_json_path), "--max-loss-rate", "1.0",
                 "--max-regressed-dims", "99", "--min-win-rate", "0.0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "gate: PASS" in out


def test_gate_fail(result_json_path, capsys):
    code = main(["gate", str(result_json_path), "--max-loss-rate", "0.0",
                 "--max-regressed-dims", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert "GATE FAIL" in out
    assert "gate: FAIL" in out


def test_gate_min_win_rate(result_json_path, capsys):
    code = main(["gate", str(result_json_path), "--max-loss-rate", "1.0",
                 "--max-regressed-dims", "99", "--min-win-rate", "1.0"])
    assert code == 1
    assert "win rate" in capsys.readouterr().out


def test_gate_missing_file_exit_2(tmp_path, capsys):
    code = main(["gate", str(tmp_path / "nope.json")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_gate_bad_json_exit_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code = main(["gate", str(bad)])
    assert code == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_gate_not_a_result_exit_2(tmp_path, capsys):
    bad = tmp_path / "arr.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    code = main(["gate", str(bad)])
    assert code == 2
    assert "not a llmdiff result" in capsys.readouterr().err


def test_gate_quiet_pass(result_json_path, capsys):
    code = main(["gate", str(result_json_path), "--max-loss-rate", "1.0",
                 "--max-regressed-dims", "99", "-q"])
    out = capsys.readouterr().out
    assert code == 0
    assert out == ""
