"""Tests for ``diff``, the record/replay/serve wrappers, and parser/main plumbing."""

from __future__ import annotations

from llmdiff.cli import main
from llmdiff.cli.corpus_io import load_corpus, save_corpus
from llmdiff.cli.diff_cmd import merge_corpora
from llmdiff.cli.examples import example_records
from llmdiff.core import Corpus


def _write_corpus(path, records):
    save_corpus(Corpus.from_records(records), path)


def test_diff_two_corpora(tmp_path, capsys):
    base = example_records()
    cand = []
    for rec in base:
        rec = dict(rec)
        rec["candidate_output"] = rec["baseline_output"]  # identical -> ties
        cand.append(rec)
    p_base = tmp_path / "base.jsonl"
    p_cand = tmp_path / "cand.jsonl"
    _write_corpus(p_base, base)
    _write_corpus(p_cand, cand)
    code = main(["diff", str(p_base), str(p_cand), "--max-loss-rate", "1.0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "verdict matrix" in out
    assert "gate: PASS" in out


def test_diff_merged_out(tmp_path, capsys):
    base = example_records()
    p_base = tmp_path / "base.jsonl"
    p_cand = tmp_path / "cand.jsonl"
    _write_corpus(p_base, base)
    _write_corpus(p_cand, base)
    merged = tmp_path / "merged.jsonl"
    code = main([
        "diff", str(p_base), str(p_cand),
        "--merged-out", str(merged), "--max-loss-rate", "1.0",
    ])
    assert code == 0
    corpus = load_corpus(merged)
    assert len(corpus) == 5
    assert "wrote merged corpus" in capsys.readouterr().out


def test_diff_no_overlap_exit_2(tmp_path, capsys):
    base = [{"id": "a", "prompt": "p", "baseline_output": "x", "candidate_output": "y"}]
    cand = [{"id": "b", "prompt": "p", "baseline_output": "x", "candidate_output": "y"}]
    p_base = tmp_path / "b.jsonl"
    p_cand = tmp_path / "c.jsonl"
    _write_corpus(p_base, base)
    _write_corpus(p_cand, cand)
    code = main(["diff", str(p_base), str(p_cand)])
    assert code == 2
    assert "no overlapping" in capsys.readouterr().err


def test_merge_corpora_unit():
    base = Corpus.from_records([
        {"id": "1", "prompt": "p1", "baseline_output": "a", "candidate_output": "a",
         "metadata": {"model": "m-base"}},
        {"id": "2", "prompt": "p2", "baseline_output": "b", "candidate_output": "b"},
    ])
    cand = Corpus.from_records([
        {"id": "1", "prompt": "p1", "baseline_output": "zzz", "candidate_output": "c",
         "metadata": {"model": "m-cand"}},
    ])
    merged = merge_corpora(base, cand)
    assert len(merged) == 1
    case = merged[0]
    assert case.baseline_output == "a"
    assert case.candidate_output == "c"
    assert case.metadata["baseline_model"] == "m-base"
    assert case.metadata["candidate_model"] == "m-cand"


def test_record_graceful_when_capture_missing(tmp_path, capsys, monkeypatch):
    # Force the "capture not importable" branch so no server is started.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("llmdiff.capture"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["record", "--proxy", "--port", "0", "--target", "http://x",
                 "--out", str(tmp_path / "r.jsonl")])
    assert code == 2
    err = capsys.readouterr().err
    assert "not importable" in err
    assert "python -m llmdiff.capture.proxy" in err


def test_replay_mock_provider(tmp_path, capsys, example_corpus_path):
    out = tmp_path / "replayed.jsonl"
    code = main([
        "replay", str(example_corpus_path),
        "--provider", "mock", "--out", str(out),
    ])
    assert code == 0
    corpus = load_corpus(out)
    assert len(corpus) == 5
    assert "replayed 5 cases" in capsys.readouterr().out


def test_replay_http_requires_base_url(example_corpus_path, capsys):
    code = main(["replay", str(example_corpus_path), "--provider", "http", "--out", "x.jsonl"])
    assert code == 2
    assert "--base-url is required" in capsys.readouterr().err


def test_serve_wrapper(capsys):
    code = main(["serve", "--port", "0"])
    # server package does not exist in this repo -> graceful exit 2
    assert code == 2
    assert "not importable" in capsys.readouterr().err


def test_main_no_args_prints_help(capsys):
    code = main([])
    out = capsys.readouterr().out
    assert code == 2
    assert "usage:" in out


def test_main_unknown_command(capsys):
    code = main(["frobnicate"])
    assert code == 2  # argparse error


def test_main_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "llmdiff" in out


def test_entry_point_importable():
    from llmdiff.cli import main as m

    assert callable(m)
