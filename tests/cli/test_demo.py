"""Tests for ``llmdiff demo`` - must run fully offline."""

from __future__ import annotations

import json

from llmdiff.cli import main


def test_demo_runs_offline_exit_0(capsys):
    code = main(["demo"])
    out = capsys.readouterr().out
    assert code == 0
    assert "verdict matrix" in out
    assert "no network was used" in out
    assert "overall verdict" in out


def test_demo_json_format_is_parseable(capsys):
    code = main(["demo", "--format", "json"])
    out = capsys.readouterr().out
    assert code == 0
    # The JSON block is embedded after the header lines; decode from the first '{'
    start = out.index("{")
    data, _ = json.JSONDecoder().raw_decode(out[start:])
    assert "matrix" in data
    assert data["matrix"]["total_cases"] == 12


def test_demo_dimensions_subset(capsys):
    code = main(["demo", "--dimensions", "exact,similarity"])
    out = capsys.readouterr().out
    assert code == 0
    assert "exact" in out and "similarity" in out
    assert "json_valid" not in out.split("== per-case summary ==")[0]


def test_demo_unknown_format_exit_2(capsys):
    code = main(["demo", "--format", "nope"])
    assert code == 2
    assert "unknown format" in capsys.readouterr().err
