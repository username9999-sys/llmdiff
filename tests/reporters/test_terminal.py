"""Tests for the terminal reporter (ANSI, NO_COLOR, tables)."""

from __future__ import annotations

import io
import re

from llmdiff.reporters.terminal import TerminalReporter, _strip_ansi

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def test_render_plain_when_color_disabled(evaluated_result):
    stream = io.StringIO()
    stream.isatty = lambda: True
    reporter = TerminalReporter(color=False, stream=stream)
    out = reporter.render(evaluated_result)
    assert ANSI_RE.search(out) is None
    assert "VERDICT:" in out
    assert "Dimension" in out
    assert "Regressed cases:" in out


def test_render_color_when_enabled(handcrafted_result):
    out = TerminalReporter(color=True).render(handcrafted_result)
    assert ANSI_RE.search(out) is not None
    assert "\033[31m" in out  # red for loss


def test_no_color_env_strips_colors(evaluated_result, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    stream.isatty = lambda: True
    reporter = TerminalReporter(stream=stream)
    out = reporter.render(evaluated_result)
    assert ANSI_RE.search(out) is None


def test_non_tty_strips_colors(evaluated_result, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    stream = io.StringIO()  # StringIO.isatty() -> False
    reporter = TerminalReporter(stream=stream)
    out = reporter.render(evaluated_result)
    assert ANSI_RE.search(out) is None


def test_force_color_env(evaluated_result, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = io.StringIO()
    reporter = TerminalReporter(stream=stream)
    out = reporter.render(evaluated_result)
    assert ANSI_RE.search(out) is not None


def test_matrix_table_contains_dimensions(evaluated_result):
    out = TerminalReporter(color=False).render(evaluated_result)
    for summary in evaluated_result.matrix.dimensions:
        assert summary.dimension in out


def test_regressed_case_details(handcrafted_result):
    out = TerminalReporter(color=False).render(handcrafted_result)
    assert "regressed-case" in out
    assert "exact" in out
    assert "->" in out


def test_strip_ansi():
    assert _strip_ansi("\033[31mred\033[0m") == "red"
    assert _strip_ansi("plain") == "plain"


def test_show_cases_false_omits_section(evaluated_result):
    out = TerminalReporter(color=False, show_cases=False).render(evaluated_result)
    assert "Regressed cases:" not in out
