"""Tests for the reporter base: protocol, registry, helpers."""

from __future__ import annotations

import pytest

from llmdiff.core.errors import ReporterError
from llmdiff.reporters.base import (
    BaseReporter,
    Reporter,
    ReporterRegistry,
    escape_md,
    escape_md_cell,
    format_delta,
    format_pct,
    format_score,
    safe_filename,
    verdict_emoji,
    verdict_label,
)


class _Dummy(BaseReporter):
    name = "dummy"

    def render(self, result) -> str:
        return f"dummy:{result.matrix.verdict}"


class TestSafeFilename:
    def test_replaces_unsafe_chars(self):
        assert safe_filename("case/one: two") == "case_one_two"

    def test_appends_extension(self):
        assert safe_filename("report", extension="md") == "report.md"
        assert safe_filename("report", extension=".md") == "report.md"

    def test_does_not_double_extension(self):
        assert safe_filename("report.md", extension="md") == "report.md"

    def test_empty_falls_back(self):
        assert safe_filename("///") == "report"
        assert safe_filename("", default="out", extension="json") == "out.json"

    def test_strips_leading_dots(self):
        assert safe_filename("..hidden") == "hidden"


class TestEscaping:
    def test_escape_md_escapes_specials(self):
        assert escape_md("a*b") == "a\\*b"
        assert escape_md("[link](http://x)") == "\\[link\\]\\(http://x\\)"

    def test_escape_md_cell_pipes_and_newlines(self):
        assert escape_md_cell("a|b\nc") == "a\\|b c"


class TestFormatters:
    def test_format_delta_sign(self):
        assert format_delta(0.25) == "+0.250"
        assert format_delta(-0.125) == "-0.125"
        assert format_delta(0.0, sign=False) == "0.000"

    def test_format_pct(self):
        assert format_pct(0.42) == "42.0%"
        assert format_pct(1.0, digits=0) == "100%"

    def test_format_score(self):
        assert format_score(0.123456) == "0.123"

    def test_verdict_helpers(self):
        assert verdict_emoji("win") == "\u2705"
        assert verdict_emoji("loss") == "\u274c"
        assert verdict_emoji("unknown") == "\u2753"
        assert verdict_label("tie") == "TIE"
        assert verdict_label("weird") == "WEIRD"


class TestReporterProtocol:
    def test_base_reporter_satisfies_protocol(self, evaluated_result):
        reporter = _Dummy()
        assert isinstance(reporter, Reporter)
        assert reporter.name == "dummy"
        assert reporter.render(evaluated_result) == f"dummy:{evaluated_result.matrix.verdict}"

    def test_render_to_file(self, evaluated_result, tmp_path):
        reporter = _Dummy()
        out = reporter.render_to_file(evaluated_result, tmp_path / "sub" / "r.txt")
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("dummy:")

    def test_render_to_file_wraps_errors(self, tmp_path):
        class _Boom(BaseReporter):
            name = "boom"

            def render(self, result) -> str:
                raise RuntimeError("kaput")

        with pytest.raises(ReporterError, match="kaput"):
            _Boom().render_to_file(object(), tmp_path / "x.txt")

    def test_base_render_not_implemented(self):
        with pytest.raises(ReporterError):
            BaseReporter().render(object())


class TestReporterRegistry:
    def test_register_get_names(self):
        reg = ReporterRegistry()
        reg.register(_Dummy())
        assert "dummy" in reg
        assert reg.names() == ("dummy",)
        assert len(reg) == 1
        assert reg.get("dummy").name == "dummy"

    def test_get_unknown_raises(self):
        reg = ReporterRegistry()
        with pytest.raises(ReporterError, match="unknown reporter"):
            reg.get("nope")

    def test_register_requires_name(self):
        class _NoName(BaseReporter):
            @property
            def name(self):  # type: ignore[override]
                return ""

        with pytest.raises(ReporterError):
            ReporterRegistry([_NoName()])

    def test_unregister(self):
        reg = ReporterRegistry([_Dummy()])
        reg.unregister("dummy")
        assert "dummy" not in reg
        reg.unregister("missing")  # no error

    def test_render_and_render_to_file(self, evaluated_result, tmp_path):
        reg = ReporterRegistry([_Dummy()])
        assert reg.render("dummy", evaluated_result).startswith("dummy:")
        out = reg.render_to_file("dummy", evaluated_result, tmp_path / "r.txt")
        assert out.exists()

    def test_render_to_file_rejects_non_base(self, evaluated_result, tmp_path):
        class _Plain:
            name = "plain"

            def render(self, result) -> str:
                return "x"

        reg = ReporterRegistry([_Plain()])
        with pytest.raises(ReporterError):
            reg.render_to_file("plain", evaluated_result, tmp_path / "r.txt")
