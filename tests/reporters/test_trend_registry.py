"""Tests for the trend reporter and registry."""

from __future__ import annotations

import json

import pytest

from llmdiff.core.errors import ReporterError
from llmdiff.reporters.json_reporter import JsonReporter
from llmdiff.reporters.registry import (
    REPORTER_NAMES,
    build_registry,
    default_reporters,
    get_reporter,
)
from llmdiff.reporters.trend import TrendReporter, load_previous_result

# ------------------------------------------------------------- trend


def test_trend_render_shows_dimensions(handcrafted_result, evaluated_result):
    out = TrendReporter().render(handcrafted_result, evaluated_result)
    assert out.startswith("# llmdiff trend")
    assert "| Dimension | Prev verdict | Curr verdict |" in out
    assert "Overall:" in out
    for summary in handcrafted_result.matrix.dimensions:
        assert f"`{summary.dimension}`" in out


def test_trend_direction(handcrafted_result, evaluated_result):
    trends = TrendReporter().compare(handcrafted_result, evaluated_result)
    assert trends
    for trend in trends:
        assert trend.direction in ("improved", "regressed", "stable")
        assert trend.delta_win_rate == pytest.approx(
            trend.curr_win_rate - trend.prev_win_rate
        )


def test_trend_render_json(handcrafted_result, evaluated_result):
    payload = json.loads(TrendReporter().render_json(handcrafted_result, evaluated_result))
    assert payload["current"]["verdict"] == handcrafted_result.matrix.verdict
    assert payload["previous"]["verdict"] == evaluated_result.matrix.verdict
    assert isinstance(payload["dimensions"], list) and payload["dimensions"]
    assert "delta_win_rate" in payload["dimensions"][0]


def test_load_previous_result_roundtrip(evaluated_result, tmp_path):
    path = tmp_path / "prev.json"
    path.write_text(JsonReporter().render(evaluated_result), encoding="utf-8")
    previous = load_previous_result(path)
    assert previous.matrix.verdict == evaluated_result.matrix.verdict
    out = TrendReporter().render_with_file(evaluated_result, path)
    assert "llmdiff trend" in out


def test_load_previous_result_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_previous_result(tmp_path / "nope.json")


def test_load_previous_result_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_previous_result(path)


# ----------------------------------------------------------- registry


def test_default_reporters_cover_all_names():
    reporters = default_reporters()
    assert tuple(r.name for r in reporters) == REPORTER_NAMES
    assert len(reporters) == 12


def test_get_reporter_by_name():
    assert get_reporter("markdown").name == "markdown"
    assert get_reporter("sarif").name == "sarif"
    with pytest.raises(ReporterError):
        get_reporter("nope")


def test_build_registry_render(evaluated_result):
    registry = build_registry()
    assert len(registry) == 12
    out = registry.render("json", evaluated_result)
    assert json.loads(out)["schema_version"] == "1.0.0"
    md = registry.render("markdown", evaluated_result)
    assert md.startswith("# llmdiff report")


def test_registry_contains_all(evaluated_result):
    registry = build_registry()
    for name in REPORTER_NAMES:
        assert name in registry
    # every built-in renders non-empty output for the evaluated result
    renderable = ("markdown", "json", "html", "junit", "sarif", "github",
                  "slack", "terminal", "csv", "badge")
    for name in renderable:
        out = registry.render(name, evaluated_result)
        assert isinstance(out, str) and out.strip(), name
