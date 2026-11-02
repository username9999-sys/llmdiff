"""Tests for the badge reporter."""

from __future__ import annotations

import json

import pytest

from llmdiff.reporters.badge import BADGE_COLORS, BadgeReporter


def test_json_style(evaluated_result):
    info = json.loads(BadgeReporter(style="json").render(evaluated_result))
    assert info["schemaVersion"] == 1
    assert info["label"] == "llmdiff"
    assert info["message"]
    assert info["color"] in BADGE_COLORS.values()


def test_markdown_style(handcrafted_result):
    out = BadgeReporter(style="markdown").render(handcrafted_result)
    assert out.startswith("![llmdiff](https://img.shields.io/badge/")
    # handcrafted overall verdict is tie -> yellow
    assert "-yellow" in out


def test_markdown_with_link(handcrafted_result):
    out = BadgeReporter(style="markdown", url="https://example.com/pr/1").render(handcrafted_result)
    assert out.startswith("[![llmdiff](https://img.shields.io/badge/")
    assert "(https://example.com/pr/1)" in out


def test_url_style(handcrafted_result):
    out = BadgeReporter(style="url").render(handcrafted_result)
    assert out.startswith("https://img.shields.io/badge/llmdiff-")
    assert "logo=git" in out


def test_verdict_colors(handcrafted_result):
    # handcrafted: 2W/5T/2L -> overall tie -> yellow
    assert BadgeReporter().badge_info(handcrafted_result)["color"] == "yellow"


def test_invalid_style():
    with pytest.raises(ValueError):
        BadgeReporter(style="yaml")


def test_message_contains_regression_pct(handcrafted_result):
    info = BadgeReporter().badge_info(handcrafted_result)
    assert "regress" in info["message"]
    assert "%" in info["message"]
