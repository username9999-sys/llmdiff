"""Tests for the Slack Block Kit reporter."""

from __future__ import annotations

import json

from llmdiff.reporters.slack import SlackReporter


def test_render_is_valid_json(evaluated_result):
    out = SlackReporter().render(evaluated_result)
    payload = json.loads(out)
    assert "text" in payload
    assert "blocks" in payload
    assert isinstance(payload["blocks"], list) and payload["blocks"]


def test_payload_structure(evaluated_result):
    payload = json.loads(SlackReporter().render(evaluated_result))
    types = [b["type"] for b in payload["blocks"]]
    assert "header" in types
    assert "section" in types
    assert "divider" in types
    header = next(b for b in payload["blocks"] if b["type"] == "header")
    assert evaluated_result.matrix.verdict.upper() in header["text"]["text"]


def test_regressed_cases_section(handcrafted_result):
    payload = json.loads(SlackReporter().render(handcrafted_result))
    text_blobs = " ".join(
        b.get("text", {}).get("text", "") for b in payload["blocks"] if b["type"] == "section"
    )
    assert "regressed-case" in text_blobs
    assert "Regressed cases" in text_blobs


def test_dimension_table_in_blocks(handcrafted_result):
    payload = json.loads(SlackReporter().render(handcrafted_result))
    text_blobs = " ".join(
        b.get("text", {}).get("text", "") for b in payload["blocks"] if b["type"] == "section"
    )
    assert "`exact`" in text_blobs
    assert "`similarity`" in text_blobs


def test_channel_and_attachment(handcrafted_result):
    payload = json.loads(SlackReporter(channel="#ci").render(handcrafted_result))
    assert payload["channel"] == "#ci"
    # handcrafted overall verdict is tie -> yellow
    assert payload["attachments"][0]["color"] == "#ecb22e"


def test_pretty_mode(handcrafted_result):
    out = SlackReporter(pretty=True).render(handcrafted_result)
    assert "\n" in out
    json.loads(out)
