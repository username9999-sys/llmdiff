"""Tests for the JSON reporter."""

from __future__ import annotations

import json

import pytest

from llmdiff.reporters.json_reporter import SCHEMA_VERSION, JsonReporter


def test_render_pretty_is_valid_json(evaluated_result):
    out = JsonReporter(pretty=True).render(evaluated_result)
    payload = json.loads(out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["matrix"]["verdict"] == evaluated_result.matrix.verdict
    assert len(payload["cases"]) == len(evaluated_result.cases)
    assert "generated_at" in payload
    assert "\n" in out


def test_render_compact_is_single_line(evaluated_result):
    out = JsonReporter(pretty=False).render(evaluated_result)
    assert "\n" not in out.strip()
    payload = json.loads(out)
    assert payload["schema_version"] == SCHEMA_VERSION


def test_payload_contains_config(evaluated_result):
    payload = JsonReporter().to_payload(evaluated_result)
    assert "config" in payload
    assert payload["config"] is not None
    assert payload["matrix"]["total_cases"] == evaluated_result.matrix.total_cases


def test_roundtrip_from_json(evaluated_result):
    out = JsonReporter().render(evaluated_result)
    restored = JsonReporter.from_json(out)
    assert restored.matrix.verdict == evaluated_result.matrix.verdict
    assert restored.matrix.wins == evaluated_result.matrix.wins
    assert {c.case_id for c in restored.cases} == {c.case_id for c in evaluated_result.cases}
    dims = {d.dimension for d in restored.matrix.dimensions}
    assert dims == {d.dimension for d in evaluated_result.matrix.dimensions}


def test_from_json_rejects_garbage():
    with pytest.raises(ValueError):
        JsonReporter.from_json("not json at all")
    with pytest.raises(ValueError):
        JsonReporter.from_json('{"foo": 1}')


def test_handcrafted_losses_present(handcrafted_result):
    payload = json.loads(JsonReporter().render(handcrafted_result))
    losses = [r for c in payload["cases"] for r in c["results"] if r["verdict"] == "loss"]
    assert len(losses) == 2
    assert payload["matrix"]["losses"] == 2
