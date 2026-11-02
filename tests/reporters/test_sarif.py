"""Tests for the SARIF reporter."""

from __future__ import annotations

import json

from llmdiff.reporters.sarif import SARIF_VERSION, SarifReporter


def test_top_level_sarif_keys(evaluated_result):
    payload = json.loads(SarifReporter().render(evaluated_result))
    assert payload["version"] == SARIF_VERSION == "2.1.0"
    assert payload["$schema"].endswith("sarif-2.1.0.json")
    assert isinstance(payload["runs"], list) and len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "llmdiff"
    assert isinstance(run["results"], list)
    assert isinstance(run["tool"]["driver"]["rules"], list)


def test_results_only_for_losses(handcrafted_result):
    payload = json.loads(SarifReporter().render(handcrafted_result))
    results = payload["runs"][0]["results"]
    assert len(results) == 2
    for result in results:
        assert result["ruleId"].startswith("llmdiff/")
        assert result["level"] in ("error", "warning")
        assert result["message"]["text"]
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert uri.startswith("case://")


def test_severity_mapping(handcrafted_result):
    payload = json.loads(SarifReporter().render(handcrafted_result))
    results = payload["runs"][0]["results"]
    by_rule = {r["ruleId"]: r["level"] for r in results}
    assert by_rule["llmdiff/exact"] == "error"
    assert by_rule["llmdiff/similarity"] == "warning"


def test_rules_cover_all_dimensions(evaluated_result):
    payload = json.loads(SarifReporter().render(evaluated_result))
    rule_ids = {r["id"] for r in payload["runs"][0]["tool"]["driver"]["rules"]}
    for summary in evaluated_result.matrix.dimensions:
        assert f"llmdiff/{summary.dimension}" in rule_ids


def test_compact_mode_single_line(evaluated_result):
    out = SarifReporter(pretty=False).render(evaluated_result)
    assert "\n" not in out.strip()
    json.loads(out)


def test_properties_contain_matrix_summary(handcrafted_result):
    payload = json.loads(SarifReporter().render(handcrafted_result))
    props = payload["runs"][0]["properties"]
    assert props["losses"] == 2
    assert props["verdict"] == "tie"  # 2W/5T/2L overall
    assert "regression_rate" in props
