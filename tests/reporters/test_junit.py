"""Tests for the JUnit XML reporter."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from llmdiff.reporters.junit import JunitReporter


def _parse(out: str) -> ET.Element:
    return ET.fromstring(out)


def test_render_is_valid_xml_with_declaration(evaluated_result):
    out = JunitReporter().render(evaluated_result)
    assert out.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    root = _parse(out)
    assert root.tag == "testsuites"
    assert root.get("name") == "llmdiff"
    total = (
        evaluated_result.matrix.wins
        + evaluated_result.matrix.ties
        + evaluated_result.matrix.losses
    )
    assert int(root.get("tests", "0")) == total
    assert int(root.get("failures", "0")) == evaluated_result.matrix.losses


def test_one_testsuite_per_dimension(evaluated_result):
    root = _parse(JunitReporter().render(evaluated_result))
    suites = root.findall("testsuite")
    assert len(suites) == len(evaluated_result.matrix.dimensions)
    names = {s.get("name") for s in suites}
    for summary in evaluated_result.matrix.dimensions:
        assert f"llmdiff.{summary.dimension}" in names


def test_losses_become_failures(handcrafted_result):
    root = _parse(JunitReporter().render(handcrafted_result))
    failures = root.findall(".//failure")
    assert len(failures) == 2
    for failure in failures:
        assert failure.get("type") == "regression"
        assert failure.text
    testcases = root.findall(".//testcase")
    assert len(testcases) == 9  # 3 cases x 3 dimensions


def test_testcase_names_are_case_ids(handcrafted_result):
    root = _parse(JunitReporter().render(handcrafted_result))
    names = {tc.get("name") for tc in root.findall(".//testcase")}
    assert {"regressed-case", "improved-case", "tied-case"} <= names


def test_suite_properties_carry_scores(evaluated_result):
    root = _parse(JunitReporter().render(evaluated_result))
    suite = root.find("testsuite")
    props = {p.get("name"): p.get("value") for p in suite.find("properties").findall("property")}
    assert "baseline_mean" in props
    assert "delta" in props
    assert "verdict" in props
