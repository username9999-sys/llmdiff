"""JUnit XML report renderer for CI systems.

Emits one ``<testsuite>`` per dimension and one ``<testcase>`` per case. A
``<failure>`` element is attached to every loss so CI surfaces regressions;
ties and wins are plain passes. The XML is built with
:mod:`xml.etree.ElementTree` and serialised with the XML declaration.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter

__all__ = ["JunitReporter"]


class JunitReporter(BaseReporter):
    """Render a :class:`DiffResult` as JUnit XML."""

    name = "junit"

    def __init__(self, *, suite_name: str = "llmdiff") -> None:
        self.suite_name = suite_name

    def render(self, result: DiffResult) -> str:
        """Render ``result`` to a JUnit XML string."""
        root = self.to_element(result)
        ET.indent(root, space="  ")
        body = ET.tostring(root, encoding="unicode")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"

    def to_element(self, result: DiffResult) -> ET.Element:
        """Build the ``<testsuites>`` element tree for ``result``."""
        matrix = result.matrix
        total_tests = matrix.wins + matrix.ties + matrix.losses
        root = ET.Element(
            "testsuites",
            {
                "name": self.suite_name,
                "tests": str(total_tests),
                "failures": str(matrix.losses),
                "skipped": "0",
                "time": "0",
            },
        )
        stamp = datetime.now(timezone.utc).isoformat()
        for summary in matrix.dimensions:
            suite = ET.SubElement(
                root,
                "testsuite",
                {
                    "name": f"llmdiff.{summary.dimension}",
                    "tests": str(summary.total),
                    "failures": str(summary.losses),
                    "skipped": "0",
                    "time": "0",
                    "timestamp": stamp,
                },
            )
            props = ET.SubElement(suite, "properties")
            for key, value in (
                ("dimension", summary.dimension),
                ("wins", str(summary.wins)),
                ("ties", str(summary.ties)),
                ("losses", str(summary.losses)),
                ("baseline_mean", f"{summary.baseline_mean:.6f}"),
                ("candidate_mean", f"{summary.candidate_mean:.6f}"),
                ("delta", f"{summary.delta:.6f}"),
                ("verdict", summary.verdict),
            ):
                ET.SubElement(props, "property", {"name": key, "value": value})
            for case in result.cases:
                for check in case.results:
                    if check.dimension != summary.dimension:
                        continue
                    attrs: dict[str, str] = {
                        "classname": f"llmdiff.{summary.dimension}",
                        "name": case.case_id,
                        "time": "0",
                    }
                    testcase = ET.SubElement(suite, "testcase", attrs)
                    if check.verdict.value == "loss":
                        ET.SubElement(
                            testcase,
                            "failure",
                            {
                                "message": self._failure_message(check),
                                "type": "regression",
                            },
                        ).text = check.detail or "candidate regressed on this dimension"
                    elif check.verdict.value == "tie":
                        ET.SubElement(testcase, "system-out").text = check.detail
                    else:
                        ET.SubElement(testcase, "system-out").text = check.detail
        return root

    @staticmethod
    def _failure_message(check: Any) -> str:
        return (
            f"{check.dimension}: candidate {check.candidate_score:.4f} < "
            f"baseline {check.baseline_score:.4f} (delta {check.delta:+.4f})"
        )
