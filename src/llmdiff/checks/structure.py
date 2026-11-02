"""Structured-output comparison dimensions (JSON validity and key structure)."""

from __future__ import annotations

import json
from typing import Any

from llmdiff.core.checks import CheckContext

__all__ = ["JsonValidCheck", "StructureCheck", "key_paths"]


def key_paths(value: Any, prefix: str = "$") -> set[str]:
    """Return the set of JSON key paths present in ``value``.

    Example: ``{"a": {"b": 1}}`` yields ``{"$", "$.a", "$.a.b"}``.
    """
    paths = {prefix}
    if isinstance(value, dict):
        for key, item in value.items():
            paths |= key_paths(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for item in value:
            paths |= key_paths(item, f"{prefix}[]")
    return paths


class JsonValidCheck:
    """1.0 when the output parses as JSON, else 0.0."""

    @property
    def name(self) -> str:
        return "json_valid"

    def score(self, output: str, context: CheckContext) -> float:
        try:
            json.loads(output)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0
        return 1.0


class StructureCheck:
    """Key-path overlap between the output and the baseline JSON.

    Catches a candidate that reworded the prose but silently dropped fields:
    the score is the Jaccard index of the two key-path sets.
    """

    @property
    def name(self) -> str:
        return "structure"

    def score(self, output: str, context: CheckContext) -> float:
        try:
            candidate = json.loads(output)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0
        try:
            baseline = json.loads(context.baseline_output)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 1.0 if isinstance(candidate, (dict, list)) else 0.0
        candidate_paths = key_paths(candidate)
        baseline_paths = key_paths(baseline)
        union = candidate_paths | baseline_paths
        if not union:
            return 1.0
        return len(candidate_paths & baseline_paths) / len(union)
