"""User-configurable regex dimensions driven by case metadata."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext
from llmdiff.core.errors import CheckError

__all__ = ["RegexMustMatchCheck", "RegexForbiddenCheck", "compile_patterns"]


def compile_patterns(raw: object, *, field: str) -> list[re.Pattern[str]]:
    """Compile a list of regex strings from metadata, raising CheckError on bad input."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise CheckError(f"metadata {field!r} must be a list of regex strings")
    out: list[re.Pattern[str]] = []
    for item in raw:
        if not isinstance(item, str):
            raise CheckError(f"metadata {field!r} entries must be strings")
        try:
            out.append(re.compile(item))
        except re.error as exc:
            raise CheckError(f"metadata {field!r}: invalid regex {item!r}: {exc}") from exc
    return out


class RegexMustMatchCheck:
    """Fraction of ``regex_must_match`` patterns found in the output.

    When the metadata key is absent the check is neutral (1.0).
    """

    @property
    def name(self) -> str:
        return "regex_must_match"

    def score(self, output: str, context: CheckContext) -> float:
        patterns = compile_patterns(
            context.metadata.get("regex_must_match"), field="regex_must_match"
        )
        if not patterns:
            return 1.0
        sample = output[: 500_000]
        hits = sum(1 for p in patterns if p.search(sample))
        return hits / len(patterns)


class RegexForbiddenCheck:
    """1.0 when no ``regex_forbidden`` pattern is found (0 hits = 1.0)."""

    @property
    def name(self) -> str:
        return "regex_forbidden"

    def score(self, output: str, context: CheckContext) -> float:
        patterns = compile_patterns(
            context.metadata.get("regex_forbidden"), field="regex_forbidden"
        )
        if not patterns:
            return 1.0
        sample = output[: 500_000]
        hits = sum(1 for p in patterns if p.search(sample))
        if hits == 0:
            return 1.0
        return max(0.0, 1.0 - hits / len(patterns))
