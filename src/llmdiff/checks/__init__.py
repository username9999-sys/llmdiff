"""Built-in comparison dimensions."""

from __future__ import annotations

from llmdiff.core.checks import Check
from llmdiff.checks.structure import JsonValidCheck, StructureCheck
from llmdiff.checks.text import ContentCheck, ExactMatchCheck, LengthCheck, SimilarityCheck

__all__ = [
    "builtin_checks",
    "ExactMatchCheck",
    "SimilarityCheck",
    "LengthCheck",
    "ContentCheck",
    "JsonValidCheck",
    "StructureCheck",
]


def builtin_checks() -> tuple[Check, ...]:
    """Return one instance of every built-in comparison dimension."""
    return (
        ExactMatchCheck(),
        SimilarityCheck(),
        LengthCheck(),
        ContentCheck(),
        JsonValidCheck(),
        StructureCheck(),
    )
