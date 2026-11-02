"""Built-in comparison dimensions."""

from __future__ import annotations

from llmdiff.checks.entropy import EntropyCheck, RepetitionCheck
from llmdiff.checks.ngram import (
    NgramF1Check,
    NgramPrecisionCheck,
    NgramRecallCheck,
    TokenNgramCheck,
)
from llmdiff.checks.structure import JsonValidCheck, StructureCheck
from llmdiff.checks.text import ContentCheck, ExactMatchCheck, LengthCheck, SimilarityCheck
from llmdiff.core.checks import Check

__all__ = [
    "builtin_checks",
    "ExactMatchCheck",
    "SimilarityCheck",
    "LengthCheck",
    "ContentCheck",
    "JsonValidCheck",
    "StructureCheck",
    "TokenNgramCheck",
    "NgramPrecisionCheck",
    "NgramRecallCheck",
    "NgramF1Check",
    "EntropyCheck",
    "RepetitionCheck",
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
        TokenNgramCheck(),
        NgramPrecisionCheck(),
        NgramRecallCheck(),
        NgramF1Check(),
        EntropyCheck(),
        RepetitionCheck(),
    )
