"""Unicode and encoding-hygiene dimensions."""

from __future__ import annotations

import unicodedata
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "UnicodeScriptCheck",
    "MojibakeCheck",
    "ControlCharCheck",
    "RtlMarkCheck",
    "script_distribution",
]

_MAX = 200_000


def script_distribution(text: str) -> dict[str, float]:
    """Return the fraction of alphabetic chars per Unicode script."""
    counts: Counter[str] = Counter()
    total = 0
    for ch in text[:_MAX]:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "?")
        script = name.split()[0] if name != "?" else "UNKNOWN"
        counts[script] += 1
        total += 1
    if total == 0:
        return {}
    return {script: c / total for script, c in counts.items()}


class UnicodeScriptCheck:
    """1.0 minus the total variation distance between script distributions."""

    @property
    def name(self) -> str:
        return "unicode_script"

    def score(self, output: str, context: CheckContext) -> float:
        base = script_distribution(context.baseline_output)
        cand = script_distribution(output)
        if not base and not cand:
            return 1.0
        keys = set(base) | set(cand)
        tvd = sum(abs(base.get(k, 0.0) - cand.get(k, 0.0)) for k in keys) / 2.0
        return max(0.0, 1.0 - tvd)


class MojibakeCheck:
    """1.0 when no classic mojibake sequences (Ã, Â, å etc.) appear."""

    @property
    def name(self) -> str:
        return "mojibake"

    _MARKERS = ("Ã", "Â", "â€", "å", "ä¸", "æ", "ï¿½")

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        hits = sum(sample.count(m) for m in self._MARKERS)
        if hits == 0:
            return 1.0
        return max(0.0, 1.0 - hits / 10.0)


class ControlCharCheck:
    """1.0 when no C0/C1 control characters appear (except \\t \\n \\r)."""

    @property
    def name(self) -> str:
        return "control_chars"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        bad = sum(
            1
            for ch in sample
            if unicodedata.category(ch) == "Cc" and ch not in "\t\n\r"
        )
        if bad == 0:
            return 1.0
        return max(0.0, 1.0 - bad / 20.0)


class RtlMarkCheck:
    """1.0 when RTL/LTR bidi marks appear iff the baseline had them."""

    @property
    def name(self) -> str:
        return "bidi_marks"

    _MARKS = ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e")

    def score(self, output: str, context: CheckContext) -> float:
        base = any(m in context.baseline_output[:_MAX] for m in self._MARKS)
        cand = any(m in output[:_MAX] for m in self._MARKS)
        return 1.0 if base == cand else 0.0
