"""QA dimensions: does the output answer the prompt, keep facts, cite, match language."""

from __future__ import annotations

import re
import unicodedata

from llmdiff.core.checks import CheckContext

__all__ = [
    "AnswersPromptCheck",
    "NumberConsistencyCheck",
    "CitationCheck",
    "LanguageMatchCheck",
    "extract_numbers",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[a-z]{3,}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:%|[kKmMbB])?")
_CITE_RE = re.compile(r"(\[\d+\]|https?://[^\s)\]]+|\(source:|doi:)", re.IGNORECASE)


def extract_numbers(text: str) -> list[float]:
    """Extract numeric literals from ``text`` as floats, capped for huge inputs."""
    sample = text[:_MAX]
    out: list[float] = []
    for m in _NUM_RE.finditer(sample):
        token = m.group(0)
        mult = 1.0
        if token.endswith(("k", "K")):
            mult, token = 1e3, token[:-1]
        elif token.endswith(("m", "M")):
            mult, token = 1e6, token[:-1]
        elif token.endswith(("b", "B")):
            mult, token = 1e9, token[:-1]
        try:
            out.append(float(token.rstrip("%")) * mult)
        except ValueError:
            continue
    return out


class AnswersPromptCheck:
    """Heuristic: fraction of significant prompt words that appear in the output."""

    @property
    def name(self) -> str:
        return "answers_prompt"

    def score(self, output: str, context: CheckContext) -> float:
        prompt_words = {w for w in _WORD_RE.findall(context.prompt[:_MAX].lower())}
        if not prompt_words:
            return 1.0
        out_lower = output[:_MAX].lower()
        hits = sum(1 for w in prompt_words if w in out_lower)
        return hits / len(prompt_words)


class NumberConsistencyCheck:
    """Fraction of baseline numbers preserved in the output within 1% tolerance."""

    @property
    def name(self) -> str:
        return "number_consistency"

    def score(self, output: str, context: CheckContext) -> float:
        base = extract_numbers(context.baseline_output)
        if not base:
            return 1.0
        cand = extract_numbers(output)
        if not cand:
            return 0.0
        preserved = 0
        for b in base:
            tol = max(abs(b) * 0.01, 1e-9)
            if any(abs(c - b) <= tol for c in cand):
                preserved += 1
        return preserved / len(base)


class CitationCheck:
    """1.0 when the output contains a citation marker iff the baseline does."""

    @property
    def name(self) -> str:
        return "citation"

    def score(self, output: str, context: CheckContext) -> float:
        base = bool(_CITE_RE.search(context.baseline_output[:_MAX]))
        cand = bool(_CITE_RE.search(output[:_MAX]))
        return 1.0 if base == cand else 0.0


class LanguageMatchCheck:
    """1.0 when the output's dominant script matches the baseline's."""

    @property
    def name(self) -> str:
        return "language_match"

    @staticmethod
    def _dominant_script(text: str) -> str:
        counts: dict[str, int] = {}
        for ch in text[:_MAX]:
            if not ch.isalpha():
                continue
            name = unicodedata.name(ch, "?")
            script = name.split()[0] if name != "?" else "LATIN"
            counts[script] = counts.get(script, 0) + 1
        if not counts:
            return "NONE"
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def score(self, output: str, context: CheckContext) -> float:
        base = self._dominant_script(context.baseline_output)
        cand = self._dominant_script(output)
        return 1.0 if base == cand else 0.0
