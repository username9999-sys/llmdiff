"""Topic drift and keyword coverage dimensions."""

from __future__ import annotations

from llmdiff.checks.ngram import tokenize
from llmdiff.core.checks import CheckContext

__all__ = ["KeywordCoverageCheck", "TopicDriftCheck", "extract_keywords"]

_MAX = 200_000
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "it", "this",
    "that", "these", "those", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "not", "no", "you", "your", "we", "our",
    "they", "their", "he", "she", "i", "me", "my", "what", "which", "who", "how",
}


def extract_keywords(text: str, *, top_k: int = 20) -> list[str]:
    """Top content words by frequency, excluding stopwords."""
    from collections import Counter

    words = [w for w in tokenize(text[:_MAX]) if w not in _STOPWORDS and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(top_k)]


class KeywordCoverageCheck:
    """Fraction of baseline keywords that also appear in the output."""

    @property
    def name(self) -> str:
        return "keyword_coverage"

    def score(self, output: str, context: CheckContext) -> float:
        base_kw = set(extract_keywords(context.baseline_output))
        if not base_kw:
            return 1.0
        out_tokens = set(tokenize(output))
        hits = sum(1 for kw in base_kw if kw in out_tokens)
        return hits / len(base_kw)


class TopicDriftCheck:
    """1.0 minus Jaccard distance between baseline and output keyword sets."""

    @property
    def name(self) -> str:
        return "topic_drift"

    def score(self, output: str, context: CheckContext) -> float:
        base = set(extract_keywords(context.baseline_output))
        cand = set(extract_keywords(output))
        if not base and not cand:
            return 1.0
        union = base | cand
        if not union:
            return 1.0
        return len(base & cand) / len(union)
