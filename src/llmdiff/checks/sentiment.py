"""Sentiment and tone dimensions (lexicon-based, offline)."""

from __future__ import annotations

from llmdiff.checks.ngram import tokenize
from llmdiff.core.checks import CheckContext

__all__ = ["SentimentCheck", "ToneMatchCheck", "SENTIMENT_LEXICON", "sentiment_polarity"]

_MAX = 200_000

_POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "loved", "awesome", "fantastic",
    "happy", "wonderful", "best", "perfect", "thanks", "thank", "helpful", "nice",
    "works", "working", "easy", "recommend", "enjoy", "enjoyed", "pleased", "glad",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "hate", "hated", "worst", "broken", "crash", "crashes",
    "bug", "bugs", "angry", "frustrating", "frustrated", "useless", "slow",
    "wrong", "error", "errors", "fail", "fails", "failed", "annoying", "disappointed",
    "disappointing", "poor", "horrible", "sucks", "refund",
}

SENTIMENT_LEXICON = {w: 1.0 for w in _POSITIVE} | {w: -1.0 for w in _NEGATIVE}


def sentiment_polarity(text: str) -> float:
    """Mean sentiment in [-1, 1]; 0.0 when no sentiment words."""
    words = tokenize(text[:_MAX])
    scores = [SENTIMENT_LEXICON[w] for w in words if w in SENTIMENT_LEXICON]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


class SentimentCheck:
    """1.0 when the output's sentiment polarity matches the baseline's sign."""

    @property
    def name(self) -> str:
        return "sentiment"

    def score(self, output: str, context: CheckContext) -> float:
        base = sentiment_polarity(context.baseline_output)
        cand = sentiment_polarity(output)
        if base == 0.0 and cand == 0.0:
            return 1.0
        if (base > 0) == (cand > 0) and base != 0 and cand != 0:
            return 1.0
        if base == 0.0 or cand == 0.0:
            return 0.5
        return 0.0


class ToneMatchCheck:
    """1.0 minus normalized absolute sentiment difference vs baseline."""

    @property
    def name(self) -> str:
        return "tone"

    def score(self, output: str, context: CheckContext) -> float:
        base = sentiment_polarity(context.baseline_output)
        cand = sentiment_polarity(output)
        return max(0.0, 1.0 - abs(cand - base) / 2.0)
