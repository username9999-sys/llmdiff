"""Tests for misc dimensions (emoji, urls, comment ratio, casing, non-ascii)."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.misc import (
    CommentRatioCheck,
    EmojiCountCheck,
    EmojiPresenceCheck,
    NonAsciiRatioCheck,
    UpperRatioCheck,
    UrlValidityCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_emoji_count_match():
    ctx = _ctx("hi \U0001F600", "hi \U0001F600")
    assert EmojiCountCheck().score("hi \U0001F600", ctx) == 1.0
    assert EmojiCountCheck().score("hi", ctx) < 1.0


def test_emoji_presence():
    ctx = _ctx("hi \U0001F600", "hi")
    assert EmojiPresenceCheck().score("hi", ctx) == 0.0
    assert EmojiPresenceCheck().score("hi \U0001F600", ctx) == 1.0


def test_url_validity():
    ctx = _ctx("see https://example.com/a", "see https://example.com/a")
    assert UrlValidityCheck().score("see https://example.com/a", ctx) == 1.0
    assert UrlValidityCheck().score("see https://", ctx) < 1.0
    assert UrlValidityCheck().score("no urls", ctx) == 1.0


def test_comment_ratio():
    ctx = _ctx("```python\n# comment\nx=1\n```", "```python\n# comment\nx=1\n```")
    assert CommentRatioCheck().score("```python\n# comment\nx=1\n```", ctx) == 1.0


def test_upper_ratio():
    ctx = _ctx("Hello World", "hello world")
    assert UpperRatioCheck().score("hello world", ctx) < 1.0


def test_non_ascii_ratio():
    ctx = _ctx("hello", "héllo wörld")
    assert NonAsciiRatioCheck().score("héllo wörld", ctx) < 1.0
    assert NonAsciiRatioCheck().score("hello", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hi \U0001F600 see https://x.co",
         "candidate_output": "hi \U0001F600 see https://x.co"},
        {"id": "b", "prompt": "p", "baseline_output": "hi \U0001F600 see https://x.co",
         "candidate_output": "plain text no emoji no url"},
    ])
    result = evaluate(corpus, dimensions=("emoji_count", "emoji_presence"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["emoji_count"].losses == 1
    assert dims["emoji_presence"].losses == 1
