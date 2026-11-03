"""Miscellaneous output-quality dimensions: emoji, URLs, comment ratio, casing."""

from __future__ import annotations

import re
import unicodedata

from llmdiff.checks.code import extract_fenced_blocks
from llmdiff.core.checks import CheckContext

__all__ = [
    "EmojiCountCheck",
    "UrlValidityCheck",
    "CommentRatioCheck",
    "UpperRatioCheck",
    "NonAsciiRatioCheck",
    "EmojiPresenceCheck",
]

_MAX = 200_000
_URL_RE = re.compile(r"https?://[^\s)\]\"']*")
_EMOJI_RE = re.compile(
    "[" + "\U0001F300-\U0001FAFF" + "\U00002600-\U000027BF" + "\U0001F1E6-\U0001F1FF" + "]"
)


def _is_emoji(ch: str) -> bool:
    return bool(_EMOJI_RE.match(ch)) or unicodedata.category(ch) in ("So", "Sk")


class EmojiCountCheck:
    """1.0 when emoji count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "emoji_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = sum(1 for c in context.baseline_output[:_MAX] if _is_emoji(c))
        cand = sum(1 for c in output[:_MAX] if _is_emoji(c))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class EmojiPresenceCheck:
    """1.0 when the output has an emoji iff the baseline does."""

    @property
    def name(self) -> str:
        return "emoji_presence"

    def score(self, output: str, context: CheckContext) -> float:
        base = any(_is_emoji(c) for c in context.baseline_output[:_MAX])
        cand = any(_is_emoji(c) for c in output[:_MAX])
        return 1.0 if base == cand else 0.0


class UrlValidityCheck:
    """Fraction of URLs in the output that have a valid http(s) shape."""

    @property
    def name(self) -> str:
        return "url_validity"

    def score(self, output: str, context: CheckContext) -> float:
        urls = _URL_RE.findall(output[:_MAX])
        if not urls:
            return 1.0
        valid = sum(1 for u in urls if re.match(r"https?://[A-Za-z0-9.-]+(:\d+)?(/\S*)?$", u))
        return valid / len(urls)


class CommentRatioCheck:
    """1.0 when the comment-line ratio inside code blocks matches the baseline."""

    @property
    def name(self) -> str:
        return "comment_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            blocks = extract_fenced_blocks(text)
            lines = [ln for _, code in blocks for ln in code.splitlines()]
            if not lines:
                return 0.0
            comments = sum(
                1 for ln in lines if ln.lstrip().startswith(("#", "//", "/*", "*"))
            )
            return comments / len(lines)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        return max(0.0, 1.0 - abs(cand - base))


class UpperRatioCheck:
    """1.0 when the ratio of uppercase letters matches the baseline."""

    @property
    def name(self) -> str:
        return "upper_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            letters = [c for c in text[:_MAX] if c.isalpha()]
            if not letters:
                return 0.0
            return sum(1 for c in letters if c.isupper()) / len(letters)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class NonAsciiRatioCheck:
    """1.0 when the non-ASCII character ratio matches the baseline."""

    @property
    def name(self) -> str:
        return "non_ascii_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            sample = text[:_MAX]
            if not sample:
                return 0.0
            return sum(1 for c in sample if ord(c) > 127) / len(sample)

        base = ratio(context.baseline_output)
        cand = ratio(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
