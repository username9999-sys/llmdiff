"""Common-pattern dimensions: repeated phrases, templates, and structural clichés."""

from __future__ import annotations

import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = [
    "RepeatedPhraseCountCheck",
    "MostCommonPhraseCheck",
    "SentenceStarterDiversityCheck",
    "TransitionWordRatioCheck",
    "ListiclePatternCountCheck",
    "NumberedHeadingCountCheck",
    "BulletHeadingCountCheck",
    "CtaCountCheck",
    "DisclaimerCountCheck",
    "ParentheticalCountCheck",
    "EmDashCountCheck",
    "SemicolonPhraseCountCheck",
    "ColonIntroCountCheck",
    "QuoteAttributionCountCheck",
    "HashtagCountCheck",
    "MentionCountCheck",
    "AllCapsLineCountCheck",
    "EllipsisLineCountCheck",
    "RepeatedLineCountCheck",
    "BoilerplateScoreCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")

_TRANSITIONS = {
    "however", "moreover", "furthermore", "therefore", "thus", "meanwhile",
    "in conclusion", "in summary", "on the other hand", "for example",
    "for instance", "in addition", "as a result", "consequently", "next",
    "first", "second", "finally", "overall",
}
_CTA_RE = re.compile(
    r"\b(?:click here|learn more|get started|sign up|subscribe|try it|buy now|"
    r"download|read more|shop now|contact us)\b",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(
    r"\b(?:disclaimer|not (?:financial|legal|medical) advice|as an ai|"
    r"i am an ai|consult a (?:professional|doctor|lawyer))\b",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\([^()]{2,80}\)")
_EMDASH_RE = re.compile(r"—|–")
_COLON_INTRO_RE = re.compile(r":\s*$", re.MULTILINE)
_QUOTE_ATTR_RE = re.compile(r"—\s*[A-Z][a-z]+|,\s*(?:he|she|they)\s+said", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_BOILERPLATE = {
    "in today's fast-paced world", "it's important to note", "delve into",
    "at its core", "game-changer", "cutting-edge", "leverage",
    "in conclusion", "overall,", "it is worth noting",
}


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(_s(text))]


def _sents(text: str) -> list[str]:
    return [x.strip() for x in _SENT_RE.findall(_s(text)) if x.strip()]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class RepeatedPhraseCountCheck:
    """Count of bigram phrases (word pairs) that occur more than once."""

    @property
    def name(self) -> str:
        return "repeated_phrase_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            ws = _words(text)
            pairs = Counter(zip(ws, ws[1:], strict=False))
            return sum(1 for c in pairs.values() if c > 1)

        return _count_match(count(context.baseline_output), count(output))


class MostCommonPhraseCheck:
    """1.0 when the most frequent bigram matches the baseline's."""

    @property
    def name(self) -> str:
        return "top_phrase"

    def score(self, output: str, context: CheckContext) -> float:
        def top(text: str) -> tuple[str, str] | None:
            ws = _words(text)
            pairs = Counter(zip(ws, ws[1:], strict=False))
            return pairs.most_common(1)[0][0] if pairs else None

        base, cand = top(context.baseline_output), top(output)
        if base is None and cand is None:
            return 1.0
        return 1.0 if base == cand else 0.0


class SentenceStarterDiversityCheck:
    """Number of distinct first words across sentences."""

    @property
    def name(self) -> str:
        return "starter_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def diversity(text: str) -> int:
            starters = []
            for sent in _sents(text):
                ws = _WORD_RE.findall(sent)
                if ws:
                    starters.append(ws[0].lower())
            return len(set(starters))

        return _count_match(diversity(context.baseline_output), diversity(output))


class TransitionWordRatioCheck:
    @property
    def name(self) -> str:
        return "transition_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            if not ws:
                return 0.0
            hits = sum(1 for w in ws if w in _TRANSITIONS)
            return hits / len(ws)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ListiclePatternCountCheck:
    """Count of 'N ways/reasons/tips' style listicle openers."""

    @property
    def name(self) -> str:
        return "listicle_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(
            r"\b\d+\s+(?:ways|reasons|tips|tricks|steps|lessons)\b",
            re.IGNORECASE,
        )
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class NumberedHeadingCountCheck:
    @property
    def name(self) -> str:
        return "numbered_heading_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^#{1,6}\s+\d+[.)]\s", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class BulletHeadingCountCheck:
    @property
    def name(self) -> str:
        return "bullet_heading_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^#{1,6}\s+[-*•]\s", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class CtaCountCheck:
    @property
    def name(self) -> str:
        return "cta_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CTA_RE.findall(_s(context.baseline_output))),
            len(_CTA_RE.findall(_s(output))),
        )


class DisclaimerCountCheck:
    @property
    def name(self) -> str:
        return "disclaimer_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DISCLAIMER_RE.findall(_s(context.baseline_output))),
            len(_DISCLAIMER_RE.findall(_s(output))),
        )


class ParentheticalCountCheck:
    @property
    def name(self) -> str:
        return "parenthetical_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PAREN_RE.findall(_s(context.baseline_output))),
            len(_PAREN_RE.findall(_s(output))),
        )


class EmDashCountCheck:
    @property
    def name(self) -> str:
        return "emdash_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EMDASH_RE.findall(_s(context.baseline_output))),
            len(_EMDASH_RE.findall(_s(output))),
        )


class SemicolonPhraseCountCheck:
    @property
    def name(self) -> str:
        return "semicolon_phrase_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for s in _sents(text) if ";" in s)

        return _count_match(count(context.baseline_output), count(output))


class ColonIntroCountCheck:
    """Count of lines ending with ':' (list introductions)."""

    @property
    def name(self) -> str:
        return "colon_intro_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COLON_INTRO_RE.findall(_s(context.baseline_output))),
            len(_COLON_INTRO_RE.findall(_s(output))),
        )


class QuoteAttributionCountCheck:
    @property
    def name(self) -> str:
        return "quote_attribution_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTE_ATTR_RE.findall(_s(context.baseline_output))),
            len(_QUOTE_ATTR_RE.findall(_s(output))),
        )


class HashtagCountCheck:
    @property
    def name(self) -> str:
        return "hashtag_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HASHTAG_RE.findall(_s(context.baseline_output))),
            len(_HASHTAG_RE.findall(_s(output))),
        )


class MentionCountCheck:
    @property
    def name(self) -> str:
        return "mention_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MENTION_RE.findall(_s(context.baseline_output))),
            len(_MENTION_RE.findall(_s(output))),
        )


class AllCapsLineCountCheck:
    @property
    def name(self) -> str:
        return "allcaps_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(
                1
                for ln in _s(text).splitlines()
                if len(ln.strip()) >= 3 and ln.strip().isupper()
            )

        return _count_match(count(context.baseline_output), count(output))


class EllipsisLineCountCheck:
    @property
    def name(self) -> str:
        return "ellipsis_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for ln in _s(text).splitlines() if "…" in ln or "..." in ln)

        return _count_match(count(context.baseline_output), count(output))


class RepeatedLineCountCheck:
    """Count of lines that appear more than once in the text."""

    @property
    def name(self) -> str:
        return "repeated_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            lines = [ln.strip() for ln in _s(text).splitlines() if ln.strip()]
            c = Counter(lines)
            return sum(1 for v in c.values() if v > 1)

        return _count_match(count(context.baseline_output), count(output))


class BoilerplateScoreCheck:
    """Ratio of known boilerplate/AI-cliché phrases."""

    @property
    def name(self) -> str:
        return "boilerplate_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text).lower()
            hits = sum(s.count(bp) for bp in _BOILERPLATE)
            ws = _words(text)
            return hits / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))
