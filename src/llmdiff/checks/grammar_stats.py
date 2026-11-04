"""Grammar-pattern dimensions: agreement hints, tense markers, and clause shapes."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "PastTenseVerbCountCheck",
    "PresentTenseVerbCountCheck",
    "FutureMarkerCountCheck",
    "ModalVerbCountCheck",
    "GerundCountCheck",
    "InfinitiveCountCheck",
    "PassiveAuxCountCheck",
    "ConditionalMarkerCountCheck",
    "SubordinateClauseCountCheck",
    "RelativePronounCountCheck",
    "ArticleACountCheck",
    "ArticleAnCountCheck",
    "ArticleTheCountCheck",
    "PrepositionCountCheck",
    "ConjunctionCountCheck",
    "PronounCountCheck",
    "AuxVerbCountCheck",
    "NegationGrammarCountCheck",
    "QuestionWordCountCheck",
    "ExclamationGrammarCountCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")

_PAST_ED = re.compile(r"\b\w+ed\b", re.IGNORECASE)
_PRESENT_3RD = re.compile(r"\b(?:he|she|it)\s+\w+s\b", re.IGNORECASE)
_FUTURE = re.compile(r"\b(?:will|shall|going to)\b", re.IGNORECASE)
_MODALS = re.compile(
    r"\b(?:can|could|may|might|must|should|would|ought to)\b",
    re.IGNORECASE,
)
_GERUND = re.compile(r"\b\w+ing\b", re.IGNORECASE)
_INFINITIVE = re.compile(r"\bto\s+[a-z]+\b", re.IGNORECASE)
_PASSIVE_AUX = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am)\s+\w+(?:ed|en)\b",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\bif\b|\bwould\b|\bunless\b|\bprovided that\b",
    re.IGNORECASE,
)
_SUBORD = re.compile(
    r"\b(?:because|although|while|since|if|when|after|before|unless|until)\b",
    re.IGNORECASE,
)
_REL_PRON = re.compile(r"\b(?:who|whom|whose|which|that)\b", re.IGNORECASE)
_ART_A = re.compile(r"\ba\b", re.IGNORECASE)
_ART_AN = re.compile(r"\ban\b", re.IGNORECASE)
_ART_THE = re.compile(r"\bthe\b", re.IGNORECASE)
_PREPS = re.compile(
    r"\b(?:in|on|at|to|for|with|of|by|from|about|into|over|under|between|"
    r"through|during|before|after)\b",
    re.IGNORECASE,
)
_CONJS = re.compile(
    r"\b(?:and|but|or|nor|so|yet|because|although|while|if|than)\b",
    re.IGNORECASE,
)
_PRONS = re.compile(
    r"\b(?:i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|its|our|their)\b",
    re.IGNORECASE,
)
_AUX = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am|do|does|did|have|has|had|will|would|"
    r"can|could|may|might|must|shall|should)\b",
    re.IGNORECASE,
)
_NEG = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot|"
    r"can't|don't|won't|isn't|aren't|wasn't|weren't|didn't|doesn't|haven't|"
    r"hasn't|hadn't)\b",
    re.IGNORECASE,
)
_QWORDS = re.compile(r"\b(?:who|what|when|where|why|how|which)\b", re.IGNORECASE)
_EXCL = re.compile(r"!")


def _s(text: str) -> str:
    return text[:_MAX]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_s(text))


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class PastTenseVerbCountCheck:
    @property
    def name(self) -> str:
        return "past_verb_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PAST_ED.findall(_s(context.baseline_output))),
            len(_PAST_ED.findall(_s(output))),
        )


class PresentTenseVerbCountCheck:
    @property
    def name(self) -> str:
        return "present_3rd_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PRESENT_3RD.findall(_s(context.baseline_output))),
            len(_PRESENT_3RD.findall(_s(output))),
        )


class FutureMarkerCountCheck:
    @property
    def name(self) -> str:
        return "future_marker_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FUTURE.findall(_s(context.baseline_output))),
            len(_FUTURE.findall(_s(output))),
        )


class ModalVerbCountCheck:
    @property
    def name(self) -> str:
        return "modal_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MODALS.findall(_s(context.baseline_output))),
            len(_MODALS.findall(_s(output))),
        )


class GerundCountCheck:
    @property
    def name(self) -> str:
        return "gerund_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GERUND.findall(_s(context.baseline_output))),
            len(_GERUND.findall(_s(output))),
        )


class InfinitiveCountCheck:
    @property
    def name(self) -> str:
        return "infinitive_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INFINITIVE.findall(_s(context.baseline_output))),
            len(_INFINITIVE.findall(_s(output))),
        )


class PassiveAuxCountCheck:
    @property
    def name(self) -> str:
        return "passive_aux_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PASSIVE_AUX.findall(_s(context.baseline_output))),
            len(_PASSIVE_AUX.findall(_s(output))),
        )


class ConditionalMarkerCountCheck:
    @property
    def name(self) -> str:
        return "conditional_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CONDITIONAL.findall(_s(context.baseline_output))),
            len(_CONDITIONAL.findall(_s(output))),
        )


class SubordinateClauseCountCheck:
    @property
    def name(self) -> str:
        return "subordinate_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUBORD.findall(_s(context.baseline_output))),
            len(_SUBORD.findall(_s(output))),
        )


class RelativePronounCountCheck:
    @property
    def name(self) -> str:
        return "rel_pronoun_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_REL_PRON.findall(_s(context.baseline_output))),
            len(_REL_PRON.findall(_s(output))),
        )


class ArticleACountCheck:
    @property
    def name(self) -> str:
        return "article_a_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ART_A.findall(_s(context.baseline_output))),
            len(_ART_A.findall(_s(output))),
        )


class ArticleAnCountCheck:
    @property
    def name(self) -> str:
        return "article_an_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ART_AN.findall(_s(context.baseline_output))),
            len(_ART_AN.findall(_s(output))),
        )


class ArticleTheCountCheck:
    @property
    def name(self) -> str:
        return "article_the_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ART_THE.findall(_s(context.baseline_output))),
            len(_ART_THE.findall(_s(output))),
        )


class PrepositionCountCheck:
    @property
    def name(self) -> str:
        return "preposition_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PREPS.findall(_s(context.baseline_output))),
            len(_PREPS.findall(_s(output))),
        )


class ConjunctionCountCheck:
    @property
    def name(self) -> str:
        return "conjunction_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CONJS.findall(_s(context.baseline_output))),
            len(_CONJS.findall(_s(output))),
        )


class PronounCountCheck:
    @property
    def name(self) -> str:
        return "pronoun_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PRONS.findall(_s(context.baseline_output))),
            len(_PRONS.findall(_s(output))),
        )


class AuxVerbCountCheck:
    @property
    def name(self) -> str:
        return "aux_verb_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_AUX.findall(_s(context.baseline_output))),
            len(_AUX.findall(_s(output))),
        )


class NegationGrammarCountCheck:
    @property
    def name(self) -> str:
        return "negation_grammar_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NEG.findall(_s(context.baseline_output))),
            len(_NEG.findall(_s(output))),
        )


class QuestionWordCountCheck:
    @property
    def name(self) -> str:
        return "qword_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QWORDS.findall(_s(context.baseline_output))),
            len(_QWORDS.findall(_s(output))),
        )


class ExclamationGrammarCountCheck:
    @property
    def name(self) -> str:
        return "exclamation_grammar_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EXCL.findall(_s(context.baseline_output))),
            len(_EXCL.findall(_s(output))),
        )
