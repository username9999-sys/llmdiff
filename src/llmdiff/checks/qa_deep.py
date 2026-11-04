"""Deep QA dimensions: answerability, consistency, and factual-support signals."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "AnswersYesNoCheck",
    "RefusalCountCheck",
    "CaveatCountCheck",
    "ConfidenceMarkerRatioCheck",
    "CitationBracketCountCheck",
    "SourceMentionCountCheck",
    "ContradictionMarkerCountCheck",
    "RepetitionInAnswerCheck",
    "AnswerLengthAdequacyCheck",
    "DirectAnswerFirstCheck",
    "ListAnswerCoverageCheck",
    "ComparisonMarkerCountCheck",
    "ProsConsBalanceCheck",
    "StepMarkerCountCheck",
    "ExampleCountCheck",
    "DefinitionPresenceCheck",
    "SummaryPresenceCheck",
    "FollowUpSuggestionCountCheck",
    "UncertaintyMarkerRatioCheck",
    "BoldClaimCountCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")

_REFUSAL_RE = re.compile(
    r"\b(?:i can't|i cannot|i'm unable|i don't have access|as an ai|i am an ai|"
    r"i'm not able|sorry, i can't)\b",
    re.IGNORECASE,
)
_CAVEAT_RE = re.compile(
    r"\b(?:however|but note|keep in mind|it's worth noting|on the other hand|"
    r"that said|in practice|caveat)\b",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"\b(?:definitely|certainly|clearly|obviously|undoubtedly|without a doubt|"
    r"for sure|100%)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|might|could be|possibly|probably|likely|not sure|"
    r"i think|i believe|it depends)\b",
    re.IGNORECASE,
)
_CITE_RE = re.compile(r"\[\d+\]|\(\d{4}\)|\[citation|\[source", re.IGNORECASE)
_SOURCE_RE = re.compile(
    r"\b(?:according to|per |source:|reference:|see also|based on)\b",
    re.IGNORECASE,
)
_CONTRA_RE = re.compile(
    r"\b(?:contradicts|inconsistent with|on the contrary|actually, no|"
    r"that's not (?:right|correct)|however, this conflicts)\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(?:compared to|in contrast|unlike|whereas|while|but|however|versus|vs\.?)\b",
    re.IGNORECASE,
)
_STEP_RE = re.compile(r"^\s*(?:step\s+\d+|\d+[.)]\s)", re.MULTILINE | re.IGNORECASE)
_EXAMPLE_RE = re.compile(
    r"\b(?:for example|for instance|e\.g\.|i\.e\.|such as)\b",
    re.IGNORECASE,
)
_DEF_RE = re.compile(
    r"\b(?:is defined as|refers to|means that|in other words|that is,|i\.e\.)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:in summary|to sum up|in short|in conclusion|to conclude|overall)\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"\b(?:would you like|let me know if|feel free to ask|happy to help|"
    r"want me to|shall i)\b.*\?",
    re.IGNORECASE,
)
_BOLD_CLAIM_RE = re.compile(
    r"\b(?:always|never|everyone|no one|all|none|guaranteed)\b",
    re.IGNORECASE,
)


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


class AnswersYesNoCheck:
    """1.0 when the prompt is a yes/no question and the output starts with yes/no."""

    @property
    def name(self) -> str:
        return "yesno_answer"

    def score(self, output: str, context: CheckContext) -> float:
        prompt = _s(context.prompt).strip().lower()
        is_yn = bool(
            re.match(r"^(is|are|can|could|do|does|did|will|would|should|has|have)\b", prompt)
        ) and prompt.endswith("?")
        if not is_yn:
            return 1.0
        out = _s(output).strip().lower()
        return 1.0 if re.match(r"^(yes|no)\b", out) else 0.0


class RefusalCountCheck:
    @property
    def name(self) -> str:
        return "refusal_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_REFUSAL_RE.findall(_s(context.baseline_output))),
            len(_REFUSAL_RE.findall(_s(output))),
        )


class CaveatCountCheck:
    @property
    def name(self) -> str:
        return "caveat_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CAVEAT_RE.findall(_s(context.baseline_output))),
            len(_CAVEAT_RE.findall(_s(output))),
        )


class ConfidenceMarkerRatioCheck:
    @property
    def name(self) -> str:
        return "confidence_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_CONFIDENCE_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CitationBracketCountCheck:
    @property
    def name(self) -> str:
        return "citation_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CITE_RE.findall(_s(context.baseline_output))),
            len(_CITE_RE.findall(_s(output))),
        )


class SourceMentionCountCheck:
    @property
    def name(self) -> str:
        return "source_mention_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SOURCE_RE.findall(_s(context.baseline_output))),
            len(_SOURCE_RE.findall(_s(output))),
        )


class ContradictionMarkerCountCheck:
    @property
    def name(self) -> str:
        return "contradiction_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CONTRA_RE.findall(_s(context.baseline_output))),
            len(_CONTRA_RE.findall(_s(output))),
        )


class RepetitionInAnswerCheck:
    """Ratio of sentences that repeat an earlier sentence (exact match)."""

    @property
    def name(self) -> str:
        return "answer_repetition"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            sents = [s.lower() for s in _sents(text)]
            if not sents:
                return 0.0
            seen: set[str] = set()
            repeated = 0
            for s in sents:
                if s in seen:
                    repeated += 1
                else:
                    seen.add(s)
            return repeated / len(sents)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class AnswerLengthAdequacyCheck:
    """1.0 when the answer is at least 20 chars (non-trivial) iff baseline is."""

    @property
    def name(self) -> str:
        return "answer_adequacy"

    def score(self, output: str, context: CheckContext) -> float:
        def adequate(text: str) -> bool:
            return len(_s(text).strip()) >= 20

        return 1.0 if adequate(context.baseline_output) == adequate(output) else 0.0


class DirectAnswerFirstCheck:
    """1.0 when the first sentence has a preamble iff the baseline does."""

    @property
    def name(self) -> str:
        return "direct_answer_first"

    def score(self, output: str, context: CheckContext) -> float:
        preamble = re.compile(
            r"^(sure|certainly|of course|great question|good question|happy to help|"
            r"let me|i'll|here is|here's)\b",
            re.IGNORECASE,
        )

        def has_preamble(text: str) -> bool:
            sents = _sents(text)
            return bool(sents and preamble.match(sents[0]))

        return 1.0 if has_preamble(context.baseline_output) == has_preamble(output) else 0.0


class ListAnswerCoverageCheck:
    """When the prompt asks for N items, coverage of list markers in the answer."""

    @property
    def name(self) -> str:
        return "list_coverage"

    def score(self, output: str, context: CheckContext) -> float:
        m = re.search(
            r"\b(\d+)\s+(?:items|things|ways|reasons|points)\b",
            _s(context.prompt),
            re.IGNORECASE,
        )
        if not m:
            return 1.0
        n = int(m.group(1))
        items = len(
            re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+\S", _s(output), re.MULTILINE)
        )
        return min(1.0, items / n)


class ComparisonMarkerCountCheck:
    @property
    def name(self) -> str:
        return "comparison_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPARE_RE.findall(_s(context.baseline_output))),
            len(_COMPARE_RE.findall(_s(output))),
        )


class ProsConsBalanceCheck:
    """Balance of pros vs cons markers (advantages/disadvantages)."""

    @property
    def name(self) -> str:
        return "pros_cons_balance"

    def score(self, output: str, context: CheckContext) -> float:
        pros = re.compile(r"\b(?:pros?|advantages?|benefits?|strengths?)\b", re.IGNORECASE)
        cons = re.compile(r"\b(?:cons?|disadvantages?|drawbacks?|weaknesses?)\b", re.IGNORECASE)

        def balance(text: str) -> float:
            s = _s(text)
            p, c = len(pros.findall(s)), len(cons.findall(s))
            total = p + c
            return p / total if total else 0.5

        return _ratio_match(balance(context.baseline_output), balance(output))


class StepMarkerCountCheck:
    @property
    def name(self) -> str:
        return "step_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STEP_RE.findall(_s(context.baseline_output))),
            len(_STEP_RE.findall(_s(output))),
        )


class ExampleCountCheck:
    @property
    def name(self) -> str:
        return "example_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EXAMPLE_RE.findall(_s(context.baseline_output))),
            len(_EXAMPLE_RE.findall(_s(output))),
        )


class DefinitionPresenceCheck:
    @property
    def name(self) -> str:
        return "definition_presence"

    def score(self, output: str, context: CheckContext) -> float:
        base = bool(_DEF_RE.search(_s(context.baseline_output)))
        cand = bool(_DEF_RE.search(_s(output)))
        return 1.0 if base == cand else 0.0


class SummaryPresenceCheck:
    @property
    def name(self) -> str:
        return "summary_presence"

    def score(self, output: str, context: CheckContext) -> float:
        base = bool(_SUMMARY_RE.search(_s(context.baseline_output)))
        cand = bool(_SUMMARY_RE.search(_s(output)))
        return 1.0 if base == cand else 0.0


class FollowUpSuggestionCountCheck:
    @property
    def name(self) -> str:
        return "followup_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FOLLOWUP_RE.findall(_s(context.baseline_output))),
            len(_FOLLOWUP_RE.findall(_s(output))),
        )


class UncertaintyMarkerRatioCheck:
    @property
    def name(self) -> str:
        return "uncertainty_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ws = _words(text)
            return len(_UNCERTAINTY_RE.findall(_s(text))) / len(ws) if ws else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class BoldClaimCountCheck:
    """Count of absolute claims (always/never/everyone...)."""

    @property
    def name(self) -> str:
        return "bold_claim_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOLD_CLAIM_RE.findall(_s(context.baseline_output))),
            len(_BOLD_CLAIM_RE.findall(_s(output))),
        )
