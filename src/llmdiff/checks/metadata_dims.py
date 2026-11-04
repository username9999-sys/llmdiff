"""Output-metadata dimensions: length bands, token estimates, and structural hints."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "LengthBandCheck",
    "TokenEstimateCheck",
    "TokenEstimateRatioCheck",
    "CharsPerTokenCheck",
    "WordsPerSentenceBandCheck",
    "LinesPerParagraphCheck",
    "AvgLineLengthCheck",
    "MaxLineLengthBandCheck",
    "BlankLineRatioCheck",
    "CodeToTextRatioCheck",
    "MarkdownDensityCheck",
    "JsonLikelihoodCheck",
    "YamlLikelihoodCheck",
    "CsvLikelihoodCheck",
    "HtmlLikelihoodCheck",
    "XmlLikelihoodCheck",
    "LogLikelihoodCheck",
    "StackTraceLikelihoodCheck",
    "DiffLikelihoodCheck",
    "TableLikelihoodCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _band(value: float, edges: tuple[float, ...]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


class LengthBandCheck:
    """1.0 when the char-length band (log scale) matches the baseline."""

    @property
    def name(self) -> str:
        return "length_band"

    def score(self, output: str, context: CheckContext) -> float:
        edges = (10, 50, 200, 1000, 5000, 20000)
        base = _band(len(_s(context.baseline_output)), edges)
        cand = _band(len(_s(output)), edges)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / len(edges))


class TokenEstimateCheck:
    """1.0 when the whitespace-token count matches the baseline."""

    @property
    def name(self) -> str:
        return "token_estimate"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_WORD_RE.findall(_s(context.baseline_output))),
            len(_WORD_RE.findall(_s(output))),
        )


class TokenEstimateRatioCheck:
    """Ratio of output tokens to baseline tokens, mapped to [0,1]."""

    @property
    def name(self) -> str:
        return "token_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_WORD_RE.findall(_s(context.baseline_output)))
        cand = len(_WORD_RE.findall(_s(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        if cand <= base:
            return cand / base
        return max(0.0, 1.0 - (cand - base) / cand)


class CharsPerTokenCheck:
    @property
    def name(self) -> str:
        return "chars_per_token"

    def score(self, output: str, context: CheckContext) -> float:
        def cpt(text: str) -> float:
            s = _s(text)
            toks = len(_WORD_RE.findall(s))
            return len(s) / toks if toks else 0.0

        base, cand = cpt(context.baseline_output), cpt(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class WordsPerSentenceBandCheck:
    @property
    def name(self) -> str:
        return "wps_band"

    def score(self, output: str, context: CheckContext) -> float:
        def band(text: str) -> int:
            sents = [x for x in re.split(r"[.!?]+", _s(text)) if x.strip()]
            if not sents:
                return 0
            avg = sum(len(x.split()) for x in sents) / len(sents)
            return _band(avg, (5, 10, 20, 35))

        base, cand = band(context.baseline_output), band(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / 4)


class LinesPerParagraphCheck:
    @property
    def name(self) -> str:
        return "lines_per_para"

    def score(self, output: str, context: CheckContext) -> float:
        def lpp(text: str) -> float:
            paras = [p for p in _s(text).split("\n\n") if p.strip()]
            return sum(len(p.splitlines()) for p in paras) / len(paras) if paras else 0.0

        base, cand = lpp(context.baseline_output), lpp(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class AvgLineLengthCheck:
    @property
    def name(self) -> str:
        return "avg_line_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            lines = _s(text).splitlines()
            return sum(len(ln) for ln in lines) / len(lines) if lines else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class MaxLineLengthBandCheck:
    @property
    def name(self) -> str:
        return "max_line_band"

    def score(self, output: str, context: CheckContext) -> float:
        def band(text: str) -> int:
            lines = _s(text).splitlines()
            mx = max((len(ln) for ln in lines), default=0)
            return _band(mx, (40, 80, 120, 200, 500))

        base, cand = band(context.baseline_output), band(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / 5)


class BlankLineRatioCheck:
    @property
    def name(self) -> str:
        return "blank_line_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = _s(text).splitlines()
            return sum(1 for ln in lines if not ln.strip()) / len(lines) if lines else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CodeToTextRatioCheck:
    @property
    def name(self) -> str:
        return "code_text_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            fences = len(_FENCE_RE.findall(s))
            if fences == 0:
                return 0.0
            return fences / max(1, len(s.splitlines()))

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class MarkdownDensityCheck:
    @property
    def name(self) -> str:
        return "md_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            s = _s(text)
            marks = len(
                re.findall(r"^#{1,6}\s|\*\*|__|\[.+\]\(.+\)|^\s*[-*]\s", s, re.MULTILINE)
            )
            return marks / max(1, len(s.splitlines()))

        return _ratio_match(density(context.baseline_output), density(output))


def _likelihood(text: str, pattern: re.Pattern[str]) -> float:
    s = _s(text)
    return len(pattern.findall(s)) / max(1, len(s.splitlines()))


class JsonLikelihoodCheck:
    @property
    def name(self) -> str:
        return "json_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^\s*[\{\[]|:\s*[^,}\]]+,?\s*$", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class YamlLikelihoodCheck:
    @property
    def name(self) -> str:
        return "yaml_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^\s*[\w.-]+:\s", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class CsvLikelihoodCheck:
    @property
    def name(self) -> str:
        return "csv_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^[^,;\n]+[,;][^,;\n]+", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class HtmlLikelihoodCheck:
    @property
    def name(self) -> str:
        return "html_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"</?[a-zA-Z][^>]*>")
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class XmlLikelihoodCheck:
    @property
    def name(self) -> str:
        return "xml_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"<\?xml|</[a-zA-Z][\w.-]*>")
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class LogLikelihoodCheck:
    @property
    def name(self) -> str:
        return "log_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(
            r"\b(?:INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
        )
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class StackTraceLikelihoodCheck:
    @property
    def name(self) -> str:
        return "stacktrace_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^\s*(?:File \"|at [\w.]+\s*\(|Traceback)", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class DiffLikelihoodCheck:
    @property
    def name(self) -> str:
        return "diff_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^(?:[+\-]{3} |@@ |diff --git )", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))


class TableLikelihoodCheck:
    @property
    def name(self) -> str:
        return "table_likelihood"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
        return _ratio_match(_likelihood(context.baseline_output, pat), _likelihood(output, pat))
