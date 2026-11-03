"""Numeric dimensions: extract numbers from text and compare their statistics."""

from __future__ import annotations

import math
import re
import statistics

from llmdiff.core.checks import CheckContext

__all__ = [
    "NumberCountCheck",
    "NumberSumCheck",
    "NumberMeanCheck",
    "NumberMedianCheck",
    "NumberMinCheck",
    "NumberMaxCheck",
    "NumberStdCheck",
    "NumberRangeCheck",
    "IntegerCountCheck",
    "FloatCountCheck",
    "NegativeCountCheck",
    "PercentCountCheck",
    "CurrencyCountCheck",
    "DateCountCheck",
    "TimeCountCheck",
    "NumberPrecisionMatchCheck",
    "NumberSetJaccardCheck",
    "NumberSortedMatchCheck",
    "AvgDigitCountCheck",
    "NumberMagnitudeRatioCheck",
]

_MAX = 200_000
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_INT_RE = re.compile(r"(?<![\d.])-?\d+(?![\d.])")
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
_PCT_RE = re.compile(r"-?\d+(?:\.\d+)?\s*%")
_CUR_RE = re.compile(
    r"[$€£¥₹]\s?-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?\s?(?:USD|EUR|IDR|JPY)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b")


def _s(text: str) -> str:
    return text[:_MAX]


def _nums(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(_s(text))]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class NumberCountCheck:
    @property
    def name(self) -> str:
        return "number_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_nums(context.baseline_output)), len(_nums(output)))


class NumberSumCheck:
    @property
    def name(self) -> str:
        return "number_sum"

    def score(self, output: str, context: CheckContext) -> float:
        base = sum(_nums(context.baseline_output))
        cand = sum(_nums(output))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))


class NumberMeanCheck:
    @property
    def name(self) -> str:
        return "number_mean"

    def score(self, output: str, context: CheckContext) -> float:
        bn, cn = _nums(context.baseline_output), _nums(output)
        if not bn or not cn:
            return 1.0 if not bn and not cn else 0.0
        return _ratio_match(statistics.fmean(bn), statistics.fmean(cn))


class NumberMedianCheck:
    @property
    def name(self) -> str:
        return "number_median"

    def score(self, output: str, context: CheckContext) -> float:
        bn, cn = _nums(context.baseline_output), _nums(output)
        if not bn or not cn:
            return 1.0 if not bn and not cn else 0.0
        return _ratio_match(statistics.median(bn), statistics.median(cn))


class NumberMinCheck:
    @property
    def name(self) -> str:
        return "number_min"

    def score(self, output: str, context: CheckContext) -> float:
        bn, cn = _nums(context.baseline_output), _nums(output)
        if not bn or not cn:
            return 1.0 if not bn and not cn else 0.0
        return _ratio_match(min(bn), min(cn))


class NumberMaxCheck:
    @property
    def name(self) -> str:
        return "number_max"

    def score(self, output: str, context: CheckContext) -> float:
        bn, cn = _nums(context.baseline_output), _nums(output)
        if not bn or not cn:
            return 1.0 if not bn and not cn else 0.0
        return _ratio_match(max(bn), max(cn))


class NumberStdCheck:
    @property
    def name(self) -> str:
        return "number_std"

    def score(self, output: str, context: CheckContext) -> float:
        bn, cn = _nums(context.baseline_output), _nums(output)
        if len(bn) < 2 or len(cn) < 2:
            return 1.0 if len(bn) < 2 and len(cn) < 2 else 0.0
        return _ratio_match(statistics.pstdev(bn), statistics.pstdev(cn))


class NumberRangeCheck:
    @property
    def name(self) -> str:
        return "number_range"

    def score(self, output: str, context: CheckContext) -> float:
        def rng(ns: list[float]) -> float:
            return max(ns) - min(ns) if ns else 0.0

        base, cand = rng(_nums(context.baseline_output)), rng(_nums(output))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class IntegerCountCheck:
    @property
    def name(self) -> str:
        return "int_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INT_RE.findall(_s(context.baseline_output))),
            len(_INT_RE.findall(_s(output))),
        )


class FloatCountCheck:
    @property
    def name(self) -> str:
        return "float_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FLOAT_RE.findall(_s(context.baseline_output))),
            len(_FLOAT_RE.findall(_s(output))),
        )


class NegativeCountCheck:
    @property
    def name(self) -> str:
        return "negative_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            sum(1 for n in _nums(context.baseline_output) if n < 0),
            sum(1 for n in _nums(output) if n < 0),
        )


class PercentCountCheck:
    @property
    def name(self) -> str:
        return "percent_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PCT_RE.findall(_s(context.baseline_output))),
            len(_PCT_RE.findall(_s(output))),
        )


class CurrencyCountCheck:
    @property
    def name(self) -> str:
        return "currency_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CUR_RE.findall(_s(context.baseline_output))),
            len(_CUR_RE.findall(_s(output))),
        )


class DateCountCheck:
    @property
    def name(self) -> str:
        return "date_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DATE_RE.findall(_s(context.baseline_output))),
            len(_DATE_RE.findall(_s(output))),
        )


class TimeCountCheck:
    @property
    def name(self) -> str:
        return "time_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TIME_RE.findall(_s(context.baseline_output))),
            len(_TIME_RE.findall(_s(output))),
        )


class NumberPrecisionMatchCheck:
    """1.0 when the multiset of decimal-place counts matches the baseline."""

    @property
    def name(self) -> str:
        return "number_precision"

    def score(self, output: str, context: CheckContext) -> float:
        def precisions(text: str) -> dict[int, int]:
            out: dict[int, int] = {}
            for m in _NUM_RE.finditer(_s(text)):
                tok = m.group(0)
                if "." in tok:
                    frac = tok.split(".")[1]
                    frac = frac.split("e")[0].split("E")[0]
                    p = len(frac)
                else:
                    p = 0
                out[p] = out.get(p, 0) + 1
            return out

        base, cand = precisions(context.baseline_output), precisions(output)
        if not base and not cand:
            return 1.0
        keys = set(base) | set(cand)
        total = sum(base.values()) + sum(cand.values())
        if total == 0:
            return 1.0
        diff = sum(abs(base.get(k, 0) - cand.get(k, 0)) for k in keys)
        return max(0.0, 1.0 - diff / total)


class NumberSetJaccardCheck:
    @property
    def name(self) -> str:
        return "number_set_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {round(n, 6) for n in _nums(context.baseline_output)}
        b = {round(n, 6) for n in _nums(output)}
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class NumberSortedMatchCheck:
    """1.0 when the sorted number sequence matches the baseline exactly."""

    @property
    def name(self) -> str:
        return "number_sorted"

    def score(self, output: str, context: CheckContext) -> float:
        a = sorted(round(n, 6) for n in _nums(context.baseline_output))
        b = sorted(round(n, 6) for n in _nums(output))
        if not a and not b:
            return 1.0
        if len(a) != len(b):
            return max(0.0, 1.0 - abs(len(a) - len(b)) / max(len(a), len(b)))
        match = sum(1 for x, y in zip(a, b, strict=False) if x == y)
        return match / len(a)


class AvgDigitCountCheck:
    """Average number of digits per numeric token."""

    @property
    def name(self) -> str:
        return "avg_digit_count"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            toks = _NUM_RE.findall(_s(text))
            if not toks:
                return 0.0
            total = 0
            for t in toks:
                digits = t.lstrip("-").replace(".", "")
                digits = digits.split("e")[0].split("E")[0]
                total += len(digits)
            return total / len(toks)

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class NumberMagnitudeRatioCheck:
    """1.0 when the log10 magnitude of the number sum matches the baseline."""

    @property
    def name(self) -> str:
        return "number_magnitude"

    def score(self, output: str, context: CheckContext) -> float:
        def mag(text: str) -> float:
            total = abs(sum(_nums(text)))
            return math.log10(total) if total > 0 else 0.0

        base, cand = mag(context.baseline_output), mag(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / max(abs(base), 1e-9))
