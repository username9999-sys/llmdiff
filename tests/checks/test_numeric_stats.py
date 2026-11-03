"""Tests for numeric_stats dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.numeric_stats import (
    AvgDigitCountCheck,
    CurrencyCountCheck,
    DateCountCheck,
    FloatCountCheck,
    IntegerCountCheck,
    NegativeCountCheck,
    NumberCountCheck,
    NumberMagnitudeRatioCheck,
    NumberMaxCheck,
    NumberMeanCheck,
    NumberMedianCheck,
    NumberMinCheck,
    NumberPrecisionMatchCheck,
    NumberRangeCheck,
    NumberSetJaccardCheck,
    NumberSortedMatchCheck,
    NumberStdCheck,
    NumberSumCheck,
    PercentCountCheck,
    TimeCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_number_count():
    ctx = _ctx("a 1 b 2 c 3", "a 1 b 2 c 3")
    assert NumberCountCheck().score("x 10 y 20 z 30", ctx) == 1.0
    assert NumberCountCheck().score("x 10", ctx) < 1.0


def test_number_sum():
    ctx = _ctx("1 2 3", "1 2 3")
    assert NumberSumCheck().score("1 2 3", ctx) == 1.0
    assert NumberSumCheck().score("1 2 9", ctx) < 1.0


def test_number_mean():
    ctx = _ctx("2 4 6", "2 4 6")
    assert NumberMeanCheck().score("2 4 6", ctx) == 1.0


def test_number_median():
    ctx = _ctx("1 5 9", "1 5 9")
    assert NumberMedianCheck().score("1 5 9", ctx) == 1.0


def test_number_min_max():
    ctx = _ctx("3 7 9", "3 7 9")
    assert NumberMinCheck().score("3 7 9", ctx) == 1.0
    assert NumberMaxCheck().score("3 7 9", ctx) == 1.0
    assert NumberMaxCheck().score("1 2 3", ctx) < 1.0


def test_number_std():
    ctx = _ctx("1 2 3 4 5", "1 2 3 4 5")
    assert NumberStdCheck().score("1 2 3 4 5", ctx) == 1.0


def test_number_range():
    ctx = _ctx("1 10", "1 10")
    assert NumberRangeCheck().score("1 10", ctx) == 1.0
    assert NumberRangeCheck().score("5 6", ctx) < 1.0


def test_int_float_count():
    ctx = _ctx("1 2 3", "1 2 3")
    assert IntegerCountCheck().score("1 2 3", ctx) == 1.0
    assert IntegerCountCheck().score("1 2", ctx) < 1.0
    fctx = _ctx("2.5 3.5", "2.5 3.5")
    assert FloatCountCheck().score("2.5 3.5", fctx) == 1.0
    assert FloatCountCheck().score("2.5", fctx) < 1.0


def test_negative_count():
    ctx = _ctx("-1 5 -2", "-1 5 -2")
    assert NegativeCountCheck().score("-1 5 -2", ctx) == 1.0
    assert NegativeCountCheck().score("1 5 2", ctx) < 1.0


def test_percent_count():
    ctx = _ctx("10% 20%", "10% 20%")
    assert PercentCountCheck().score("10% 20%", ctx) == 1.0
    assert PercentCountCheck().score("10%", ctx) < 1.0


def test_currency_count():
    ctx = _ctx("$100 $200", "$100 $200")
    assert CurrencyCountCheck().score("$100 $200", ctx) == 1.0
    assert CurrencyCountCheck().score("$100", ctx) < 1.0


def test_date_time_count():
    ctx = _ctx("2026-01-01 10:30", "2026-01-01 10:30")
    assert DateCountCheck().score("2026-01-01", ctx) == 1.0
    assert TimeCountCheck().score("10:30", ctx) == 1.0


def test_number_precision():
    ctx = _ctx("1.5 2.25", "1.5 2.25")
    assert NumberPrecisionMatchCheck().score("1.5 2.25", ctx) == 1.0


def test_number_set_jaccard():
    ctx = _ctx("1 2 3", "1 2 3")
    assert NumberSetJaccardCheck().score("1 2 3", ctx) == 1.0
    assert NumberSetJaccardCheck().score("1 2 9", ctx) < 1.0


def test_number_sorted():
    ctx = _ctx("3 1 2", "3 1 2")
    assert NumberSortedMatchCheck().score("1 2 3", ctx) == 1.0
    assert NumberSortedMatchCheck().score("9 8 7", ctx) < 1.0


def test_avg_digit_count():
    ctx = _ctx("12 345", "12 345")
    assert AvgDigitCountCheck().score("12 345", ctx) == 1.0


def test_number_magnitude():
    ctx = _ctx("100 200", "100 200")
    assert NumberMagnitudeRatioCheck().score("100 200", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "total 100 200",
         "candidate_output": "total 100 200"},
        {"id": "b", "prompt": "p", "baseline_output": "total 100 200",
         "candidate_output": "total 5"},
    ])
    result = evaluate(corpus, dimensions=("number_count", "number_sum"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["number_count"].losses == 1
    assert dims["number_sum"].losses == 1
