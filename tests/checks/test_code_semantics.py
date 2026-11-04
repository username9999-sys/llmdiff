"""Tests for code_semantics dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code_semantics import (
    ArgCountMatchCheck,
    AssignmentCountCheck,
    AsyncAwaitCountCheck,
    CallCountCheck,
    ComparisonCountCheck,
    ConditionalCountCheck,
    DecoratorCountCheck,
    ElifCountCheck,
    GlobalNonlocalCountCheck,
    IdentifierSetJaccardCheck,
    KeywordCountCheck,
    LoopCountCheck,
    MatchCaseCountCheck,
    OperatorDensityCheck,
    RaiseCountCheck,
    ReturnCountCheck,
    StringLiteralCountCheck,
    StringLiteralOverlapCheck,
    WithCountCheck,
    YieldCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

PY = "```python\n{}\n```"
CODE = "def f(a, b):\n    x = a + b\n    if x > 0:\n        return x\n    return -x"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_ident_jaccard():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert IdentifierSetJaccardCheck().score(PY.format(CODE), ctx) == 1.0
    assert IdentifierSetJaccardCheck().score(PY.format("def g():\n    pass"), ctx) < 1.0


def test_str_literal_count():
    ctx = _ctx(PY.format('x = "a" + "b"'), PY.format('x = "a" + "b"'))
    assert StringLiteralCountCheck().score(PY.format('x = "a" + "b"'), ctx) == 1.0
    assert StringLiteralCountCheck().score(PY.format('x = "a"'), ctx) < 1.0


def test_str_literal_overlap():
    ctx = _ctx(PY.format('x = "a"'), PY.format('x = "a"'))
    assert StringLiteralOverlapCheck().score(PY.format('x = "a"'), ctx) == 1.0
    assert StringLiteralOverlapCheck().score(PY.format('x = "z"'), ctx) == 0.0


def test_keyword_count():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert KeywordCountCheck().score(PY.format(CODE), ctx) == 1.0


def test_operator_density():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert OperatorDensityCheck().score(PY.format(CODE), ctx) == 1.0


def test_assign_compare_count():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert AssignmentCountCheck().score(PY.format(CODE), ctx) == 1.0
    assert ComparisonCountCheck().score(PY.format(CODE), ctx) == 1.0


def test_loop_cond_count():
    code = "for i in range(3):\n    if i:\n        print(i)"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert LoopCountCheck().score(PY.format(code), ctx) == 1.0
    assert ConditionalCountCheck().score(PY.format(code), ctx) == 1.0


def test_return_call_count():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert ReturnCountCheck().score(PY.format(CODE), ctx) == 1.0
    cctx = _ctx(PY.format("print(1, 2)"), PY.format("print(1, 2)"))
    assert CallCountCheck().score(PY.format("print(1, 2)"), cctx) == 1.0


def test_arg_count():
    ctx = _ctx(PY.format("f(1, 2, 3)"), PY.format("f(1, 2, 3)"))
    assert ArgCountMatchCheck().score(PY.format("f(1, 2, 3)"), ctx) == 1.0
    assert ArgCountMatchCheck().score(PY.format("f(1)"), ctx) < 1.0


def test_decorator_count():
    code = "@property\ndef x(self):\n    return 1"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert DecoratorCountCheck().score(PY.format(code), ctx) == 1.0


def test_yield_raise_count():
    ycode = "def g():\n    yield 1"
    yctx = _ctx(PY.format(ycode), PY.format(ycode))
    assert YieldCountCheck().score(PY.format(ycode), yctx) == 1.0
    rcode = "raise ValueError('x')"
    rctx = _ctx(PY.format(rcode), PY.format(rcode))
    assert RaiseCountCheck().score(PY.format(rcode), rctx) == 1.0


def test_with_global_count():
    wcode = "with open('f') as fh:\n    pass"
    wctx = _ctx(PY.format(wcode), PY.format(wcode))
    assert WithCountCheck().score(PY.format(wcode), wctx) == 1.0
    gcode = "def f():\n    global x"
    gctx = _ctx(PY.format(gcode), PY.format(gcode))
    assert GlobalNonlocalCountCheck().score(PY.format(gcode), gctx) == 1.0


def test_async_await_count():
    code = "async def f():\n    await g()"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert AsyncAwaitCountCheck().score(PY.format(code), ctx) == 1.0


def test_elif_match_count():
    code = "if a:\n    pass\nelif b:\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert ElifCountCheck().score(PY.format(code), ctx) == 1.0
    mcode = "match x:\n    case 1:\n        pass"
    mctx = _ctx(PY.format(mcode), PY.format(mcode))
    assert MatchCaseCountCheck().score(PY.format(mcode), mctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": PY.format(CODE),
         "candidate_output": PY.format(CODE)},
        {"id": "b", "prompt": "p", "baseline_output": PY.format(CODE),
         "candidate_output": PY.format("def g():\n    return 0")},
    ])
    result = evaluate(corpus, dimensions=("ident_jaccard", "return_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["ident_jaccard"].losses == 1
    assert dims["return_count"].losses == 1
