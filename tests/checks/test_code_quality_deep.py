"""Tests for code_quality_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code_quality_deep import (
    AvgFunctionLengthCheck,
    BareExceptCountCheck,
    BooleanParamCountCheck,
    CommentedOutCodeCountCheck,
    CyclomaticComplexityCheck,
    DeepConditionalCountCheck,
    EvalExecCountCheck,
    FunctionCountDensityCheck,
    GlobalVarCountCheck,
    HardcodedIpCountCheck,
    HardcodedUrlCountCheck,
    LongParameterListCountCheck,
    MagicStringCountCheck,
    MaxFunctionLengthCheck,
    MaxNestingDepthCheck,
    MutableDefaultCountCheck,
    PrintDebugCountCheck,
    SleepCallCountCheck,
    TodoFixmeCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

PY = "```python\n{}\n```"
CODE = (
    "def f(a, b):\n"
    "    if a:\n"
    "        if b:\n"
    "            return 'result'\n"
    "    return None"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_cyclomatic():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert CyclomaticComplexityCheck().score(PY.format(CODE), ctx) == 1.0
    assert CyclomaticComplexityCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_max_nesting():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert MaxNestingDepthCheck().score(PY.format(CODE), ctx) == 1.0


def test_func_lengths():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert AvgFunctionLengthCheck().score(PY.format(CODE), ctx) == 1.0
    assert MaxFunctionLengthCheck().score(PY.format(CODE), ctx) == 1.0


def test_func_density():
    ctx = _ctx(PY.format(CODE), PY.format(CODE))
    assert FunctionCountDensityCheck().score(PY.format(CODE), ctx) == 1.0


def test_todo_fixme():
    ctx = _ctx(PY.format("# TODO: fix\nx = 1"), PY.format("# TODO: fix\nx = 1"))
    assert TodoFixmeCountCheck().score(PY.format("# TODO a\n# FIXME b"), ctx) < 1.0
    assert TodoFixmeCountCheck().score(PY.format("# TODO: fix\nx = 1"), ctx) == 1.0


def test_hardcoded_ip_url():
    ctx = _ctx(PY.format('x = "192.168.1.1"'), PY.format('x = "192.168.1.1"'))
    assert HardcodedIpCountCheck().score(PY.format('x = "192.168.1.1"'), ctx) == 1.0
    noip = _ctx(PY.format("x = 1"), PY.format("x = 1"))
    assert HardcodedIpCountCheck().score(PY.format('x = "10.0.0.1"'), noip) < 1.0
    uctx = _ctx(PY.format('u = "https://api.x.com"'), PY.format('u = "https://api.x.com"'))
    assert HardcodedUrlCountCheck().score(PY.format('u = "https://api.x.com"'), uctx) == 1.0


def test_magic_string():
    ctx = _ctx(PY.format('x = "hello world"'), PY.format('x = "hello world"'))
    assert MagicStringCountCheck().score(PY.format('x = "hello world"'), ctx) == 1.0
    assert MagicStringCountCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_long_param_list():
    code = "def f(a, b, c, d, e, f, g):\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert LongParameterListCountCheck().score(PY.format(code), ctx) == 1.0
    assert LongParameterListCountCheck().score(PY.format("def f(a):\n    pass"), ctx) < 1.0


def test_bool_param():
    code = "def f(debug: bool):\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert BooleanParamCountCheck().score(PY.format(code), ctx) == 1.0


def test_deep_conditional():
    code = "if a:\n    if b:\n        if c:\n            if d:\n                pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert DeepConditionalCountCheck().score(PY.format(code), ctx) == 1.0


def test_goto_global():
    ctx = _ctx(PY.format("def f():\n    global x"), PY.format("def f():\n    global x"))
    assert GlobalVarCountCheck().score(PY.format("def f():\n    global x"), ctx) == 1.0


def test_mutable_default():
    code = "def f(x=[]):\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert MutableDefaultCountCheck().score(PY.format(code), ctx) == 1.0


def test_bare_except():
    code = "try:\n    pass\nexcept:\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert BareExceptCountCheck().score(PY.format(code), ctx) == 1.0


def test_print_sleep_eval():
    ctx = _ctx(PY.format("print('x')"), PY.format("print('x')"))
    assert PrintDebugCountCheck().score(PY.format("print('x')"), ctx) == 1.0
    sctx = _ctx(PY.format("import time\ntime.sleep(1)"), PY.format("import time\ntime.sleep(1)"))
    assert SleepCallCountCheck().score(PY.format("import time\ntime.sleep(1)"), sctx) == 1.0
    ectx = _ctx(PY.format("eval('1+1')"), PY.format("eval('1+1')"))
    assert EvalExecCountCheck().score(PY.format("eval('1+1')"), ectx) == 1.0


def test_commented_out_code():
    code = "# def old():\n#     return 1\nx = 1"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert CommentedOutCodeCountCheck().score(PY.format(code), ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": PY.format(CODE),
         "candidate_output": PY.format(CODE)},
        {"id": "b", "prompt": "p", "baseline_output": PY.format(CODE),
         "candidate_output": PY.format("x = 1")},
    ])
    result = evaluate(corpus, dimensions=("cyclomatic", "todo_fixme_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["cyclomatic"].losses == 1
    assert dims["todo_fixme_count"].losses == 0
