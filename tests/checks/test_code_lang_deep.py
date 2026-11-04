"""Tests for code_lang_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code_lang_deep import (
    CSharpAsyncCountCheck,
    GoChannelCountCheck,
    GoDeferCountCheck,
    GoErrorCheckCountCheck,
    GoGoroutineCountCheck,
    JavaAnnotationCountCheck,
    JavaStreamCountCheck,
    JsArrowFuncCountCheck,
    JsConstCountCheck,
    JsDestructuringCountCheck,
    JsSpreadCountCheck,
    JsTemplateLiteralCountCheck,
    PythonAsyncDefCountCheck,
    PythonDecoratorCountCheck,
    PythonFstringCountCheck,
    PythonListcompCountCheck,
    PythonTypeHintCountCheck,
    RustLifetimeCountCheck,
    RustMatchArmCountCheck,
    RustResultUnwrapCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

PY = "```python\n{}\n```"
JS = "```javascript\n{}\n```"
GO = "```go\n{}\n```"
RS = "```rust\n{}\n```"
JAVA = "```java\n{}\n```"
CS = "```csharp\n{}\n```"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_python_idioms():
    code = 'x = f"hi {name}"\ny = [i for i in range(3)]\n@property\ndef f(a: int) -> int:\n    pass'
    ctx = _ctx(PY.format(code), PY.format(code))
    assert PythonFstringCountCheck().score(PY.format(code), ctx) == 1.0
    assert PythonListcompCountCheck().score(PY.format(code), ctx) == 1.0
    assert PythonDecoratorCountCheck().score(PY.format(code), ctx) == 1.0
    assert PythonTypeHintCountCheck().score(PY.format(code), ctx) == 1.0
    acode = "async def f():\n    pass"
    actx = _ctx(PY.format(acode), PY.format(acode))
    assert PythonAsyncDefCountCheck().score(PY.format(acode), actx) == 1.0


def test_js_idioms():
    code = "const f = (a) => `v${a}`;\nconst {x} = obj;\nfoo(...args);"
    ctx = _ctx(JS.format(code), JS.format(code))
    assert JsConstCountCheck().score(JS.format(code), ctx) == 1.0
    assert JsArrowFuncCountCheck().score(JS.format(code), ctx) == 1.0
    assert JsTemplateLiteralCountCheck().score(JS.format(code), ctx) == 1.0
    assert JsDestructuringCountCheck().score(JS.format(code), ctx) == 1.0
    assert JsSpreadCountCheck().score(JS.format(code), ctx) == 1.0


def test_go_idioms():
    code = "go f()\ndefer g()\nch := make(chan int)\nif err != nil {\n}"
    ctx = _ctx(GO.format(code), GO.format(code))
    assert GoGoroutineCountCheck().score(GO.format(code), ctx) == 1.0
    assert GoDeferCountCheck().score(GO.format(code), ctx) == 1.0
    assert GoChannelCountCheck().score(GO.format(code), ctx) == 1.0
    assert GoErrorCheckCountCheck().score(GO.format(code), ctx) == 1.0


def test_rust_idioms():
    code = "match x {\n    1 => a,\n    _ => b,\n}\nlet y = r.unwrap();\nfn f<'a>(s: &'a str) {}"
    ctx = _ctx(RS.format(code), RS.format(code))
    assert RustMatchArmCountCheck().score(RS.format(code), ctx) == 1.0
    assert RustResultUnwrapCountCheck().score(RS.format(code), ctx) == 1.0
    assert RustLifetimeCountCheck().score(RS.format(code), ctx) == 1.0


def test_java_csharp_idioms():
    jcode = "@Override\npublic void x() { list.stream().map(y -> y).collect(); }"
    jctx = _ctx(JAVA.format(jcode), JAVA.format(jcode))
    assert JavaAnnotationCountCheck().score(JAVA.format(jcode), jctx) == 1.0
    assert JavaStreamCountCheck().score(JAVA.format(jcode), jctx) == 1.0
    ccode = "async Task F() { await G(); }"
    cctx = _ctx(CS.format(ccode), CS.format(ccode))
    assert CSharpAsyncCountCheck().score(CS.format(ccode), cctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": PY.format('x = f"{y}"'),
         "candidate_output": PY.format('x = f"{y}"')},
        {"id": "b", "prompt": "p", "baseline_output": PY.format('x = f"{y}"'),
         "candidate_output": PY.format("x = 1")},
    ])
    result = evaluate(corpus, dimensions=("py_fstring_count", "py_decorator_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["py_fstring_count"].losses == 1
    assert dims["py_decorator_count"].losses == 0
