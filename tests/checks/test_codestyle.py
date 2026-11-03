"""Tests for codestyle dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.codestyle import (
    AssertCountCheck,
    BlankLineInCodeCheck,
    CamelCaseRatioCheck,
    ClassCountCheck,
    CommentLineRatioCheck,
    DocstringPresenceCheck,
    FenceLanguageMatchCheck,
    FunctionCountCheck,
    ImportCountCheck,
    LongIdentifierRatioCheck,
    MagicNumberRatioCheck,
    NestedCallDepthCheck,
    PascalCaseRatioCheck,
    PrintStatementCountCheck,
    ScreamingCaseRatioCheck,
    SemicolonLineEndCheck,
    ShortIdentifierRatioCheck,
    SingleLetterVarCheck,
    SnakeCaseRatioCheck,
    TryExceptCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

PY = "```python\n{}\n```"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_snake_case_ratio():
    code = "def my_func():\n    my_var = 1"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert SnakeCaseRatioCheck().score(PY.format(code), ctx) == 1.0


def test_camel_case_ratio():
    ctx = _ctx(PY.format("myVar = 1"), PY.format("myVar = 1"))
    assert CamelCaseRatioCheck().score(PY.format("myVar = 1"), ctx) == 1.0
    assert CamelCaseRatioCheck().score(PY.format("my_var = 1"), ctx) < 1.0


def test_pascal_case_ratio():
    code = "class MyClass:\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert PascalCaseRatioCheck().score(PY.format(code), ctx) == 1.0


def test_screaming_case_ratio():
    ctx = _ctx(PY.format("MAX_SIZE = 10"), PY.format("MAX_SIZE = 10"))
    assert ScreamingCaseRatioCheck().score(PY.format("MAX_SIZE = 10"), ctx) == 1.0


def test_comment_line_ratio():
    ctx = _ctx(PY.format("# comment\nx = 1"), PY.format("# comment\nx = 1"))
    assert CommentLineRatioCheck().score(PY.format("# comment\nx = 1"), ctx) == 1.0
    assert CommentLineRatioCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_docstring_presence():
    with_doc = 'def f():\n    """doc"""\n    pass'
    ctx = _ctx(PY.format(with_doc), PY.format("def f():\n    pass"))
    assert DocstringPresenceCheck().score(PY.format(with_doc), ctx) == 1.0
    assert DocstringPresenceCheck().score(PY.format("def f():\n    pass"), ctx) == 0.0


def test_import_count():
    ctx = _ctx(PY.format("import os\nimport sys"), PY.format("import os\nimport sys"))
    assert ImportCountCheck().score(PY.format("import os\nimport sys"), ctx) == 1.0
    assert ImportCountCheck().score(PY.format("import os"), ctx) < 1.0


def test_function_count():
    two = "def a():\n    pass\ndef b():\n    pass"
    ctx = _ctx(PY.format(two), PY.format("def a():\n    pass"))
    assert FunctionCountCheck().score(PY.format(two), ctx) == 1.0
    assert FunctionCountCheck().score(PY.format("def a():\n    pass"), ctx) < 1.0


def test_class_count():
    ctx = _ctx(PY.format("class A:\n    pass"), PY.format("class A:\n    pass\nclass B:\n    pass"))
    assert ClassCountCheck().score(PY.format("class A:\n    pass"), ctx) == 1.0
    assert ClassCountCheck().score(PY.format("class A:\n    pass\nclass B:\n    pass"), ctx) < 1.0


def test_magic_number_ratio():
    ctx = _ctx(PY.format("x = 42"), PY.format("x = 42"))
    assert MagicNumberRatioCheck().score(PY.format("x = 42"), ctx) == 1.0


def test_blank_line_in_code():
    ctx = _ctx(PY.format("x = 1\n\ny = 2"), PY.format("x = 1\n\ny = 2"))
    assert BlankLineInCodeCheck().score(PY.format("x = 1\n\ny = 2"), ctx) == 1.0


def test_semicolon_line_end():
    ctx = _ctx(PY.format("x = 1;"), PY.format("x = 1;"))
    assert SemicolonLineEndCheck().score(PY.format("x = 1;"), ctx) == 1.0
    assert SemicolonLineEndCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_single_letter_vars():
    ctx = _ctx(PY.format("x = 1\ny = 2"), PY.format("x = 1\ny = 2"))
    assert SingleLetterVarCheck().score(PY.format("x = 1\ny = 2"), ctx) == 1.0


def test_long_ident_ratio():
    code = "very_long_identifier_name_here = 1"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert LongIdentifierRatioCheck().score(PY.format(code), ctx) == 1.0


def test_short_ident_ratio():
    ctx = _ctx(PY.format("x = 1"), PY.format("x = 1"))
    assert ShortIdentifierRatioCheck().score(PY.format("x = 1"), ctx) == 1.0


def test_nested_call_depth():
    ctx = _ctx(PY.format("f(g(h(1)))"), PY.format("f(g(h(1)))"))
    assert NestedCallDepthCheck().score(PY.format("f(g(h(1)))"), ctx) == 1.0
    assert NestedCallDepthCheck().score(PY.format("f(1)"), ctx) < 1.0


def test_print_count():
    ctx = _ctx(PY.format("print('a')\nprint('b')"), PY.format("print('a')\nprint('b')"))
    assert PrintStatementCountCheck().score(PY.format("print('a')\nprint('b')"), ctx) == 1.0
    assert PrintStatementCountCheck().score(PY.format("print('a')"), ctx) < 1.0


def test_assert_count():
    ctx = _ctx(PY.format("assert x"), PY.format("assert x"))
    assert AssertCountCheck().score(PY.format("assert x"), ctx) == 1.0
    assert AssertCountCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_try_except_count():
    code = "try:\n    pass\nexcept Exception:\n    pass"
    ctx = _ctx(PY.format(code), PY.format(code))
    assert TryExceptCountCheck().score(PY.format(code), ctx) == 1.0


def test_fence_language():
    ctx = _ctx("```python\nx=1\n```", "```python\nx=1\n```")
    assert FenceLanguageMatchCheck().score("```python\nx=1\n```", ctx) == 1.0
    assert FenceLanguageMatchCheck().score("```js\nx=1\n```", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": PY.format("def f():\n    return 1"),
         "candidate_output": PY.format("def f():\n    return 1")},
        {"id": "b", "prompt": "p", "baseline_output": PY.format("def f():\n    return 1"),
         "candidate_output": PY.format("def g():\n    return 2\ndef h():\n    return 3")},
    ])
    result = evaluate(corpus, dimensions=("function_count", "import_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["function_count"].losses == 1
    assert dims["import_count"].losses == 0
