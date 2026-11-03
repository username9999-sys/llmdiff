"""Tests for code_lang dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code_lang import (
    ArrowFuncCountCheck,
    BraceDensityCheck,
    CIncludeStyleCheck,
    ClassKeywordCountCheck,
    CSharpUsingCheck,
    GoImportStyleCheck,
    InterfaceKeywordCountCheck,
    JavaPackageCheck,
    JsImportStyleCheck,
    KotlinImportCheck,
    LambdaCountCheck,
    LangDetectMatchCheck,
    ParenDensityCheck,
    PhpUseCheck,
    PythonImportStyleCheck,
    RubyRequireCheck,
    RustUseStyleCheck,
    SemicolonDensityCheck,
    ShebangMatchCheck,
    SwiftImportCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

PY = "```python\n{}\n```"
JS = "```javascript\n{}\n```"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_py_imports():
    ctx = _ctx(PY.format("import os\nimport sys"), PY.format("import os\nimport sys"))
    assert PythonImportStyleCheck().score(PY.format("import os\nimport sys"), ctx) == 1.0
    assert PythonImportStyleCheck().score(PY.format("import os"), ctx) < 1.0


def test_js_imports():
    ctx = _ctx(JS.format("import x from 'y'"), JS.format("import x from 'y'"))
    assert JsImportStyleCheck().score(JS.format("import x from 'y'"), ctx) == 1.0
    assert JsImportStyleCheck().score(JS.format("const x = 1"), ctx) < 1.0


def test_go_imports():
    ctx = _ctx("```go\n" + 'import "fmt"\n```', "```go\n" + 'import "fmt"\n```')
    assert GoImportStyleCheck().score('```go\nimport "fmt"\n```', ctx) == 1.0


def test_rust_use():
    ctx = _ctx("```rust\nuse std::io;\n```", "```rust\nuse std::io;\n```")
    assert RustUseStyleCheck().score("```rust\nuse std::io;\n```", ctx) == 1.0


def test_java_package():
    ctx = _ctx("```java\npackage com.x;\n```", "```java\npackage com.x;\n```")
    assert JavaPackageCheck().score("```java\npackage com.x;\n```", ctx) == 1.0


def test_c_includes():
    ctx = _ctx("```c\n#include <stdio.h>\n```", "```c\n#include <stdio.h>\n```")
    assert CIncludeStyleCheck().score("```c\n#include <stdio.h>\n```", ctx) == 1.0
    assert CIncludeStyleCheck().score("```c\nint x;\n```", ctx) < 1.0


def test_ruby_require():
    ctx = _ctx("```ruby\nrequire 'json'\n```", "```ruby\nrequire 'json'\n```")
    assert RubyRequireCheck().score("```ruby\nrequire 'json'\n```", ctx) == 1.0


def test_php_use():
    ctx = _ctx("```php\nuse App\\Model;\n```", "```php\nuse App\\Model;\n```")
    assert PhpUseCheck().score("```php\nuse App\\Model;\n```", ctx) == 1.0


def test_csharp_using():
    ctx = _ctx("```csharp\nusing System;\n```", "```csharp\nusing System;\n```")
    assert CSharpUsingCheck().score("```csharp\nusing System;\n```", ctx) == 1.0


def test_kotlin_imports():
    ctx = _ctx("```kotlin\nimport java.util.*\n```", "```kotlin\nimport java.util.*\n```")
    assert KotlinImportCheck().score("```kotlin\nimport java.util.*\n```", ctx) == 1.0


def test_swift_imports():
    ctx = _ctx("```swift\nimport Foundation\n```", "```swift\nimport Foundation\n```")
    assert SwiftImportCheck().score("```swift\nimport Foundation\n```", ctx) == 1.0


def test_lang_detect():
    ctx = _ctx(PY.format("def f():\n    print(1)"), PY.format("def g():\n    print(2)"))
    assert LangDetectMatchCheck().score(PY.format("def h():\n    print(3)"), ctx) == 1.0
    assert LangDetectMatchCheck().score(JS.format("const x = 1;"), ctx) == 0.0


def test_shebang():
    ctx = _ctx("#!/usr/bin/env python3\nprint(1)", "#!/usr/bin/env python3\nprint(1)")
    assert ShebangMatchCheck().score("#!/usr/bin/env python3\nx=1", ctx) == 1.0
    assert ShebangMatchCheck().score("#!/bin/bash\necho hi", ctx) == 0.0


def test_semicolon_density():
    ctx = _ctx(JS.format("const a = 1;\nconst b = 2;"), JS.format("const a = 1;\nconst b = 2;"))
    assert SemicolonDensityCheck().score(JS.format("const a = 1;\nconst b = 2;"), ctx) == 1.0


def test_brace_density():
    ctx = _ctx(JS.format("function f() { return 1; }"), JS.format("function f() { return 1; }"))
    assert BraceDensityCheck().score(JS.format("function f() { return 1; }"), ctx) == 1.0


def test_paren_density():
    ctx = _ctx(PY.format("f(g(1))"), PY.format("f(g(1))"))
    assert ParenDensityCheck().score(PY.format("f(g(1))"), ctx) == 1.0


def test_arrow_func_count():
    ctx = _ctx(JS.format("const f = () => 1;"), JS.format("const f = () => 1;"))
    assert ArrowFuncCountCheck().score(JS.format("const f = () => 1;"), ctx) == 1.0
    assert ArrowFuncCountCheck().score(JS.format("const f = 1;"), ctx) < 1.0


def test_lambda_count():
    ctx = _ctx(PY.format("f = lambda x: x"), PY.format("f = lambda x: x"))
    assert LambdaCountCheck().score(PY.format("f = lambda x: x"), ctx) == 1.0


def test_class_kw_count():
    ctx = _ctx(PY.format("class A:\n    pass"), PY.format("class A:\n    pass"))
    assert ClassKeywordCountCheck().score(PY.format("class A:\n    pass"), ctx) == 1.0
    assert ClassKeywordCountCheck().score(PY.format("x = 1"), ctx) < 1.0


def test_interface_kw_count():
    ctx = _ctx("```java\ninterface I {}\n```", "```java\ninterface I {}\n```")
    assert InterfaceKeywordCountCheck().score("```java\ninterface I {}\n```", ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": PY.format("import os\nx = 1"),
         "candidate_output": PY.format("import os\nx = 1")},
        {"id": "b", "prompt": "p", "baseline_output": PY.format("import os\nx = 1"),
         "candidate_output": PY.format("x = 1")},
    ])
    result = evaluate(corpus, dimensions=("py_imports", "class_kw_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["py_imports"].losses == 1
    assert dims["class_kw_count"].losses == 0
