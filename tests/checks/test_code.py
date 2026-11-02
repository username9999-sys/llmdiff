"""Tests for code-output dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.code import (
    BraceBalanceCheck,
    CodeBlockCountCheck,
    CodeHasDefinitionsCheck,
    FencedCodeCheck,
    PythonSyntaxCheck,
    extract_fenced_blocks,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_extract_fenced_blocks():
    text = "here:\n```python\nprint(1)\n```\nand\n```\nplain\n```\n"
    blocks = extract_fenced_blocks(text)
    assert blocks == [("python", "print(1)\n"), ("", "plain\n")]


def test_python_syntax_valid():
    ctx = _ctx("x = 1", "def f():\n    return 1\n")
    assert PythonSyntaxCheck().score("def f():\n    return 1\n", ctx) == 1.0


def test_python_syntax_invalid():
    ctx = _ctx("x = 1", "def broken(:\n")
    assert PythonSyntaxCheck().score("def broken(:\n", ctx) == 0.0


def test_python_syntax_fenced():
    ctx = _ctx("", "```python\ndef ok():\n    pass\n```")
    assert PythonSyntaxCheck().score("```python\ndef ok():\n    pass\n```", ctx) == 1.0


def test_brace_balance():
    ctx = _ctx("{}", "{}")
    assert BraceBalanceCheck().score("{}", ctx) == 1.0
    assert BraceBalanceCheck().score("{", ctx) < 1.0
    assert BraceBalanceCheck().score("", ctx) == 1.0


def test_code_block_count():
    ctx = _ctx("```\na\n```", "```\na\n```")
    assert CodeBlockCountCheck().score("```\na\n```", ctx) == 1.0
    assert CodeBlockCountCheck().score("no blocks", ctx) < 1.0


def test_definitions():
    ctx = _ctx("def f(): pass", "def g(): pass")
    assert CodeHasDefinitionsCheck().score("def g(): pass", ctx) == 1.0
    assert CodeHasDefinitionsCheck().score("plain text", ctx) == 0.0


def test_fenced_code():
    ctx = _ctx("", "```\nx\n```")
    assert FencedCodeCheck().score("```\nx\n```", ctx) == 1.0
    assert FencedCodeCheck().score("plain", ctx) == 0.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "def f():\n    return 1",
         "candidate_output": "def f():\n    return 1"},
        {"id": "b", "prompt": "p", "baseline_output": "def f():\n    return 1",
         "candidate_output": "def broken(:\n"},
    ])
    result = evaluate(corpus, dimensions=("python_syntax", "brace_balance"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["python_syntax"].losses == 1
    assert dims["brace_balance"].losses == 1
