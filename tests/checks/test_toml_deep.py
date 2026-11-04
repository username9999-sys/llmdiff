"""Tests for toml_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.toml_deep import (
    TomlArrayTableCountCheck,
    TomlArrayValueCountCheck,
    TomlBasicStringCountCheck,
    TomlBoolValueCountCheck,
    TomlCommentCountCheck,
    TomlDateValueCountCheck,
    TomlDottedKeyCountCheck,
    TomlEmptyTableCountCheck,
    TomlFloatValueCountCheck,
    TomlIndentPresenceCheck,
    TomlInlineTableCountCheck,
    TomlIntegerValueCountCheck,
    TomlKeyCountDeepCheck,
    TomlLiteralStringCountCheck,
    TomlMaxTableDepthCheck,
    TomlMultilineStringCountCheck,
    TomlNestedArrayCountCheck,
    TomlQuotedKeyCountCheck,
    TomlStringValueCountCheck,
    TomlTableHeaderCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

TOML = (
    "# comment\n"
    "[package]\n"
    'name = "app"\n'
    'version = "1.0"\n'
    "[server]\n"
    "host = 'localhost'\n"
    "port = 8080\n"
    "ratio = 0.5\n"
    "enabled = true\n"
    "date = 2026-01-01\n"
    'tags = ["a", "b"]\n'
    "matrix = [[1, 2], [3, 4]]\n"
    "point = { x = 1, y = 2 }\n"
    "a.b = 1\n"
    '"quoted key" = 1\n'
    'ml = """\nline1\n"""\n'
    "[[products]]\n"
    'name = "x"\n'
    "[[products]]\n"
    'name = "y"\n'
    "[empty]\n"
    "[server.advanced]\n"
    "opt = 1\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_tables_and_arrays():
    ctx = _ctx(TOML, TOML)
    assert TomlTableHeaderCountCheck().score(TOML, ctx) == 1.0
    assert TomlArrayTableCountCheck().score(TOML, ctx) == 1.0
    assert TomlInlineTableCountCheck().score(TOML, ctx) == 1.0
    assert TomlMaxTableDepthCheck().score(TOML, ctx) == 1.0
    assert TomlEmptyTableCountCheck().score(TOML, ctx) == 1.0


def test_keys_and_comments():
    ctx = _ctx(TOML, TOML)
    assert TomlDottedKeyCountCheck().score(TOML, ctx) == 1.0
    assert TomlCommentCountCheck().score(TOML, ctx) == 1.0
    assert TomlKeyCountDeepCheck().score(TOML, ctx) == 1.0
    assert TomlQuotedKeyCountCheck().score(TOML, ctx) == 1.0
    assert TomlIndentPresenceCheck().score(TOML, ctx) == 1.0


def test_value_types():
    ctx = _ctx(TOML, TOML)
    assert TomlStringValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlBasicStringCountCheck().score(TOML, ctx) == 1.0
    assert TomlLiteralStringCountCheck().score(TOML, ctx) == 1.0
    assert TomlMultilineStringCountCheck().score(TOML, ctx) == 1.0
    assert TomlIntegerValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlFloatValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlBoolValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlDateValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlArrayValueCountCheck().score(TOML, ctx) == 1.0
    assert TomlNestedArrayCountCheck().score(TOML, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": TOML,
         "candidate_output": TOML},
        {"id": "b", "prompt": "p", "baseline_output": TOML,
         "candidate_output": "key = 1\n"},
    ])
    result = evaluate(corpus, dimensions=("toml_table_count", "toml_key_count_deep"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["toml_table_count"].losses == 1
    assert dims["toml_key_count_deep"].losses == 1
