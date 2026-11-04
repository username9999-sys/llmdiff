"""Tests for env_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.env_deep import (
    EnvAvgNameLenCheck,
    EnvBlankLineCountCheck,
    EnvBoolValueCountCheck,
    EnvCommentCountDeepCheck,
    EnvDashNameCountCheck,
    EnvDottedNameCountCheck,
    EnvDoubleQuotedCountCheck,
    EnvDuplicateNameCountCheck,
    EnvEmptyValueCountCheck,
    EnvExportPrefixCountCheck,
    EnvInlineCommentCountCheck,
    EnvLongValueCountCheck,
    EnvNumericValueCountCheck,
    EnvQuotedValueCountCheck,
    EnvSecretNameCountCheck,
    EnvSingleQuotedCountCheck,
    EnvSnakeNameCountCheck,
    EnvUpperNameCountCheck,
    EnvUrlValueCountCheck,
    EnvVarCountDeepCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

ENV = (
    "# top comment\n"
    "APP_NAME=myapp\n"
    "export DEBUG=true\n"
    'API_KEY="abcdef123456"\n'
    "DB_URL=https://db.example.com\n"
    "PORT=8080\n"
    "EMPTY=\n"
    "feature.flag=on  # inline\n"
    "my-var=1\n"
    "lower_case=2\n"
    "LONG_SECRET_TOKEN_" + "x" * 40 + "=1\n"
    "APP_NAME=dup\n"
    "\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_counts():
    ctx = _ctx(ENV, ENV)
    assert EnvVarCountDeepCheck().score(ENV, ctx) == 1.0
    assert EnvCommentCountDeepCheck().score(ENV, ctx) == 1.0
    assert EnvBlankLineCountCheck().score(ENV, ctx) == 1.0
    assert EnvInlineCommentCountCheck().score(ENV, ctx) == 1.0
    assert EnvVarCountDeepCheck().score("A=1\n", ctx) < 1.0


def test_quoting_and_export():
    ctx = _ctx(ENV, ENV)
    assert EnvQuotedValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvSingleQuotedCountCheck().score(ENV, ctx) == 1.0
    assert EnvDoubleQuotedCountCheck().score(ENV, ctx) == 1.0
    assert EnvExportPrefixCountCheck().score(ENV, ctx) == 1.0


def test_name_styles():
    ctx = _ctx(ENV, ENV)
    assert EnvUpperNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvSnakeNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvDottedNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvDashNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvAvgNameLenCheck().score(ENV, ctx) == 1.0


def test_values_and_hygiene():
    ctx = _ctx(ENV, ENV)
    assert EnvEmptyValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvSecretNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvUrlValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvNumericValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvBoolValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvLongValueCountCheck().score(ENV, ctx) == 1.0
    assert EnvDuplicateNameCountCheck().score(ENV, ctx) == 1.0
    assert EnvDuplicateNameCountCheck().score("A=1\nB=2\n", ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": ENV,
         "candidate_output": ENV},
        {"id": "b", "prompt": "p", "baseline_output": ENV,
         "candidate_output": "A=1\n"},
    ])
    result = evaluate(corpus, dimensions=("env_var_count_deep", "env_secret_name_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["env_var_count_deep"].losses == 1
    assert dims["env_secret_name_count"].losses == 1
