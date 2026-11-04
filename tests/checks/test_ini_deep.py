"""Tests for ini_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.ini_deep import (
    IniAvgKeyNameLenCheck,
    IniBlankLineCountCheck,
    IniBoolValueCountCheck,
    IniBracketSectionCountCheck,
    IniCommentCountCheck,
    IniDuplicateKeyCountCheck,
    IniDuplicateSectionCountCheck,
    IniEmptyValueCountCheck,
    IniInlineCommentCountCheck,
    IniKeyCountDeepCheck,
    IniKeyNameJaccardCheck,
    IniMaxSectionNameLenCheck,
    IniNoSectionKeyCountCheck,
    IniNumericValueCountCheck,
    IniPathValueCountCheck,
    IniQuotedValueCountCheck,
    IniSectionCountDeepCheck,
    IniSectionNameJaccardCheck,
    IniSemicolonCommentCountCheck,
    IniUrlValueCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

INI = (
    "; semicolon comment\n"
    "# hash comment\n"
    "GLOBAL_KEY=1\n"
    "\n"
    "[server]\n"
    'host = "localhost"\n'
    "port = 8080\n"
    "debug = true\n"
    "empty =\n"
    "url = https://x.com\n"
    "path = /etc/app\n"
    "note = hi  # inline\n"
    "[server]\n"
    "dup = 1\n"
    "dup = 2\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_sections_and_keys():
    ctx = _ctx(INI, INI)
    assert IniSectionCountDeepCheck().score(INI, ctx) == 1.0
    assert IniKeyCountDeepCheck().score(INI, ctx) == 1.0
    assert IniBracketSectionCountCheck().score(INI, ctx) == 1.0
    assert IniNoSectionKeyCountCheck().score(INI, ctx) == 1.0
    assert IniSectionNameJaccardCheck().score(INI, ctx) == 1.0
    assert IniKeyNameJaccardCheck().score(INI, ctx) == 1.0
    assert IniSectionNameJaccardCheck().score("[other]\nx=1\n", ctx) < 1.0


def test_comments_and_blanks():
    ctx = _ctx(INI, INI)
    assert IniCommentCountCheck().score(INI, ctx) == 1.0
    assert IniSemicolonCommentCountCheck().score(INI, ctx) == 1.0
    assert IniInlineCommentCountCheck().score(INI, ctx) == 1.0
    assert IniBlankLineCountCheck().score(INI, ctx) == 1.0


def test_value_shapes():
    ctx = _ctx(INI, INI)
    assert IniQuotedValueCountCheck().score(INI, ctx) == 1.0
    assert IniEmptyValueCountCheck().score(INI, ctx) == 1.0
    assert IniBoolValueCountCheck().score(INI, ctx) == 1.0
    assert IniNumericValueCountCheck().score(INI, ctx) == 1.0
    assert IniUrlValueCountCheck().score(INI, ctx) == 1.0
    assert IniPathValueCountCheck().score(INI, ctx) == 1.0


def test_dups_and_name_stats():
    ctx = _ctx(INI, INI)
    assert IniDuplicateSectionCountCheck().score(INI, ctx) == 1.0
    assert IniDuplicateKeyCountCheck().score(INI, ctx) == 1.0
    assert IniDuplicateKeyCountCheck().score("[a]\nx=1\ny=2\n", ctx) < 1.0
    assert IniMaxSectionNameLenCheck().score(INI, ctx) == 1.0
    assert IniAvgKeyNameLenCheck().score(INI, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": INI,
         "candidate_output": INI},
        {"id": "b", "prompt": "p", "baseline_output": INI,
         "candidate_output": "[only]\nk=1\n"},
    ])
    result = evaluate(corpus, dimensions=("ini_section_count", "ini_key_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["ini_section_count"].losses == 1
    assert dims["ini_key_count"].losses == 1
