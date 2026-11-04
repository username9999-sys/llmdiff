"""Tests for csv_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.csv_deep import (
    CsvAvgCellLengthCheck,
    CsvBomPresenceCheck,
    CsvCellLengthStdCheck,
    CsvColumnCountDeepCheck,
    CsvCommentLineCountCheck,
    CsvDelimiterDetectCheck,
    CsvDuplicateRowCountCheck,
    CsvEmptyCellCountCheck,
    CsvHeaderNameJaccardCheck,
    CsvMaxCellLengthCheck,
    CsvNumericColumnRatioCheck,
    CsvPipeDelimCheck,
    CsvProperParseCheck,
    CsvQuotedFieldRatioCheck,
    CsvRowCountDeepCheck,
    CsvRowWidthConsistencyCheck,
    CsvSemicolonDelimCheck,
    CsvTabDelimCheck,
    CsvTrailingNewlineCheck,
    CsvTypeInferenceMatchCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

CSV = 'name,age,score\n"ann, jr",30,9.5\nbob,25,7.0\n'


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_parse_and_width():
    ctx = _ctx(CSV, CSV)
    assert CsvProperParseCheck().score(CSV, ctx) == 1.0
    assert CsvRowWidthConsistencyCheck().score(CSV, ctx) == 1.0


def test_quoted_and_delim():
    ctx = _ctx(CSV, CSV)
    assert CsvQuotedFieldRatioCheck().score(CSV, ctx) == 1.0
    assert CsvDelimiterDetectCheck().score(CSV, ctx) == 1.0
    sctx = _ctx("a;b\n1;2", "a;b\n1;2")
    assert CsvSemicolonDelimCheck().score("a;b\n1;2", sctx) == 1.0
    assert CsvSemicolonDelimCheck().score(CSV, sctx) == 0.0
    tctx = _ctx("a\tb\n1\t2", "a\tb\n1\t2")
    assert CsvTabDelimCheck().score("a\tb\n1\t2", tctx) == 1.0
    pctx = _ctx("a|b\n1|2", "a|b\n1|2")
    assert CsvPipeDelimCheck().score("a|b\n1|2", pctx) == 1.0


def test_header_and_numeric():
    ctx = _ctx(CSV, CSV)
    assert CsvHeaderNameJaccardCheck().score(CSV, ctx) == 1.0
    assert CsvHeaderNameJaccardCheck().score("x,y,z\n1,2,3", ctx) < 1.0
    assert CsvNumericColumnRatioCheck().score(CSV, ctx) == 1.0


def test_cells_and_dups():
    ctx = _ctx(CSV, CSV)
    assert CsvEmptyCellCountCheck().score(CSV, ctx) == 1.0
    dup = "a,b\n1,2\n1,2\n"
    dctx = _ctx(dup, dup)
    assert CsvDuplicateRowCountCheck().score(dup, dctx) == 1.0
    assert CsvDuplicateRowCountCheck().score(CSV, dctx) < 1.0
    assert CsvCellLengthStdCheck().score(CSV, ctx) == 1.0
    assert CsvMaxCellLengthCheck().score(CSV, ctx) == 1.0
    assert CsvAvgCellLengthCheck().score(CSV, ctx) == 1.0


def test_counts_and_misc():
    ctx = _ctx(CSV, CSV)
    assert CsvColumnCountDeepCheck().score(CSV, ctx) == 1.0
    assert CsvRowCountDeepCheck().score(CSV, ctx) == 1.0
    assert CsvTrailingNewlineCheck().score(CSV, ctx) == 1.0
    assert CsvTrailingNewlineCheck().score("a,b", ctx) == 0.0
    cctx = _ctx("# comment\na,b\n", "# comment\na,b\n")
    assert CsvCommentLineCountCheck().score("# comment\na,b\n", cctx) == 1.0
    assert CsvTypeInferenceMatchCheck().score(CSV, ctx) == 1.0
    bctx = _ctx("﻿a,b\n1,2\n", "﻿a,b\n1,2\n")
    assert CsvBomPresenceCheck().score("﻿a,b\n1,2\n", bctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": CSV,
         "candidate_output": CSV},
        {"id": "b", "prompt": "p", "baseline_output": CSV,
         "candidate_output": "x\n"},
    ])
    result = evaluate(corpus, dimensions=("csv_row_count_deep", "csv_header_jaccard"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["csv_row_count_deep"].losses == 1
    assert dims["csv_header_jaccard"].losses == 1
