"""Tests for log_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.log_deep import (
    LogBracketLevelCountCheck,
    LogDebugCountCheck,
    LogEpochTimestampCountCheck,
    LogErrorCountCheck,
    LogExceptionNameCountCheck,
    LogFatalCountCheck,
    LogInfoCountCheck,
    LogIsoTimestampCountCheck,
    LogJsonLineCountCheck,
    LogKeyValueCountCheck,
    LogLevelCountCheck,
    LogMessageLenAvgCheck,
    LogSourceFileRefCountCheck,
    LogSpanIdCountCheck,
    LogStackTraceLineCountCheck,
    LogStructuredFieldCountCheck,
    LogTimestampCountCheck,
    LogTraceCountCheck,
    LogTraceIdCountCheck,
    LogWarnCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

LOG = (
    "2026-11-04T10:00:00Z [INFO] start user=alice trace_id=abcdef0123456789\n"
    "2026-11-04T10:00:01Z [WARN] slow query=1.2s\n"
    "2026-11-04T10:00:02Z [ERROR] boom ValueError\n"
    "Traceback:\n  File \"app.py\"\n    at main()\n"
    '{"level": "info", "msg": "json line"}\n'
    "1762233600 epoch\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_levels():
    ctx = _ctx(LOG, LOG)
    assert LogLevelCountCheck().score(LOG, ctx) == 1.0
    assert LogWarnCountCheck().score(LOG, ctx) == 1.0
    assert LogErrorCountCheck().score(LOG, ctx) == 1.0
    assert LogInfoCountCheck().score(LOG, ctx) == 1.0
    assert LogDebugCountCheck().score(LOG, ctx) == 1.0
    assert LogTraceCountCheck().score(LOG, ctx) == 1.0
    assert LogFatalCountCheck().score(LOG, ctx) == 1.0
    assert LogBracketLevelCountCheck().score(LOG, ctx) == 1.0


def test_timestamps():
    ctx = _ctx(LOG, LOG)
    assert LogTimestampCountCheck().score(LOG, ctx) == 1.0
    assert LogIsoTimestampCountCheck().score(LOG, ctx) == 1.0
    assert LogEpochTimestampCountCheck().score(LOG, ctx) == 1.0


def test_structured_fields():
    ctx = _ctx(LOG, LOG)
    assert LogStructuredFieldCountCheck().score(LOG, ctx) == 1.0
    assert LogKeyValueCountCheck().score(LOG, ctx) == 1.0
    assert LogJsonLineCountCheck().score(LOG, ctx) == 1.0
    assert LogTraceIdCountCheck().score(LOG, ctx) == 1.0
    assert LogSpanIdCountCheck().score(LOG, ctx) == 1.0


def test_stack_and_exceptions():
    ctx = _ctx(LOG, LOG)
    assert LogStackTraceLineCountCheck().score(LOG, ctx) == 1.0
    assert LogExceptionNameCountCheck().score(LOG, ctx) == 1.0
    assert LogSourceFileRefCountCheck().score(LOG, ctx) == 1.0
    assert LogMessageLenAvgCheck().score(LOG, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": LOG,
         "candidate_output": LOG},
        {"id": "b", "prompt": "p", "baseline_output": LOG,
         "candidate_output": "plain text"},
    ])
    result = evaluate(corpus, dimensions=("log_error_count", "log_iso_ts_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["log_error_count"].losses == 1
    assert dims["log_iso_ts_count"].losses == 1
