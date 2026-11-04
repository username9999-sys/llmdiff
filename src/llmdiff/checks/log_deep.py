"""Deep log dimensions: levels, timestamps, structured fields, and stack traces."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "LogLevelCountCheck",
    "LogTimestampCountCheck",
    "LogIsoTimestampCountCheck",
    "LogEpochTimestampCountCheck",
    "LogStructuredFieldCountCheck",
    "LogJsonLineCountCheck",
    "LogKeyValueCountCheck",
    "LogTraceIdCountCheck",
    "LogSpanIdCountCheck",
    "LogStackTraceLineCountCheck",
    "LogExceptionNameCountCheck",
    "LogWarnCountCheck",
    "LogErrorCountCheck",
    "LogInfoCountCheck",
    "LogDebugCountCheck",
    "LogTraceCountCheck",
    "LogFatalCountCheck",
    "LogBracketLevelCountCheck",
    "LogMessageLenAvgCheck",
    "LogSourceFileRefCountCheck",
]

_MAX = 200_000
_LEVEL_RE = re.compile(r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b")
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_EPOCH_TS_RE = re.compile(r"\b1[0-9]{9}(?:\d{0,3})\b")
_KV_RE = re.compile(r"\b\w+=(?:\"[^\"]*\"|'[^']*'|\S+)")
_JSON_LINE_RE = re.compile(r"^\s*\{.*\}\s*$", re.MULTILINE)
_TRACE_ID_RE = re.compile(
    r"\b(?:trace[_-]?id|traceId)\s*[:=]\s*[0-9a-fA-F-]{8,}\b",
    re.IGNORECASE,
)
_SPAN_ID_RE = re.compile(
    r"\b(?:span[_-]?id|spanId)\s*[:=]\s*[0-9a-fA-F-]{8,}\b",
    re.IGNORECASE,
)
_STACK_LINE_RE = re.compile(
    r"^\s*(?:at\s+[\w.$]+\s*\(|File\s+\"[^\"]+\"|Traceback)",
    re.MULTILINE,
)
_EXC_RE = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Warning|Exit|Interrupt))\b")
_BRACKET_LEVEL_RE = re.compile(
    r"^\s*\[(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\]",
    re.MULTILINE | re.IGNORECASE,
)
_SRC_FILE_RE = re.compile(r"\b[\w-]+\.(?:py|js|ts|go|rs|java|c|cpp|h)\b")


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _level_count(text: str, level: str) -> int:
    return len(re.findall(rf"\b{level}\b", _s(text), re.IGNORECASE))


class LogLevelCountCheck:
    @property
    def name(self) -> str:
        return "log_level_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LEVEL_RE.findall(_s(context.baseline_output))),
            len(_LEVEL_RE.findall(_s(output))),
        )


class LogTimestampCountCheck:
    @property
    def name(self) -> str:
        return "log_ts_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return len(_ISO_TS_RE.findall(s)) + len(_EPOCH_TS_RE.findall(s))

        return _count_match(count(context.baseline_output), count(output))


class LogIsoTimestampCountCheck:
    @property
    def name(self) -> str:
        return "log_iso_ts_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ISO_TS_RE.findall(_s(context.baseline_output))),
            len(_ISO_TS_RE.findall(_s(output))),
        )


class LogEpochTimestampCountCheck:
    @property
    def name(self) -> str:
        return "log_epoch_ts_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EPOCH_TS_RE.findall(_s(context.baseline_output))),
            len(_EPOCH_TS_RE.findall(_s(output))),
        )


class LogStructuredFieldCountCheck:
    @property
    def name(self) -> str:
        return "log_structured_fields"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KV_RE.findall(_s(context.baseline_output))),
            len(_KV_RE.findall(_s(output))),
        )


class LogJsonLineCountCheck:
    @property
    def name(self) -> str:
        return "log_json_lines"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_JSON_LINE_RE.findall(_s(context.baseline_output))),
            len(_JSON_LINE_RE.findall(_s(output))),
        )


class LogKeyValueCountCheck:
    @property
    def name(self) -> str:
        return "log_kv_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KV_RE.findall(_s(context.baseline_output))),
            len(_KV_RE.findall(_s(output))),
        )


class LogTraceIdCountCheck:
    @property
    def name(self) -> str:
        return "log_trace_id_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TRACE_ID_RE.findall(_s(context.baseline_output))),
            len(_TRACE_ID_RE.findall(_s(output))),
        )


class LogSpanIdCountCheck:
    @property
    def name(self) -> str:
        return "log_span_id_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SPAN_ID_RE.findall(_s(context.baseline_output))),
            len(_SPAN_ID_RE.findall(_s(output))),
        )


class LogStackTraceLineCountCheck:
    @property
    def name(self) -> str:
        return "log_stack_lines"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STACK_LINE_RE.findall(_s(context.baseline_output))),
            len(_STACK_LINE_RE.findall(_s(output))),
        )


class LogExceptionNameCountCheck:
    @property
    def name(self) -> str:
        return "log_exc_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EXC_RE.findall(_s(context.baseline_output))),
            len(_EXC_RE.findall(_s(output))),
        )


class LogWarnCountCheck:
    @property
    def name(self) -> str:
        return "log_warn_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return _level_count(text, "WARN") + _level_count(text, "WARNING")

        return _count_match(count(context.baseline_output), count(output))


class LogErrorCountCheck:
    @property
    def name(self) -> str:
        return "log_error_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _level_count(context.baseline_output, "ERROR"),
            _level_count(output, "ERROR"),
        )


class LogInfoCountCheck:
    @property
    def name(self) -> str:
        return "log_info_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _level_count(context.baseline_output, "INFO"),
            _level_count(output, "INFO"),
        )


class LogDebugCountCheck:
    @property
    def name(self) -> str:
        return "log_debug_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _level_count(context.baseline_output, "DEBUG"),
            _level_count(output, "DEBUG"),
        )


class LogTraceCountCheck:
    @property
    def name(self) -> str:
        return "log_trace_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            _level_count(context.baseline_output, "TRACE"),
            _level_count(output, "TRACE"),
        )


class LogFatalCountCheck:
    @property
    def name(self) -> str:
        return "log_fatal_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return _level_count(text, "FATAL") + _level_count(text, "CRITICAL")

        return _count_match(count(context.baseline_output), count(output))


class LogBracketLevelCountCheck:
    @property
    def name(self) -> str:
        return "log_bracket_level_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BRACKET_LEVEL_RE.findall(_s(context.baseline_output))),
            len(_BRACKET_LEVEL_RE.findall(_s(output))),
        )


class LogMessageLenAvgCheck:
    @property
    def name(self) -> str:
        return "log_msg_len_avg"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            lines = [ln for ln in _s(text).splitlines() if ln.strip()]
            return sum(len(ln) for ln in lines) / len(lines) if lines else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class LogSourceFileRefCountCheck:
    @property
    def name(self) -> str:
        return "log_srcfile_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SRC_FILE_RE.findall(_s(context.baseline_output))),
            len(_SRC_FILE_RE.findall(_s(output))),
        )
