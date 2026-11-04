"""Deep .env dimensions: var naming, quoting, comments, and secret hygiene."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "EnvVarCountDeepCheck",
    "EnvCommentCountDeepCheck",
    "EnvBlankLineCountCheck",
    "EnvQuotedValueCountCheck",
    "EnvSingleQuotedCountCheck",
    "EnvDoubleQuotedCountCheck",
    "EnvExportPrefixCountCheck",
    "EnvUpperNameCountCheck",
    "EnvSnakeNameCountCheck",
    "EnvDottedNameCountCheck",
    "EnvDashNameCountCheck",
    "EnvEmptyValueCountCheck",
    "EnvInlineCommentCountCheck",
    "EnvSecretNameCountCheck",
    "EnvUrlValueCountCheck",
    "EnvNumericValueCountCheck",
    "EnvBoolValueCountCheck",
    "EnvLongValueCountCheck",
    "EnvDuplicateNameCountCheck",
    "EnvAvgNameLenCheck",
]

_MAX = 200_000
_VAR_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=", re.MULTILINE)
_COMMENT_RE = re.compile(r"^\s*#", re.MULTILINE)
_INLINE_COMMENT_RE = re.compile(r"\S\s+#")
_DQ_RE = re.compile(r'=\s*"[^"]*"\s*$', re.MULTILINE)
_SQ_RE = re.compile(r"=\s*'[^']*'\s*$", re.MULTILINE)
_EXPORT_RE = re.compile(r"^\s*export\s+[A-Za-z_]", re.MULTILINE)
_UPPER_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_DOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.-]+$")
_DASH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*-[A-Za-z0-9_.-]+$")
_EMPTY_RE = re.compile(r"=\s*$", re.MULTILINE)
_SECRET_NAME_RE = re.compile(
    r"^(?:.*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL|AUTH))",
    re.IGNORECASE,
)
_URL_VAL_RE = re.compile(r'=\s*"?https?://', re.MULTILINE | re.IGNORECASE)
_NUM_VAL_RE = re.compile(r'=\s*"?-?\d+(?:\.\d+)?"?\s*$', re.MULTILINE)
_BOOL_VAL_RE = re.compile(r'=\s*"?(?:true|false)"?\s*$', re.MULTILINE | re.IGNORECASE)
_LONG_VAL_RE = re.compile(r'=\s*"?.{32,}"?\s*$', re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _var_lines(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(0)) for m in _VAR_RE.finditer(_s(text))]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class EnvVarCountDeepCheck:
    @property
    def name(self) -> str:
        return "env_var_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_var_lines(context.baseline_output)),
            len(_var_lines(output)),
        )


class EnvCommentCountDeepCheck:
    @property
    def name(self) -> str:
        return "env_comment_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_COMMENT_RE.findall(_s(output))),
        )


class EnvBlankLineCountCheck:
    @property
    def name(self) -> str:
        return "env_blank_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for ln in _s(text).splitlines() if not ln.strip())

        return _count_match(count(context.baseline_output), count(output))


class EnvQuotedValueCountCheck:
    @property
    def name(self) -> str:
        return "env_quoted_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return len(_DQ_RE.findall(s)) + len(_SQ_RE.findall(s))

        return _count_match(count(context.baseline_output), count(output))


class EnvSingleQuotedCountCheck:
    @property
    def name(self) -> str:
        return "env_single_quoted_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SQ_RE.findall(_s(context.baseline_output))),
            len(_SQ_RE.findall(_s(output))),
        )


class EnvDoubleQuotedCountCheck:
    @property
    def name(self) -> str:
        return "env_double_quoted_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DQ_RE.findall(_s(context.baseline_output))),
            len(_DQ_RE.findall(_s(output))),
        )


class EnvExportPrefixCountCheck:
    @property
    def name(self) -> str:
        return "env_export_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EXPORT_RE.findall(_s(context.baseline_output))),
            len(_EXPORT_RE.findall(_s(output))),
        )


class EnvUpperNameCountCheck:
    @property
    def name(self) -> str:
        return "env_upper_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for name, _ in _var_lines(text) if _UPPER_RE.match(name))

        return _count_match(count(context.baseline_output), count(output))


class EnvSnakeNameCountCheck:
    @property
    def name(self) -> str:
        return "env_snake_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for name, _ in _var_lines(text) if _SNAKE_RE.match(name))

        return _count_match(count(context.baseline_output), count(output))


class EnvDottedNameCountCheck:
    @property
    def name(self) -> str:
        return "env_dotted_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for name, _ in _var_lines(text) if _DOT_RE.match(name))

        return _count_match(count(context.baseline_output), count(output))


class EnvDashNameCountCheck:
    @property
    def name(self) -> str:
        return "env_dash_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for name, _ in _var_lines(text) if _DASH_RE.match(name))

        return _count_match(count(context.baseline_output), count(output))


class EnvEmptyValueCountCheck:
    @property
    def name(self) -> str:
        return "env_empty_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EMPTY_RE.findall(_s(context.baseline_output))),
            len(_EMPTY_RE.findall(_s(output))),
        )


class EnvInlineCommentCountCheck:
    @property
    def name(self) -> str:
        return "env_inline_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_INLINE_COMMENT_RE.findall(_s(output))),
        )


class EnvSecretNameCountCheck:
    @property
    def name(self) -> str:
        return "env_secret_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for name, _ in _var_lines(text) if _SECRET_NAME_RE.match(name))

        return _count_match(count(context.baseline_output), count(output))


class EnvUrlValueCountCheck:
    @property
    def name(self) -> str:
        return "env_url_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_URL_VAL_RE.findall(_s(context.baseline_output))),
            len(_URL_VAL_RE.findall(_s(output))),
        )


class EnvNumericValueCountCheck:
    @property
    def name(self) -> str:
        return "env_num_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NUM_VAL_RE.findall(_s(context.baseline_output))),
            len(_NUM_VAL_RE.findall(_s(output))),
        )


class EnvBoolValueCountCheck:
    @property
    def name(self) -> str:
        return "env_bool_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOOL_VAL_RE.findall(_s(context.baseline_output))),
            len(_BOOL_VAL_RE.findall(_s(output))),
        )


class EnvLongValueCountCheck:
    @property
    def name(self) -> str:
        return "env_long_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LONG_VAL_RE.findall(_s(context.baseline_output))),
            len(_LONG_VAL_RE.findall(_s(output))),
        )


class EnvDuplicateNameCountCheck:
    @property
    def name(self) -> str:
        return "env_dup_name_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            names = [n for n, _ in _var_lines(text)]
            seen: set[str] = set()
            dups = 0
            for n in names:
                if n in seen:
                    dups += 1
                else:
                    seen.add(n)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class EnvAvgNameLenCheck:
    @property
    def name(self) -> str:
        return "env_avg_name_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            names = [n for n, _ in _var_lines(text)]
            return sum(len(n) for n in names) / len(names) if names else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
