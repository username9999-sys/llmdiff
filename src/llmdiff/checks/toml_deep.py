"""Deep TOML dimensions: tables, arrays of tables, inline tables, and value types."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "TomlTableHeaderCountCheck",
    "TomlArrayTableCountCheck",
    "TomlInlineTableCountCheck",
    "TomlDottedKeyCountCheck",
    "TomlCommentCountCheck",
    "TomlStringValueCountCheck",
    "TomlBasicStringCountCheck",
    "TomlLiteralStringCountCheck",
    "TomlMultilineStringCountCheck",
    "TomlIntegerValueCountCheck",
    "TomlFloatValueCountCheck",
    "TomlBoolValueCountCheck",
    "TomlDateValueCountCheck",
    "TomlArrayValueCountCheck",
    "TomlNestedArrayCountCheck",
    "TomlKeyCountDeepCheck",
    "TomlIndentPresenceCheck",
    "TomlMaxTableDepthCheck",
    "TomlQuotedKeyCountCheck",
    "TomlEmptyTableCountCheck",
]

_MAX = 200_000
_TABLE_RE = re.compile(r"^\s*\[\s*[\w.\"-]+\s*\]\s*$", re.MULTILINE)
_ARR_TABLE_RE = re.compile(r"^\s*\[\[\s*[\w.\"-]+\s*\]\]\s*$", re.MULTILINE)
_INLINE_TABLE_RE = re.compile(r"\{[^{}]*\}")
_DOTTED_KEY_RE = re.compile(r"^\s*[\w-]+\.[\w.-]+\s*=", re.MULTILINE)
_COMMENT_RE = re.compile(r"^\s*#", re.MULTILINE)
_KEY_RE = re.compile(r"^\s*(?:\"[^\"]+\"|[\w-]+)\s*=", re.MULTILINE)
_BASIC_STR_RE = re.compile(r":\s*\"(?:[^\"\\]|\\.)*\"")
_LIT_STR_RE = re.compile(r":\s*'(?:[^'])*'")
_ML_STR_RE = re.compile(r"(?:\"\"\"|''')")
_INT_RE = re.compile(
    r":\s*[+-]?(?:0x[0-9a-fA-F_]+|0o[0-7_]+|0b[01_]+|\d[\d_]*)\s*$",
    re.MULTILINE,
)
_FLOAT_RE = re.compile(
    r":\s*[+-]?(?:\d[\d_]*\.\d[\d_]*(?:[eE][+-]?\d+)?|inf|nan)\s*$",
    re.MULTILINE,
)
_BOOL_RE = re.compile(r":\s*(?:true|false)\s*$", re.MULTILINE)
_DATE_RE = re.compile(
    r":\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2})?\s*$",
    re.MULTILINE,
)
_ARRAY_RE = re.compile(r":\s*\[[^\]]*\]")
_NESTED_ARRAY_RE = re.compile(r"\[[^\]]*\[[^\]]*\]")
_QUOTED_KEY_RE = re.compile(r'^\s*"[^"]+"\s*=')


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class TomlTableHeaderCountCheck:
    @property
    def name(self) -> str:
        return "toml_table_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TABLE_RE.findall(_s(context.baseline_output))),
            len(_TABLE_RE.findall(_s(output))),
        )


class TomlArrayTableCountCheck:
    @property
    def name(self) -> str:
        return "toml_array_table_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ARR_TABLE_RE.findall(_s(context.baseline_output))),
            len(_ARR_TABLE_RE.findall(_s(output))),
        )


class TomlInlineTableCountCheck:
    @property
    def name(self) -> str:
        return "toml_inline_table_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_TABLE_RE.findall(_s(context.baseline_output))),
            len(_INLINE_TABLE_RE.findall(_s(output))),
        )


class TomlDottedKeyCountCheck:
    @property
    def name(self) -> str:
        return "toml_dotted_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOTTED_KEY_RE.findall(_s(context.baseline_output))),
            len(_DOTTED_KEY_RE.findall(_s(output))),
        )


class TomlCommentCountCheck:
    @property
    def name(self) -> str:
        return "toml_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_COMMENT_RE.findall(_s(output))),
        )


class TomlStringValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_string_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return len(_BASIC_STR_RE.findall(s)) + len(_LIT_STR_RE.findall(s))

        return _count_match(count(context.baseline_output), count(output))


class TomlBasicStringCountCheck:
    @property
    def name(self) -> str:
        return "toml_basic_str_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BASIC_STR_RE.findall(_s(context.baseline_output))),
            len(_BASIC_STR_RE.findall(_s(output))),
        )


class TomlLiteralStringCountCheck:
    @property
    def name(self) -> str:
        return "toml_literal_str_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LIT_STR_RE.findall(_s(context.baseline_output))),
            len(_LIT_STR_RE.findall(_s(output))),
        )


class TomlMultilineStringCountCheck:
    @property
    def name(self) -> str:
        return "toml_ml_str_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ML_STR_RE.findall(_s(context.baseline_output))) // 2,
            len(_ML_STR_RE.findall(_s(output))) // 2,
        )


class TomlIntegerValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_int_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INT_RE.findall(_s(context.baseline_output))),
            len(_INT_RE.findall(_s(output))),
        )


class TomlFloatValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_float_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FLOAT_RE.findall(_s(context.baseline_output))),
            len(_FLOAT_RE.findall(_s(output))),
        )


class TomlBoolValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_bool_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOOL_RE.findall(_s(context.baseline_output))),
            len(_BOOL_RE.findall(_s(output))),
        )


class TomlDateValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_date_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DATE_RE.findall(_s(context.baseline_output))),
            len(_DATE_RE.findall(_s(output))),
        )


class TomlArrayValueCountCheck:
    @property
    def name(self) -> str:
        return "toml_array_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ARRAY_RE.findall(_s(context.baseline_output))),
            len(_ARRAY_RE.findall(_s(output))),
        )


class TomlNestedArrayCountCheck:
    @property
    def name(self) -> str:
        return "toml_nested_array_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NESTED_ARRAY_RE.findall(_s(context.baseline_output))),
            len(_NESTED_ARRAY_RE.findall(_s(output))),
        )


class TomlKeyCountDeepCheck:
    @property
    def name(self) -> str:
        return "toml_key_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KEY_RE.findall(_s(context.baseline_output))),
            len(_KEY_RE.findall(_s(output))),
        )


class TomlIndentPresenceCheck:
    @property
    def name(self) -> str:
        return "toml_indent_presence"

    def score(self, output: str, context: CheckContext) -> float:
        def has_indent(text: str) -> bool:
            return any(ln.startswith((" ", "\t")) for ln in _s(text).splitlines())

        return 1.0 if has_indent(context.baseline_output) == has_indent(output) else 0.0


class TomlMaxTableDepthCheck:
    @property
    def name(self) -> str:
        return "toml_table_depth"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = 0
            for m in re.finditer(r"^\s*\[+([^\]]+)\]+\s*$", _s(text), re.MULTILINE):
                best = max(best, m.group(1).count(".") + 1)
            return best

        return _count_match(depth(context.baseline_output), depth(output))


class TomlQuotedKeyCountCheck:
    @property
    def name(self) -> str:
        return "toml_quoted_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTED_KEY_RE.findall(_s(context.baseline_output))),
            len(_QUOTED_KEY_RE.findall(_s(output))),
        )


class TomlEmptyTableCountCheck:
    """Count of table headers immediately followed by another header or EOF."""

    @property
    def name(self) -> str:
        return "toml_empty_table_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            lines = _s(text).splitlines()
            n = 0
            for i, ln in enumerate(lines):
                if _TABLE_RE.match(ln) or _ARR_TABLE_RE.match(ln):
                    nxt = next(
                        (
                            x
                            for x in lines[i + 1 :]
                            if x.strip() and not x.strip().startswith("#")
                        ),
                        "",
                    )
                    if not nxt or _TABLE_RE.match(nxt) or _ARR_TABLE_RE.match(nxt):
                        n += 1
            return n

        return _count_match(count(context.baseline_output), count(output))
