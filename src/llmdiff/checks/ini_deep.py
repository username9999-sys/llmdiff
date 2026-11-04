"""Deep INI dimensions: sections, keys, comments, and value shapes."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "IniSectionCountDeepCheck",
    "IniKeyCountDeepCheck",
    "IniCommentCountCheck",
    "IniBlankLineCountCheck",
    "IniQuotedValueCountCheck",
    "IniEmptyValueCountCheck",
    "IniBoolValueCountCheck",
    "IniNumericValueCountCheck",
    "IniUrlValueCountCheck",
    "IniPathValueCountCheck",
    "IniSemicolonCommentCountCheck",
    "IniInlineCommentCountCheck",
    "IniDuplicateSectionCountCheck",
    "IniDuplicateKeyCountCheck",
    "IniSectionNameJaccardCheck",
    "IniKeyNameJaccardCheck",
    "IniMaxSectionNameLenCheck",
    "IniAvgKeyNameLenCheck",
    "IniBracketSectionCountCheck",
    "IniNoSectionKeyCountCheck",
]

_MAX = 200_000
_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.MULTILINE)
_KEY_RE = re.compile(r"^\s*([\w.\-]+)\s*=", re.MULTILINE)
_COMMENT_HASH_RE = re.compile(r"^\s*#", re.MULTILINE)
_COMMENT_SEMI_RE = re.compile(r"^\s*;", re.MULTILINE)
_INLINE_COMMENT_RE = re.compile(r"\S\s+[#;]")
_QUOTED_RE = re.compile(r'=\s*"[^"]*"\s*$|=\s*\'[^\']*\'\s*$', re.MULTILINE)
_EMPTY_RE = re.compile(r"=\s*$", re.MULTILINE)
_BOOL_RE = re.compile(
    r"=\s*(?:true|false|yes|no|on|off)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_NUM_RE = re.compile(r"=\s*-?\d+(?:\.\d+)?\s*$", re.MULTILINE)
_URL_RE = re.compile(r'=\s*"?https?://', re.MULTILINE | re.IGNORECASE)
_PATH_RE = re.compile(r'=\s*"?(?:/|[A-Za-z]:\\|\.\.?/)[^\s]*"?\s*$', re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _lines(text: str) -> list[str]:
    return _s(text).splitlines()


class IniSectionCountDeepCheck:
    @property
    def name(self) -> str:
        return "ini_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SECTION_RE.findall(_s(context.baseline_output))),
            len(_SECTION_RE.findall(_s(output))),
        )


class IniKeyCountDeepCheck:
    @property
    def name(self) -> str:
        return "ini_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KEY_RE.findall(_s(context.baseline_output))),
            len(_KEY_RE.findall(_s(output))),
        )


class IniCommentCountCheck:
    @property
    def name(self) -> str:
        return "ini_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return len(_COMMENT_HASH_RE.findall(s)) + len(_COMMENT_SEMI_RE.findall(s))

        return _count_match(count(context.baseline_output), count(output))


class IniBlankLineCountCheck:
    @property
    def name(self) -> str:
        return "ini_blank_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for ln in _lines(text) if not ln.strip())

        return _count_match(count(context.baseline_output), count(output))


class IniQuotedValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_quoted_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTED_RE.findall(_s(context.baseline_output))),
            len(_QUOTED_RE.findall(_s(output))),
        )


class IniEmptyValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_empty_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EMPTY_RE.findall(_s(context.baseline_output))),
            len(_EMPTY_RE.findall(_s(output))),
        )


class IniBoolValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_bool_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOOL_RE.findall(_s(context.baseline_output))),
            len(_BOOL_RE.findall(_s(output))),
        )


class IniNumericValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_num_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NUM_RE.findall(_s(context.baseline_output))),
            len(_NUM_RE.findall(_s(output))),
        )


class IniUrlValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_url_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_URL_RE.findall(_s(context.baseline_output))),
            len(_URL_RE.findall(_s(output))),
        )


class IniPathValueCountCheck:
    @property
    def name(self) -> str:
        return "ini_path_value_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PATH_RE.findall(_s(context.baseline_output))),
            len(_PATH_RE.findall(_s(output))),
        )


class IniSemicolonCommentCountCheck:
    @property
    def name(self) -> str:
        return "ini_semicolon_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENT_SEMI_RE.findall(_s(context.baseline_output))),
            len(_COMMENT_SEMI_RE.findall(_s(output))),
        )


class IniInlineCommentCountCheck:
    @property
    def name(self) -> str:
        return "ini_inline_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_INLINE_COMMENT_RE.findall(_s(output))),
        )


class IniDuplicateSectionCountCheck:
    @property
    def name(self) -> str:
        return "ini_dup_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            names = [m.group(1).strip().lower() for m in _SECTION_RE.finditer(_s(text))]
            seen: set[str] = set()
            dups = 0
            for n in names:
                if n in seen:
                    dups += 1
                else:
                    seen.add(n)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class IniDuplicateKeyCountCheck:
    @property
    def name(self) -> str:
        return "ini_dup_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            keys = [m.group(1).strip().lower() for m in _KEY_RE.finditer(_s(text))]
            seen: set[str] = set()
            dups = 0
            for k in keys:
                if k in seen:
                    dups += 1
                else:
                    seen.add(k)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class IniSectionNameJaccardCheck:
    @property
    def name(self) -> str:
        return "ini_section_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {m.group(1).strip().lower() for m in _SECTION_RE.finditer(_s(context.baseline_output))}
        b = {m.group(1).strip().lower() for m in _SECTION_RE.finditer(_s(output))}
        return _jaccard(a, b)


class IniKeyNameJaccardCheck:
    @property
    def name(self) -> str:
        return "ini_key_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {m.group(1).strip().lower() for m in _KEY_RE.finditer(_s(context.baseline_output))}
        b = {m.group(1).strip().lower() for m in _KEY_RE.finditer(_s(output))}
        return _jaccard(a, b)


class IniMaxSectionNameLenCheck:
    @property
    def name(self) -> str:
        return "ini_max_section_len"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            names = [m.group(1) for m in _SECTION_RE.finditer(_s(text))]
            return max((len(n) for n in names), default=0)

        return _count_match(mx(context.baseline_output), mx(output))


class IniAvgKeyNameLenCheck:
    @property
    def name(self) -> str:
        return "ini_avg_key_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            keys = [m.group(1) for m in _KEY_RE.finditer(_s(text))]
            return sum(len(k) for k in keys) / len(keys) if keys else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class IniBracketSectionCountCheck:
    @property
    def name(self) -> str:
        return "ini_bracket_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SECTION_RE.findall(_s(context.baseline_output))),
            len(_SECTION_RE.findall(_s(output))),
        )


class IniNoSectionKeyCountCheck:
    """Count of key=value lines appearing before the first section header."""

    @property
    def name(self) -> str:
        return "ini_global_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            n = 0
            for ln in _lines(text):
                if _SECTION_RE.match(ln):
                    break
                if _KEY_RE.match(ln):
                    n += 1
            return n

        return _count_match(count(context.baseline_output), count(output))
