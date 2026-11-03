"""Code-style dimensions: naming, comments, imports, and structure of code blocks."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "SnakeCaseRatioCheck",
    "CamelCaseRatioCheck",
    "PascalCaseRatioCheck",
    "ScreamingCaseRatioCheck",
    "CommentLineRatioCheck",
    "DocstringPresenceCheck",
    "ImportCountCheck",
    "FunctionCountCheck",
    "ClassCountCheck",
    "MagicNumberRatioCheck",
    "BlankLineInCodeCheck",
    "SemicolonLineEndCheck",
    "SingleLetterVarCheck",
    "LongIdentifierRatioCheck",
    "ShortIdentifierRatioCheck",
    "NestedCallDepthCheck",
    "PrintStatementCountCheck",
    "AssertCountCheck",
    "TryExceptCountCheck",
    "FenceLanguageMatchCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_COMMENT_RE = re.compile(r"^\s*#")
_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+\w+", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+\w+", re.MULTILINE)
_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", re.MULTILINE)
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _code_blocks(text: str) -> list[str]:
    return [m.group(2) for m in _FENCE_RE.finditer(text[:_MAX])]


def _all_code(text: str) -> str:
    blocks = _code_blocks(text)
    return "\n".join(blocks) if blocks else text[:_MAX]


def _idents(text: str) -> list[str]:
    return _IDENT_RE.findall(text)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class SnakeCaseRatioCheck:
    """1.0 when the ratio of snake_case identifiers matches the baseline."""

    @property
    def name(self) -> str:
        return "snake_case_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            all_ids = _idents(_all_code(text))
            if not all_ids:
                return 0.0
            snake = [i for i in all_ids if "_" in i and i.islower()]
            return len(snake) / len(all_ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CamelCaseRatioCheck:
    """1.0 when the ratio of camelCase identifiers matches the baseline."""

    @property
    def name(self) -> str:
        return "camel_case_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ids = _idents(_all_code(text))
            if not ids:
                return 0.0
            camel = [i for i in ids if i[0].islower() and any(c.isupper() for c in i[1:])]
            return len(camel) / len(ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PascalCaseRatioCheck:
    """1.0 when the ratio of PascalCase identifiers matches the baseline."""

    @property
    def name(self) -> str:
        return "pascal_case_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ids = _idents(_all_code(text))
            if not ids:
                return 0.0
            pascal = [i for i in ids if i[0].isupper() and "_" not in i]
            return len(pascal) / len(ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ScreamingCaseRatioCheck:
    """1.0 when the ratio of SCREAMING_CASE constants matches the baseline."""

    @property
    def name(self) -> str:
        return "screaming_case_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ids = _idents(_all_code(text))
            if not ids:
                return 0.0
            screaming = [i for i in ids if i.isupper() and "_" in i]
            return len(screaming) / len(ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CommentLineRatioCheck:
    """1.0 when the comment-line ratio inside code matches the baseline."""

    @property
    def name(self) -> str:
        return "comment_line_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = _all_code(text).splitlines()
            if not lines:
                return 0.0
            return sum(1 for ln in lines if _COMMENT_RE.match(ln)) / len(lines)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class DocstringPresenceCheck:
    """1.0 when docstring presence (triple-quote in code) matches the baseline."""

    @property
    def name(self) -> str:
        return "docstring_presence"

    def score(self, output: str, context: CheckContext) -> float:
        def has_doc(text: str) -> bool:
            code = _all_code(text)
            return '"""' in code or "'''" in code

        return 1.0 if has_doc(context.baseline_output) == has_doc(output) else 0.0


class ImportCountCheck:
    """1.0 when the number of import statements matches the baseline."""

    @property
    def name(self) -> str:
        return "import_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_IMPORT_RE.findall(_all_code(context.baseline_output)))
        cand = len(_IMPORT_RE.findall(_all_code(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class FunctionCountCheck:
    """1.0 when the number of function definitions matches the baseline."""

    @property
    def name(self) -> str:
        return "function_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_FUNC_RE.findall(_all_code(context.baseline_output)))
        cand = len(_FUNC_RE.findall(_all_code(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class ClassCountCheck:
    """1.0 when the number of class definitions matches the baseline."""

    @property
    def name(self) -> str:
        return "class_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(_CLASS_RE.findall(_all_code(context.baseline_output)))
        cand = len(_CLASS_RE.findall(_all_code(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class MagicNumberRatioCheck:
    """1.0 when the ratio of bare numeric literals matches the baseline."""

    @property
    def name(self) -> str:
        return "magic_number_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            code = _all_code(text)
            tokens = re.findall(r"\S+", code)
            if not tokens:
                return 0.0
            return len(_NUM_RE.findall(code)) / len(tokens)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class BlankLineInCodeCheck:
    """1.0 when the blank-line ratio inside code matches the baseline."""

    @property
    def name(self) -> str:
        return "blank_line_in_code"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = _all_code(text).splitlines()
            if not lines:
                return 0.0
            return sum(1 for ln in lines if not ln.strip()) / len(lines)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SemicolonLineEndCheck:
    """1.0 when the ratio of lines ending with ';' matches the baseline."""

    @property
    def name(self) -> str:
        return "semicolon_line_end"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = [ln for ln in _all_code(text).splitlines() if ln.strip()]
            if not lines:
                return 0.0
            return sum(1 for ln in lines if ln.rstrip().endswith(";")) / len(lines)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class SingleLetterVarCheck:
    """1.0 when the count of single-letter identifiers matches the baseline."""

    @property
    def name(self) -> str:
        return "single_letter_vars"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for i in _idents(_all_code(text)) if len(i) == 1)

        base, cand = count(context.baseline_output), count(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class LongIdentifierRatioCheck:
    """1.0 when the ratio of identifiers longer than 20 chars matches the baseline."""

    @property
    def name(self) -> str:
        return "long_ident_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ids = _idents(_all_code(text))
            if not ids:
                return 0.0
            return sum(1 for i in ids if len(i) > 20) / len(ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ShortIdentifierRatioCheck:
    """1.0 when the ratio of identifiers of 3 chars or fewer matches the baseline."""

    @property
    def name(self) -> str:
        return "short_ident_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            ids = _idents(_all_code(text))
            if not ids:
                return 0.0
            return sum(1 for i in ids if len(i) <= 3) / len(ids)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class NestedCallDepthCheck:
    """1.0 when the max parenthesis nesting depth in code matches the baseline."""

    @property
    def name(self) -> str:
        return "nested_call_depth"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = cur = 0
            for ch in _all_code(text):
                if ch == "(":
                    cur += 1
                    best = max(best, cur)
                elif ch == ")":
                    cur = max(0, cur - 1)
            return best

        base, cand = depth(context.baseline_output), depth(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class PrintStatementCountCheck:
    """1.0 when the number of print() calls matches the baseline."""

    @property
    def name(self) -> str:
        return "print_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(re.findall(r"\bprint\s*\(", _all_code(context.baseline_output)))
        cand = len(re.findall(r"\bprint\s*\(", _all_code(output)))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class AssertCountCheck:
    """1.0 when the number of assert statements matches the baseline."""

    @property
    def name(self) -> str:
        return "assert_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(re.findall(r"^\s*assert\s", _all_code(context.baseline_output), re.MULTILINE))
        cand = len(re.findall(r"^\s*assert\s", _all_code(output), re.MULTILINE))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class TryExceptCountCheck:
    """1.0 when the number of try blocks matches the baseline."""

    @property
    def name(self) -> str:
        return "try_except_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(re.findall(r"^\s*try\s*:", _all_code(context.baseline_output), re.MULTILINE))
        cand = len(re.findall(r"^\s*try\s*:", _all_code(output), re.MULTILINE))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class FenceLanguageMatchCheck:
    """1.0 when the set of fenced-code languages matches the baseline (Jaccard)."""

    @property
    def name(self) -> str:
        return "fence_language"

    def score(self, output: str, context: CheckContext) -> float:
        def langs(text: str) -> set[str]:
            return {m.group(1).lower() for m in _FENCE_RE.finditer(text[:_MAX])}

        base, cand = langs(context.baseline_output), langs(output)
        if not base and not cand:
            return 1.0
        if base == cand:
            return 1.0
        union = base | cand
        return len(base & cand) / len(union) if union else 1.0
