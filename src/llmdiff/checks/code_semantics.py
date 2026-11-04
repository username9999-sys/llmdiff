"""Code-semantics dimensions: identifiers, literals, operators, and control flow."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "IdentifierSetJaccardCheck",
    "StringLiteralCountCheck",
    "StringLiteralOverlapCheck",
    "KeywordCountCheck",
    "OperatorDensityCheck",
    "AssignmentCountCheck",
    "ComparisonCountCheck",
    "LoopCountCheck",
    "ConditionalCountCheck",
    "ReturnCountCheck",
    "CallCountCheck",
    "ArgCountMatchCheck",
    "DecoratorCountCheck",
    "YieldCountCheck",
    "RaiseCountCheck",
    "WithCountCheck",
    "GlobalNonlocalCountCheck",
    "AsyncAwaitCountCheck",
    "ElifCountCheck",
    "MatchCaseCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_STR_RE = re.compile(r"(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)")
_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "return", "import",
    "from", "as", "try", "except", "finally", "with", "lambda", "yield",
    "raise", "assert", "pass", "break", "continue", "global", "nonlocal",
    "async", "await", "in", "is", "not", "and", "or", "match", "case",
}
_OPERATORS = set("+-*/%=<>!&|^~@")
_CALL_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_.]*\s*\(")
_ASSIGN_RE = re.compile(r"(?<![=!<>])=(?!=)")
_COMPARE_RE = re.compile(r"==|!=|<=|>=|<|>")
_LOOP_RE = re.compile(r"^\s*(?:for|while)\s", re.MULTILINE)
_COND_RE = re.compile(r"^\s*(?:if|elif)\s", re.MULTILINE)
_RETURN_RE = re.compile(r"^\s*return\b", re.MULTILINE)
_DECOR_RE = re.compile(r"^\s*@\w+", re.MULTILINE)
_YIELD_RE = re.compile(r"\byield\b")
_RAISE_RE = re.compile(r"^\s*raise\b", re.MULTILINE)
_WITH_RE = re.compile(r"^\s*with\s", re.MULTILINE)
_GLOBAL_RE = re.compile(r"^\s*(?:global|nonlocal)\s", re.MULTILINE)
_ASYNC_RE = re.compile(r"\b(?:async|await)\b")
_ELIF_RE = re.compile(r"^\s*elif\s", re.MULTILINE)
_MATCH_RE = re.compile(r"^\s*(?:match|case)\s", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _code(text: str) -> str:
    blocks = [m.group(1) for m in _FENCE_RE.finditer(_s(text))]
    return "\n".join(blocks) if blocks else _s(text)


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


class IdentifierSetJaccardCheck:
    """Jaccard similarity of identifier sets between baseline and output code."""

    @property
    def name(self) -> str:
        return "ident_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {i for i in _IDENT_RE.findall(_code(context.baseline_output)) if i not in _KEYWORDS}
        b = {i for i in _IDENT_RE.findall(_code(output)) if i not in _KEYWORDS}
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class StringLiteralCountCheck:
    @property
    def name(self) -> str:
        return "str_literal_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STR_RE.findall(_code(context.baseline_output))),
            len(_STR_RE.findall(_code(output))),
        )


class StringLiteralOverlapCheck:
    @property
    def name(self) -> str:
        return "str_literal_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        a = set(_STR_RE.findall(_code(context.baseline_output)))
        b = set(_STR_RE.findall(_code(output)))
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a)


class KeywordCountCheck:
    @property
    def name(self) -> str:
        return "keyword_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for i in _IDENT_RE.findall(_code(text)) if i in _KEYWORDS)

        return _count_match(count(context.baseline_output), count(output))


class OperatorDensityCheck:
    @property
    def name(self) -> str:
        return "operator_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            return sum(1 for c in code if c in _OPERATORS) / max(1, len(code))

        return _ratio_match(density(context.baseline_output), density(output))


class AssignmentCountCheck:
    @property
    def name(self) -> str:
        return "assign_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ASSIGN_RE.findall(_code(context.baseline_output))),
            len(_ASSIGN_RE.findall(_code(output))),
        )


class ComparisonCountCheck:
    @property
    def name(self) -> str:
        return "compare_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMPARE_RE.findall(_code(context.baseline_output))),
            len(_COMPARE_RE.findall(_code(output))),
        )


class LoopCountCheck:
    @property
    def name(self) -> str:
        return "loop_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LOOP_RE.findall(_code(context.baseline_output))),
            len(_LOOP_RE.findall(_code(output))),
        )


class ConditionalCountCheck:
    @property
    def name(self) -> str:
        return "cond_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COND_RE.findall(_code(context.baseline_output))),
            len(_COND_RE.findall(_code(output))),
        )


class ReturnCountCheck:
    @property
    def name(self) -> str:
        return "return_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RETURN_RE.findall(_code(context.baseline_output))),
            len(_RETURN_RE.findall(_code(output))),
        )


class CallCountCheck:
    @property
    def name(self) -> str:
        return "call_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CALL_RE.findall(_code(context.baseline_output))),
            len(_CALL_RE.findall(_code(output))),
        )


class ArgCountMatchCheck:
    """1.0 when the total number of call arguments matches the baseline."""

    @property
    def name(self) -> str:
        return "arg_count"

    def score(self, output: str, context: CheckContext) -> float:
        def total_args(text: str) -> int:
            total = 0
            for m in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_.]*\s*\(([^()]*)\)", _code(text)):
                inner = m.group(1).strip()
                total += 0 if not inner else inner.count(",") + 1
            return total

        return _count_match(total_args(context.baseline_output), total_args(output))


class DecoratorCountCheck:
    @property
    def name(self) -> str:
        return "decorator_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DECOR_RE.findall(_code(context.baseline_output))),
            len(_DECOR_RE.findall(_code(output))),
        )


class YieldCountCheck:
    @property
    def name(self) -> str:
        return "yield_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_YIELD_RE.findall(_code(context.baseline_output))),
            len(_YIELD_RE.findall(_code(output))),
        )


class RaiseCountCheck:
    @property
    def name(self) -> str:
        return "raise_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_RAISE_RE.findall(_code(context.baseline_output))),
            len(_RAISE_RE.findall(_code(output))),
        )


class WithCountCheck:
    @property
    def name(self) -> str:
        return "with_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_WITH_RE.findall(_code(context.baseline_output))),
            len(_WITH_RE.findall(_code(output))),
        )


class GlobalNonlocalCountCheck:
    @property
    def name(self) -> str:
        return "global_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GLOBAL_RE.findall(_code(context.baseline_output))),
            len(_GLOBAL_RE.findall(_code(output))),
        )


class AsyncAwaitCountCheck:
    @property
    def name(self) -> str:
        return "async_await_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ASYNC_RE.findall(_code(context.baseline_output))),
            len(_ASYNC_RE.findall(_code(output))),
        )


class ElifCountCheck:
    @property
    def name(self) -> str:
        return "elif_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ELIF_RE.findall(_code(context.baseline_output))),
            len(_ELIF_RE.findall(_code(output))),
        )


class MatchCaseCountCheck:
    @property
    def name(self) -> str:
        return "match_case_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MATCH_RE.findall(_code(context.baseline_output))),
            len(_MATCH_RE.findall(_code(output))),
        )
