"""Deep code-quality dimensions: complexity, naming, and anti-pattern signals."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "CyclomaticComplexityCheck",
    "MaxNestingDepthCheck",
    "AvgFunctionLengthCheck",
    "MaxFunctionLengthCheck",
    "FunctionCountDensityCheck",
    "TodoFixmeCountCheck",
    "HardcodedIpCountCheck",
    "HardcodedUrlCountCheck",
    "MagicStringCountCheck",
    "LongParameterListCountCheck",
    "BooleanParamCountCheck",
    "DeepConditionalCountCheck",
    "GotoCountCheck",
    "GlobalVarCountCheck",
    "MutableDefaultCountCheck",
    "BareExceptCountCheck",
    "PrintDebugCountCheck",
    "SleepCallCountCheck",
    "EvalExecCountCheck",
    "CommentedOutCodeCountCheck",
]

_MAX = 200_000
_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
_TODO_RE = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_URL_RE = re.compile(r"https?://[^\s\"']+")
_STR_RE = re.compile(r"(?:\"(?:[^\"\\]|\\.){3,}\"|'(?:[^'\\]|\\.){3,}')")
_BOOL_PARAM_RE = re.compile(
    r"\b(?:is_|has_|enable_|use_|allow_)?\w*(?:flag|enabled|verbose|debug|force|strict)\w*\s*:\s*bool\b",
    re.IGNORECASE,
)
_COND_RE = re.compile(r"^\s*(?:if|elif|while|for|except|with)\b", re.MULTILINE)
_GOTO_RE = re.compile(r"\bgoto\b")
_GLOBAL_RE = re.compile(r"^\s*global\s+\w+", re.MULTILINE)
_MUTABLE_DEFAULT_RE = re.compile(
    r"\)\s*(?:->\s*[^:]+)?\s*:\s*(?:=\s*[\[\{]|:\s*(?:list|dict|set)\s*=\s*[\[\{])",
    re.MULTILINE,
)
_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:", re.MULTILINE)
_PRINT_RE = re.compile(r"\bprint\s*\(")
_SLEEP_RE = re.compile(r"\b(?:time\.sleep|asyncio\.sleep|sleep)\s*\(")
_EVAL_RE = re.compile(r"\b(?:eval|exec)\s*\(")
_COMMENTED_CODE_RE = re.compile(
    r"^\s*#\s*(?:def |class |if |for |return |import |x\s*=|print\()",
    re.MULTILINE,
)


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


class CyclomaticComplexityCheck:
    """Approximate cyclomatic complexity: 1 + number of decision keywords."""

    @property
    def name(self) -> str:
        return "cyclomatic"

    def score(self, output: str, context: CheckContext) -> float:
        def complexity(text: str) -> int:
            code = _code(text)
            decisions = len(
                re.findall(r"\b(?:if|elif|for|while|except|with|and|or|case)\b", code)
            )
            return 1 + decisions

        return _count_match(complexity(context.baseline_output), complexity(output))


class MaxNestingDepthCheck:
    @property
    def name(self) -> str:
        return "max_nesting"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = cur = 0
            for ln in _code(text).splitlines():
                stripped = ln.strip()
                if stripped.endswith(":") and not stripped.startswith(("#", "@")):
                    cur += 1
                    best = max(best, cur)
                elif stripped and not stripped.startswith(("#", "@")):
                    indent = len(ln) - len(ln.lstrip())
                    cur = max(0, indent // 4)
            return best

        return _count_match(depth(context.baseline_output), depth(output))


class AvgFunctionLengthCheck:
    @property
    def name(self) -> str:
        return "avg_func_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg_len(text: str) -> float:
            code = _code(text)
            matches = list(_FUNC_RE.finditer(code))
            if not matches:
                return 0.0
            lines = code.splitlines()
            total = 0
            for m in matches:
                start = code[: m.start()].count("\n")
                total += 1
                for i in range(start + 1, len(lines)):
                    if lines[i].strip() and not lines[i].startswith((" ", "\t")):
                        break
                    total += 1
            return total / len(matches)

        base, cand = avg_len(context.baseline_output), avg_len(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class MaxFunctionLengthCheck:
    @property
    def name(self) -> str:
        return "max_func_len"

    def score(self, output: str, context: CheckContext) -> float:
        def max_len(text: str) -> int:
            code = _code(text)
            lines = code.splitlines()
            best = 0
            for m in _FUNC_RE.finditer(code):
                start = code[: m.start()].count("\n")
                length = 1
                for i in range(start + 1, len(lines)):
                    if lines[i].strip() and not lines[i].startswith((" ", "\t")):
                        break
                    length += 1
                best = max(best, length)
            return best

        return _count_match(max_len(context.baseline_output), max_len(output))


class FunctionCountDensityCheck:
    @property
    def name(self) -> str:
        return "func_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            code = _code(text)
            lines = [ln for ln in code.splitlines() if ln.strip()]
            if not lines:
                return 0.0
            return len(_FUNC_RE.findall(code)) / len(lines)

        return _ratio_match(density(context.baseline_output), density(output))


class TodoFixmeCountCheck:
    @property
    def name(self) -> str:
        return "todo_fixme_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TODO_RE.findall(_code(context.baseline_output))),
            len(_TODO_RE.findall(_code(output))),
        )


class HardcodedIpCountCheck:
    @property
    def name(self) -> str:
        return "hardcoded_ip_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IP_RE.findall(_code(context.baseline_output))),
            len(_IP_RE.findall(_code(output))),
        )


class HardcodedUrlCountCheck:
    @property
    def name(self) -> str:
        return "hardcoded_url_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_URL_RE.findall(_code(context.baseline_output))),
            len(_URL_RE.findall(_code(output))),
        )


class MagicStringCountCheck:
    """Count of string literals of length >= 4 (potential magic strings)."""

    @property
    def name(self) -> str:
        return "magic_string_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STR_RE.findall(_code(context.baseline_output))),
            len(_STR_RE.findall(_code(output))),
        )


class LongParameterListCountCheck:
    """Count of functions with more than 5 parameters."""

    @property
    def name(self) -> str:
        return "long_param_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            n = 0
            for m in _FUNC_RE.finditer(_code(text)):
                params = [
                    p for p in m.group(2).split(",") if p.strip() and p.strip() != "self"
                ]
                if len(params) > 5:
                    n += 1
            return n

        return _count_match(count(context.baseline_output), count(output))


class BooleanParamCountCheck:
    @property
    def name(self) -> str:
        return "bool_param_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOOL_PARAM_RE.findall(_code(context.baseline_output))),
            len(_BOOL_PARAM_RE.findall(_code(output))),
        )


class DeepConditionalCountCheck:
    """Count of conditionals nested deeper than 3 levels (by indent)."""

    @property
    def name(self) -> str:
        return "deep_cond_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            n = 0
            for ln in _code(text).splitlines():
                if _COND_RE.match(ln):
                    indent = len(ln) - len(ln.lstrip())
                    if indent >= 12:
                        n += 1
            return n

        return _count_match(count(context.baseline_output), count(output))


class GotoCountCheck:
    @property
    def name(self) -> str:
        return "goto_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GOTO_RE.findall(_code(context.baseline_output))),
            len(_GOTO_RE.findall(_code(output))),
        )


class GlobalVarCountCheck:
    @property
    def name(self) -> str:
        return "global_var_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_GLOBAL_RE.findall(_code(context.baseline_output))),
            len(_GLOBAL_RE.findall(_code(output))),
        )


class MutableDefaultCountCheck:
    @property
    def name(self) -> str:
        return "mutable_default_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MUTABLE_DEFAULT_RE.findall(_code(context.baseline_output))),
            len(_MUTABLE_DEFAULT_RE.findall(_code(output))),
        )


class BareExceptCountCheck:
    @property
    def name(self) -> str:
        return "bare_except_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BARE_EXCEPT_RE.findall(_code(context.baseline_output))),
            len(_BARE_EXCEPT_RE.findall(_code(output))),
        )


class PrintDebugCountCheck:
    @property
    def name(self) -> str:
        return "print_debug_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PRINT_RE.findall(_code(context.baseline_output))),
            len(_PRINT_RE.findall(_code(output))),
        )


class SleepCallCountCheck:
    @property
    def name(self) -> str:
        return "sleep_call_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SLEEP_RE.findall(_code(context.baseline_output))),
            len(_SLEEP_RE.findall(_code(output))),
        )


class EvalExecCountCheck:
    @property
    def name(self) -> str:
        return "eval_exec_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EVAL_RE.findall(_code(context.baseline_output))),
            len(_EVAL_RE.findall(_code(output))),
        )


class CommentedOutCodeCountCheck:
    @property
    def name(self) -> str:
        return "commented_code_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENTED_CODE_RE.findall(_code(context.baseline_output))),
            len(_COMMENTED_CODE_RE.findall(_code(output))),
        )
