"""Code-output comparison dimensions."""

from __future__ import annotations

import ast
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "PythonSyntaxCheck",
    "BraceBalanceCheck",
    "CodeBlockCountCheck",
    "CodeHasDefinitionsCheck",
    "FencedCodeCheck",
]

_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_MAX = 500_000


def extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(language, code)`` pairs for every fenced code block."""
    sample = text[:_MAX]
    return [(m.group(1) or "", m.group(2)) for m in _FENCE_RE.finditer(sample)]


class PythonSyntaxCheck:
    """1.0 when the output parses as valid Python, else 0.0.

    If the output contains fenced blocks, only ```python blocks are parsed.
    """

    @property
    def name(self) -> str:
        return "python_syntax"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        blocks = [
            code for lang, code in extract_fenced_blocks(sample) if lang in ("python", "py", "")
        ]
        candidates = blocks if blocks else [sample]
        # A candidate is valid if every extracted python block parses, or the
        # whole output parses when there are no blocks.
        ok = 0
        for cand in candidates:
            stripped = cand.strip()
            if not stripped:
                ok += 1
                continue
            try:
                ast.parse(stripped)
                ok += 1
            except SyntaxError:
                pass
        return ok / len(candidates) if candidates else 1.0


class BraceBalanceCheck:
    """1.0 when (), [], {} are balanced in the output, else proportional."""

    @property
    def name(self) -> str:
        return "brace_balance"

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        pairs = {")": "(", "]": "[", "}": "{"}
        stack: list[str] = []
        total = 0
        for ch in sample:
            if ch in "([{":
                stack.append(ch)
                total += 1
            elif ch in pairs:
                total += 1
                if stack and stack[-1] == pairs[ch]:
                    stack.pop()
        if total == 0:
            return 1.0
        matched = total - 2 * len(stack) if stack else total
        return max(0.0, matched / total)


class CodeBlockCountCheck:
    """1.0 when the number of fenced code blocks matches the baseline."""

    @property
    def name(self) -> str:
        return "code_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(extract_fenced_blocks(context.baseline_output))
        cand = len(extract_fenced_blocks(output))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class CodeHasDefinitionsCheck:
    """1.0 when the output contains def/class/import (or baseline had none)."""

    @property
    def name(self) -> str:
        return "code_definitions"

    _RE = re.compile(r"^\s*(def |class |import |from )", re.MULTILINE)

    def score(self, output: str, context: CheckContext) -> float:
        sample = output[:_MAX]
        has = bool(self._RE.search(sample))
        base_has = bool(self._RE.search(context.baseline_output[:_MAX]))
        if has == base_has:
            return 1.0
        return 0.0


class FencedCodeCheck:
    """1.0 when the output contains at least one fenced code block."""

    @property
    def name(self) -> str:
        return "fenced_code"

    def score(self, output: str, context: CheckContext) -> float:
        return 1.0 if extract_fenced_blocks(output) else 0.0
