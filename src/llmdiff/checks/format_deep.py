"""Deep format dimensions: indentation, alignment, quoting, and delimiter balance."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "IndentWidthMatchCheck",
    "IndentConsistencyCheck",
    "MixedIndentCheck",
    "TrailingWhitespaceLinesCheck",
    "DoubleSpaceCountCheck",
    "SpaceBeforePunctCountCheck",
    "SpaceAfterPunctCountCheck",
    "BracketSpacingCheck",
    "CommaSpacingCheck",
    "ColonSpacingCheck",
    "QuotePairBalanceCheck",
    "BracketBalanceCheck",
    "BraceBalanceDeepCheck",
    "SquareBalanceCheck",
    "AngleBalanceCheck",
    "BacktickBalanceCheck",
    "AsteriskBalanceCheck",
    "UnderscoreBalanceCheck",
    "PipeBalanceCheck",
    "DollarBalanceCheck",
]

_MAX = 200_000
_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    "`": "`",
    "*": "*",
    "_": "_",
    "|": "|",
    "$": "$",
    '"': '"',
    "'": "'",
}


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _balance(text: str, ch: str) -> int:
    return _s(text).count(ch) - _s(text).count(_PAIRS[ch])


class IndentWidthMatchCheck:
    """1.0 when the most common indent width matches the baseline."""

    @property
    def name(self) -> str:
        return "indent_width"

    def score(self, output: str, context: CheckContext) -> float:
        def width(text: str) -> int:
            counts: dict[int, int] = {}
            for ln in _s(text).splitlines():
                stripped = ln.lstrip(" ")
                if stripped and len(ln) != len(stripped):
                    w = len(ln) - len(stripped)
                    counts[w] = counts.get(w, 0) + 1
            return max(counts, key=lambda k: counts[k]) if counts else 0

        base, cand = width(context.baseline_output), width(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class IndentConsistencyCheck:
    """1.0 when the ratio of space-indented lines matches the baseline."""

    @property
    def name(self) -> str:
        return "indent_consistency"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            indented = [ln for ln in _s(text).splitlines() if ln[:1] in (" ", "\t")]
            if not indented:
                return 1.0
            return sum(1 for ln in indented if ln.startswith(" ")) / len(indented)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class MixedIndentCheck:
    """1.0 when tab/space mixing presence matches the baseline."""

    @property
    def name(self) -> str:
        return "mixed_indent"

    def score(self, output: str, context: CheckContext) -> float:
        def has_mixed(text: str) -> bool:
            has_space = has_tab = False
            for ln in _s(text).splitlines():
                if ln.startswith(" "):
                    has_space = True
                if ln.startswith("\t"):
                    has_tab = True
            return has_space and has_tab

        base, cand = has_mixed(context.baseline_output), has_mixed(output)
        return 1.0 if base == cand else 0.0


class TrailingWhitespaceLinesCheck:
    @property
    def name(self) -> str:
        return "trailing_ws_lines"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for ln in _s(text).splitlines() if ln != ln.rstrip())

        return _count_match(count(context.baseline_output), count(output))


class DoubleSpaceCountCheck:
    @property
    def name(self) -> str:
        return "double_space_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"(?<=\S)  (?=\S)")
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class SpaceBeforePunctCountCheck:
    @property
    def name(self) -> str:
        return "space_before_punct"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\s+([,.!?;:])")
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class SpaceAfterPunctCountCheck:
    @property
    def name(self) -> str:
        return "space_after_punct"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"([,.!?;:])\S")
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class BracketSpacingCheck:
    """Ratio of brackets with a space just inside."""

    @property
    def name(self) -> str:
        return "bracket_spacing"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            opens = s.count("(") + s.count("[") + s.count("{")
            if opens == 0:
                return 1.0
            spaced = len(re.findall(r"[(\[{]\s", s))
            return spaced / opens

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class CommaSpacingCheck:
    @property
    def name(self) -> str:
        return "comma_spacing"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            commas = s.count(",")
            if commas == 0:
                return 1.0
            return len(re.findall(r",\s", s)) / commas

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class ColonSpacingCheck:
    @property
    def name(self) -> str:
        return "colon_spacing"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            colons = s.count(":")
            if colons == 0:
                return 1.0
            return len(re.findall(r":\s", s)) / colons

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class QuotePairBalanceCheck:
    @property
    def name(self) -> str:
        return "quote_balance"

    def score(self, output: str, context: CheckContext) -> float:
        def quote_count(text: str) -> int:
            s = _s(text)
            return s.count('"') + s.count("'")

        base, cand = quote_count(context.baseline_output), quote_count(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class BracketBalanceCheck:
    @property
    def name(self) -> str:
        return "paren_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _balance(context.baseline_output, "("), _balance(output, "(")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class BraceBalanceDeepCheck:
    @property
    def name(self) -> str:
        return "brace_balance_deep"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _balance(context.baseline_output, "{"), _balance(output, "{")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class SquareBalanceCheck:
    @property
    def name(self) -> str:
        return "square_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _balance(context.baseline_output, "["), _balance(output, "[")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class AngleBalanceCheck:
    @property
    def name(self) -> str:
        return "angle_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _balance(context.baseline_output, "<"), _balance(output, "<")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class BacktickBalanceCheck:
    @property
    def name(self) -> str:
        return "backtick_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _s(context.baseline_output).count("`"), _s(output).count("`")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class AsteriskBalanceCheck:
    @property
    def name(self) -> str:
        return "asterisk_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _s(context.baseline_output).count("*"), _s(output).count("*")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class UnderscoreBalanceCheck:
    @property
    def name(self) -> str:
        return "underscore_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _s(context.baseline_output).count("_"), _s(output).count("_")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class PipeBalanceCheck:
    @property
    def name(self) -> str:
        return "pipe_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _s(context.baseline_output).count("|"), _s(output).count("|")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class DollarBalanceCheck:
    @property
    def name(self) -> str:
        return "dollar_balance"

    def score(self, output: str, context: CheckContext) -> float:
        base, cand = _s(context.baseline_output).count("$"), _s(output).count("$")
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))
