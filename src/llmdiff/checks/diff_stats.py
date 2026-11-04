"""Diff-statistics dimensions: line-level add/remove/change and hunk structure."""

from __future__ import annotations

import difflib
import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "AddedLineCountCheck",
    "RemovedLineCountCheck",
    "ChangedLineCountCheck",
    "AddRemoveRatioCheck",
    "DiffHunkCountDeepCheck",
    "DiffLineCountCheck",
    "UnifiedDiffLineCountCheck",
    "ContextLineRatioCheck",
    "DiffSimilarityRatioCheck",
    "LongestAddedLineCheck",
    "LongestRemovedLineCheck",
    "AvgAddedLineLenCheck",
    "AvgRemovedLineLenCheck",
    "AddedWordCountCheck",
    "RemovedWordCountCheck",
    "WordChurnRatioCheck",
    "PlusMinusBalanceCheck",
    "DiffHeaderCountCheck",
    "FilePairCountCheck",
    "ChangeDensityCheck",
]

_MAX = 200_000
_WORD_RE = re.compile(r"\w+")


def _s(text: str) -> str:
    return text[:_MAX]


def _lines(text: str) -> list[str]:
    return _s(text).splitlines()


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _ratio_match(base: float, cand: float, tol: float = 0.05) -> float:
    if abs(base - cand) <= tol:
        return 1.0
    denom = max(abs(base), 1e-9)
    return max(0.0, 1.0 - abs(cand - base) / denom)


def _added(text: str) -> list[str]:
    return [ln for ln in _lines(text) if ln.startswith("+") and not ln.startswith("+++")]


def _removed(text: str) -> list[str]:
    return [ln for ln in _lines(text) if ln.startswith("-") and not ln.startswith("---")]


def _changed(text: str) -> list[str]:
    return [ln for ln in _lines(text) if ln.startswith("!")]


class AddedLineCountCheck:
    @property
    def name(self) -> str:
        return "added_lines"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_added(context.baseline_output)), len(_added(output)))


class RemovedLineCountCheck:
    @property
    def name(self) -> str:
        return "removed_lines"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_removed(context.baseline_output)), len(_removed(output)))


class ChangedLineCountCheck:
    @property
    def name(self) -> str:
        return "changed_lines"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(len(_changed(context.baseline_output)), len(_changed(output)))


class AddRemoveRatioCheck:
    @property
    def name(self) -> str:
        return "add_remove_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            a, r = len(_added(text)), len(_removed(text))
            total = a + r
            return a / total if total else 0.0

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class DiffHunkCountDeepCheck:
    @property
    def name(self) -> str:
        return "diff_hunk_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^@@", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class DiffLineCountCheck:
    """Total lines that are part of the diff (start with +, -, or !)."""

    @property
    def name(self) -> str:
        return "diff_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for ln in _lines(text) if ln[:1] in ("+", "-", "!"))

        return _count_match(count(context.baseline_output), count(output))


class UnifiedDiffLineCountCheck:
    """Count of lines in a unified diff including headers and context."""

    @property
    def name(self) -> str:
        return "unified_line_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            if "@@" not in s and not s.startswith(("--- ", "+++ ")):
                return 0
            return len(_lines(text))

        return _count_match(count(context.baseline_output), count(output))


class ContextLineRatioCheck:
    """Ratio of unchanged (context) lines to total lines in a unified diff."""

    @property
    def name(self) -> str:
        return "context_line_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            lines = _lines(text)
            if not lines:
                return 0.0
            context = sum(1 for ln in lines if ln[:1] == " ")
            return context / len(lines)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class DiffSimilarityRatioCheck:
    """difflib SequenceMatcher ratio between baseline and output."""

    @property
    def name(self) -> str:
        return "diff_similarity"

    def score(self, output: str, context: CheckContext) -> float:
        return difflib.SequenceMatcher(None, _s(context.baseline_output), _s(output)).ratio()


class LongestAddedLineCheck:
    @property
    def name(self) -> str:
        return "longest_added"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            a = _added(text)
            return max((len(ln) for ln in a), default=0)

        return _count_match(mx(context.baseline_output), mx(output))


class LongestRemovedLineCheck:
    @property
    def name(self) -> str:
        return "longest_removed"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            r = _removed(text)
            return max((len(ln) for ln in r), default=0)

        return _count_match(mx(context.baseline_output), mx(output))


class AvgAddedLineLenCheck:
    @property
    def name(self) -> str:
        return "avg_added_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            a = _added(text)
            return sum(len(ln) for ln in a) / len(a) if a else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class AvgRemovedLineLenCheck:
    @property
    def name(self) -> str:
        return "avg_removed_len"

    def score(self, output: str, context: CheckContext) -> float:
        def avg(text: str) -> float:
            r = _removed(text)
            return sum(len(ln) for ln in r) / len(r) if r else 0.0

        base, cand = avg(context.baseline_output), avg(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)


class AddedWordCountCheck:
    @property
    def name(self) -> str:
        return "added_words"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(len(_WORD_RE.findall(ln)) for ln in _added(text))

        return _count_match(count(context.baseline_output), count(output))


class RemovedWordCountCheck:
    @property
    def name(self) -> str:
        return "removed_words"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(len(_WORD_RE.findall(ln)) for ln in _removed(text))

        return _count_match(count(context.baseline_output), count(output))


class WordChurnRatioCheck:
    """(added+removed words) / total words in the diff text."""

    @property
    def name(self) -> str:
        return "word_churn"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            s = _s(text)
            total = len(_WORD_RE.findall(s))
            if total == 0:
                return 0.0
            churn = sum(len(_WORD_RE.findall(ln)) for ln in _added(text) + _removed(text))
            return churn / total

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class PlusMinusBalanceCheck:
    """balance = added - removed; compare to baseline."""

    @property
    def name(self) -> str:
        return "plus_minus_balance"

    def score(self, output: str, context: CheckContext) -> float:
        def balance(text: str) -> int:
            return len(_added(text)) - len(_removed(text))

        base, cand = balance(context.baseline_output), balance(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, abs(base)))


class DiffHeaderCountCheck:
    """Count of '--- '/'+++ ' file header lines."""

    @property
    def name(self) -> str:
        return "diff_header_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^(?:--- |\+\+\+ )", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class FilePairCountCheck:
    """Count of 'diff --git' markers (number of file pairs)."""

    @property
    def name(self) -> str:
        return "file_pair_count"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^diff --git ", re.MULTILINE)
        return _count_match(
            len(pat.findall(_s(context.baseline_output))),
            len(pat.findall(_s(output))),
        )


class ChangeDensityCheck:
    """Changed lines per hunk."""

    @property
    def name(self) -> str:
        return "change_density"

    def score(self, output: str, context: CheckContext) -> float:
        def density(text: str) -> float:
            hunks = len(re.findall(r"^@@", _s(text), re.MULTILINE))
            if hunks == 0:
                return 0.0
            changed = len(_added(text)) + len(_removed(text))
            return changed / hunks

        base, cand = density(context.baseline_output), density(output)
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return max(0.0, 1.0 - abs(cand - base) / base)
