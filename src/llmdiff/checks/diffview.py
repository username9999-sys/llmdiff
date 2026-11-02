"""Diff-view dimensions: edit distance and hunk counts vs the baseline."""

from __future__ import annotations

import difflib

from llmdiff.core.checks import CheckContext

__all__ = ["EditDistanceCheck", "DiffHunkCountCheck", "LineChangeRatioCheck"]

_MAX = 200_000


class EditDistanceCheck:
    """1.0 minus the normalized edit distance (difflib opcodes) vs baseline."""

    @property
    def name(self) -> str:
        return "edit_distance"

    def score(self, output: str, context: CheckContext) -> float:
        base = context.baseline_output[:_MAX]
        cand = output[:_MAX]
        if not base and not cand:
            return 1.0
        sm = difflib.SequenceMatcher(None, base, cand, autojunk=False)
        changed = sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag != "equal"
        )
        total = max(len(base), len(cand))
        if total == 0:
            return 1.0
        return max(0.0, 1.0 - changed / total)


class DiffHunkCountCheck:
    """1.0 when the candidate is identical to the baseline (0 hunks)."""

    @property
    def name(self) -> str:
        return "diff_hunks"

    def score(self, output: str, context: CheckContext) -> float:
        base = context.baseline_output[:_MAX].splitlines(keepends=True)
        cand = output[:_MAX].splitlines(keepends=True)
        hunks = sum(
            1
            for group in difflib.SequenceMatcher(
                None, base, cand, autojunk=False
            ).get_grouped_opcodes(3)
            if any(tag != "equal" for tag, *_ in group)
        )
        if hunks == 0:
            return 1.0
        return max(0.0, 1.0 - hunks / 20.0)


class LineChangeRatioCheck:
    """Fraction of baseline lines that survive unchanged in the candidate."""

    @property
    def name(self) -> str:
        return "line_survival"

    def score(self, output: str, context: CheckContext) -> float:
        base_lines = [
            ln.strip() for ln in context.baseline_output[:_MAX].splitlines() if ln.strip()
        ]
        if not base_lines:
            return 1.0
        cand_set = {ln.strip() for ln in output[:_MAX].splitlines()}
        survived = sum(1 for ln in base_lines if ln in cand_set)
        return survived / len(base_lines)
