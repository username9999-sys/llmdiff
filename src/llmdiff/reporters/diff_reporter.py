"""Unified-diff report renderer.

For every regressed case, renders a unified diff (via
:func:`difflib.unified_diff`) between the baseline and candidate outputs,
wrapped in fenced code blocks so the result is paste-able into markdown.

:class:`CaseDiff` only carries check results, not the raw outputs, so the
reporter takes an ``outputs`` mapping (``case_id -> (baseline, candidate)``)
— typically built from the :class:`~llmdiff.core.corpus.Corpus` that was
evaluated. Use :meth:`from_corpus` to build one conveniently.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping

from llmdiff.core.result import CaseDiff, DiffResult
from llmdiff.reporters.base import BaseReporter, escape_md_cell

__all__ = ["DiffReporter"]

Outputs = Mapping[str, tuple[str, str]]


class DiffReporter(BaseReporter):
    """Render unified diffs of baseline vs candidate for regressed cases."""

    name = "diff"

    def __init__(
        self,
        outputs: Outputs | None = None,
        *,
        context_lines: int = 3,
        fence: bool = True,
        max_cases: int | None = None,
    ) -> None:
        self.outputs = dict(outputs or {})
        self.context_lines = context_lines
        self.fence = fence
        self.max_cases = max_cases

    @classmethod
    def from_corpus(cls, corpus, **kwargs) -> DiffReporter:
        """Build a reporter whose outputs come from ``corpus``.

        ``corpus`` is any iterable of :class:`~llmdiff.core.case.Case` (or a
        :class:`~llmdiff.core.corpus.Corpus`).
        """
        outputs = {case.id: (case.baseline_output, case.candidate_output) for case in corpus}
        return cls(outputs, **kwargs)

    def render(self, result: DiffResult) -> str:
        """Render markdown with a fenced unified diff per regressed case."""
        lines: list[str] = ["# Output diffs (regressed cases)", ""]
        regressed = result.regressions
        if self.max_cases is not None:
            regressed = regressed[: self.max_cases]
        if not regressed:
            lines.append("_No regressed cases — nothing to diff._")
            return "\n".join(lines) + "\n"
        for case in regressed:
            lines.extend(self._case_diff(case))
        return "\n".join(lines).rstrip() + "\n"

    def render_plain(self, result: DiffResult) -> str:
        """Render the same diffs without markdown fences (for terminals/logs)."""
        saved = self.fence
        self.fence = False
        try:
            return self.render(result)
        finally:
            self.fence = saved

    def diff_case(self, case: CaseDiff) -> list[str]:
        """Return the unified-diff lines for one regressed case."""
        baseline, candidate = self.outputs.get(case.case_id, ("", ""))
        return list(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                candidate.splitlines(keepends=True),
                fromfile=f"baseline/{case.case_id}",
                tofile=f"candidate/{case.case_id}",
                n=self.context_lines,
            )
        )

    def _case_diff(self, case: CaseDiff) -> list[str]:
        diff_lines = self.diff_case(case)
        lines = [f"## `{escape_md_cell(case.case_id)}` ({case.verdict})", ""]
        if self.fence:
            lines.append("```diff")
            lines.extend(line.rstrip("\n") for line in diff_lines)
            lines.append("```")
        else:
            lines.extend(line.rstrip("\n") for line in diff_lines)
        lines.append("")
        return lines
