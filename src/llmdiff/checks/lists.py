"""List and table structure dimensions."""

from __future__ import annotations

import re
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = ["ListItemCountCheck", "ListItemOverlapCheck", "ListOrderCheck", "TableShapeCheck"]

_MAX = 200_000
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def extract_list_items(text: str) -> list[str]:
    """Return normalized list item texts, capped for huge inputs."""
    return [m.group(1).strip().lower()[:200] for m in _LIST_RE.finditer(text[:_MAX])]


def extract_table_rows(text: str) -> list[list[str]]:
    """Return markdown table rows as lists of cell strings."""
    rows: list[list[str]] = []
    for line in text[:_MAX].splitlines():
        if _TABLE_ROW_RE.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            rows.append(cells)
    return rows


class ListItemCountCheck:
    """1.0 when list item count matches the baseline, else proportional."""

    @property
    def name(self) -> str:
        return "list_count"

    def score(self, output: str, context: CheckContext) -> float:
        base = len(extract_list_items(context.baseline_output))
        cand = len(extract_list_items(output))
        if base == 0:
            return 1.0 if cand == 0 else 0.0
        return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class ListItemOverlapCheck:
    """Jaccard overlap of normalized list items vs the baseline."""

    @property
    def name(self) -> str:
        return "list_overlap"

    def score(self, output: str, context: CheckContext) -> float:
        base = Counter(extract_list_items(context.baseline_output))
        cand = Counter(extract_list_items(output))
        if not base and not cand:
            return 1.0
        union = base | cand
        if not union:
            return 1.0
        inter = base & cand
        return sum(inter.values()) / sum(union.values())


class ListOrderCheck:
    """1.0 when the shared items appear in the same relative order."""

    @property
    def name(self) -> str:
        return "list_order"

    def score(self, output: str, context: CheckContext) -> float:
        base = extract_list_items(context.baseline_output)
        cand = extract_list_items(output)
        shared = [item for item in base if item in cand]
        if len(shared) < 2:
            return 1.0
        positions = [cand.index(item) for item in shared]
        in_order = all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))
        return 1.0 if in_order else 0.0


class TableShapeCheck:
    """1.0 when row and column counts of the markdown table match the baseline."""

    @property
    def name(self) -> str:
        return "table_shape"

    def score(self, output: str, context: CheckContext) -> float:
        base_rows = extract_table_rows(context.baseline_output)
        cand_rows = extract_table_rows(output)
        if not base_rows:
            return 1.0 if not cand_rows else 0.0
        if not cand_rows:
            return 0.0
        base_cols = len(base_rows[0])
        cand_cols = len(cand_rows[0])
        col_score = 1.0 if base_cols == cand_cols else max(
            0.0, 1.0 - abs(cand_cols - base_cols) / base_cols
        )
        row_score = 1.0 if len(base_rows) == len(cand_rows) else max(
            0.0, 1.0 - abs(len(cand_rows) - len(base_rows)) / len(base_rows)
        )
        return (col_score + row_score) / 2.0
