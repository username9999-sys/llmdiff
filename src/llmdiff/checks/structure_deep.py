"""Deep structural dimensions: nesting, ordering, and hierarchy of text blocks."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "MaxHeadingLevelCheck",
    "MinHeadingLevelCheck",
    "HeadingLevelSequenceCheck",
    "HeadingSkipCountCheck",
    "ListNestingDepthCheck",
    "ListItemIndentConsistencyCheck",
    "CodeFenceBalanceCheck",
    "QuoteNestingDepthCheck",
    "TableColumnConsistencyCheck",
    "ParagraphOrderSignatureCheck",
    "SentenceOrderSignatureCheck",
    "BlockOrderSignatureCheck",
    "FirstLineMatchCheck",
    "LastLineMatchCheck",
    "LineOrderSignatureCheck",
    "WordOrderSignatureCheck",
    "SectionOrderJaccardCheck",
    "DuplicateSectionCheck",
    "MonotonicNumberCheck",
    "IndentLevelDiversityCheck",
]

_MAX = 200_000
_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$", re.MULTILINE)
_LIST_RE = re.compile(r"^(\s*)[-*+]\s+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_QUOTE_RE = re.compile(r"^(>+)\s?\S", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")
_WORD_RE = re.compile(r"\w+")


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


def _headings(text: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(_s(text))]


class MaxHeadingLevelCheck:
    @property
    def name(self) -> str:
        return "max_heading_level"

    def score(self, output: str, context: CheckContext) -> float:
        def mx(text: str) -> int:
            hs = _headings(text)
            return max((h[0] for h in hs), default=0)

        return _count_match(mx(context.baseline_output), mx(output))


class MinHeadingLevelCheck:
    @property
    def name(self) -> str:
        return "min_heading_level"

    def score(self, output: str, context: CheckContext) -> float:
        def mn(text: str) -> int:
            hs = _headings(text)
            return min((h[0] for h in hs), default=0)

        return _count_match(mn(context.baseline_output), mn(output))


class HeadingLevelSequenceCheck:
    """1.0 when the sequence of heading levels matches the baseline."""

    @property
    def name(self) -> str:
        return "heading_seq"

    def score(self, output: str, context: CheckContext) -> float:
        base = [h[0] for h in _headings(context.baseline_output)]
        cand = [h[0] for h in _headings(output)]
        if not base and not cand:
            return 1.0
        if len(base) != len(cand):
            return max(0.0, 1.0 - abs(len(base) - len(cand)) / max(len(base), len(cand)))
        match = sum(1 for a, b in zip(base, cand, strict=False) if a == b)
        return match / len(base)


class HeadingSkipCountCheck:
    """Count of heading-level jumps greater than 1 (e.g. # to ###)."""

    @property
    def name(self) -> str:
        return "heading_skips"

    def score(self, output: str, context: CheckContext) -> float:
        def skips(text: str) -> int:
            levels = [h[0] for h in _headings(text)]
            return sum(1 for a, b in zip(levels, levels[1:], strict=False) if b > a + 1)

        return _count_match(skips(context.baseline_output), skips(output))


class ListNestingDepthCheck:
    @property
    def name(self) -> str:
        return "list_nesting"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = 0
            for m in _LIST_RE.finditer(_s(text)):
                indent = len(m.group(1))
                best = max(best, indent // 2 + 1)
            return best

        return _count_match(depth(context.baseline_output), depth(output))


class ListItemIndentConsistencyCheck:
    """1.0 when the set of list indent widths matches the baseline."""

    @property
    def name(self) -> str:
        return "list_indent_set"

    def score(self, output: str, context: CheckContext) -> float:
        def indents(text: str) -> set[int]:
            return {len(m.group(1)) for m in _LIST_RE.finditer(_s(text))}

        a, b = indents(context.baseline_output), indents(output)
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class CodeFenceBalanceCheck:
    """1.0 when the number of code fences is even in both (balanced blocks)."""

    @property
    def name(self) -> str:
        return "fence_balance"

    def score(self, output: str, context: CheckContext) -> float:
        def balanced(text: str) -> bool:
            return len(_FENCE_RE.findall(_s(text))) % 2 == 0

        return 1.0 if balanced(context.baseline_output) == balanced(output) else 0.0


class QuoteNestingDepthCheck:
    @property
    def name(self) -> str:
        return "quote_nesting"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            return max((len(m.group(1)) for m in _QUOTE_RE.finditer(_s(text))), default=0)

        return _count_match(depth(context.baseline_output), depth(output))


class TableColumnConsistencyCheck:
    """1.0 when all table rows have the same column count as the header."""

    @property
    def name(self) -> str:
        return "table_col_consistency"

    def score(self, output: str, context: CheckContext) -> float:
        def consistency(text: str) -> float:
            rows = [m.group(0) for m in _TABLE_ROW_RE.finditer(_s(text))]
            if len(rows) < 2:
                return 1.0
            cols = [r.count("|") for r in rows]
            header = cols[0]
            return sum(1 for c in cols if c == header) / len(cols)

        return _ratio_match(consistency(context.baseline_output), consistency(output))


class ParagraphOrderSignatureCheck:
    """Jaccard of paragraph first-word sets (order-insensitive signature)."""

    @property
    def name(self) -> str:
        return "para_order_sig"

    def score(self, output: str, context: CheckContext) -> float:
        def sig(text: str) -> set[str]:
            paras = [p.strip() for p in _s(text).split("\n\n") if p.strip()]
            return {p.split()[0].lower() for p in paras if p.split()}

        a, b = sig(context.baseline_output), sig(output)
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class SentenceOrderSignatureCheck:
    @property
    def name(self) -> str:
        return "sent_order_sig"

    def score(self, output: str, context: CheckContext) -> float:
        def sig(text: str) -> list[str]:
            return [s.strip()[:20].lower() for s in _SENT_RE.findall(_s(text)) if s.strip()]

        base, cand = sig(context.baseline_output), sig(output)
        if not base and not cand:
            return 1.0
        if len(base) != len(cand):
            return max(0.0, 1.0 - abs(len(base) - len(cand)) / max(len(base), len(cand)))
        match = sum(1 for a, b in zip(base, cand, strict=False) if a == b)
        return match / len(base)


class BlockOrderSignatureCheck:
    """Order signature of non-empty lines (first 10 chars each)."""

    @property
    def name(self) -> str:
        return "block_order_sig"

    def score(self, output: str, context: CheckContext) -> float:
        def sig(text: str) -> list[str]:
            return [ln.strip()[:10] for ln in _s(text).splitlines() if ln.strip()]

        base, cand = sig(context.baseline_output), sig(output)
        if not base and not cand:
            return 1.0
        if len(base) != len(cand):
            return max(0.0, 1.0 - abs(len(base) - len(cand)) / max(len(base), len(cand)))
        match = sum(1 for a, b in zip(base, cand, strict=False) if a == b)
        return match / len(base)


class FirstLineMatchCheck:
    @property
    def name(self) -> str:
        return "first_line"

    def score(self, output: str, context: CheckContext) -> float:
        def first(text: str) -> str:
            lines = [ln for ln in _s(text).splitlines() if ln.strip()]
            return lines[0].strip() if lines else ""

        base, cand = first(context.baseline_output), first(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class LastLineMatchCheck:
    @property
    def name(self) -> str:
        return "last_line"

    def score(self, output: str, context: CheckContext) -> float:
        def last(text: str) -> str:
            lines = [ln for ln in _s(text).splitlines() if ln.strip()]
            return lines[-1].strip() if lines else ""

        base, cand = last(context.baseline_output), last(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class LineOrderSignatureCheck:
    """Fraction of lines that appear in the same relative order as baseline."""

    @property
    def name(self) -> str:
        return "line_order_sig"

    def score(self, output: str, context: CheckContext) -> float:
        def lines(text: str) -> list[str]:
            return [ln.strip() for ln in _s(text).splitlines() if ln.strip()]

        base, cand = lines(context.baseline_output), lines(output)
        if not base and not cand:
            return 1.0
        base_pos = {ln: i for i, ln in enumerate(base)}
        common = [ln for ln in cand if ln in base_pos]
        if not common:
            return 0.0 if base else 1.0
        positions = [base_pos[ln] for ln in common]
        inversions = sum(
            1
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
            if positions[i] > positions[j]
        )
        max_inv = len(positions) * (len(positions) - 1) // 2
        return 1.0 - inversions / max_inv if max_inv else 1.0


class WordOrderSignatureCheck:
    @property
    def name(self) -> str:
        return "word_order_sig"

    def score(self, output: str, context: CheckContext) -> float:
        def words(text: str) -> list[str]:
            return [w.lower() for w in _WORD_RE.findall(_s(text))]

        base, cand = words(context.baseline_output), words(output)
        if not base and not cand:
            return 1.0
        base_pos = {w: i for i, w in enumerate(base)}
        common = [w for w in cand if w in base_pos]
        if not common:
            return 0.0 if base else 1.0
        positions = [base_pos[w] for w in common]
        inversions = sum(
            1
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
            if positions[i] > positions[j]
        )
        max_inv = len(positions) * (len(positions) - 1) // 2
        return 1.0 - inversions / max_inv if max_inv else 1.0


class SectionOrderJaccardCheck:
    """Jaccard of heading-text sets between baseline and output."""

    @property
    def name(self) -> str:
        return "section_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {h[1].lower() for h in _headings(context.baseline_output)}
        b = {h[1].lower() for h in _headings(output)}
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class DuplicateSectionCheck:
    """Count of duplicated heading texts."""

    @property
    def name(self) -> str:
        return "dup_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            titles = [h[1].lower() for h in _headings(text)]
            seen: set[str] = set()
            dups = 0
            for t in titles:
                if t in seen:
                    dups += 1
                else:
                    seen.add(t)
            return dups

        return _count_match(count(context.baseline_output), count(output))


class MonotonicNumberCheck:
    """1.0 when the sequence of leading numbers in lines is monotonic in both."""

    @property
    def name(self) -> str:
        return "monotonic_numbers"

    def score(self, output: str, context: CheckContext) -> float:
        def is_monotonic(text: str) -> bool:
            nums = [
                int(m.group(1))
                for ln in _s(text).splitlines()
                if (m := re.match(r"^\s*(\d+)[.)]\s", ln))
            ]
            return all(b >= a for a, b in zip(nums, nums[1:], strict=False)) if nums else True

        return 1.0 if is_monotonic(context.baseline_output) == is_monotonic(output) else 0.0


class IndentLevelDiversityCheck:
    """Number of distinct indentation levels used."""

    @property
    def name(self) -> str:
        return "indent_diversity"

    def score(self, output: str, context: CheckContext) -> float:
        def diversity(text: str) -> int:
            return len(
                {len(ln) - len(ln.lstrip()) for ln in _s(text).splitlines() if ln.strip()}
            )

        return _count_match(diversity(context.baseline_output), diversity(output))
