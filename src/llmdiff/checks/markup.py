"""Markup/structure dimensions: headings, links, images, emphasis, and blocks."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "HeadingCountCheck",
    "HeadingLevelDistributionCheck",
    "BoldCountCheck",
    "ItalicCountCheck",
    "InlineCodeCountCheck",
    "CodeBlockCountMatchCheck",
    "LinkCountMatchCheck",
    "ImageCountCheck",
    "BlockquoteCountCheck",
    "HrCountCheck",
    "OrderedListCountCheck",
    "UnorderedListCountCheck",
    "TaskListItemCountCheck",
    "TableRowCountCheck",
    "TableColumnCountCheck",
    "HtmlTagCountCheck",
    "LatexInlineCountCheck",
    "LatexBlockCountCheck",
    "FootnoteCountCheck",
    "DefinitionListCountCheck",
]

_MAX = 200_000
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
_ITALIC_RE = re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]+\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_QUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_OL_RE = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)
_UL_RE = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
_TASK_RE = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s+\S", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>")
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")
_LATEX_BLOCK_RE = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
_DEFLIST_RE = re.compile(r"^\s*\S.*\n:\s+\S", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: int, cand: int) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class HeadingCountCheck:
    @property
    def name(self) -> str:
        return "heading_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HEADING_RE.findall(_s(context.baseline_output))),
            len(_HEADING_RE.findall(_s(output))),
        )


class HeadingLevelDistributionCheck:
    @property
    def name(self) -> str:
        return "heading_levels"

    def score(self, output: str, context: CheckContext) -> float:
        def dist(text: str) -> dict[int, int]:
            d: dict[int, int] = {}
            for m in re.finditer(r"^(#{1,6})\s", _s(text), re.MULTILINE):
                lvl = len(m.group(1))
                d[lvl] = d.get(lvl, 0) + 1
            return d

        base, cand = dist(context.baseline_output), dist(output)
        if not base and not cand:
            return 1.0
        keys = set(base) | set(cand)
        total = sum(base.values()) + sum(cand.values())
        if total == 0:
            return 1.0
        diff = sum(abs(base.get(k, 0) - cand.get(k, 0)) for k in keys)
        return max(0.0, 1.0 - diff / total)


class BoldCountCheck:
    @property
    def name(self) -> str:
        return "bold_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOLD_RE.findall(_s(context.baseline_output))),
            len(_BOLD_RE.findall(_s(output))),
        )


class ItalicCountCheck:
    @property
    def name(self) -> str:
        return "italic_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ITALIC_RE.findall(_s(context.baseline_output))),
            len(_ITALIC_RE.findall(_s(output))),
        )


class InlineCodeCountCheck:
    @property
    def name(self) -> str:
        return "inline_code_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_CODE_RE.findall(_s(context.baseline_output))),
            len(_INLINE_CODE_RE.findall(_s(output))),
        )


class CodeBlockCountMatchCheck:
    @property
    def name(self) -> str:
        return "code_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return len(_FENCE_RE.findall(_s(text))) // 2

        return _count_match(count(context.baseline_output), count(output))


class LinkCountMatchCheck:
    @property
    def name(self) -> str:
        return "link_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LINK_RE.findall(_s(context.baseline_output))),
            len(_LINK_RE.findall(_s(output))),
        )


class ImageCountCheck:
    @property
    def name(self) -> str:
        return "image_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IMAGE_RE.findall(_s(context.baseline_output))),
            len(_IMAGE_RE.findall(_s(output))),
        )


class BlockquoteCountCheck:
    @property
    def name(self) -> str:
        return "blockquote_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTE_RE.findall(_s(context.baseline_output))),
            len(_QUOTE_RE.findall(_s(output))),
        )


class HrCountCheck:
    @property
    def name(self) -> str:
        return "hr_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HR_RE.findall(_s(context.baseline_output))),
            len(_HR_RE.findall(_s(output))),
        )


class OrderedListCountCheck:
    @property
    def name(self) -> str:
        return "ordered_list_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_OL_RE.findall(_s(context.baseline_output))),
            len(_OL_RE.findall(_s(output))),
        )


class UnorderedListCountCheck:
    @property
    def name(self) -> str:
        return "unordered_list_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_UL_RE.findall(_s(context.baseline_output))),
            len(_UL_RE.findall(_s(output))),
        )


class TaskListItemCountCheck:
    @property
    def name(self) -> str:
        return "task_list_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TASK_RE.findall(_s(context.baseline_output))),
            len(_TASK_RE.findall(_s(output))),
        )


class TableRowCountCheck:
    @property
    def name(self) -> str:
        return "table_row_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TABLE_ROW_RE.findall(_s(context.baseline_output))),
            len(_TABLE_ROW_RE.findall(_s(output))),
        )


class TableColumnCountCheck:
    @property
    def name(self) -> str:
        return "table_col_count"

    def score(self, output: str, context: CheckContext) -> float:
        def max_cols(text: str) -> int:
            best = 0
            for m in _TABLE_ROW_RE.finditer(_s(text)):
                row = m.group(0)
                best = max(best, row.count("|") - 1)
            return best

        base, cand = max_cols(context.baseline_output), max_cols(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class HtmlTagCountCheck:
    @property
    def name(self) -> str:
        return "html_tag_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HTML_TAG_RE.findall(_s(context.baseline_output))),
            len(_HTML_TAG_RE.findall(_s(output))),
        )


class LatexInlineCountCheck:
    @property
    def name(self) -> str:
        return "latex_inline_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LATEX_INLINE_RE.findall(_s(context.baseline_output))),
            len(_LATEX_INLINE_RE.findall(_s(output))),
        )


class LatexBlockCountCheck:
    @property
    def name(self) -> str:
        return "latex_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LATEX_BLOCK_RE.findall(_s(context.baseline_output))),
            len(_LATEX_BLOCK_RE.findall(_s(output))),
        )


class FootnoteCountCheck:
    @property
    def name(self) -> str:
        return "footnote_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FOOTNOTE_RE.findall(_s(context.baseline_output))),
            len(_FOOTNOTE_RE.findall(_s(output))),
        )


class DefinitionListCountCheck:
    @property
    def name(self) -> str:
        return "deflist_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DEFLIST_RE.findall(_s(context.baseline_output))),
            len(_DEFLIST_RE.findall(_s(output))),
        )
