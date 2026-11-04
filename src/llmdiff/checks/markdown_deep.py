"""Deep markdown dimensions: reference links, task lists, admonitions, and GFM extras."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "MdRefLinkDefCountCheck",
    "MdRefLinkUseCountCheck",
    "MdCollapsedRefCountCheck",
    "MdTaskDoneCountCheck",
    "MdTaskTodoCountCheck",
    "MdAdmonitionCountCheck",
    "MdAlertCountCheck",
    "MdTablePipeCountCheck",
    "MdTableSeparatorRowCountCheck",
    "MdFootnoteUseCountCheck",
    "MdFootnoteDefCountCheck",
    "MdHeadingATXCountCheck",
    "MdHeadingSetextCountCheck",
    "MdCodeFenceLangCountCheck",
    "MdInlineCodeCountCheck",
    "MdLinkAutolinkCountCheck",
    "MdImageMdCountCheck",
    "MdHtmlBlockCountCheck",
    "MdHrThematicCountCheck",
    "MdBlockquoteNestCountCheck",
]

_MAX = 200_000
_REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S+", re.MULTILINE)
_REF_USE_RE = re.compile(r"\[[^\]]+\]\[[^\]]*\]")
_COLLAPSED_RE = re.compile(r"\[[^\]]+\]\[\]")
_TASK_DONE_RE = re.compile(r"^\s*[-*+]\s+\[[xX]\]\s", re.MULTILINE)
_TASK_TODO_RE = re.compile(r"^\s*[-*+]\s+\[ \]\s", re.MULTILINE)
_ADMON_RE = re.compile(
    r"^:::\s*(?:note|warning|tip|caution|important|danger)",
    re.MULTILINE | re.IGNORECASE,
)
_ALERT_RE = re.compile(
    r"^>\s*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]",
    re.MULTILINE | re.IGNORECASE,
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$", re.MULTILINE)
_FN_USE_RE = re.compile(r"\[\^[^\]]+\](?!:)")
_FN_DEF_RE = re.compile(r"^\s{0,3}\[\^[^\]]+\]:", re.MULTILINE)
_ATX_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_SETEXT_RE = re.compile(r"^\S.*\n^\s*(?:=+|-+)\s*$", re.MULTILINE)
_FENCE_LANG_RE = re.compile(r"^```(\w+)", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_AUTOLINK_RE = re.compile(r"<(?:https?://[^>]+|[\w.+-]+@[\w-]+\.[\w.]+)>")
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_BLOCK_RE = re.compile(
    r"^\s*<(?:div|table|details|summary|p|section|article)\b",
    re.MULTILINE | re.IGNORECASE,
)
_HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_QUOTE_NEST_RE = re.compile(r"^(>+)\s?\S", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class MdRefLinkDefCountCheck:
    @property
    def name(self) -> str:
        return "md_reflink_def_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_REF_DEF_RE.findall(_s(context.baseline_output))),
            len(_REF_DEF_RE.findall(_s(output))),
        )


class MdRefLinkUseCountCheck:
    @property
    def name(self) -> str:
        return "md_reflink_use_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_REF_USE_RE.findall(_s(context.baseline_output))),
            len(_REF_USE_RE.findall(_s(output))),
        )


class MdCollapsedRefCountCheck:
    @property
    def name(self) -> str:
        return "md_collapsed_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COLLAPSED_RE.findall(_s(context.baseline_output))),
            len(_COLLAPSED_RE.findall(_s(output))),
        )


class MdTaskDoneCountCheck:
    @property
    def name(self) -> str:
        return "md_task_done_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TASK_DONE_RE.findall(_s(context.baseline_output))),
            len(_TASK_DONE_RE.findall(_s(output))),
        )


class MdTaskTodoCountCheck:
    @property
    def name(self) -> str:
        return "md_task_todo_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TASK_TODO_RE.findall(_s(context.baseline_output))),
            len(_TASK_TODO_RE.findall(_s(output))),
        )


class MdAdmonitionCountCheck:
    @property
    def name(self) -> str:
        return "md_admonition_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ADMON_RE.findall(_s(context.baseline_output))),
            len(_ADMON_RE.findall(_s(output))),
        )


class MdAlertCountCheck:
    @property
    def name(self) -> str:
        return "md_alert_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ALERT_RE.findall(_s(context.baseline_output))),
            len(_ALERT_RE.findall(_s(output))),
        )


class MdTablePipeCountCheck:
    @property
    def name(self) -> str:
        return "md_table_pipe_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TABLE_ROW_RE.findall(_s(context.baseline_output))),
            len(_TABLE_ROW_RE.findall(_s(output))),
        )


class MdTableSeparatorRowCountCheck:
    @property
    def name(self) -> str:
        return "md_table_sep_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TABLE_SEP_RE.findall(_s(context.baseline_output))),
            len(_TABLE_SEP_RE.findall(_s(output))),
        )


class MdFootnoteUseCountCheck:
    @property
    def name(self) -> str:
        return "md_fn_use_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FN_USE_RE.findall(_s(context.baseline_output))),
            len(_FN_USE_RE.findall(_s(output))),
        )


class MdFootnoteDefCountCheck:
    @property
    def name(self) -> str:
        return "md_fn_def_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FN_DEF_RE.findall(_s(context.baseline_output))),
            len(_FN_DEF_RE.findall(_s(output))),
        )


class MdHeadingATXCountCheck:
    @property
    def name(self) -> str:
        return "md_atx_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ATX_RE.findall(_s(context.baseline_output))),
            len(_ATX_RE.findall(_s(output))),
        )


class MdHeadingSetextCountCheck:
    @property
    def name(self) -> str:
        return "md_setext_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SETEXT_RE.findall(_s(context.baseline_output))),
            len(_SETEXT_RE.findall(_s(output))),
        )


class MdCodeFenceLangCountCheck:
    @property
    def name(self) -> str:
        return "md_fence_lang_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FENCE_LANG_RE.findall(_s(context.baseline_output))),
            len(_FENCE_LANG_RE.findall(_s(output))),
        )


class MdInlineCodeCountCheck:
    @property
    def name(self) -> str:
        return "md_inline_code_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_CODE_RE.findall(_s(context.baseline_output))),
            len(_INLINE_CODE_RE.findall(_s(output))),
        )


class MdLinkAutolinkCountCheck:
    @property
    def name(self) -> str:
        return "md_autolink_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_AUTOLINK_RE.findall(_s(context.baseline_output))),
            len(_AUTOLINK_RE.findall(_s(output))),
        )


class MdImageMdCountCheck:
    @property
    def name(self) -> str:
        return "md_image_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IMG_MD_RE.findall(_s(context.baseline_output))),
            len(_IMG_MD_RE.findall(_s(output))),
        )


class MdHtmlBlockCountCheck:
    @property
    def name(self) -> str:
        return "md_html_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HTML_BLOCK_RE.findall(_s(context.baseline_output))),
            len(_HTML_BLOCK_RE.findall(_s(output))),
        )


class MdHrThematicCountCheck:
    @property
    def name(self) -> str:
        return "md_hr_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HR_RE.findall(_s(context.baseline_output))),
            len(_HR_RE.findall(_s(output))),
        )


class MdBlockquoteNestCountCheck:
    @property
    def name(self) -> str:
        return "md_quote_nest_count"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            return max(
                (len(m.group(1)) for m in _QUOTE_NEST_RE.finditer(_s(text))),
                default=0,
            )

        return _count_match(depth(context.baseline_output), depth(output))
