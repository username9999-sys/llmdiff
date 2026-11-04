"""Deep YAML dimensions: anchors, merge keys, multiline blocks, and nesting."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "YamlAnchorCountCheck",
    "YamlAliasCountCheck",
    "YamlMergeKeyCountCheck",
    "YamlMultilineBlockCountCheck",
    "YamlLiteralBlockCountCheck",
    "YamlFoldedBlockCountCheck",
    "YamlCommentCountCheck",
    "YamlDocMarkerCountCheck",
    "YamlTagCountCheck",
    "YamlQuotedStringCountCheck",
    "YamlFlowStyleCountCheck",
    "YamlNestedMapCountCheck",
    "YamlNestedSeqCountCheck",
    "YamlKeyWithDotCountCheck",
    "YamlBoolValueCountCheck",
    "YamlNullValueCountCheck",
    "YamlNumberValueCountCheck",
    "YamlIndentWidthCheck",
    "YamlMaxDepthCheck",
    "YamlKeyCountDeepCheck",
]

_MAX = 200_000
_ANCHOR_RE = re.compile(r"&[\w-]+")
_ALIAS_RE = re.compile(r"\*[\w-]+")
_MERGE_RE = re.compile(r"<<\s*:")
_LITERAL_RE = re.compile(r":\s*\|[+-]?\s*$", re.MULTILINE)
_FOLDED_RE = re.compile(r":\s*>[+-]?\s*$", re.MULTILINE)
_COMMENT_RE = re.compile(r"^\s*#", re.MULTILINE)
_DOC_RE = re.compile(r"^(?:---|\.\.\.)\s*$", re.MULTILINE)
_TAG_RE = re.compile(r"!!?\w+")
_QUOTED_RE = re.compile(r":\s*(?:\"[^\"]*\"|'[^']*')")
_FLOW_RE = re.compile(r"[\{\[]", re.MULTILINE)
_MAP_RE = re.compile(r"^\s*[\w.-]+:\s*$", re.MULTILINE)
_SEQ_RE = re.compile(r"^\s*-\s", re.MULTILINE)
_DOTKEY_RE = re.compile(r"^\s*[\w-]+\.[\w.-]+\s*:", re.MULTILINE)
_BOOLVAL_RE = re.compile(
    r":\s*(?:true|false|yes|no|on|off)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_NULLVAL_RE = re.compile(r":\s*(?:null|~|Null|NULL)\s*$", re.MULTILINE)
_NUMVAL_RE = re.compile(r":\s*-?\d+(?:\.\d+)?\s*$", re.MULTILINE)
_KEY_RE = re.compile(r"^\s*[\w.-]+\s*:", re.MULTILINE)


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _lines(text: str) -> list[str]:
    return [
        ln for ln in _s(text).splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]


class YamlAnchorCountCheck:
    @property
    def name(self) -> str:
        return "yaml_anchor_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ANCHOR_RE.findall(_s(context.baseline_output))),
            len(_ANCHOR_RE.findall(_s(output))),
        )


class YamlAliasCountCheck:
    @property
    def name(self) -> str:
        return "yaml_alias_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ALIAS_RE.findall(_s(context.baseline_output))),
            len(_ALIAS_RE.findall(_s(output))),
        )


class YamlMergeKeyCountCheck:
    @property
    def name(self) -> str:
        return "yaml_merge_key_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MERGE_RE.findall(_s(context.baseline_output))),
            len(_MERGE_RE.findall(_s(output))),
        )


class YamlMultilineBlockCountCheck:
    @property
    def name(self) -> str:
        return "yaml_multiline_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            s = _s(text)
            return len(_LITERAL_RE.findall(s)) + len(_FOLDED_RE.findall(s))

        return _count_match(count(context.baseline_output), count(output))


class YamlLiteralBlockCountCheck:
    @property
    def name(self) -> str:
        return "yaml_literal_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LITERAL_RE.findall(_s(context.baseline_output))),
            len(_LITERAL_RE.findall(_s(output))),
        )


class YamlFoldedBlockCountCheck:
    @property
    def name(self) -> str:
        return "yaml_folded_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FOLDED_RE.findall(_s(context.baseline_output))),
            len(_FOLDED_RE.findall(_s(output))),
        )


class YamlCommentCountCheck:
    @property
    def name(self) -> str:
        return "yaml_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_COMMENT_RE.findall(_s(output))),
        )


class YamlDocMarkerCountCheck:
    @property
    def name(self) -> str:
        return "yaml_doc_marker_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOC_RE.findall(_s(context.baseline_output))),
            len(_DOC_RE.findall(_s(output))),
        )


class YamlTagCountCheck:
    @property
    def name(self) -> str:
        return "yaml_tag_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TAG_RE.findall(_s(context.baseline_output))),
            len(_TAG_RE.findall(_s(output))),
        )


class YamlQuotedStringCountCheck:
    @property
    def name(self) -> str:
        return "yaml_quoted_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_QUOTED_RE.findall(_s(context.baseline_output))),
            len(_QUOTED_RE.findall(_s(output))),
        )


class YamlFlowStyleCountCheck:
    @property
    def name(self) -> str:
        return "yaml_flow_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FLOW_RE.findall(_s(context.baseline_output))),
            len(_FLOW_RE.findall(_s(output))),
        )


class YamlNestedMapCountCheck:
    @property
    def name(self) -> str:
        return "yaml_nested_map_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MAP_RE.findall(_s(context.baseline_output))),
            len(_MAP_RE.findall(_s(output))),
        )


class YamlNestedSeqCountCheck:
    @property
    def name(self) -> str:
        return "yaml_nested_seq_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SEQ_RE.findall(_s(context.baseline_output))),
            len(_SEQ_RE.findall(_s(output))),
        )


class YamlKeyWithDotCountCheck:
    @property
    def name(self) -> str:
        return "yaml_dotkey_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOTKEY_RE.findall(_s(context.baseline_output))),
            len(_DOTKEY_RE.findall(_s(output))),
        )


class YamlBoolValueCountCheck:
    @property
    def name(self) -> str:
        return "yaml_boolval_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BOOLVAL_RE.findall(_s(context.baseline_output))),
            len(_BOOLVAL_RE.findall(_s(output))),
        )


class YamlNullValueCountCheck:
    @property
    def name(self) -> str:
        return "yaml_nullval_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NULLVAL_RE.findall(_s(context.baseline_output))),
            len(_NULLVAL_RE.findall(_s(output))),
        )


class YamlNumberValueCountCheck:
    @property
    def name(self) -> str:
        return "yaml_numval_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NUMVAL_RE.findall(_s(context.baseline_output))),
            len(_NUMVAL_RE.findall(_s(output))),
        )


class YamlIndentWidthCheck:
    @property
    def name(self) -> str:
        return "yaml_indent_width"

    def score(self, output: str, context: CheckContext) -> float:
        def width(text: str) -> int:
            counts: dict[int, int] = {}
            for ln in _lines(text):
                stripped = ln.lstrip(" ")
                if stripped and len(ln) != len(stripped):
                    w = len(ln) - len(stripped)
                    counts[w] = counts.get(w, 0) + 1
            return max(counts, key=lambda k: counts[k]) if counts else 0

        base, cand = width(context.baseline_output), width(output)
        return 1.0 if base == cand else max(0.0, 1.0 - abs(cand - base) / max(1, base))


class YamlMaxDepthCheck:
    @property
    def name(self) -> str:
        return "yaml_max_depth"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = 0
            for ln in _lines(text):
                indent = len(ln) - len(ln.lstrip())
                best = max(best, indent // 2 + 1)
            return best

        return _count_match(depth(context.baseline_output), depth(output))


class YamlKeyCountDeepCheck:
    @property
    def name(self) -> str:
        return "yaml_key_count_deep"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KEY_RE.findall(_s(context.baseline_output))),
            len(_KEY_RE.findall(_s(output))),
        )
