"""Deep XML/HTML dimensions: tags, attributes, nesting, and entities."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from llmdiff.core.checks import CheckContext

__all__ = [
    "XmlParseOkCheck",
    "XmlRootTagMatchCheck",
    "XmlTagSetJaccardCheck",
    "XmlMaxDepthCheck",
    "XmlAttrCountCheck",
    "XmlAttrNameJaccardCheck",
    "XmlTextContentLenCheck",
    "XmlEntityCountCheck",
    "XmlCdataCountCheck",
    "XmlCommentCountCheck",
    "XmlPiCountCheck",
    "XmlSelfClosingCountCheck",
    "XmlVoidTagCountCheck",
    "HtmlDoctypePresenceCheck",
    "HtmlHeadPresenceCheck",
    "HtmlBodyPresenceCheck",
    "HtmlScriptCountCheck",
    "HtmlStyleCountCheck",
    "HtmlMetaCountCheck",
    "HtmlSemanticTagCountCheck",
]

_MAX = 200_000
_TAG_RE = re.compile(r"</?([a-zA-Z][\w:-]*)")
_ATTR_RE = re.compile(r"\s([a-zA-Z-]+)\s*=\s*(\"[^\"]*\"|'[^']*')")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
_CDATA_RE = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PI_RE = re.compile(r"<\?.*?\?>", re.DOTALL)
_SELFCLOSE_RE = re.compile(r"<[a-zA-Z][^>]*/>")
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
_SCRIPT_RE = re.compile(r"<script[^>]*>", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>", re.IGNORECASE)
_META_RE = re.compile(r"<meta[^>]*>", re.IGNORECASE)
_SEMANTIC_RE = re.compile(
    r"<(?:header|footer|nav|main|article|section|aside|figure|figcaption)\b",
    re.IGNORECASE,
)


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class XmlParseOkCheck:
    """1.0 when the output parses as XML iff the baseline does."""

    @property
    def name(self) -> str:
        return "xml_parse_ok"

    def score(self, output: str, context: CheckContext) -> float:
        def ok(text: str) -> bool:
            try:
                ET.fromstring(_s(text))
                return True
            except ET.ParseError:
                return False

        return 1.0 if ok(context.baseline_output) == ok(output) else 0.0


class XmlRootTagMatchCheck:
    @property
    def name(self) -> str:
        return "xml_root_tag"

    def score(self, output: str, context: CheckContext) -> float:
        def root(text: str) -> str:
            try:
                return ET.fromstring(_s(text)).tag
            except ET.ParseError:
                m = _TAG_RE.search(_s(text))
                return m.group(1) if m else ""

        base, cand = root(context.baseline_output), root(output)
        if not base and not cand:
            return 1.0
        return 1.0 if base == cand else 0.0


class XmlTagSetJaccardCheck:
    @property
    def name(self) -> str:
        return "xml_tag_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {t.lower() for t in _TAG_RE.findall(_s(context.baseline_output))}
        b = {t.lower() for t in _TAG_RE.findall(_s(output))}
        return _jaccard(a, b)


class XmlMaxDepthCheck:
    @property
    def name(self) -> str:
        return "xml_max_depth"

    def score(self, output: str, context: CheckContext) -> float:
        def depth(text: str) -> int:
            best = cur = 0
            for m in re.finditer(r"<(/?)([a-zA-Z][\w:-]*)([^>]*?)(/?)>", _s(text)):
                closing, _, _, selfclose = m.groups()
                if closing:
                    cur = max(0, cur - 1)
                elif not selfclose:
                    cur += 1
                    best = max(best, cur)
            return best

        return _count_match(depth(context.baseline_output), depth(output))


class XmlAttrCountCheck:
    @property
    def name(self) -> str:
        return "xml_attr_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ATTR_RE.findall(_s(context.baseline_output))),
            len(_ATTR_RE.findall(_s(output))),
        )


class XmlAttrNameJaccardCheck:
    @property
    def name(self) -> str:
        return "xml_attr_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a = {m.group(1).lower() for m in _ATTR_RE.finditer(_s(context.baseline_output))}
        b = {m.group(1).lower() for m in _ATTR_RE.finditer(_s(output))}
        return _jaccard(a, b)


class XmlTextContentLenCheck:
    @property
    def name(self) -> str:
        return "xml_text_len"

    def score(self, output: str, context: CheckContext) -> float:
        def textlen(text: str) -> int:
            s = re.sub(r"<[^>]+>", "", _s(text))
            return len(s.strip())

        return _count_match(textlen(context.baseline_output), textlen(output))


class XmlEntityCountCheck:
    @property
    def name(self) -> str:
        return "xml_entity_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ENTITY_RE.findall(_s(context.baseline_output))),
            len(_ENTITY_RE.findall(_s(output))),
        )


class XmlCdataCountCheck:
    @property
    def name(self) -> str:
        return "xml_cdata_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CDATA_RE.findall(_s(context.baseline_output))),
            len(_CDATA_RE.findall(_s(output))),
        )


class XmlCommentCountCheck:
    @property
    def name(self) -> str:
        return "xml_comment_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMENT_RE.findall(_s(context.baseline_output))),
            len(_COMMENT_RE.findall(_s(output))),
        )


class XmlPiCountCheck:
    @property
    def name(self) -> str:
        return "xml_pi_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PI_RE.findall(_s(context.baseline_output))),
            len(_PI_RE.findall(_s(output))),
        )


class XmlSelfClosingCountCheck:
    @property
    def name(self) -> str:
        return "xml_selfclose_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SELFCLOSE_RE.findall(_s(context.baseline_output))),
            len(_SELFCLOSE_RE.findall(_s(output))),
        )


class XmlVoidTagCountCheck:
    @property
    def name(self) -> str:
        return "xml_void_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for t in _TAG_RE.findall(_s(text)) if t.lower() in _VOID_TAGS)

        return _count_match(count(context.baseline_output), count(output))


class HtmlDoctypePresenceCheck:
    @property
    def name(self) -> str:
        return "html_doctype"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"<!DOCTYPE\s+html>", re.IGNORECASE)
        base = bool(pat.search(_s(context.baseline_output)))
        cand = bool(pat.search(_s(output)))
        return 1.0 if base == cand else 0.0


class HtmlHeadPresenceCheck:
    @property
    def name(self) -> str:
        return "html_head"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"<head[\s>]", re.IGNORECASE)
        base = bool(pat.search(_s(context.baseline_output)))
        cand = bool(pat.search(_s(output)))
        return 1.0 if base == cand else 0.0


class HtmlBodyPresenceCheck:
    @property
    def name(self) -> str:
        return "html_body"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"<body[\s>]", re.IGNORECASE)
        base = bool(pat.search(_s(context.baseline_output)))
        cand = bool(pat.search(_s(output)))
        return 1.0 if base == cand else 0.0


class HtmlScriptCountCheck:
    @property
    def name(self) -> str:
        return "html_script_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SCRIPT_RE.findall(_s(context.baseline_output))),
            len(_SCRIPT_RE.findall(_s(output))),
        )


class HtmlStyleCountCheck:
    @property
    def name(self) -> str:
        return "html_style_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STYLE_RE.findall(_s(context.baseline_output))),
            len(_STYLE_RE.findall(_s(output))),
        )


class HtmlMetaCountCheck:
    @property
    def name(self) -> str:
        return "html_meta_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_META_RE.findall(_s(context.baseline_output))),
            len(_META_RE.findall(_s(output))),
        )


class HtmlSemanticTagCountCheck:
    @property
    def name(self) -> str:
        return "html_semantic_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SEMANTIC_RE.findall(_s(context.baseline_output))),
            len(_SEMANTIC_RE.findall(_s(output))),
        )
