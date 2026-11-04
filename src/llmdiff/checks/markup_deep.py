"""Deep markup dimensions: inline formatting, embeds, and HTML attributes."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "StrikethroughCountCheck",
    "SubscriptCountCheck",
    "SuperscriptCountCheck",
    "MarkHighlightCountCheck",
    "KbdKeyCountCheck",
    "InlineMathCountCheck",
    "DisplayMathCountCheck",
    "FootnoteRefCountCheck",
    "FootnoteDefCountCheck",
    "AbbrTagCountCheck",
    "DetailsBlockCountCheck",
    "SummaryTagCountCheck",
    "HtmlAttrCountCheck",
    "HtmlClassCountCheck",
    "HtmlIdCountCheck",
    "ImgAltCountCheck",
    "LinkRelCountCheck",
    "LinkTargetCountCheck",
    "VideoEmbedCountCheck",
    "IframeEmbedCountCheck",
]

_MAX = 200_000
_STRIKE_RE = re.compile(r"~~[^~\n]+~~")
_SUB_RE = re.compile(r"~[^~\s]+~")
_SUP_RE = re.compile(r"\^[^\^\s]+\^")
_MARK_RE = re.compile(r"==[^=\n]+==")
_KBD_RE = re.compile(r"<kbd>[^<]+</kbd>")
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$[^$\n]+\$(?!\$)")
_DISPLAY_MATH_RE = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
_FN_REF_RE = re.compile(r"\[\^[^\]]+\](?!:)")
_FN_DEF_RE = re.compile(r"^\[\^[^\]]+\]:", re.MULTILINE)
_ABBR_RE = re.compile(r"<abbr[^>]*>")
_DETAILS_RE = re.compile(r"<details[^>]*>")
_SUMMARY_RE = re.compile(r"<summary[^>]*>")
_HTML_ATTR_RE = re.compile(r"\s[a-zA-Z-]+\s*=\s*(\"[^\"]*\"|'[^']*')")
_CLASS_RE = re.compile(r'class\s*=\s*["\']([^"\']*)["\']')
_ID_RE = re.compile(r'\bid\s*=\s*["\']([^"\']*)["\']')
_IMG_RE = re.compile(r"<img[^>]*>")
_LINK_REL_RE = re.compile(r"<a\s[^>]*\brel\s*=\s*[\"'][^\"']*[\"']")
_LINK_TARGET_RE = re.compile(r"<a\s[^>]*\btarget\s*=\s*[\"'][^\"']*[\"']")
_VIDEO_RE = re.compile(r"<(?:video|iframe)[^>]*(?:youtube|youtu\.be|vimeo)[^>]*>", re.IGNORECASE)
_IFRAME_RE = re.compile(r"<iframe[^>]*>")


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


class StrikethroughCountCheck:
    @property
    def name(self) -> str:
        return "strike_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_STRIKE_RE.findall(_s(context.baseline_output))),
            len(_STRIKE_RE.findall(_s(output))),
        )


class SubscriptCountCheck:
    @property
    def name(self) -> str:
        return "sub_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUB_RE.findall(_s(context.baseline_output))),
            len(_SUB_RE.findall(_s(output))),
        )


class SuperscriptCountCheck:
    @property
    def name(self) -> str:
        return "sup_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUP_RE.findall(_s(context.baseline_output))),
            len(_SUP_RE.findall(_s(output))),
        )


class MarkHighlightCountCheck:
    @property
    def name(self) -> str:
        return "mark_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MARK_RE.findall(_s(context.baseline_output))),
            len(_MARK_RE.findall(_s(output))),
        )


class KbdKeyCountCheck:
    @property
    def name(self) -> str:
        return "kbd_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_KBD_RE.findall(_s(context.baseline_output))),
            len(_KBD_RE.findall(_s(output))),
        )


class InlineMathCountCheck:
    @property
    def name(self) -> str:
        return "inline_math_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_INLINE_MATH_RE.findall(_s(context.baseline_output))),
            len(_INLINE_MATH_RE.findall(_s(output))),
        )


class DisplayMathCountCheck:
    @property
    def name(self) -> str:
        return "display_math_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DISPLAY_MATH_RE.findall(_s(context.baseline_output))),
            len(_DISPLAY_MATH_RE.findall(_s(output))),
        )


class FootnoteRefCountCheck:
    @property
    def name(self) -> str:
        return "footnote_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FN_REF_RE.findall(_s(context.baseline_output))),
            len(_FN_REF_RE.findall(_s(output))),
        )


class FootnoteDefCountCheck:
    @property
    def name(self) -> str:
        return "footnote_def_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_FN_DEF_RE.findall(_s(context.baseline_output))),
            len(_FN_DEF_RE.findall(_s(output))),
        )


class AbbrTagCountCheck:
    @property
    def name(self) -> str:
        return "abbr_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ABBR_RE.findall(_s(context.baseline_output))),
            len(_ABBR_RE.findall(_s(output))),
        )


class DetailsBlockCountCheck:
    @property
    def name(self) -> str:
        return "details_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DETAILS_RE.findall(_s(context.baseline_output))),
            len(_DETAILS_RE.findall(_s(output))),
        )


class SummaryTagCountCheck:
    @property
    def name(self) -> str:
        return "summary_tag_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SUMMARY_RE.findall(_s(context.baseline_output))),
            len(_SUMMARY_RE.findall(_s(output))),
        )


class HtmlAttrCountCheck:
    @property
    def name(self) -> str:
        return "html_attr_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HTML_ATTR_RE.findall(_s(context.baseline_output))),
            len(_HTML_ATTR_RE.findall(_s(output))),
        )


class HtmlClassCountCheck:
    @property
    def name(self) -> str:
        return "html_class_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(len(c.split()) for c in _CLASS_RE.findall(_s(text)))

        return _count_match(count(context.baseline_output), count(output))


class HtmlIdCountCheck:
    @property
    def name(self) -> str:
        return "html_id_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ID_RE.findall(_s(context.baseline_output))),
            len(_ID_RE.findall(_s(output))),
        )


class ImgAltCountCheck:
    @property
    def name(self) -> str:
        return "img_alt_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(1 for tag in _IMG_RE.findall(_s(text)) if "alt=" in tag)

        return _count_match(count(context.baseline_output), count(output))


class LinkRelCountCheck:
    @property
    def name(self) -> str:
        return "link_rel_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LINK_REL_RE.findall(_s(context.baseline_output))),
            len(_LINK_REL_RE.findall(_s(output))),
        )


class LinkTargetCountCheck:
    @property
    def name(self) -> str:
        return "link_target_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_LINK_TARGET_RE.findall(_s(context.baseline_output))),
            len(_LINK_TARGET_RE.findall(_s(output))),
        )


class VideoEmbedCountCheck:
    @property
    def name(self) -> str:
        return "video_embed_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_VIDEO_RE.findall(_s(context.baseline_output))),
            len(_VIDEO_RE.findall(_s(output))),
        )


class IframeEmbedCountCheck:
    @property
    def name(self) -> str:
        return "iframe_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IFRAME_RE.findall(_s(context.baseline_output))),
            len(_IFRAME_RE.findall(_s(output))),
        )
