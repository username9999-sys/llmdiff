"""Documentation-structure dimensions: sections, TOC, code refs, and API mentions."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "SectionCountCheck",
    "TocEntryCountCheck",
    "CodeRefCountCheck",
    "ApiMentionCountCheck",
    "ExampleBlockCountCheck",
    "WarningBlockCountCheck",
    "NoteBlockCountCheck",
    "AdmonitionCountCheck",
    "ReadmeBadgeCountCheck",
    "InstallSectionPresenceCheck",
    "UsageSectionPresenceCheck",
    "LicenseMentionCheck",
    "ChangelogMentionCheck",
    "ContributingMentionCheck",
    "CodeOfConductMentionCheck",
    "SecurityPolicyMentionCheck",
    "DiagramCountCheck",
    "MermaidBlockCountCheck",
    "ImageEmbedCountCheck",
    "AnchorLinkCountCheck",
]

_MAX = 200_000
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_TOC_RE = re.compile(r"^\s*[-*]\s+\[[^\]]+\]\(#[^)]+\)", re.MULTILINE)
_CODE_REF_RE = re.compile(
    r"`[\w./-]+\.(?:py|js|ts|go|rs|java|c|cpp|sh|json|yaml|yml|toml|md)`"
)
_API_RE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[\w/{}-]+")
_EXAMPLE_RE = re.compile(
    r"^#{2,4}\s+(?:example|usage|quickstart)", re.MULTILINE | re.IGNORECASE
)
_WARN_RE = re.compile(
    r"^>\s*(?:⚠️|WARNING|NOTE|IMPORTANT|TIP|CAUTION)",
    re.MULTILINE | re.IGNORECASE,
)
_NOTE_RE = re.compile(r"^>\s*NOTE", re.MULTILINE | re.IGNORECASE)
_ADMON_RE = re.compile(
    r"^:::\s*(?:note|warning|tip|caution|important)",
    re.MULTILINE | re.IGNORECASE,
)
_BADGE_RE = re.compile(
    r"!\[[^\]]*\]\([^)]*(?:badge|shields?\.io|\.svg)[^)]*\)",
    re.IGNORECASE,
)
_DIAGRAM_RE = re.compile(r"```(?:mermaid|dot|graphviz|plantuml|ascii)", re.IGNORECASE)
_MERMAID_RE = re.compile(r"```mermaid", re.IGNORECASE)
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_ANCHOR_RE = re.compile(r"\[[^\]]+\]\(#[^)]+\)")


def _s(text: str) -> str:
    return text[:_MAX]


def _count_match(base: float, cand: float) -> float:
    if base == 0:
        return 1.0 if cand == 0 else 0.0
    return 1.0 if cand == base else max(0.0, 1.0 - abs(cand - base) / base)


def _presence_match(base: bool, cand: bool) -> float:
    return 1.0 if base == cand else 0.0


class SectionCountCheck:
    @property
    def name(self) -> str:
        return "doc_section_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HEADING_RE.findall(_s(context.baseline_output))),
            len(_HEADING_RE.findall(_s(output))),
        )


class TocEntryCountCheck:
    @property
    def name(self) -> str:
        return "toc_entry_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_TOC_RE.findall(_s(context.baseline_output))),
            len(_TOC_RE.findall(_s(output))),
        )


class CodeRefCountCheck:
    @property
    def name(self) -> str:
        return "code_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_CODE_REF_RE.findall(_s(context.baseline_output))),
            len(_CODE_REF_RE.findall(_s(output))),
        )


class ApiMentionCountCheck:
    @property
    def name(self) -> str:
        return "api_mention_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_API_RE.findall(_s(context.baseline_output))),
            len(_API_RE.findall(_s(output))),
        )


class ExampleBlockCountCheck:
    @property
    def name(self) -> str:
        return "example_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EXAMPLE_RE.findall(_s(context.baseline_output))),
            len(_EXAMPLE_RE.findall(_s(output))),
        )


class WarningBlockCountCheck:
    @property
    def name(self) -> str:
        return "warning_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_WARN_RE.findall(_s(context.baseline_output))),
            len(_WARN_RE.findall(_s(output))),
        )


class NoteBlockCountCheck:
    @property
    def name(self) -> str:
        return "note_block_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_NOTE_RE.findall(_s(context.baseline_output))),
            len(_NOTE_RE.findall(_s(output))),
        )


class AdmonitionCountCheck:
    @property
    def name(self) -> str:
        return "admonition_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ADMON_RE.findall(_s(context.baseline_output))),
            len(_ADMON_RE.findall(_s(output))),
        )


class ReadmeBadgeCountCheck:
    @property
    def name(self) -> str:
        return "badge_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_BADGE_RE.findall(_s(context.baseline_output))),
            len(_BADGE_RE.findall(_s(output))),
        )


class InstallSectionPresenceCheck:
    @property
    def name(self) -> str:
        return "install_section"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"^#{1,4}\s+.*install", re.MULTILINE | re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class UsageSectionPresenceCheck:
    @property
    def name(self) -> str:
        return "usage_section"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(
            r"^#{1,4}\s+.*(?:usage|quickstart|getting started)",
            re.MULTILINE | re.IGNORECASE,
        )
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class LicenseMentionCheck:
    @property
    def name(self) -> str:
        return "license_mention"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\blicen[cs]e\b", re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class ChangelogMentionCheck:
    @property
    def name(self) -> str:
        return "changelog_mention"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\bchangelog\b", re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class ContributingMentionCheck:
    @property
    def name(self) -> str:
        return "contributing_mention"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"\bcontributing\b", re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class CodeOfConductMentionCheck:
    @property
    def name(self) -> str:
        return "coc_mention"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"code of conduct", re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class SecurityPolicyMentionCheck:
    @property
    def name(self) -> str:
        return "security_policy_mention"

    def score(self, output: str, context: CheckContext) -> float:
        pat = re.compile(r"security\s+policy", re.IGNORECASE)
        return _presence_match(
            bool(pat.search(_s(context.baseline_output))),
            bool(pat.search(_s(output))),
        )


class DiagramCountCheck:
    @property
    def name(self) -> str:
        return "diagram_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DIAGRAM_RE.findall(_s(context.baseline_output))),
            len(_DIAGRAM_RE.findall(_s(output))),
        )


class MermaidBlockCountCheck:
    @property
    def name(self) -> str:
        return "mermaid_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MERMAID_RE.findall(_s(context.baseline_output))),
            len(_MERMAID_RE.findall(_s(output))),
        )


class ImageEmbedCountCheck:
    @property
    def name(self) -> str:
        return "image_embed_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_IMG_RE.findall(_s(context.baseline_output))),
            len(_IMG_RE.findall(_s(output))),
        )


class AnchorLinkCountCheck:
    @property
    def name(self) -> str:
        return "anchor_link_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ANCHOR_RE.findall(_s(context.baseline_output))),
            len(_ANCHOR_RE.findall(_s(output))),
        )
