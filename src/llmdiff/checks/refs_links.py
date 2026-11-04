"""Reference dimensions: URLs, emails, file paths, issue/PR refs, and identifiers."""

from __future__ import annotations

import re

from llmdiff.core.checks import CheckContext

__all__ = [
    "UrlCountCheck",
    "UrlDomainSetJaccardCheck",
    "HttpVsHttpsRatioCheck",
    "EmailRefCountCheck",
    "FilePathCountCheck",
    "AbsolutePathCountCheck",
    "RelativePathCountCheck",
    "IssueRefCountCheck",
    "PrRefCountCheck",
    "CommitHashCountCheck",
    "DoiCountCheck",
    "ArxivIdCountCheck",
    "IsbnCountCheck",
    "UuidCountCheck",
    "SemverCountCheck",
    "HexColorCountCheck",
    "MacAddressCountCheck",
    "Base64UrlCountCheck",
    "MarkdownRefLinkCountCheck",
    "ShortLinkCountCheck",
]

_MAX = 200_000
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.\w+|~[/\w.-]+")
_ABS_PATH_RE = re.compile(r"(?:/[\w.-]+)+/?")
_ISSUE_RE = re.compile(r"#\d+")
_PR_RE = re.compile(r"\b(?:pull request|PR)\s*#?\d+", re.IGNORECASE)
_COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b(?=\s|$)")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_ARXIV_RE = re.compile(r"\barXiv:\s?\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_ISBN_RE = re.compile(r"\b(?:97[89][- ]?)?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?\d\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SEMVER_RE = re.compile(r"\b\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b")
_HEXCOLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_MAC_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
_B64URL_RE = re.compile(r"https?://[^\s]*[A-Za-z0-9_-]{20,}={0,2}")
_MDREF_RE = re.compile(r"\[[^\]]+\]\[[^\]]+\]")
_SHORTLINK_RE = re.compile(r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|youtu\.be)/[^\s]+")


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


def _domains(text: str) -> set[str]:
    out = set()
    for m in _URL_RE.finditer(_s(text)):
        host = m.group(0).split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        out.add(host.lower())
    return out


class UrlCountCheck:
    @property
    def name(self) -> str:
        return "url_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_URL_RE.findall(_s(context.baseline_output))),
            len(_URL_RE.findall(_s(output))),
        )


class UrlDomainSetJaccardCheck:
    @property
    def name(self) -> str:
        return "url_domain_jaccard"

    def score(self, output: str, context: CheckContext) -> float:
        a, b = _domains(context.baseline_output), _domains(output)
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0


class HttpVsHttpsRatioCheck:
    @property
    def name(self) -> str:
        return "https_ratio"

    def score(self, output: str, context: CheckContext) -> float:
        def ratio(text: str) -> float:
            urls = _URL_RE.findall(_s(text))
            if not urls:
                return 1.0
            return sum(1 for u in urls if u.startswith("https://")) / len(urls)

        return _ratio_match(ratio(context.baseline_output), ratio(output))


class EmailRefCountCheck:
    @property
    def name(self) -> str:
        return "email_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_EMAIL_RE.findall(_s(context.baseline_output))),
            len(_EMAIL_RE.findall(_s(output))),
        )


class FilePathCountCheck:
    @property
    def name(self) -> str:
        return "filepath_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PATH_RE.findall(_s(context.baseline_output))),
            len(_PATH_RE.findall(_s(output))),
        )


class AbsolutePathCountCheck:
    @property
    def name(self) -> str:
        return "abspath_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ABS_PATH_RE.findall(_s(context.baseline_output))),
            len(_ABS_PATH_RE.findall(_s(output))),
        )


class RelativePathCountCheck:
    """Paths containing a slash but not starting with / or ~."""

    @property
    def name(self) -> str:
        return "relpath_count"

    def score(self, output: str, context: CheckContext) -> float:
        def count(text: str) -> int:
            return sum(
                1
                for m in _PATH_RE.finditer(_s(text))
                if not m.group(0).startswith(("/", "~"))
            )

        return _count_match(count(context.baseline_output), count(output))


class IssueRefCountCheck:
    @property
    def name(self) -> str:
        return "issue_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ISSUE_RE.findall(_s(context.baseline_output))),
            len(_ISSUE_RE.findall(_s(output))),
        )


class PrRefCountCheck:
    @property
    def name(self) -> str:
        return "pr_ref_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_PR_RE.findall(_s(context.baseline_output))),
            len(_PR_RE.findall(_s(output))),
        )


class CommitHashCountCheck:
    @property
    def name(self) -> str:
        return "commit_hash_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_COMMIT_RE.findall(_s(context.baseline_output))),
            len(_COMMIT_RE.findall(_s(output))),
        )


class DoiCountCheck:
    @property
    def name(self) -> str:
        return "doi_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_DOI_RE.findall(_s(context.baseline_output))),
            len(_DOI_RE.findall(_s(output))),
        )


class ArxivIdCountCheck:
    @property
    def name(self) -> str:
        return "arxiv_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ARXIV_RE.findall(_s(context.baseline_output))),
            len(_ARXIV_RE.findall(_s(output))),
        )


class IsbnCountCheck:
    @property
    def name(self) -> str:
        return "isbn_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_ISBN_RE.findall(_s(context.baseline_output))),
            len(_ISBN_RE.findall(_s(output))),
        )


class UuidCountCheck:
    @property
    def name(self) -> str:
        return "uuid_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_UUID_RE.findall(_s(context.baseline_output))),
            len(_UUID_RE.findall(_s(output))),
        )


class SemverCountCheck:
    @property
    def name(self) -> str:
        return "semver_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SEMVER_RE.findall(_s(context.baseline_output))),
            len(_SEMVER_RE.findall(_s(output))),
        )


class HexColorCountCheck:
    @property
    def name(self) -> str:
        return "hexcolor_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_HEXCOLOR_RE.findall(_s(context.baseline_output))),
            len(_HEXCOLOR_RE.findall(_s(output))),
        )


class MacAddressCountCheck:
    @property
    def name(self) -> str:
        return "mac_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MAC_RE.findall(_s(context.baseline_output))),
            len(_MAC_RE.findall(_s(output))),
        )


class Base64UrlCountCheck:
    @property
    def name(self) -> str:
        return "b64url_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_B64URL_RE.findall(_s(context.baseline_output))),
            len(_B64URL_RE.findall(_s(output))),
        )


class MarkdownRefLinkCountCheck:
    @property
    def name(self) -> str:
        return "md_reflink_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_MDREF_RE.findall(_s(context.baseline_output))),
            len(_MDREF_RE.findall(_s(output))),
        )


class ShortLinkCountCheck:
    @property
    def name(self) -> str:
        return "shortlink_count"

    def score(self, output: str, context: CheckContext) -> float:
        return _count_match(
            len(_SHORTLINK_RE.findall(_s(context.baseline_output))),
            len(_SHORTLINK_RE.findall(_s(output))),
        )
