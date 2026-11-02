"""Jira integration: REST API v3 client for issues and comments.

llmdiff can file regression issues in Jira or comment on existing ones.
This module implements:

- :class:`JiraClient` — REST client for issue create/search/comment with
  basic auth (email + API token), injectable transport,
- :func:`build_issue_payload` / :func:`build_comment_payload` /
  :func:`build_jql` — payload builders (ADF format for descriptions),
- :class:`JiraReporter` — maps a DiffResult to a Jira issue and creates or
  comments on it.

Auth: HTTP Basic with ``email:api_token`` base64-encoded.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "JiraError",
    "HttpResponseLike",
    "HttpClientLike",
    "JiraIssue",
    "JiraClient",
    "JiraReporter",
    "build_issue_payload",
    "build_comment_payload",
    "build_jql",
    "markdown_to_adf",
    "build_verdict_adf",
]


class JiraError(LLMDiffError):
    """Raised for Jira configuration, parsing or API errors."""


class HttpResponseLike(Protocol):
    status_code: int

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


class HttpClientLike(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Any = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


@dataclass(frozen=True, slots=True)
class JiraIssue:
    """A parsed Jira issue (subset of fields)."""

    id: str
    key: str = ""
    summary: str = ""
    status: str = ""
    url: str = ""

    @classmethod
    def from_api(cls, data: Mapping[str, Any], *, base_url: str = "") -> JiraIssue:
        fields = data.get("fields") if isinstance(data.get("fields"), Mapping) else {}
        status = fields.get("status") or {}
        key = str(data.get("key", ""))
        return cls(
            id=str(data.get("id", "")),
            key=key,
            summary=str(fields.get("summary", "")),
            status=str(status.get("name", "")) if isinstance(status, Mapping) else "",
            url=f"{base_url}/browse/{key}" if base_url and key else "",
        )


def markdown_to_adf(markdown: str) -> dict[str, Any]:
    """Convert a small subset of markdown to Jira's Atlassian Document Format.

    Supports: headings (``#``..``######``), bold (``**``), inline code
    (```` `` ````), bullet lists (``- ``), numbered lists (``1. ``) and
    paragraphs. Tables are rendered as plain text lines.
    """
    doc: dict[str, Any] = {"type": "doc", "version": 1, "content": []}
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        heading_match = None
        for level in range(6, 0, -1):
            prefix = "#" * level + " "
            if stripped.startswith(prefix):
                heading_match = (level, stripped[len(prefix) :])
                break
        if heading_match is not None:
            level, text = heading_match
            doc["content"].append(
                {
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": _inline_nodes(text),
                }
            )
            i += 1
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            items: list[dict[str, Any]] = []
            while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": _inline_nodes(lines[i].strip()[2:])}
                        ],
                    }
                )
                i += 1
            doc["content"].append({"type": "bulletList", "content": items})
            continue
        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == "." and stripped[2] == " ":
            items = []
            while i < len(lines):
                item_line = lines[i].strip()
                if len(item_line) > 2 and item_line[0].isdigit() and item_line[1] == "." and item_line[2] == " ":
                    items.append(
                        {
                            "type": "listItem",
                            "content": [
                                {"type": "paragraph", "content": _inline_nodes(item_line[3:])}
                            ],
                        }
                    )
                    i += 1
                else:
                    break
            doc["content"].append({"type": "orderedList", "content": items})
            continue
        # paragraph: gather consecutive non-empty, non-special lines
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "- ", "* ")) or (len(nxt) > 2 and nxt[0].isdigit() and nxt[1] == "."):
                break
            para_lines.append(nxt)
            i += 1
        doc["content"].append(
            {"type": "paragraph", "content": _inline_nodes(" ".join(para_lines))}
        )
    return doc


def _inline_nodes(text: str) -> list[dict[str, Any]]:
    """Parse inline ``**bold**`` and ```` ``code`` ```` into ADF inline nodes."""
    nodes: list[dict[str, Any]] = []
    # Split on bold and code markers, keeping the markers.
    import re

    token_re = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")
    pos = 0
    for match in token_re.finditer(text):
        if match.start() > pos:
            nodes.append({"type": "text", "text": text[pos : match.start()]})
        token = match.group(0)
        if token.startswith("**"):
            nodes.append(
                {
                    "type": "text",
                    "text": token[2:-2],
                    "marks": [{"type": "strong"}],
                }
            )
        else:
            nodes.append(
                {
                    "type": "text",
                    "text": token[1:-1],
                    "marks": [{"type": "code"}],
                }
            )
        pos = match.end()
    if pos < len(text):
        nodes.append({"type": "text", "text": text[pos:]})
    return nodes or [{"type": "text", "text": text}]


def build_verdict_adf(result: Any, *, heading: str = "llmdiff verdict") -> dict[str, Any]:
    """Render a DiffResult as a Jira ADF document with a table."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    doc: dict[str, Any] = {"type": "doc", "version": 1, "content": []}
    doc["content"].append(
        {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": heading}]}
    )
    verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
    doc["content"].append(
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Overall verdict: "},
                {"type": "text", "text": verdict, "marks": [{"type": "strong"}]},
            ],
        }
    )
    dimensions = list(getattr(matrix, "dimensions", ()) or ())
    if dimensions:
        header_cells = [
            _table_cell([{"type": "text", "text": h, "marks": [{"type": "strong"}]}])
            for h in ("dimension", "win", "tie", "loss", "baseline", "candidate", "delta")
        ]
        rows: list[dict[str, Any]] = [{"type": "tableRow", "content": header_cells}]
        for summary in dimensions:
            rows.append(
                {
                    "type": "tableRow",
                    "content": [
                        _table_cell([{"type": "text", "text": str(summary.dimension)}]),
                        _table_cell([{"type": "text", "text": str(summary.wins)}]),
                        _table_cell([{"type": "text", "text": str(summary.ties)}]),
                        _table_cell([{"type": "text", "text": str(summary.losses)}]),
                        _table_cell([{"type": "text", "text": f"{summary.baseline_mean:.3f}"}]),
                        _table_cell([{"type": "text", "text": f"{summary.candidate_mean:.3f}"}]),
                        _table_cell([{"type": "text", "text": f"{summary.delta:+.3f}"}]),
                    ],
                }
            )
        doc["content"].append({"type": "table", "attrs": {"isNumberColumnEnabled": False}, "content": rows})
    regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
    if regressed:
        doc["content"].append(
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Regressed dimensions: "},
                    {"type": "text", "text": ", ".join(regressed), "marks": [{"type": "code"}]},
                ],
            }
        )
    return doc


def _table_cell(inline: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "tableCell", "content": [{"type": "paragraph", "content": inline}]}


def build_jql(*, project: str, labels: Sequence[str] = (), text: str = "", max_results: int = 20) -> str:
    """Build a JQL query string."""
    clauses: list[str] = []
    if project:
        clauses.append(f'project = "{project}"')
    for label in labels:
        clauses.append(f'labels = "{label}"')
    if text:
        clauses.append(f'text ~ "{text}"')
    if not clauses:
        raise JiraError("build_jql needs at least one of project/labels/text")
    return " AND ".join(clauses) + " ORDER BY created DESC"


def build_issue_payload(
    *,
    project_key: str,
    summary: str,
    description_adf: Mapping[str, Any] | None = None,
    issue_type: str = "Bug",
    labels: Sequence[str] = (),
    priority: str = "",
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``POST /rest/api/3/issue`` payload."""
    if not project_key:
        raise JiraError("project_key is required")
    if not summary.strip():
        raise JiraError("summary must be non-empty")
    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "summary": summary[:255],
        "issuetype": {"name": issue_type},
    }
    if description_adf is not None:
        fields["description"] = dict(description_adf)
    if labels:
        fields["labels"] = [str(label) for label in labels]
    if priority:
        fields["priority"] = {"name": priority}
    if extra_fields:
        fields.update(extra_fields)
    return {"fields": fields}


def build_comment_payload(body_adf: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ``POST /rest/api/3/issue/{key}/comment`` payload."""
    return {"body": dict(body_adf)}


@dataclass(slots=True)
class JiraClient:
    """Minimal Jira Cloud REST v3 client.

    Attributes:
        base_url: e.g. ``https://your-site.atlassian.net``.
        email: The account email used for basic auth.
        api_token: The API token.
        client: Injectable transport.
        timeout: Per-request timeout.
    """

    base_url: str
    email: str
    api_token: str
    client: HttpClientLike
    timeout: float = 15.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise JiraError(f"base_url must be http(s), got {self.base_url!r}")
        if not self.email or "@" not in self.email:
            raise JiraError(f"email must look like an email address, got {self.email!r}")
        if not self.api_token:
            raise JiraError("api_token must be non-empty")
        self.base_url = self.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        credentials = base64.b64encode(f"{self.email}:{self.api_token}".encode()).decode()
        return {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _check(self, response: HttpResponseLike, *, what: str, allow: tuple[int, ...] = (200, 201, 204)) -> Any:
        if response.status_code in allow:
            if response.status_code == 204:
                return None
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise JiraError(
            f"Jira error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # issues
    # ------------------------------------------------------------------ #

    def create_issue(self, payload: Mapping[str, Any]) -> JiraIssue:
        """Create an issue; returns the created issue."""
        response = self.client.post(
            f"{self.base_url}/rest/api/3/issue",
            headers=self._headers(),
            json=dict(payload),
            timeout=self.timeout,
        )
        data = self._check(response, what="create issue", allow=(200, 201))
        if not isinstance(data, Mapping) or not data.get("key"):
            raise JiraError(f"unexpected create-issue response: {data}")
        return JiraIssue.from_api(data, base_url=self.base_url)

    def search(self, jql: str, *, max_results: int = 20, fields: Sequence[str] = ("summary", "status")) -> list[JiraIssue]:
        """GET /rest/api/3/search with a JQL query."""
        if not jql.strip():
            raise JiraError("jql must be non-empty")
        response = self.client.get(
            f"{self.base_url}/rest/api/3/search",
            headers=self._headers(),
            params={
                "jql": jql,
                "maxResults": str(max_results),
                "fields": ",".join(fields),
            },
            timeout=self.timeout,
        )
        data = self._check(response, what="search issues")
        issues = data.get("issues", []) if isinstance(data, Mapping) else []
        return [JiraIssue.from_api(item, base_url=self.base_url) for item in issues if isinstance(item, Mapping)]

    def get_issue(self, key: str) -> JiraIssue:
        """GET /rest/api/3/issue/{key}."""
        if not key.strip():
            raise JiraError("issue key is required")
        response = self.client.get(
            f"{self.base_url}/rest/api/3/issue/{quote(key, safe='')}",
            headers=self._headers(),
            params={"fields": "summary,status,description"},
            timeout=self.timeout,
        )
        data = self._check(response, what=f"get issue {key}")
        return JiraIssue.from_api(data if isinstance(data, Mapping) else {}, base_url=self.base_url)

    # ------------------------------------------------------------------ #
    # comments
    # ------------------------------------------------------------------ #

    def add_comment(self, issue_key: str, body_adf: Mapping[str, Any]) -> dict[str, Any]:
        """POST a comment on an issue."""
        if not issue_key.strip():
            raise JiraError("issue_key is required")
        response = self.client.post(
            f"{self.base_url}/rest/api/3/issue/{quote(issue_key, safe='')}/comment",
            headers=self._headers(),
            json=build_comment_payload(body_adf),
            timeout=self.timeout,
        )
        data = self._check(response, what=f"comment on {issue_key}", allow=(200, 201))
        return data if isinstance(data, Mapping) else {}

    def list_comments(self, issue_key: str) -> list[dict[str, Any]]:
        """GET comments of an issue."""
        response = self.client.get(
            f"{self.base_url}/rest/api/3/issue/{quote(issue_key, safe='')}/comment",
            headers=self._headers(),
            timeout=self.timeout,
        )
        data = self._check(response, what=f"list comments {issue_key}")
        comments = data.get("comments", []) if isinstance(data, Mapping) else []
        return [c for c in comments if isinstance(c, Mapping)]


@dataclass(slots=True)
class JiraReporter:
    """Files verdict regressions as Jira issues.

    Attributes:
        client: The :class:`JiraClient`.
        project_key: Target project.
        labels: Labels attached to created issues.
        marker: Text included in the summary so existing llmdiff issues can
            be found via JQL.
        issue_type: Jira issue type name (default ``Bug``).
        priority: Priority name for regression issues (``High`` etc.).
    """

    client: JiraClient
    project_key: str
    labels: tuple[str, ...] = ("llmdiff",)
    marker: str = "llmdiff"
    issue_type: str = "Bug"
    priority: str = "High"

    def __post_init__(self) -> None:
        if not self.project_key:
            raise JiraError("project_key is required")

    def find_existing(self) -> JiraIssue | None:
        """Find an existing llmdiff issue via JQL text search."""
        jql = build_jql(project=self.project_key, labels=self.labels, text=self.marker)
        issues = self.client.search(jql, max_results=10)
        for issue in issues:
            if self.marker in issue.summary:
                return issue
        return None

    def report(self, result: Any, *, update_existing: bool = True) -> JiraIssue:
        """Create an issue for the verdict, or comment on the existing one."""
        matrix = getattr(result, "matrix", None)
        verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        summary = f"[{self.marker}] verdict={verdict} loss_rate={rate:.1%} regressed={','.join(regressed) or '-'}"
        description = build_verdict_adf(result)
        if update_existing:
            existing = self.find_existing()
            if existing is not None and existing.key:
                self.client.add_comment(existing.key, description)
                return existing
        payload = build_issue_payload(
            project_key=self.project_key,
            summary=summary,
            description_adf=description,
            issue_type=self.issue_type,
            labels=self.labels,
            priority=self.priority if regressed else "",
        )
        return self.client.create_issue(payload)
