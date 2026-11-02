"""Linear integration: GraphQL API client for issues, comments and labels.

llmdiff can file regression issues in Linear or comment on existing ones.
This module implements:

- :class:`LinearClient` — a small GraphQL client (``POST https://api.linear.app/graphql``)
  with queries/mutations for issue create, comment create, issue search and
  label listing; transport is injectable (``httpx.Client``-like),
- :func:`build_issue_create_mutation` / :func:`build_comment_create_mutation`
  / :func:`build_issue_search_query` — payload builders,
- :class:`LinearReporter` — maps a DiffResult to a Linear issue (title,
  description markdown, labels, team/project) and creates or updates it.

Auth: ``Authorization: <api_key>`` header (Linear API keys are bearer-style).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "LinearError",
    "HttpResponseLike",
    "HttpClientLike",
    "LinearIssue",
    "LinearClient",
    "LinearReporter",
    "build_issue_create_mutation",
    "build_comment_create_mutation",
    "build_issue_search_query",
    "build_verdict_description",
]


class LinearError(LLMDiffError):
    """Raised for Linear configuration, parsing or API errors."""


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


GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"


@dataclass(frozen=True, slots=True)
class LinearIssue:
    """A parsed Linear issue (subset of fields)."""

    id: str
    identifier: str = ""
    title: str = ""
    description: str = ""
    state: str = ""
    url: str = ""
    labels: tuple[str, ...] = ()

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> LinearIssue:
        labels_data = data.get("labels") or {}
        nodes = labels_data.get("nodes", []) if isinstance(labels_data, Mapping) else []
        labels = tuple(str(n.get("name", "")) for n in nodes if isinstance(n, Mapping))
        state = data.get("state") or {}
        return cls(
            id=str(data.get("id", "")),
            identifier=str(data.get("identifier", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "") or ""),
            state=str(state.get("name", "")) if isinstance(state, Mapping) else "",
            url=str(data.get("url", "")),
            labels=labels,
        )


def _gql(document: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": document}
    if variables:
        payload["variables"] = dict(variables)
    return payload


def build_issue_search_query(*, term: str, limit: int = 10) -> dict[str, Any]:
    """Build a GraphQL query searching issues by ``term``."""
    if not term.strip():
        raise LinearError("search term must be non-empty")
    document = """
    query SearchIssues($term: String!, $limit: Int!) {
      issueSearch(query: $term, first: $limit) {
        nodes {
          id identifier title description url
          state { name }
          labels { nodes { name } }
        }
      }
    }
    """
    return _gql(document, {"term": term, "limit": limit})


def build_issue_create_mutation(
    *,
    team_id: str,
    title: str,
    description: str = "",
    label_ids: Sequence[str] = (),
    project_id: str = "",
    priority: int = 0,
) -> dict[str, Any]:
    """Build the ``issueCreate`` mutation.

    ``priority`` uses Linear's scale: 0 = no priority, 1 = urgent,
    2 = high, 3 = medium, 4 = low.
    """
    if not team_id:
        raise LinearError("team_id is required")
    if not title.strip():
        raise LinearError("issue title must be non-empty")
    if not 0 <= priority <= 4:
        raise LinearError(f"priority must be in 0..4, got {priority}")
    document = """
    mutation CreateIssue($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success
        issue { id identifier title url }
      }
    }
    """
    input_data: dict[str, Any] = {"teamId": team_id, "title": title[:2000]}
    if description:
        input_data["description"] = description
    if label_ids:
        input_data["labelIds"] = list(label_ids)
    if project_id:
        input_data["projectId"] = project_id
    if priority:
        input_data["priority"] = priority
    return _gql(document, {"input": input_data})


def build_comment_create_mutation(*, issue_id: str, body: str) -> dict[str, Any]:
    """Build the ``commentCreate`` mutation."""
    if not issue_id:
        raise LinearError("issue_id is required")
    if not body.strip():
        raise LinearError("comment body must be non-empty")
    document = """
    mutation CreateComment($input: CommentCreateInput!) {
      commentCreate(input: $input) {
        success
        comment { id body }
      }
    }
    """
    return _gql(document, {"input": {"issueId": issue_id, "body": body}})


def build_verdict_description(result: Any, *, heading: str = "## llmdiff verdict") -> str:
    """Render a markdown issue/comment body from a DiffResult-like object."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    lines = [heading, "", "| dimension | win | tie | loss | baseline | candidate | delta |", "|---|---|---|---|---|---|---|"]
    for summary in getattr(matrix, "dimensions", ()) or []:
        lines.append(
            f"| {summary.dimension} | {summary.wins} | {summary.ties} | {summary.losses} "
            f"| {summary.baseline_mean:.3f} | {summary.candidate_mean:.3f} | {summary.delta:+.3f} |"
        )
    verdict = getattr(matrix, "verdict", "unknown")
    lines += ["", f"**Overall verdict: {verdict}**"]
    regressed = getattr(matrix, "regressed_dimensions", ()) or ()
    if regressed:
        lines += ["", "Regressed dimensions: " + ", ".join(f"`{d}`" for d in regressed)]
    return "\n".join(lines)


@dataclass(slots=True)
class LinearClient:
    """Minimal Linear GraphQL client.

    Attributes:
        api_key: A Linear API key.
        client: Injectable transport (``httpx.Client``-like).
        endpoint: GraphQL endpoint (override for tests/proxies).
        timeout: Per-request timeout.
    """

    api_key: str
    client: HttpClientLike
    endpoint: str = GRAPHQL_ENDPOINT
    timeout: float = 15.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LinearError("api_key must be non-empty")
        if not self.endpoint.startswith(("http://", "https://")):
            raise LinearError(f"endpoint must be http(s), got {self.endpoint!r}")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "llmdiff/0.1",
        }

    def graphql(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a GraphQL payload and return the ``data`` mapping.

        Raises:
            LinearError: On HTTP errors or GraphQL ``errors``.
        """
        response = self.client.post(
            self.endpoint,
            headers=self._headers(),
            json=dict(payload),
            timeout=self.timeout,
        )
        if not 200 <= response.status_code < 300:
            raise LinearError(
                f"Linear HTTP error: {response.status_code}: {response.text[:300]}"
            )
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise LinearError(f"Linear returned non-JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise LinearError("Linear response must be a JSON object")
        errors = body.get("errors")
        if errors:
            first = errors[0] if isinstance(errors, list) and errors else {}
            message = first.get("message", "unknown") if isinstance(first, Mapping) else str(first)
            raise LinearError(f"Linear GraphQL error: {message}")
        data = body.get("data")
        if not isinstance(data, dict):
            raise LinearError("Linear response missing 'data'")
        return data

    # ------------------------------------------------------------------ #
    # high-level operations
    # ------------------------------------------------------------------ #

    def search_issues(self, term: str, *, limit: int = 10) -> list[LinearIssue]:
        """Search issues by term."""
        data = self.graphql(build_issue_search_query(term=term, limit=limit))
        search = data.get("issueSearch") or {}
        nodes = search.get("nodes", []) if isinstance(search, Mapping) else []
        return [LinearIssue.from_api(n) for n in nodes if isinstance(n, Mapping)]

    def create_issue(
        self,
        *,
        team_id: str,
        title: str,
        description: str = "",
        label_ids: Sequence[str] = (),
        project_id: str = "",
        priority: int = 0,
    ) -> LinearIssue:
        """Create an issue and return it."""
        data = self.graphql(
            build_issue_create_mutation(
                team_id=team_id,
                title=title,
                description=description,
                label_ids=label_ids,
                project_id=project_id,
                priority=priority,
            )
        )
        created = data.get("issueCreate") or {}
        if isinstance(created, Mapping) and created.get("success") is False:
            raise LinearError(f"issueCreate reported success=false: {created}")
        issue = created.get("issue") if isinstance(created, Mapping) else None
        if not isinstance(issue, Mapping):
            raise LinearError("issueCreate response missing 'issue'")
        return LinearIssue.from_api(issue)

    def add_comment(self, *, issue_id: str, body: str) -> dict[str, Any]:
        """Add a comment to an issue."""
        data = self.graphql(build_comment_create_mutation(issue_id=issue_id, body=body))
        created = data.get("commentCreate") or {}
        if isinstance(created, Mapping) and created.get("success") is False:
            raise LinearError(f"commentCreate reported success=false: {created}")
        return created if isinstance(created, Mapping) else {}

    def list_labels(self, *, team_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """List labels of a team (for resolving label names to ids)."""
        document = """
        query TeamLabels($teamId: String!, $limit: Int!) {
          team(id: $teamId) {
            labels(first: $limit) { nodes { id name } }
          }
        }
        """
        data = self.graphql(_gql(document, {"teamId": team_id, "limit": limit}))
        team = data.get("team") or {}
        labels = team.get("labels") or {} if isinstance(team, Mapping) else {}
        nodes = labels.get("nodes", []) if isinstance(labels, Mapping) else []
        return [n for n in nodes if isinstance(n, Mapping)]

    def resolve_label_ids(self, team_id: str, names: Sequence[str]) -> list[str]:
        """Map label names to ids; unknown names are skipped."""
        if not names:
            return []
        wanted = {name.strip().lower() for name in names if name.strip()}
        ids: list[str] = []
        for label in self.list_labels(team_id=team_id):
            label_name = str(label.get("name", ""))
            if label_name.lower() in wanted:
                ids.append(str(label.get("id", "")))
        return ids


@dataclass(slots=True)
class LinearReporter:
    """Files verdict regressions as Linear issues.

    Attributes:
        client: The :class:`LinearClient`.
        team_id: Default team for new issues.
        project_id: Optional default project.
        labels: Label names to attach (resolved via the API).
        priority: Priority for regression issues (default 2 = high).
        marker: A marker string included in the description so existing
            llmdiff issues can be found by search.
    """

    client: LinearClient
    team_id: str
    project_id: str = ""
    labels: tuple[str, ...] = ("llmdiff", "regression")
    priority: int = 2
    marker: str = "llmdiff-verdict"

    def __post_init__(self) -> None:
        if not self.team_id:
            raise LinearError("team_id is required")

    def find_existing(self, *, term: str | None = None) -> LinearIssue | None:
        """Find an existing llmdiff issue by marker term."""
        issues = self.client.search_issues(term or self.marker, limit=10)
        for issue in issues:
            if self.marker in issue.title or self.marker in issue.description:
                return issue
        return None

    def report(
        self,
        result: Any,
        *,
        title_prefix: str = "llmdiff regression",
        update_existing: bool = True,
    ) -> LinearIssue:
        """Create (or comment on) a Linear issue for the verdict.

        When the matrix has no regressed dimensions and ``update_existing``
        is False, an issue is still created (verdicts are worth recording).
        """
        matrix = getattr(result, "matrix", None)
        verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        title = f"[{self.marker}] {title_prefix}: verdict={verdict} loss_rate={rate:.1%}"
        description = build_verdict_description(result)
        if update_existing:
            existing = self.find_existing()
            if existing is not None and existing.id:
                self.client.add_comment(issue_id=existing.id, body=description)
                return existing
        label_ids = self.client.resolve_label_ids(self.team_id, self.labels)
        return self.client.create_issue(
            team_id=self.team_id,
            title=title,
            description=description,
            label_ids=label_ids,
            project_id=self.project_id,
            priority=self.priority if regressed else 4,
        )
