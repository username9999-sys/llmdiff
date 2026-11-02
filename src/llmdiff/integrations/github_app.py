"""GitHub App integration: webhook receiver, PR comments, commit statuses.

This module implements the GitHub side of llmdiff CI reporting:

- :func:`verify_webhook_signature` — HMAC-SHA256 verification of the
  ``X-Hub-Signature-256`` header (constant-time compare),
- :func:`parse_webhook_event` — parse the ``X-GitHub-Event`` payload into a
  typed :class:`GitHubEvent`,
- :class:`GitHubClient` — thin REST wrapper for posting PR comments and
  commit statuses, with an injectable transport (``httpx.Client``-like) so
  tests run offline,
- :class:`GitHubApp` — a FastAPI app exposing ``POST /webhooks/github`` that
  verifies, parses and dispatches events to registered handlers.

Only the pieces llmdiff needs are implemented: ``pull_request``,
``pull_request_review``, ``check_run``, ``workflow_run``, ``push`` and
``ping`` events; PR comment creation; and commit status creation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "GitHubError",
    "HttpResponseLike",
    "HttpClientLike",
    "GitHubEvent",
    "GitHubApp",
    "GitHubClient",
    "verify_webhook_signature",
    "compute_signature",
    "parse_webhook_event",
    "build_pr_comment_body",
    "build_status_payload",
]


class GitHubError(LLMDiffError):
    """Raised for GitHub App configuration, parsing or API errors."""


class HttpResponseLike(Protocol):
    status_code: int

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


class HttpClientLike(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Any = None,
        content: bytes | None = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


def compute_signature(secret: str, body: bytes) -> str:
    """Return the ``sha256=<hex>`` signature GitHub sends in webhooks."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_webhook_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """Constant-time verification of the ``X-Hub-Signature-256`` header.

    Returns False for missing/malformed headers. Never raises.
    """
    if not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = compute_signature(secret, body)
    return hmac.compare_digest(expected, signature_header)


@dataclass(frozen=True, slots=True)
class GitHubEvent:
    """A parsed GitHub webhook event.

    Attributes:
        name: The ``X-GitHub-Event`` value (``pull_request``, ``push``, ...).
        action: The payload's ``action`` field, when present.
        delivery_id: The ``X-GitHub-Delivery`` UUID.
        payload: The raw decoded JSON payload.
        repository: ``full_name`` of the repository, when present.
        sender: Login of the user who triggered the event, when present.
    """

    name: str
    payload: Mapping[str, Any]
    delivery_id: str = ""
    action: str = ""

    @property
    def repository(self) -> str:
        repo = self.payload.get("repository") or {}
        return str(repo.get("full_name", "")) if isinstance(repo, Mapping) else ""

    @property
    def sender(self) -> str:
        sender = self.payload.get("sender") or {}
        return str(sender.get("login", "")) if isinstance(sender, Mapping) else ""

    @property
    def pull_request_number(self) -> int | None:
        pr = self.payload.get("pull_request") or {}
        if isinstance(pr, Mapping) and isinstance(pr.get("number"), int):
            return pr["number"]
        return None

    @property
    def pull_request(self) -> Mapping[str, Any]:
        pr = self.payload.get("pull_request")
        return pr if isinstance(pr, Mapping) else {}

    @property
    def head_sha(self) -> str:
        """The commit SHA this event is about (PR head or push head)."""
        pr = self.pull_request
        if pr:
            head = pr.get("head") or {}
            if isinstance(head, Mapping):
                return str(head.get("sha", ""))
        return str(self.payload.get("after", ""))


def parse_webhook_event(
    event_name: str | None,
    body: bytes,
    *,
    delivery_id: str = "",
) -> GitHubEvent:
    """Parse a raw webhook body into a :class:`GitHubEvent`.

    Raises:
        GitHubError: If the event name is missing or the body is not JSON.
    """
    if not event_name:
        raise GitHubError("missing X-GitHub-Event header")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubError(f"webhook body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GitHubError("webhook payload must be a JSON object")
    action = payload.get("action")
    return GitHubEvent(
        name=event_name,
        payload=payload,
        delivery_id=delivery_id,
        action=str(action) if action is not None else "",
    )


def build_pr_comment_body(result: Any, *, template: str = "") -> str:
    """Render a markdown PR comment from a :class:`DiffResult`-like object.

    The default template summarises the verdict matrix; a custom Jinja2
    template may be supplied (rendered with ``result`` in scope).
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if template:
        try:
            import jinja2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise GitHubError("jinja2 is required for custom comment templates") from exc
        env = jinja2.Environment(autoescape=True)
        return env.from_string(template).render(result=result)
    lines = [
        "## llmdiff verdict",
        "",
        "| dimension | win | tie | loss | baseline | candidate | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    dimensions = getattr(matrix, "dimensions", None) or []
    for summary in dimensions:
        lines.append(
            f"| {summary.dimension} | {summary.wins} | {summary.ties} | {summary.losses} "
            f"| {summary.baseline_mean:.3f} | {summary.candidate_mean:.3f} | {summary.delta:+.3f} |"
        )
    verdict = getattr(matrix, "verdict", "unknown")
    lines += ["", f"**Overall verdict: {verdict}**"]
    regressed = getattr(matrix, "regressed_dimensions", ())
    if regressed:
        lines += ["", "Regressed dimensions: " + ", ".join(f"`{d}`" for d in regressed)]
    return "\n".join(lines)


def build_status_payload(
    *,
    state: str,
    context: str = "llmdiff",
    description: str = "",
    target_url: str = "",
) -> dict[str, Any]:
    """Build the body for ``POST /repos/{owner}/{repo}/statuses/{sha}``.

    ``state`` must be one of ``pending``, ``success``, ``error``, ``failure``.
    """
    allowed = {"pending", "success", "error", "failure"}
    if state not in allowed:
        raise GitHubError(f"invalid commit status state {state!r}; expected one of {sorted(allowed)}")
    payload: dict[str, Any] = {"state": state, "context": context, "description": description[:140]}
    if target_url:
        payload["target_url"] = target_url
    return payload


@dataclass(slots=True)
class GitHubClient:
    """Minimal GitHub REST client for comments and statuses.

    Attributes:
        token: A GitHub token (``ghp_...`` or fine-grained PAT).
        repo: ``owner/repo`` full name.
        client: Injectable transport (``httpx.Client``-like).
        api_base: API base URL (override for GitHub Enterprise).
        timeout: Per-request timeout.
    """

    token: str
    repo: str
    client: HttpClientLike
    api_base: str = "https://api.github.com"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.token:
            raise GitHubError("GitHub token must be non-empty")
        if "/" not in self.repo:
            raise GitHubError(f"repo must be 'owner/name', got {self.repo!r}")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "llmdiff/0.1",
        }

    def _url(self, path: str) -> str:
        return f"{self.api_base}/repos/{self.repo}{path}"

    def _check(self, response: HttpResponseLike, *, what: str) -> Any:
        if 200 <= response.status_code < 300:
            try:
                return response.json()
            except Exception:  # noqa: BLE001 - empty body is fine for 204
                return None
        raise GitHubError(
            f"GitHub API error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # PR comments
    # ------------------------------------------------------------------ #

    def create_pr_comment(self, pr_number: int, body: str) -> dict[str, Any]:
        """Post a comment on PR ``pr_number``. Returns the created comment."""
        if pr_number < 1:
            raise GitHubError(f"invalid PR number: {pr_number}")
        if not body.strip():
            raise GitHubError("comment body must be non-empty")
        response = self.client.request(
            "POST",
            self._url(f"/issues/{pr_number}/comments"),
            headers=self._headers(),
            json={"body": body},
            timeout=self.timeout,
        )
        result = self._check(response, what=f"create PR comment #{pr_number}")
        return result if isinstance(result, dict) else {}

    def list_pr_comments(self, pr_number: int) -> list[dict[str, Any]]:
        """List comments on PR ``pr_number``."""
        response = self.client.request(
            "GET",
            self._url(f"/issues/{pr_number}/comments"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        result = self._check(response, what=f"list PR comments #{pr_number}")
        return list(result) if isinstance(result, list) else []

    def update_pr_comment(self, comment_id: int, body: str) -> dict[str, Any]:
        """Update an existing comment (used to refresh the verdict comment)."""
        if comment_id < 1:
            raise GitHubError(f"invalid comment id: {comment_id}")
        response = self.client.request(
            "PATCH",
            self._url(f"/issues/comments/{comment_id}"),
            headers=self._headers(),
            json={"body": body},
            timeout=self.timeout,
        )
        result = self._check(response, what=f"update comment {comment_id}")
        return result if isinstance(result, dict) else {}

    def find_bot_comment(self, pr_number: int, *, marker: str = "## llmdiff verdict") -> dict[str, Any] | None:
        """Return the first PR comment containing ``marker``, or None."""
        for comment in self.list_pr_comments(pr_number):
            body = comment.get("body")
            if isinstance(body, str) and marker in body:
                return comment
        return None

    def upsert_pr_comment(self, pr_number: int, body: str, *, marker: str = "## llmdiff verdict") -> dict[str, Any]:
        """Update the existing llmdiff comment on a PR, or create one."""
        existing = self.find_bot_comment(pr_number, marker=marker)
        if existing is not None and isinstance(existing.get("id"), int):
            return self.update_pr_comment(existing["id"], body)
        return self.create_pr_comment(pr_number, body)

    # ------------------------------------------------------------------ #
    # commit statuses
    # ------------------------------------------------------------------ #

    def create_commit_status(
        self,
        sha: str,
        *,
        state: str,
        context: str = "llmdiff",
        description: str = "",
        target_url: str = "",
    ) -> dict[str, Any]:
        """Set a commit status on ``sha``."""
        if not sha or len(sha) < 7:
            raise GitHubError(f"invalid commit sha: {sha!r}")
        payload = build_status_payload(
            state=state, context=context, description=description, target_url=target_url
        )
        response = self.client.request(
            "POST",
            self._url(f"/statuses/{sha}"),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        result = self._check(response, what=f"create status on {sha[:8]}")
        return result if isinstance(result, dict) else {}

    def list_commit_statuses(self, sha: str) -> list[dict[str, Any]]:
        """List statuses for ``sha``."""
        response = self.client.request(
            "GET",
            self._url(f"/commits/{sha}/statuses"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        result = self._check(response, what=f"list statuses for {sha[:8]}")
        return list(result) if isinstance(result, list) else []

    # ------------------------------------------------------------------ #
    # convenience: report a DiffResult
    # ------------------------------------------------------------------ #

    def report_result(
        self,
        result: Any,
        *,
        pr_number: int | None = None,
        sha: str = "",
        target_url: str = "",
        comment_template: str = "",
    ) -> dict[str, Any]:
        """Post the verdict as a PR comment and/or a commit status.

        The commit status state is derived from the matrix: ``failure`` when
        any dimension regressed, else ``success``.
        """
        matrix = getattr(result, "matrix", None)
        regressed = bool(getattr(matrix, "regressed_dimensions", ())) if matrix else False
        out: dict[str, Any] = {"status": None, "comment": None}
        if sha:
            out["status"] = self.create_commit_status(
                sha,
                state="failure" if regressed else "success",
                description="regressions detected" if regressed else "no regressions",
                target_url=target_url,
            )
        if pr_number is not None:
            body = build_pr_comment_body(result, template=comment_template)
            out["comment"] = self.upsert_pr_comment(pr_number, body)
        return out


# ---------------------------------------------------------------------- #
# FastAPI webhook receiver
# ---------------------------------------------------------------------- #

Handler = Callable[[GitHubEvent], Any]


@dataclass(slots=True)
class GitHubApp:
    """A FastAPI app that verifies and dispatches GitHub webhooks.

    Attributes:
        webhook_secret: The app's webhook secret.
        handlers: Mapping of event name to handler callables.
        on_error: Optional error handler ``(event, error) -> None``.
    """

    webhook_secret: str
    handlers: dict[str, list[Handler]] = field(default_factory=dict)
    on_error: Callable[[GitHubEvent | None, BaseException], None] | None = None

    def __post_init__(self) -> None:
        if not self.webhook_secret:
            raise GitHubError("webhook_secret must be non-empty")

    def on(self, event_name: str, handler: Handler) -> Handler:
        """Decorator: register ``handler`` for ``event_name``."""
        self.handlers.setdefault(event_name, []).append(handler)
        return handler

    def dispatch(self, event: GitHubEvent) -> list[Any]:
        """Run every handler registered for ``event.name`` (plus ``*``)."""
        results: list[Any] = []
        for handler in [*self.handlers.get(event.name, []), *self.handlers.get("*", [])]:
            try:
                results.append(handler(event))
            except Exception as exc:  # noqa: BLE001
                if self.on_error is not None:
                    self.on_error(event, exc)
                else:
                    raise
        return results

    def handle_request(
        self,
        *,
        event_name: str | None,
        signature: str | None,
        delivery_id: str | None,
        body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        """Verify + parse + dispatch one raw webhook request.

        Returns:
            ``(status_code, json_body)`` suitable for a FastAPI response.
        """
        if not verify_webhook_signature(self.webhook_secret, body, signature):
            return 401, {"error": "invalid signature"}
        try:
            event = parse_webhook_event(event_name, body, delivery_id=delivery_id or "")
        except GitHubError as exc:
            return 400, {"error": str(exc)}
        if event.name == "ping":
            return 200, {"ok": True, "zen": event.payload.get("zen", "")}
        results = self.dispatch(event)
        return 200, {"ok": True, "event": event.name, "action": event.action, "handled": len(results)}

    def create_fastapi_app(self) -> Any:
        """Build and return a FastAPI app exposing ``POST /webhooks/github``.

        Raises:
            GitHubError: If FastAPI is not installed.
        """
        try:
            from fastapi import FastAPI, Request  # type: ignore[import-untyped]
        except ImportError as exc:
            raise GitHubError("fastapi is required for the webhook receiver; install llmdiff[server]") from exc

        app = FastAPI(title="llmdiff GitHub App")

        @app.post("/webhooks/github")
        async def github_webhook(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_request(
                event_name=request.headers.get("X-GitHub-Event"),
                signature=request.headers.get("X-Hub-Signature-256"),
                delivery_id=request.headers.get("X-GitHub-Delivery"),
                body=body,
            )
            from fastapi.responses import JSONResponse  # type: ignore[import-untyped]

            return JSONResponse(payload, status_code=status)

        return app
