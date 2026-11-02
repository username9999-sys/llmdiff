"""GitLab integration: webhook receiver, MR notes and pipeline statuses.

Mirrors :mod:`llmdiff.integrations.github_app` for GitLab:

- :func:`verify_gitlab_token` — constant-time check of the ``X-Gitlab-Token``
  header against the configured secret token,
- :func:`parse_gitlab_event` — parse ``X-Gitlab-Event`` payloads into a typed
  :class:`GitLabEvent`,
- :class:`GitLabClient` — REST wrapper for MR notes and commit statuses
  (injectable transport, offline-testable),
- :class:`GitLabWebhookApp` — FastAPI app exposing ``POST /webhooks/gitlab``.

Supported events: ``Merge Request Hook``, ``Push Hook``, ``Pipeline Hook``,
``Note Hook``, ``System Hook``.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "GitLabError",
    "HttpResponseLike",
    "HttpClientLike",
    "GitLabEvent",
    "GitLabClient",
    "GitLabWebhookApp",
    "verify_gitlab_token",
    "parse_gitlab_event",
    "build_mr_note_body",
    "build_commit_status_payload",
]


class GitLabError(LLMDiffError):
    """Raised for GitLab configuration, parsing or API errors."""


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
        params: Mapping[str, str] | None = None,
        content: bytes | None = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


def verify_gitlab_token(secret: str, header_token: str | None) -> bool:
    """Constant-time comparison of the ``X-Gitlab-Token`` header."""
    if not header_token:
        return False
    return hmac.compare_digest(secret, header_token)


@dataclass(frozen=True, slots=True)
class GitLabEvent:
    """A parsed GitLab webhook event.

    Attributes:
        name: The ``X-Gitlab-Event`` value, e.g. ``Merge Request Hook``.
        object_kind: The payload's ``object_kind`` field.
        payload: The raw decoded JSON payload.
    """

    name: str
    payload: Mapping[str, Any]
    object_kind: str = ""

    @property
    def project(self) -> Mapping[str, Any]:
        project = self.payload.get("project")
        return project if isinstance(project, Mapping) else {}

    @property
    def project_path(self) -> str:
        """``path_with_namespace`` of the project, when present."""
        return str(self.project.get("path_with_namespace", ""))

    @property
    def project_id(self) -> int | None:
        value = self.project.get("id")
        return int(value) if isinstance(value, int) else None

    @property
    def user(self) -> str:
        user = self.payload.get("user") or self.payload.get("user_username")
        if isinstance(user, Mapping):
            return str(user.get("username", ""))
        return str(user or "")

    @property
    def merge_request(self) -> Mapping[str, Any]:
        mr = self.payload.get("object_attributes") or self.payload.get("merge_request")
        return mr if isinstance(mr, Mapping) else {}

    @property
    def merge_request_iid(self) -> int | None:
        value = self.merge_request.get("iid")
        return int(value) if isinstance(value, int) else None

    @property
    def object_attributes(self) -> Mapping[str, Any]:
        attrs = self.payload.get("object_attributes")
        return attrs if isinstance(attrs, Mapping) else {}

    @property
    def action(self) -> str:
        """MR action (``open``/``merge``/``update``...) or pipeline status."""
        attrs = self.object_attributes
        if "action" in attrs:
            return str(attrs.get("action", ""))
        if "status" in attrs:
            return str(attrs.get("status", ""))
        return ""

    @property
    def head_sha(self) -> str:
        """The commit SHA the event is about."""
        mr = self.merge_request
        if mr:
            sha = mr.get("sha") or (mr.get("last_commit") or {}).get("id")  # type: ignore[union-attr]
            return str(sha or "")
        return str(self.payload.get("after", ""))

    @property
    def pipeline_status(self) -> str:
        """Pipeline status for ``Pipeline Hook`` events."""
        return str(self.object_attributes.get("status", ""))


def parse_gitlab_event(event_name: str | None, body: bytes) -> GitLabEvent:
    """Parse a raw GitLab webhook body into a :class:`GitLabEvent`.

    Raises:
        GitLabError: If the event name is missing or the body is not JSON.
    """
    if not event_name:
        raise GitLabError("missing X-Gitlab-Event header")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitLabError(f"webhook body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GitLabError("webhook payload must be a JSON object")
    return GitLabEvent(
        name=event_name,
        payload=payload,
        object_kind=str(payload.get("object_kind", "")),
    )


def build_mr_note_body(result: Any, *, template: str = "") -> str:
    """Render a markdown MR note from a DiffResult-like object."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if template:
        try:
            import jinja2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise GitLabError("jinja2 is required for custom note templates") from exc
        env = jinja2.Environment(autoescape=True)
        return env.from_string(template).render(result=result)
    lines = [
        "## llmdiff verdict",
        "",
        "| dimension | win | tie | loss | baseline | candidate | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in getattr(matrix, "dimensions", ()) or []:
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


def build_commit_status_payload(
    *,
    state: str,
    name: str = "llmdiff",
    description: str = "",
    target_url: str = "",
) -> dict[str, Any]:
    """Build the body for ``POST /projects/:id/statuses/:sha``.

    ``state`` must be one of ``pending``, ``success``, ``failed``, ``canceled``.
    """
    allowed = {"pending", "success", "failed", "canceled"}
    if state not in allowed:
        raise GitLabError(f"invalid status state {state!r}; expected one of {sorted(allowed)}")
    payload: dict[str, Any] = {"state": state, "name": name, "description": description[:255]}
    if target_url:
        payload["target_url"] = target_url
    return payload


@dataclass(slots=True)
class GitLabClient:
    """Minimal GitLab REST client (v4 API) for MR notes and statuses.

    Attributes:
        token: A GitLab personal/project access token.
        project_id: Numeric project id or URL-encoded path.
        client: Injectable transport (``httpx.Client``-like).
        api_base: API base URL (override for self-managed instances).
        timeout: Per-request timeout.
    """

    token: str
    project_id: str | int
    client: HttpClientLike
    api_base: str = "https://gitlab.com/api/v4"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.token:
            raise GitLabError("GitLab token must be non-empty")
        if self.project_id in ("", None):
            raise GitLabError("project_id must be set")

    def _headers(self) -> dict[str, str]:
        return {
            "PRIVATE-TOKEN": self.token,
            "Accept": "application/json",
            "User-Agent": "llmdiff/0.1",
        }

    def _project(self) -> str:
        pid = str(self.project_id)
        if pid.isdigit():
            return pid
        from urllib.parse import quote

        return quote(pid, safe="")

    def _url(self, path: str) -> str:
        return f"{self.api_base}/projects/{self._project()}{path}"

    def _check(self, response: HttpResponseLike, *, what: str) -> Any:
        if 200 <= response.status_code < 300:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise GitLabError(
            f"GitLab API error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # MR notes
    # ------------------------------------------------------------------ #

    def create_mr_note(self, mr_iid: int, body: str) -> dict[str, Any]:
        """Post a note on merge request ``mr_iid``."""
        if mr_iid < 1:
            raise GitLabError(f"invalid MR iid: {mr_iid}")
        if not body.strip():
            raise GitLabError("note body must be non-empty")
        response = self.client.request(
            "POST",
            self._url(f"/merge_requests/{mr_iid}/notes"),
            headers=self._headers(),
            json={"body": body},
            timeout=self.timeout,
        )
        result = self._check(response, what=f"create MR note !{mr_iid}")
        return result if isinstance(result, dict) else {}

    def list_mr_notes(self, mr_iid: int) -> list[dict[str, Any]]:
        """List notes on merge request ``mr_iid``."""
        response = self.client.request(
            "GET",
            self._url(f"/merge_requests/{mr_iid}/notes"),
            headers=self._headers(),
            params={"sort": "desc", "order_by": "created_at"},
            timeout=self.timeout,
        )
        result = self._check(response, what=f"list MR notes !{mr_iid}")
        return list(result) if isinstance(result, list) else []

    def find_bot_note(self, mr_iid: int, *, marker: str = "## llmdiff verdict") -> dict[str, Any] | None:
        """Return the first MR note containing ``marker``, or None."""
        for note in self.list_mr_notes(mr_iid):
            body = note.get("body")
            if isinstance(body, str) and marker in body and not note.get("system"):
                return note
        return None

    def upsert_mr_note(self, mr_iid: int, body: str, *, marker: str = "## llmdiff verdict") -> dict[str, Any]:
        """Update the existing llmdiff note on an MR, or create one."""
        existing = self.find_bot_note(mr_iid, marker=marker)
        if existing is not None and isinstance(existing.get("id"), int):
            response = self.client.request(
                "PUT",
                self._url(f"/merge_requests/{mr_iid}/notes/{existing['id']}"),
                headers=self._headers(),
                json={"body": body},
                timeout=self.timeout,
            )
            result = self._check(response, what=f"update MR note {existing['id']}")
            return result if isinstance(result, dict) else {}
        return self.create_mr_note(mr_iid, body)

    # ------------------------------------------------------------------ #
    # commit statuses
    # ------------------------------------------------------------------ #

    def create_commit_status(
        self,
        sha: str,
        *,
        state: str,
        name: str = "llmdiff",
        description: str = "",
        target_url: str = "",
    ) -> dict[str, Any]:
        """Set a commit status on ``sha``."""
        if not sha or len(sha) < 7:
            raise GitLabError(f"invalid commit sha: {sha!r}")
        payload = build_commit_status_payload(
            state=state, name=name, description=description, target_url=target_url
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
            self._url(f"/repository/commits/{sha}/statuses"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        result = self._check(response, what=f"list statuses for {sha[:8]}")
        return list(result) if isinstance(result, list) else []

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #

    def report_result(
        self,
        result: Any,
        *,
        mr_iid: int | None = None,
        sha: str = "",
        target_url: str = "",
        note_template: str = "",
    ) -> dict[str, Any]:
        """Post the verdict as an MR note and/or a commit status."""
        matrix = getattr(result, "matrix", None)
        regressed = bool(getattr(matrix, "regressed_dimensions", ())) if matrix else False
        out: dict[str, Any] = {"status": None, "note": None}
        if sha:
            out["status"] = self.create_commit_status(
                sha,
                state="failed" if regressed else "success",
                description="regressions detected" if regressed else "no regressions",
                target_url=target_url,
            )
        if mr_iid is not None:
            body = build_mr_note_body(result, template=note_template)
            out["note"] = self.upsert_mr_note(mr_iid, body)
        return out


# ---------------------------------------------------------------------- #
# FastAPI webhook receiver
# ---------------------------------------------------------------------- #

Handler = Callable[[GitLabEvent], Any]


@dataclass(slots=True)
class GitLabWebhookApp:
    """A FastAPI app that verifies and dispatches GitLab webhooks.

    Attributes:
        secret_token: The webhook secret token configured in GitLab.
        handlers: Mapping of event name to handler callables.
        on_error: Optional error handler ``(event, error) -> None``.
    """

    secret_token: str
    handlers: dict[str, list[Handler]] = field(default_factory=dict)
    on_error: Callable[[GitLabEvent | None, BaseException], None] | None = None

    def __post_init__(self) -> None:
        if not self.secret_token:
            raise GitLabError("secret_token must be non-empty")

    def on(self, event_name: str, handler: Handler) -> Handler:
        """Decorator: register ``handler`` for ``event_name``."""
        self.handlers.setdefault(event_name, []).append(handler)
        return handler

    def dispatch(self, event: GitLabEvent) -> list[Any]:
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
        token: str | None,
        body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        """Verify + parse + dispatch one raw webhook request."""
        if not verify_gitlab_token(self.secret_token, token):
            return 401, {"error": "invalid token"}
        try:
            event = parse_gitlab_event(event_name, body)
        except GitLabError as exc:
            return 400, {"error": str(exc)}
        if event.name == "System Hook" and event.object_kind == "ping":
            return 200, {"ok": True}
        results = self.dispatch(event)
        return 200, {
            "ok": True,
            "event": event.name,
            "object_kind": event.object_kind,
            "action": event.action,
            "handled": len(results),
        }

    def create_fastapi_app(self) -> Any:
        """Build and return a FastAPI app exposing ``POST /webhooks/gitlab``."""
        try:
            from fastapi import FastAPI, Request  # type: ignore[import-untyped]
            from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise GitLabError("fastapi is required for the webhook receiver") from exc

        app = FastAPI(title="llmdiff GitLab integration")

        @app.post("/webhooks/gitlab")
        async def gitlab_webhook(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_request(
                event_name=request.headers.get("X-Gitlab-Event"),
                token=request.headers.get("X-Gitlab-Token"),
                body=body,
            )
            return JSONResponse(payload, status_code=status)

        return app
