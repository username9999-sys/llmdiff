"""Slack app integration: Events API, slash commands and Block Kit messages.

llmdiff posts verdict summaries to Slack channels and answers the
``/llmdiff`` slash command. This module implements:

- :func:`verify_slack_signature` — HMAC-SHA256 verification of the
  ``X-Slack-Signature`` header using the signing secret and a 5-minute
  freshness window (replay protection),
- :func:`parse_slack_event` — parse an Events API envelope into a typed
  :class:`SlackEvent`,
- :func:`parse_slash_command` — parse a ``application/x-www-form-urlencoded``
  slash-command body into :class:`SlashCommand`,
- :func:`build_verdict_blocks` — render a DiffResult as Slack Block Kit JSON,
- :class:`SlackClient` — chat.postMessage / chat.update / reactions.add with
  an injectable transport,
- :class:`SlackApp` — FastAPI app exposing ``/slack/events`` and
  ``/slack/commands``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qs

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "SlackError",
    "HttpResponseLike",
    "HttpClientLike",
    "SlackEvent",
    "SlashCommand",
    "SlackClient",
    "SlackApp",
    "verify_slack_signature",
    "compute_slack_signature",
    "parse_slack_event",
    "parse_slash_command",
    "build_verdict_blocks",
    "build_verdict_text",
    "escape_slack_text",
]


class SlackError(LLMDiffError):
    """Raised for Slack configuration, parsing or API errors."""


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
        data: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


SIGNATURE_VERSION = "v0"
REPLAY_WINDOW_SECONDS = 300


def compute_slack_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Return the ``v0=<hex>`` signature Slack sends in ``X-Slack-Signature``."""
    base = f"{SIGNATURE_VERSION}:{timestamp}:".encode() + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"{SIGNATURE_VERSION}={digest}"


def verify_slack_signature(
    secret: str,
    timestamp: str | None,
    body: bytes,
    signature: str | None,
    *,
    now: float | None = None,
) -> bool:
    """Verify a Slack request signature with replay protection.

    The timestamp must be within 5 minutes of ``now`` (defaults to the
    current time). Returns False for missing/malformed/stale values.
    """
    if not timestamp or not signature:
        return False
    try:
        ts = float(timestamp)
    except ValueError:
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > REPLAY_WINDOW_SECONDS:
        return False
    expected = compute_slack_signature(secret, timestamp, body)
    return hmac.compare_digest(expected, signature)


@dataclass(frozen=True, slots=True)
class SlackEvent:
    """A parsed Events API envelope.

    Attributes:
        type: The envelope ``type`` (``event_callback``, ``url_verification``...).
        event: The inner event mapping.
        team_id / channel_id / user_id: convenience accessors.
        event_type: inner ``type`` (``app_mention``, ``message``...).
        text: inner ``text`` when present.
        ts: inner ``ts`` (message timestamp) when present.
    """

    type: str
    event: Mapping[str, Any]
    team_id: str = ""
    event_id: str = ""
    api_app_id: str = ""

    @property
    def event_type(self) -> str:
        value = self.event.get("type")
        return str(value) if value is not None else ""

    @property
    def channel_id(self) -> str:
        value = self.event.get("channel")
        return str(value) if value is not None else ""

    @property
    def user_id(self) -> str:
        value = self.event.get("user")
        return str(value) if value is not None else ""

    @property
    def text(self) -> str:
        value = self.event.get("text")
        return str(value) if value is not None else ""

    @property
    def ts(self) -> str:
        value = self.event.get("ts")
        return str(value) if value is not None else ""


def parse_slack_event(body: bytes) -> SlackEvent:
    """Parse a raw Events API body.

    Raises:
        SlackError: If the body is not a JSON object.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlackError(f"event body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SlackError("event payload must be a JSON object")
    event = payload.get("event")
    if payload.get("type") == "event_callback" and not isinstance(event, dict):
        raise SlackError("event_callback envelope missing 'event' object")
    return SlackEvent(
        type=str(payload.get("type", "")),
        event=event if isinstance(event, dict) else {},
        team_id=str(payload.get("team_id", "")),
        event_id=str(payload.get("event_id", "")),
        api_app_id=str(payload.get("api_app_id", "")),
    )


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """A parsed slash-command invocation.

    Attributes:
        command: The command name, e.g. ``/llmdiff``.
        text: The free-text arguments.
        user_id / user_name: The invoking user.
        channel_id / channel_name: Where it was invoked.
        team_id: The workspace id.
        response_url: The webhook URL for delayed responses.
        trigger_id: For opening modals.
    """

    command: str
    text: str
    user_id: str = ""
    user_name: str = ""
    channel_id: str = ""
    channel_name: str = ""
    team_id: str = ""
    response_url: str = ""
    trigger_id: str = ""

    @property
    def args(self) -> tuple[str, ...]:
        """The text split on whitespace."""
        return tuple(self.text.split())


def parse_slash_command(body: bytes) -> SlashCommand:
    """Parse an ``application/x-www-form-urlencoded`` slash-command body.

    Raises:
        SlackError: If the body cannot be parsed.
    """
    try:
        text_body = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SlackError(f"slash command body is not UTF-8: {exc}") from exc
    fields = parse_qs(text_body, keep_blank_values=True)

    def one(key: str) -> str:
        values = fields.get(key)
        return values[0] if values else ""

    return SlashCommand(
        command=one("command"),
        text=one("text"),
        user_id=one("user_id"),
        user_name=one("user_name"),
        channel_id=one("channel_id"),
        channel_name=one("channel_name"),
        team_id=one("team_id"),
        response_url=one("response_url"),
        trigger_id=one("trigger_id"),
    )


def escape_slack_text(text: str) -> str:
    """Escape ``&``, ``<``, ``>`` for Slack mrkdwn."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_verdict_text(result: Any) -> str:
    """One-line plain-text verdict summary for Slack."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        return "llmdiff: no result"
    verdict = getattr(matrix, "verdict", "unknown")
    wins = getattr(matrix, "wins", 0)
    ties = getattr(matrix, "ties", 0)
    losses = getattr(matrix, "losses", 0)
    rate = getattr(matrix, "regression_rate", 0.0)
    return (
        f"llmdiff verdict: *{verdict}* "
        f"(W{wins} T{ties} L{losses}, regression rate {rate:.1%})"
    )


def build_verdict_blocks(result: Any, *, channel: str = "") -> dict[str, Any]:
    """Render a DiffResult as a Slack Block Kit message payload.

    Returns a dict with ``text`` (fallback) and ``blocks``.
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    summary_line = build_verdict_text(result)
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "llmdiff verdict", "emoji": True},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": summary_line},
        },
    ]
    dimensions = getattr(matrix, "dimensions", ()) or []
    if dimensions:
        rows = [
            f"`{escape_slack_text(s.dimension)}`  W{s.wins} T{s.ties} L{s.losses}  "
            f"delta {s.delta:+.3f}"
            for s in dimensions
        ]
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(rows)},
            }
        )
    regressed = getattr(matrix, "regressed_dimensions", ()) or ()
    if regressed:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Regressed: "
                        + ", ".join(f"`{escape_slack_text(d)}`" for d in regressed),
                    }
                ],
            }
        )
    if channel:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"posted to <#{channel}>"}],
            }
        )
    return {"text": summary_line, "blocks": blocks}


@dataclass(slots=True)
class SlackClient:
    """Minimal Slack Web API client.

    Attributes:
        token: A bot token (``xoxb-...``).
        client: Injectable transport (``httpx.Client``-like).
        api_base: API base URL (override for tests/proxies).
        timeout: Per-request timeout.
    """

    token: str
    client: HttpClientLike
    api_base: str = "https://slack.com/api"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.token.startswith("xox"):
            raise SlackError(f"token must look like a Slack token (xox...), got {self.token[:6]!r}...")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _check(self, response: HttpResponseLike, *, what: str) -> dict[str, Any]:
        if 200 <= response.status_code < 300:
            try:
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                raise SlackError(f"Slack API returned non-JSON for {what}: {exc}") from exc
            if isinstance(data, dict) and data.get("ok") is False:
                raise SlackError(f"Slack API error on {what}: {data.get('error', 'unknown')}")
            return data if isinstance(data, dict) else {"ok": True, "data": data}
        raise SlackError(
            f"Slack API HTTP error on {what}: {response.status_code}: {response.text[:300]}"
        )

    def post_message(
        self,
        channel: str,
        *,
        text: str = "",
        blocks: Sequence[Mapping[str, Any]] | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """chat.postMessage."""
        if not channel:
            raise SlackError("channel is required")
        payload: dict[str, Any] = {"channel": channel, "text": text or " "}
        if blocks:
            payload["blocks"] = [dict(b) for b in blocks]
        if thread_ts:
            payload["thread_ts"] = thread_ts
        response = self.client.post(
            f"{self.api_base}/chat.postMessage",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        return self._check(response, what="chat.postMessage")

    def update_message(
        self,
        channel: str,
        ts: str,
        *,
        text: str = "",
        blocks: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """chat.update — edit an existing message."""
        if not ts:
            raise SlackError("ts is required to update a message")
        payload: dict[str, Any] = {"channel": channel, "ts": ts, "text": text or " "}
        if blocks:
            payload["blocks"] = [dict(b) for b in blocks]
        response = self.client.post(
            f"{self.api_base}/chat.update",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        return self._check(response, what="chat.update")

    def add_reaction(self, channel: str, ts: str, name: str) -> dict[str, Any]:
        """reactions.add."""
        if not name:
            raise SlackError("reaction name is required")
        response = self.client.post(
            f"{self.api_base}/reactions.add",
            headers=self._headers(),
            json={"channel": channel, "timestamp": ts, "name": name},
            timeout=self.timeout,
        )
        return self._check(response, what="reactions.add")

    def post_verdict(
        self,
        channel: str,
        result: Any,
        *,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Post a verdict Block Kit message to ``channel``."""
        payload = build_verdict_blocks(result, channel=channel)
        return self.post_message(
            channel,
            text=payload["text"],
            blocks=payload["blocks"],
            thread_ts=thread_ts,
        )

    def reply_to_command(self, response_url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Send a delayed response to a slash command's ``response_url``."""
        if not response_url.startswith(("http://", "https://")):
            raise SlackError(f"response_url must be http(s), got {response_url!r}")
        response = self.client.post(
            response_url,
            headers={"Content-Type": "application/json"},
            json=dict(payload),
            timeout=self.timeout,
        )
        if 200 <= response.status_code < 300:
            return {"ok": True}
        raise SlackError(
            f"response_url POST failed: HTTP {response.status_code}: {response.text[:200]}"
        )


# ---------------------------------------------------------------------- #
# FastAPI app
# ---------------------------------------------------------------------- #

EventHandler = Callable[[SlackEvent], Any]
CommandHandler = Callable[[SlashCommand], Any]


@dataclass(slots=True)
class SlackApp:
    """FastAPI app for Slack Events API + slash commands.

    Attributes:
        signing_secret: The app's signing secret.
        event_handlers: Mapping of inner event type (``app_mention``...) to
            handlers; ``*`` receives everything.
        command_handlers: Mapping of command name (``/llmdiff``) to handlers.
        on_error: Optional ``(kind, payload, error) -> None``.
    """

    signing_secret: str
    event_handlers: dict[str, list[EventHandler]] = field(default_factory=dict)
    command_handlers: dict[str, list[CommandHandler]] = field(default_factory=dict)
    on_error: Callable[[str, Any, BaseException], None] | None = None

    def __post_init__(self) -> None:
        if not self.signing_secret:
            raise SlackError("signing_secret must be non-empty")

    def on_event(self, event_type: str, handler: EventHandler) -> EventHandler:
        """Decorator: register a handler for an inner event type."""
        self.event_handlers.setdefault(event_type, []).append(handler)
        return handler

    def on_command(self, command: str, handler: CommandHandler) -> CommandHandler:
        """Decorator: register a handler for a slash command."""
        self.command_handlers.setdefault(command, []).append(handler)
        return handler

    def dispatch_event(self, event: SlackEvent) -> list[Any]:
        results: list[Any] = []
        for handler in [
            *self.event_handlers.get(event.event_type, []),
            *self.event_handlers.get("*", []),
        ]:
            try:
                results.append(handler(event))
            except Exception as exc:  # noqa: BLE001
                if self.on_error is not None:
                    self.on_error("event", event, exc)
                else:
                    raise
        return results

    def dispatch_command(self, command: SlashCommand) -> list[Any]:
        results: list[Any] = []
        for handler in [
            *self.command_handlers.get(command.command, []),
            *self.command_handlers.get("*", []),
        ]:
            try:
                results.append(handler(command))
            except Exception as exc:  # noqa: BLE001
                if self.on_error is not None:
                    self.on_error("command", command, exc)
                else:
                    raise
        return results

    def handle_events_request(
        self,
        *,
        timestamp: str | None,
        signature: str | None,
        body: bytes,
    ) -> tuple[int, Any]:
        """Verify + parse + dispatch an Events API request."""
        if not verify_slack_signature(self.signing_secret, timestamp, body, signature):
            return 401, {"error": "invalid signature"}
        try:
            envelope = parse_slack_event(body)
        except SlackError as exc:
            return 400, {"error": str(exc)}
        if envelope.type == "url_verification":
            return 200, {"challenge": envelope.event.get("challenge", "")}
        self.dispatch_event(envelope)
        return 200, {"ok": True}

    def handle_command_request(
        self,
        *,
        timestamp: str | None,
        signature: str | None,
        body: bytes,
    ) -> tuple[int, Any]:
        """Verify + parse + dispatch a slash-command request."""
        if not verify_slack_signature(self.signing_secret, timestamp, body, signature):
            return 401, {"error": "invalid signature"}
        try:
            command = parse_slash_command(body)
        except SlackError as exc:
            return 400, {"error": str(exc)}
        results = self.dispatch_command(command)
        # If a handler returned a dict, it is the immediate response payload.
        for result in results:
            if isinstance(result, dict):
                return 200, result
        return 200, {"response_type": "ephemeral", "text": "ok"}

    def create_fastapi_app(self) -> Any:
        """Build a FastAPI app with ``/slack/events`` and ``/slack/commands``."""
        try:
            from fastapi import FastAPI, Request  # type: ignore[import-untyped]
            from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SlackError("fastapi is required for the Slack app") from exc

        app = FastAPI(title="llmdiff Slack app")

        @app.post("/slack/events")
        async def slack_events(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_events_request(
                timestamp=request.headers.get("X-Slack-Request-Timestamp"),
                signature=request.headers.get("X-Slack-Signature"),
                body=body,
            )
            return JSONResponse(payload, status_code=status)

        @app.post("/slack/commands")
        async def slack_commands(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_command_request(
                timestamp=request.headers.get("X-Slack-Request-Timestamp"),
                signature=request.headers.get("X-Slack-Signature"),
                body=body,
            )
            return JSONResponse(payload, status_code=status)

        return app
