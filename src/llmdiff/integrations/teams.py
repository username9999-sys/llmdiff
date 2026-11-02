"""Microsoft Teams integration: incoming webhook + outgoing webhook + cards.

llmdiff posts verdict summaries to a Teams channel. This module implements:

- :func:`build_adaptive_card` — render a DiffResult as an Adaptive Card
  (v1.4) payload for an outgoing webhook,
- :func:`parse_teams_message` — parse an incoming webhook / bot message into
  a typed :class:`TeamsMessage`,
- :class:`TeamsClient` — post cards via an outgoing webhook URL and via the
  Bot Framework (injectable transport, offline-testable),
- :class:`TeamsApp` — FastAPI app exposing ``POST /teams/messages`` for
  incoming messages (with optional shared-secret verification).

Adaptive Card schema notes: we emit a minimal, widely-supported subset
(``TextBlock``, ``FactSet``, ``Container`` with ``style``) so the card renders
in Teams, and we keep the JSON small enough for the 28 KB webhook limit.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "TeamsError",
    "HttpResponseLike",
    "HttpClientLike",
    "TeamsMessage",
    "TeamsClient",
    "TeamsApp",
    "verify_teams_secret",
    "parse_teams_message",
    "build_adaptive_card",
    "build_verdict_facts",
    "CARD_SCHEMA",
    "CARD_TYPE",
]

CARD_SCHEMA = "http://adaptivecards.io/schemas/adaptive-card.json"
CARD_TYPE = "AdaptiveCard"
CARD_VERSION = "1.4"
MAX_CARD_BYTES = 28 * 1024  # Teams outgoing webhook payload limit


class TeamsError(LLMDiffError):
    """Raised for Teams configuration, parsing or API errors."""


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


def verify_teams_secret(secret: str, header_token: str | None) -> bool:
    """Constant-time comparison of an incoming webhook's shared secret.

    Teams incoming webhooks support an ``?secret=...`` query parameter or a
    custom ``X-Teams-Secret`` header when configured via Power Automate; we
    accept either via ``header_token``.
    """
    if not header_token:
        return False
    return hmac.compare_digest(secret, header_token)


@dataclass(frozen=True, slots=True)
class TeamsMessage:
    """A parsed incoming Teams message (bot framework activity).

    Attributes:
        type: Activity type (``message``, ``messageReaction``, ...).
        text: The message text (HTML stripped for ``message`` activities).
        from_id / from_name: Sender id and name.
        channel_id / conversation_id: Where it happened.
        timestamp: ISO timestamp when present.
        raw: The full payload.
    """

    type: str
    text: str
    raw: Mapping[str, Any]
    from_id: str = ""
    from_name: str = ""
    channel_id: str = ""
    conversation_id: str = ""
    timestamp: str = ""

    @property
    def is_message(self) -> bool:
        return self.type == "message"

    @property
    def command_args(self) -> tuple[str, ...]:
        """Text split on whitespace (after stripping a bot mention)."""
        text = self.text
        # Strip a leading "<at>BotName</at>" mention if present.
        if text.startswith("<at>"):
            end = text.find("</at>")
            if end != -1:
                text = text[end + len("</at>") :]
        return tuple(text.split())


def _strip_html(text: str) -> str:
    """Very small HTML stripper for Teams message text."""
    out: list[str] = []
    inside = False
    for ch in text:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out.append(ch)
    return "".join(out).strip()


def parse_teams_message(body: bytes) -> TeamsMessage:
    """Parse a raw bot-framework activity body.

    Raises:
        TeamsError: If the body is not a JSON object.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeamsError(f"message body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TeamsError("message payload must be a JSON object")
    sender = payload.get("from") if isinstance(payload.get("from"), dict) else {}
    conversation = payload.get("conversation") if isinstance(payload.get("conversation"), dict) else {}
    raw_text = payload.get("text")
    text = _strip_html(str(raw_text)) if isinstance(raw_text, str) else ""
    return TeamsMessage(
        type=str(payload.get("type", "")),
        text=text,
        raw=payload,
        from_id=str(sender.get("id", "")),
        from_name=str(sender.get("name", "")),
        channel_id=str(payload.get("channelId", "")),
        conversation_id=str(conversation.get("id", "")),
        timestamp=str(payload.get("timestamp", "")),
    )


def build_verdict_facts(result: Any) -> list[dict[str, str]]:
    """Return Adaptive Card ``FactSet`` facts for a DiffResult-like object."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    facts: list[dict[str, str]] = [
        {"title": "Verdict", "value": str(getattr(matrix, "verdict", "unknown"))},
        {"title": "Wins", "value": str(getattr(matrix, "wins", 0))},
        {"title": "Ties", "value": str(getattr(matrix, "ties", 0))},
        {"title": "Losses", "value": str(getattr(matrix, "losses", 0))},
        {"title": "Regression rate", "value": f"{getattr(matrix, 'regression_rate', 0.0):.1%}"},
    ]
    regressed = getattr(matrix, "regressed_dimensions", ()) or ()
    if regressed:
        facts.append({"title": "Regressed", "value": ", ".join(regressed)})
    for summary in getattr(matrix, "dimensions", ()) or []:
        facts.append(
            {
                "title": str(summary.dimension),
                "value": f"W{summary.wins} T{summary.ties} L{summary.losses} ({summary.delta:+.3f})",
            }
        )
    return facts


def build_adaptive_card(result: Any, *, title: str = "llmdiff verdict") -> dict[str, Any]:
    """Render a DiffResult as an Adaptive Card payload.

    The returned dict is the full webhook body::

        {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": {...}}]}

    Raises:
        TeamsError: If the serialised card exceeds the Teams size limit.
    """
    matrix = getattr(result, "matrix", None)
    verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
    regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
    accent = "attention" if regressed else ("good" if verdict == "win" else "default")
    card: dict[str, Any] = {
        "type": CARD_TYPE,
        "version": CARD_VERSION,
        "body": [
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "text": title,
                "wrap": True,
            },
            {
                "type": "Container",
                "style": accent,
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"Overall verdict: {verdict}",
                        "weight": "Bolder",
                        "wrap": True,
                    },
                    {"type": "FactSet", "facts": build_verdict_facts(result)},
                ],
            },
        ],
    }
    body = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }
        ],
    }
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_CARD_BYTES:
        raise TeamsError(
            f"adaptive card is {len(encoded)} bytes, over the {MAX_CARD_BYTES} byte Teams limit"
        )
    return body


@dataclass(slots=True)
class TeamsClient:
    """Posts messages to Teams via outgoing webhook or Bot Framework.

    Attributes:
        webhook_url: An outgoing webhook URL (``https://...webhook.office.com/...``).
        client: Injectable transport.
        bot_token: Optional Bot Framework bearer token for ``/v3/conversations/...``.
        api_base: Bot Framework base URL.
        timeout: Per-request timeout.
    """

    webhook_url: str = ""
    client: HttpClientLike | None = None
    bot_token: str = ""
    api_base: str = "https://smba.trafficmanager.net/amer"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.webhook_url and not self.bot_token:
            raise TeamsError("either webhook_url or bot_token is required")

    def _transport(self) -> HttpClientLike:
        if self.client is None:
            import httpx  # type: ignore[import-untyped]

            self.client = httpx.Client()
        return self.client

    def _headers(self, *, bot: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if bot:
            if not self.bot_token:
                raise TeamsError("bot_token required for Bot Framework calls")
            headers["Authorization"] = f"Bearer {self.bot_token}"
        return headers

    def _check(self, response: HttpResponseLike, *, what: str) -> Any:
        # Teams outgoing webhooks return 200/202 with "1" body on success.
        if 200 <= response.status_code < 300:
            text = response.text.strip()
            if text in ("1", ""):
                return {"ok": True}
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return {"ok": True, "raw": text}
        raise TeamsError(
            f"Teams API error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    def post_card(self, card_body: Mapping[str, Any], *, url: str | None = None) -> dict[str, Any]:
        """Post an adaptive-card message body to the outgoing webhook."""
        target = url or self.webhook_url
        if not target:
            raise TeamsError("no webhook url configured")
        if not target.startswith("https://"):
            raise TeamsError(f"webhook url must be https, got {target!r}")
        response = self._transport().post(
            target,
            headers=self._headers(),
            json=dict(card_body),
            timeout=self.timeout,
        )
        return self._check(response, what="post card")

    def post_verdict(self, result: Any, *, title: str = "llmdiff verdict") -> dict[str, Any]:
        """Build and post the verdict adaptive card."""
        return self.post_card(build_adaptive_card(result, title=title))

    def send_to_conversation(
        self,
        conversation_id: str,
        card_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Post a card to a conversation via the Bot Framework."""
        if not conversation_id:
            raise TeamsError("conversation_id is required")
        url = f"{self.api_base}/v3/conversations/{conversation_id}/activities"
        response = self._transport().post(
            url,
            headers=self._headers(bot=True),
            json=dict(card_body),
            timeout=self.timeout,
        )
        result = self._check(response, what="send to conversation")
        return result if isinstance(result, dict) else {"ok": True}


# ---------------------------------------------------------------------- #
# FastAPI app for incoming messages
# ---------------------------------------------------------------------- #

MessageHandler = Callable[[TeamsMessage], Any]


@dataclass(slots=True)
class TeamsApp:
    """FastAPI app for incoming Teams messages.

    Attributes:
        secret: Optional shared secret; when set, requests must carry it in
            the ``X-Teams-Secret`` header.
        handlers: Mapping of activity type (``message``) to handlers; ``*``
            catches everything.
        on_error: Optional ``(message, error) -> None``.
    """

    secret: str = ""
    handlers: dict[str, list[MessageHandler]] = field(default_factory=dict)
    on_error: Callable[[TeamsMessage | None, BaseException], None] | None = None

    def on(self, activity_type: str, handler: MessageHandler) -> MessageHandler:
        """Decorator: register a handler for an activity type."""
        self.handlers.setdefault(activity_type, []).append(handler)
        return handler

    def dispatch(self, message: TeamsMessage) -> list[Any]:
        """Run handlers for the message's type (plus ``*``)."""
        results: list[Any] = []
        for handler in [
            *self.handlers.get(message.type, []),
            *self.handlers.get("*", []),
        ]:
            try:
                results.append(handler(message))
            except Exception as exc:  # noqa: BLE001
                if self.on_error is not None:
                    self.on_error(message, exc)
                else:
                    raise
        return results

    def handle_request(
        self,
        *,
        secret_header: str | None,
        body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        """Verify + parse + dispatch one incoming message."""
        if self.secret and not verify_teams_secret(self.secret, secret_header):
            return 401, {"error": "invalid secret"}
        try:
            message = parse_teams_message(body)
        except TeamsError as exc:
            return 400, {"error": str(exc)}
        results = self.dispatch(message)
        return 200, {
            "ok": True,
            "type": message.type,
            "handled": len(results),
            "results": [r for r in results if isinstance(r, (dict, str, int, float, bool))],
        }

    def create_fastapi_app(self) -> Any:
        """Build a FastAPI app exposing ``POST /teams/messages``."""
        try:
            from fastapi import FastAPI, Request  # type: ignore[import-untyped]
            from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TeamsError("fastapi is required for the Teams app") from exc

        app = FastAPI(title="llmdiff Teams integration")

        @app.post("/teams/messages")
        async def teams_messages(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_request(
                secret_header=request.headers.get("X-Teams-Secret"),
                body=body,
            )
            return JSONResponse(payload, status_code=status)

        return app
