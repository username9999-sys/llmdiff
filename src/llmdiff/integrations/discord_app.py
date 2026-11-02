"""Discord app integration: gateway webhook receiver and message posting.

llmdiff posts verdict summaries to Discord channels. This module implements:

- :func:`verify_discord_signature` — Ed25519 verification of the
  ``X-Signature-Ed25519`` / ``X-Signature-Timestamp`` headers using the
  application's public key (via ``cryptography`` when available, with a pure
  fallback disabled by default — see notes),
- :func:`parse_discord_interaction` — parse an interaction payload
  (type 1 = PING, 2 = APPLICATION_COMMAND, 3 = MESSAGE_COMPONENT) into a typed
  :class:`DiscordInteraction`,
- :func:`build_verdict_embed` — render a DiffResult as a Discord embed,
- :class:`DiscordClient` — create messages / followups with an injectable
  transport,
- :class:`DiscordApp` — FastAPI app exposing ``POST /discord/interactions``.

Signature verification uses Ed25519. The ``cryptography`` package is used
when installed; otherwise verification raises unless ``allow_unverified`` is
set (useful only for local development).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "DiscordError",
    "HttpResponseLike",
    "HttpClientLike",
    "DiscordInteraction",
    "DiscordClient",
    "DiscordApp",
    "verify_discord_signature",
    "parse_discord_interaction",
    "build_verdict_embed",
    "build_verdict_content",
    "interaction_response_ping",
    "interaction_response_message",
]


class DiscordError(LLMDiffError):
    """Raised for Discord configuration, parsing or API errors."""


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

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Any = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


def verify_discord_signature(
    public_key_hex: str,
    signature_hex: str | None,
    timestamp: str | None,
    body: bytes,
    *,
    allow_unverified: bool = False,
) -> bool:
    """Verify a Discord interaction request's Ed25519 signature.

    Discord signs ``timestamp + body`` with the application private key; we
    verify with the public key. Returns False for missing/malformed input.
    When ``cryptography`` is unavailable, returns ``allow_unverified``.

    Raises:
        DiscordError: If ``public_key_hex`` is malformed.
    """
    if not signature_hex or not timestamp:
        return False
    try:
        public_key_bytes = bytes.fromhex(public_key_hex)
        signature_bytes = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise DiscordError(f"malformed hex in signature or public key: {exc}") from exc
    if len(public_key_bytes) != 32:
        raise DiscordError("Discord public key must be 32 bytes (64 hex chars)")
    message = timestamp.encode("utf-8") + body
    try:
        from cryptography.exceptions import InvalidSignature  # type: ignore[import-untyped]
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-untyped]
            Ed25519PublicKey,
        )
    except ImportError:
        return allow_unverified
    key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        key.verify(signature_bytes, message)
        return True
    except InvalidSignature:
        return False


@dataclass(frozen=True, slots=True)
class DiscordInteraction:
    """A parsed Discord interaction payload.

    Attributes:
        type: Interaction type (1=PING, 2=APPLICATION_COMMAND, 3=MESSAGE_COMPONENT, ...).
        id / token: Interaction id and token (needed for followups).
        application_id: The application id.
        guild_id / channel_id: Where it happened, when present.
        user_id: The invoking user's id.
        command_name: Slash command name for APPLICATION_COMMAND.
        options: Resolved options mapping for APPLICATION_COMMAND.
        raw: The full payload.
    """

    type: int
    raw: Mapping[str, Any]
    id: str = ""
    token: str = ""
    application_id: str = ""
    guild_id: str = ""
    channel_id: str = ""
    user_id: str = ""
    command_name: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_ping(self) -> bool:
        return self.type == 1

    @property
    def is_command(self) -> bool:
        return self.type == 2

    @property
    def is_component(self) -> bool:
        return self.type == 3

    def option(self, name: str) -> Any:
        """Return the value of slash-command option ``name``, or None."""
        return self.options.get(name)


def parse_discord_interaction(body: bytes) -> DiscordInteraction:
    """Parse a raw interaction body.

    Raises:
        DiscordError: If the body is not a JSON object or lacks ``type``.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscordError(f"interaction body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiscordError("interaction payload must be a JSON object")
    itype = payload.get("type")
    if not isinstance(itype, int):
        raise DiscordError("interaction payload missing integer 'type'")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    member = payload.get("member") if isinstance(payload.get("member"), dict) else {}
    user = payload.get("user") if isinstance(payload.get("user"), dict) else member.get("user", {})
    options: dict[str, Any] = {}
    raw_options = data.get("options")
    if isinstance(raw_options, list):
        for opt in raw_options:
            if isinstance(opt, dict) and "name" in opt:
                options[str(opt["name"])] = opt.get("value")
    return DiscordInteraction(
        type=itype,
        raw=payload,
        id=str(payload.get("id", "")),
        token=str(payload.get("token", "")),
        application_id=str(payload.get("application_id", "")),
        guild_id=str(payload.get("guild_id", "")),
        channel_id=str(payload.get("channel_id", "")),
        user_id=str(user.get("id", "")) if isinstance(user, dict) else "",
        command_name=str(data.get("name", "")),
        options=options,
    )


def interaction_response_ping() -> dict[str, Any]:
    """The PONG response for interaction type 1."""
    return {"type": 1}


def interaction_response_message(
    *,
    content: str = "",
    embeds: Sequence[Mapping[str, Any]] | None = None,
    ephemeral: bool = False,
) -> dict[str, Any]:
    """A DEFERRED/ immediate channel message response (type 4)."""
    data: dict[str, Any] = {"content": content}
    if embeds:
        data["embeds"] = [dict(e) for e in embeds]
    if ephemeral:
        data["flags"] = 64
    return {"type": 4, "data": data}


def build_verdict_content(result: Any) -> str:
    """One-line plain-text verdict for a Discord message."""
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
    return f"llmdiff verdict: **{verdict}** (W{wins} T{ties} L{losses}, regression rate {rate:.1%})"


def build_verdict_embed(result: Any, *, title: str = "llmdiff verdict") -> dict[str, Any]:
    """Render a DiffResult as a Discord embed object."""
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
    regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
    color = 0xE74C3C if regressed else (0x2ECC71 if verdict == "win" else 0x95A5A6)
    fields: list[dict[str, Any]] = []
    for summary in getattr(matrix, "dimensions", ()) or []:
        fields.append(
            {
                "name": str(summary.dimension),
                "value": (
                    f"W{summary.wins} T{summary.ties} L{summary.losses} "
                    f"delta {summary.delta:+.3f}"
                ),
                "inline": True,
            }
        )
    embed: dict[str, Any] = {
        "title": title,
        "description": build_verdict_content(result),
        "color": color,
        "fields": fields,
    }
    if regressed:
        embed["footer"] = {"text": "Regressed: " + ", ".join(regressed)}
    return embed


@dataclass(slots=True)
class DiscordClient:
    """Minimal Discord REST client for channel messages and followups.

    Attributes:
        bot_token: Bot token (``Bot <token>`` auth).
        client: Injectable transport.
        api_base: API base URL.
        timeout: Per-request timeout.
    """

    bot_token: str
    client: HttpClientLike
    api_base: str = "https://discord.com/api/v10"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.bot_token:
            raise DiscordError("bot_token must be non-empty")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
            "User-Agent": "llmdiff/0.1 (discord)",
        }

    def _check(self, response: HttpResponseLike, *, what: str) -> Any:
        if 200 <= response.status_code < 300:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise DiscordError(
            f"Discord API error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    def create_message(
        self,
        channel_id: str,
        *,
        content: str = "",
        embeds: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """POST /channels/{id}/messages."""
        if not channel_id:
            raise DiscordError("channel_id is required")
        payload: dict[str, Any] = {"content": content}
        if embeds:
            payload["embeds"] = [dict(e) for e in embeds]
        response = self.client.post(
            f"{self.api_base}/channels/{channel_id}/messages",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        result = self._check(response, what="create message")
        return result if isinstance(result, dict) else {}

    def post_verdict(self, channel_id: str, result: Any) -> dict[str, Any]:
        """Post a verdict embed to a channel."""
        return self.create_message(
            channel_id,
            content=build_verdict_content(result),
            embeds=[build_verdict_embed(result)],
        )

    def create_followup(
        self,
        application_id: str,
        interaction_token: str,
        *,
        content: str = "",
        embeds: Sequence[Mapping[str, Any]] | None = None,
        ephemeral: bool = False,
    ) -> dict[str, Any]:
        """POST /webhooks/{app_id}/{token} — respond to an interaction later."""
        if not application_id or not interaction_token:
            raise DiscordError("application_id and interaction_token are required")
        payload: dict[str, Any] = {"content": content}
        if embeds:
            payload["embeds"] = [dict(e) for e in embeds]
        if ephemeral:
            payload["flags"] = 64
        response = self.client.post(
            f"{self.api_base}/webhooks/{application_id}/{interaction_token}",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        result = self._check(response, what="followup message")
        return result if isinstance(result, dict) else {}

    def edit_interaction_response(
        self,
        application_id: str,
        interaction_token: str,
        *,
        content: str = "",
        embeds: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """PATCH /webhooks/{app_id}/{token}/messages/@original."""
        payload: dict[str, Any] = {"content": content}
        if embeds:
            payload["embeds"] = [dict(e) for e in embeds]
        response = self.client.request(
            "PATCH",
            f"{self.api_base}/webhooks/{application_id}/{interaction_token}/messages/@original",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        result = self._check(response, what="edit original response")
        return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------- #
# FastAPI app
# ---------------------------------------------------------------------- #

InteractionHandler = Callable[[DiscordInteraction], Any]


@dataclass(slots=True)
class DiscordApp:
    """FastAPI app for Discord interactions.

    Attributes:
        public_key: Hex-encoded Ed25519 public key of the application.
        handlers: Mapping of slash-command name to handlers; ``*`` catches all.
        on_error: Optional ``(interaction, error) -> None``.
        allow_unverified: Dev-only flag to skip Ed25519 when cryptography is
            unavailable.
    """

    public_key: str
    handlers: dict[str, list[InteractionHandler]] = field(default_factory=dict)
    on_error: Callable[[DiscordInteraction | None, BaseException], None] | None = None
    allow_unverified: bool = False

    def __post_init__(self) -> None:
        if len(self.public_key) != 64:
            raise DiscordError("public_key must be 64 hex chars (32 bytes)")

    def on_command(self, name: str, handler: InteractionHandler) -> InteractionHandler:
        """Decorator: register a handler for slash command ``name``."""
        self.handlers.setdefault(name, []).append(handler)
        return handler

    def dispatch(self, interaction: DiscordInteraction) -> list[Any]:
        """Run handlers for the interaction's command name (plus ``*``)."""
        results: list[Any] = []
        for handler in [
            *self.handlers.get(interaction.command_name, []),
            *self.handlers.get("*", []),
        ]:
            try:
                results.append(handler(interaction))
            except Exception as exc:  # noqa: BLE001
                if self.on_error is not None:
                    self.on_error(interaction, exc)
                else:
                    raise
        return results

    def handle_request(
        self,
        *,
        signature: str | None,
        timestamp: str | None,
        body: bytes,
    ) -> tuple[int, Any]:
        """Verify + parse + dispatch one interaction request.

        PING interactions are answered automatically with a PONG.
        """
        if not verify_discord_signature(
            self.public_key, signature, timestamp, body, allow_unverified=self.allow_unverified
        ):
            return 401, {"error": "invalid request signature"}
        try:
            interaction = parse_discord_interaction(body)
        except DiscordError as exc:
            return 400, {"error": str(exc)}
        if interaction.is_ping:
            return 200, interaction_response_ping()
        results = self.dispatch(interaction)
        for result in results:
            if isinstance(result, dict):
                return 200, result
        return 200, interaction_response_message(content="ok", ephemeral=True)

    def create_fastapi_app(self) -> Any:
        """Build a FastAPI app exposing ``POST /discord/interactions``."""
        try:
            from fastapi import FastAPI, Request  # type: ignore[import-untyped]
            from fastapi.responses import JSONResponse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DiscordError("fastapi is required for the Discord app") from exc

        app = FastAPI(title="llmdiff Discord app")

        @app.post("/discord/interactions")
        async def discord_interactions(request: Request) -> Any:
            body = await request.body()
            status, payload = self.handle_request(
                signature=request.headers.get("X-Signature-Ed25519"),
                timestamp=request.headers.get("X-Signature-Timestamp"),
                body=body,
            )
            return JSONResponse(payload, status_code=status)

        return app
