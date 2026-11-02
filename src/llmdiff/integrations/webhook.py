"""Outbound webhook dispatcher with HMAC signing, retries and a SQLite log.

llmdiff can push verdict results to any HTTP endpoint. This module implements
the sender side:

- every request body is JSON and signed with ``X-LLMDiff-Signature``
  (``sha256=<hex hmac>`` of the raw body, using the shared secret),
- deliveries are retried with exponential backoff + jitter,
- every attempt is recorded in a SQLite delivery log so nothing is lost
  silently and replays are auditable,
- the dispatcher is transport-agnostic: it takes an ``httpx.Client``-like
  object, so tests can inject ``httpx.MockTransport``.

Delivery log schema::

    CREATE TABLE deliveries (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        url           TEXT NOT NULL,
        event         TEXT NOT NULL,
        payload       TEXT NOT NULL,
        status        TEXT NOT NULL,          -- pending|delivered|failed
        attempts      INTEGER NOT NULL DEFAULT 0,
        last_status   INTEGER,
        last_error    TEXT,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "WebhookError",
    "HttpResponseLike",
    "HttpClientLike",
    "DeliveryRecord",
    "DeliveryLog",
    "RetryPolicy",
    "WebhookDispatcher",
    "sign_payload",
    "verify_signature",
    "default_headers",
]


class WebhookError(LLMDiffError):
    """Raised when a webhook delivery ultimately fails or is misconfigured."""


class HttpResponseLike(Protocol):
    """Minimal response protocol the dispatcher depends on."""

    status_code: int

    @property
    def text(self) -> str: ...


class HttpClientLike(Protocol):
    """Minimal client protocol (satisfied by ``httpx.Client``)."""

    def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponseLike: ...


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sign_payload(body: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` HMAC-SHA256 signature of ``body``."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(body: bytes, secret: str, signature: str) -> bool:
    """Constant-time check of a ``sha256=<hex>`` signature header."""
    if not signature.startswith("sha256="):
        return False
    expected = sign_payload(body, secret)
    return hmac.compare_digest(expected, signature)


def default_headers(secret: str, body: bytes, *, event: str, delivery_id: str | None = None) -> dict[str, str]:
    """Build the standard header set for an outbound webhook request."""
    headers = {
        "Content-Type": "application/json",
        "X-LLMDiff-Event": event,
        "X-LLMDiff-Signature": sign_payload(body, secret),
        "X-LLMDiff-Delivery": delivery_id or "",
        "User-Agent": "llmdiff-webhook/0.1",
    }
    return headers


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry configuration for failed deliveries.

    Attributes:
        max_attempts: Total attempts including the first one.
        base_delay: Seconds before the first retry; doubles each attempt.
        max_delay: Cap for the backoff delay.
        jitter: Fraction of random jitter added to each delay (0..1).
        retry_on_status: HTTP status codes that trigger a retry.
    """

    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.25
    retry_on_status: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise WebhookError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.base_delay <= 0:
            raise WebhookError(f"base_delay must be positive, got {self.base_delay}")
        if self.max_delay < self.base_delay:
            raise WebhookError(
                f"max_delay ({self.max_delay}) must be >= base_delay ({self.base_delay})"
            )
        if not 0.0 <= self.jitter <= 1.0:
            raise WebhookError(f"jitter must be in [0, 1], got {self.jitter}")

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """Return the backoff delay before retry ``attempt`` (1-based).

        ``attempt=1`` means "first retry", so the delay is ``base_delay``.
        """
        rng = rng or random.Random()
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        spread = delay * self.jitter
        return delay - spread + (2 * spread * rng.random())


@dataclass(slots=True)
class DeliveryRecord:
    """One row of the delivery log.

    Attributes:
        id: Autoincrement id, None until inserted.
        url: Target URL.
        event: Event name (e.g. ``verdict.completed``).
        payload: JSON body as a string.
        status: ``pending``, ``delivered`` or ``failed``.
        attempts: Number of delivery attempts made.
        last_status: Last HTTP status code, if any.
        last_error: Last error message, if any.
        created_at / updated_at: ISO timestamps.
    """

    url: str
    event: str
    payload: str
    status: str = "pending"
    attempts: int = 0
    last_status: int | None = None
    last_error: str | None = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "id": self.id,
            "url": self.url,
            "event": self.event,
            "payload": self.payload,
            "status": self.status,
            "attempts": self.attempts,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    event       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_status INTEGER,
    last_error  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deliveries_status ON deliveries(status);
CREATE INDEX IF NOT EXISTS idx_deliveries_event ON deliveries(event);
"""


class DeliveryLog:
    """SQLite-backed log of webhook deliveries.

    The log is safe to share between threads within one process: every
    operation opens a short-lived connection with a busy timeout.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #

    def insert(self, record: DeliveryRecord) -> DeliveryRecord:
        """Insert a new pending record and return it with its id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO deliveries
                    (url, event, payload, status, attempts, last_status, last_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.url,
                    record.event,
                    record.payload,
                    record.status,
                    record.attempts,
                    record.last_status,
                    record.last_error,
                    record.created_at,
                    record.updated_at,
                ),
            )
            record.id = int(cursor.lastrowid or 0)
        return record

    def update(
        self,
        record_id: int,
        *,
        status: str | None = None,
        attempts: int | None = None,
        last_status: int | None = None,
        last_error: str | None = None,
    ) -> None:
        """Patch fields of an existing record."""
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if attempts is not None:
            fields.append("attempts = ?")
            values.append(attempts)
        if last_status is not None:
            fields.append("last_status = ?")
            values.append(last_status)
        if last_error is not None:
            fields.append("last_error = ?")
            values.append(last_error)
        fields.append("updated_at = ?")
        values.append(utcnow_iso())
        values.append(record_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE deliveries SET {', '.join(fields)} WHERE id = ?",
                values,
            )

    def mark_attempt(
        self,
        record_id: int,
        *,
        attempts: int,
        status_code: int | None,
        error: str | None,
        delivered: bool,
    ) -> None:
        """Record one delivery attempt's outcome."""
        self.update(
            record_id,
            attempts=attempts,
            last_status=status_code,
            last_error=error,
            status="delivered" if delivered else "pending",
        )

    def mark_failed(self, record_id: int, *, attempts: int, error: str) -> None:
        """Mark a record as permanently failed after all retries."""
        self.update(record_id, attempts=attempts, last_error=error, status="failed")

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #

    def _row_to_record(self, row: sqlite3.Row) -> DeliveryRecord:
        return DeliveryRecord(
            id=row["id"],
            url=row["url"],
            event=row["event"],
            payload=row["payload"],
            status=row["status"],
            attempts=row["attempts"],
            last_status=row["last_status"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, record_id: int) -> DeliveryRecord | None:
        """Return one record by id, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM deliveries WHERE id = ?", (record_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def list(
        self,
        *,
        status: str | None = None,
        event: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DeliveryRecord]:
        """List records, newest first, with optional filters."""
        query = "SELECT * FROM deliveries"
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if event is not None:
            clauses.append("event = ?")
            values.append(event)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        values.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._row_to_record(row) for row in rows]

    def pending(self, limit: int = 100) -> list[DeliveryRecord]:
        """Return records still awaiting delivery (for a retry worker)."""
        return self.list(status="pending", limit=limit)

    def count_by_status(self) -> dict[str, int]:
        """Return ``{status: count}`` across all records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM deliveries GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}


@dataclass(slots=True)
class WebhookDispatcher:
    """Sends signed JSON webhooks with retries and a delivery log.

    Attributes:
        secret: Shared HMAC secret.
        client: An ``httpx.Client``-like object.
        log: The :class:`DeliveryLog`.
        retry: The :class:`RetryPolicy`.
        timeout: Per-request timeout in seconds.
        sleep: Injectable sleep function (tests pass a fake).
        rng: Injectable random generator for deterministic jitter.
    """

    secret: str
    client: HttpClientLike
    log: DeliveryLog = field(default_factory=DeliveryLog)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 10.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if not self.secret:
            raise WebhookError("webhook secret must be a non-empty string")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def send(
        self,
        url: str,
        event: str,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        extra_headers: Mapping[str, str] | None = None,
        record: bool = True,
    ) -> DeliveryRecord:
        """Send ``payload`` to ``url`` as event ``event``.

        The payload is serialised with ``json.dumps(..., separators=(",", ":"),
        sort_keys=True)`` so the signature is reproducible. When ``record``
        is True the delivery is written to the log first and updated as
        attempts happen.

        Returns:
            The final :class:`DeliveryRecord`.

        Raises:
            WebhookError: When every attempt fails.
        """
        if not url.startswith(("http://", "https://")):
            raise WebhookError(f"webhook url must be http(s), got {url!r}")
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        record_obj = DeliveryRecord(url=url, event=event, payload=body.decode("utf-8"))
        if record:
            record_obj = self.log.insert(record_obj)
        assert record_obj.id is not None

        last_error: str | None = None
        last_status: int | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            headers = default_headers(
                self.secret, body, event=event, delivery_id=str(record_obj.id)
            )
            if extra_headers:
                headers.update(extra_headers)
            try:
                response = self.client.post(
                    url, content=body, headers=headers, timeout=self.timeout
                )
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                last_error = f"{type(exc).__name__}: {exc}"
                last_status = None
                self.log.mark_attempt(
                    record_obj.id,
                    attempts=attempt,
                    status_code=None,
                    error=last_error,
                    delivered=False,
                )
            else:
                last_status = response.status_code
                if 200 <= response.status_code < 300:
                    self.log.mark_attempt(
                        record_obj.id,
                        attempts=attempt,
                        status_code=response.status_code,
                        error=None,
                        delivered=True,
                    )
                    record_obj.status = "delivered"
                    record_obj.attempts = attempt
                    record_obj.last_status = response.status_code
                    return record_obj
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                self.log.mark_attempt(
                    record_obj.id,
                    attempts=attempt,
                    status_code=response.status_code,
                    error=last_error,
                    delivered=False,
                )
                if response.status_code not in self.retry.retry_on_status:
                    # Non-retryable status: fail immediately.
                    break
            if attempt < self.retry.max_attempts:
                self.sleep(self.retry.delay_for(attempt, self.rng))
        self.log.mark_failed(record_obj.id, attempts=self.retry.max_attempts, error=last_error or "unknown")
        record_obj.status = "failed"
        record_obj.attempts = self.retry.max_attempts
        record_obj.last_status = last_status
        record_obj.last_error = last_error
        raise WebhookError(
            f"webhook delivery to {url} failed after {self.retry.max_attempts} attempt(s): {last_error}"
        )

    def send_result(self, url: str, result: Any, *, event: str = "verdict.completed") -> DeliveryRecord:
        """Convenience: send a :class:`~llmdiff.core.result.DiffResult` as JSON."""
        payload = result.to_dict() if hasattr(result, "to_dict") else {"result": result}
        return self.send(url, event, payload)

    def retry_pending(self, *, limit: int = 100) -> list[DeliveryRecord]:
        """Re-attempt every pending delivery in the log.

        Returns:
            The records that are now delivered.

        Raises:
            WebhookError: Never; failures are recorded and skipped.
        """
        delivered: list[DeliveryRecord] = []
        for record in self.log.pending(limit=limit):
            try:
                body = record.payload.encode("utf-8")
                headers = default_headers(
                    self.secret, body, event=record.event, delivery_id=str(record.id)
                )
                response = self.client.post(
                    record.url, content=body, headers=headers, timeout=self.timeout
                )
                ok = 200 <= response.status_code < 300
                self.log.mark_attempt(
                    record.id or 0,
                    attempts=record.attempts + 1,
                    status_code=response.status_code,
                    error=None if ok else f"HTTP {response.status_code}",
                    delivered=ok,
                )
                if ok:
                    record.status = "delivered"
                    delivered.append(record)
            except Exception as exc:  # noqa: BLE001
                self.log.mark_attempt(
                    record.id or 0,
                    attempts=record.attempts + 1,
                    status_code=None,
                    error=f"{type(exc).__name__}: {exc}",
                    delivered=False,
                )
        return delivered
