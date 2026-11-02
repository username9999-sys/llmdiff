"""Opsgenie integration: create/close alerts from verdict regressions.

Mirrors :mod:`llmdiff.integrations.pagerduty` for Opsgenie:

- :class:`OpsgenieClient` — Alerts API v2 (``POST /v2/alerts``) with
  ``Authorization: GenieKey <key>`` auth, plus alert listing/closing,
- :func:`build_alert_payload` — payload builder with alias-based dedup,
- :class:`OpsgenieAlerter` — maps a DiffResult to an alert: priority derived
  from the regression rate (P1..P5), alias from the regressed dimensions.

The transport is injectable (``httpx.Client``-like) so tests run offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "OpsgenieError",
    "HttpResponseLike",
    "HttpClientLike",
    "OpsgenieAlert",
    "OpsgenieClient",
    "OpsgenieAlerter",
    "build_alert_payload",
    "build_close_payload",
    "priority_for_result",
    "alias_for_result",
]


class OpsgenieError(LLMDiffError):
    """Raised for Opsgenie configuration, parsing or API errors."""


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


PRIORITIES = ("P1", "P2", "P3", "P4", "P5")


def priority_for_result(result: Any, *, p1_rate: float = 0.5, p2_rate: float = 0.2, p3_rate: float = 0.05) -> str:
    """Map a DiffResult's regression rate to an Opsgenie priority.

    - rate >= ``p1_rate`` -> P1
    - rate >= ``p2_rate`` -> P2
    - rate >= ``p3_rate`` -> P3
    - any regressed dimension -> P4
    - otherwise -> P5
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        return "P5"
    rate = float(getattr(matrix, "regression_rate", 0.0))
    if rate >= p1_rate:
        return "P1"
    if rate >= p2_rate:
        return "P2"
    if rate >= p3_rate:
        return "P3"
    if getattr(matrix, "regressed_dimensions", ()):
        return "P4"
    return "P5"


def alias_for_result(result: Any, *, prefix: str = "llmdiff") -> str:
    """Stable alert alias: hash of the regressed dimension names.

    Opsgenie dedups on ``alias``, so identical regression sets update the
    same alert instead of creating duplicates.
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    regressed = sorted(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
    digest = hashlib.sha256(",".join(regressed).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def build_alert_payload(
    *,
    message: str,
    alias: str,
    priority: str = "P3",
    description: str = "",
    source: str = "llmdiff",
    tags: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
    responders: Sequence[Mapping[str, Any]] = (),
    entity: str = "llmdiff-ci",
) -> dict[str, Any]:
    """Build an Opsgenie ``POST /v2/alerts`` payload.

    Raises:
        OpsgenieError: On invalid priority or empty message/alias.
    """
    if not message.strip():
        raise OpsgenieError("message must be non-empty")
    if not alias.strip():
        raise OpsgenieError("alias must be non-empty")
    if priority not in PRIORITIES:
        raise OpsgenieError(f"priority must be one of {PRIORITIES}, got {priority!r}")
    payload: dict[str, Any] = {
        "message": message[:130],
        "alias": alias[:512],
        "priority": priority,
        "source": source,
        "entity": entity,
    }
    if description:
        payload["description"] = description[:15000]
    if tags:
        payload["tags"] = [str(t) for t in tags]
    if details:
        payload["details"] = dict(details)
    if responders:
        payload["responders"] = [dict(r) for r in responders]
    return payload


def build_close_payload(*, alias: str, note: str = "", source: str = "llmdiff") -> dict[str, Any]:
    """Build the body for ``POST /v2/alerts/{alias}/close``."""
    if not alias.strip():
        raise OpsgenieError("alias is required to close an alert")
    payload: dict[str, Any] = {"source": source}
    if note:
        payload["note"] = note[:1000]
    return payload


@dataclass(frozen=True, slots=True)
class OpsgenieAlert:
    """A parsed Opsgenie alert (subset of fields)."""

    id: str
    alias: str = ""
    message: str = ""
    status: str = ""
    priority: str = ""
    created_at: int = 0

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> OpsgenieAlert:
        return cls(
            id=str(data.get("id", "")),
            alias=str(data.get("alias", "")),
            message=str(data.get("message", "")),
            status=str(data.get("status", "")),
            priority=str(data.get("priority", "")),
            created_at=int(data.get("createdAt", 0) or 0),
        )


@dataclass(slots=True)
class OpsgenieClient:
    """Opsgenie Alerts API v2 client.

    Attributes:
        api_key: A GenieKey with alert create/close rights.
        client: Injectable transport.
        api_base: API base URL (``https://api.opsgenie.com`` or EU
            ``https://api.eu.opsgenie.com``).
        timeout: Per-request timeout.
    """

    api_key: str
    client: HttpClientLike
    api_base: str = "https://api.opsgenie.com"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise OpsgenieError("api_key (GenieKey) must be non-empty")
        if not self.api_base.startswith(("http://", "https://")):
            raise OpsgenieError(f"api_base must be http(s), got {self.api_base!r}")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"GenieKey {self.api_key}",
            "Content-Type": "application/json",
        }

    def _check(self, response: HttpResponseLike, *, what: str, allow: tuple[int, ...] = (200, 201, 202)) -> Any:
        if response.status_code in allow:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise OpsgenieError(
            f"Opsgenie error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # alerts
    # ------------------------------------------------------------------ #

    def create_alert(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST /v2/alerts — create (or dedup-update) an alert."""
        response = self.client.post(
            f"{self.api_base}/v2/alerts",
            headers=self._headers(),
            json=dict(payload),
            timeout=self.timeout,
        )
        result = self._check(response, what="create alert", allow=(200, 201, 202))
        return result if isinstance(result, dict) else {}

    def alert(
        self,
        message: str,
        *,
        alias: str,
        priority: str = "P3",
        description: str = "",
        tags: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper around :func:`build_alert_payload` + create."""
        return self.create_alert(
            build_alert_payload(
                message=message,
                alias=alias,
                priority=priority,
                description=description,
                tags=tags,
                details=details,
            )
        )

    def close_alert(self, alias: str, *, note: str = "") -> dict[str, Any]:
        """POST /v2/alerts/{alias}/close."""
        from urllib.parse import quote

        response = self.client.post(
            f"{self.api_base}/v2/alerts/{quote(alias, safe='')}/close",
            headers=self._headers(),
            json=build_close_payload(alias=alias, note=note),
            timeout=self.timeout,
        )
        result = self._check(response, what=f"close alert {alias}", allow=(200, 202))
        return result if isinstance(result, dict) else {}

    def list_alerts(self, *, query: str = "status:open", limit: int = 20) -> list[OpsgenieAlert]:
        """GET /v2/alerts — list alerts matching ``query``."""
        response = self.client.get(
            f"{self.api_base}/v2/alerts",
            headers=self._headers(),
            params={"query": query, "limit": str(limit)},
            timeout=self.timeout,
        )
        data = self._check(response, what="list alerts")
        items = data.get("data", []) if isinstance(data, Mapping) else []
        return [OpsgenieAlert.from_api(item) for item in items if isinstance(item, Mapping)]

    def get_alert(self, alias: str) -> OpsgenieAlert:
        """GET /v2/alerts/{alias}."""
        from urllib.parse import quote

        response = self.client.get(
            f"{self.api_base}/v2/alerts/{quote(alias, safe='')}",
            headers=self._headers(),
            params={"identifierType": "alias"},
            timeout=self.timeout,
        )
        data = self._check(response, what=f"get alert {alias}")
        alert = data.get("data", {}) if isinstance(data, Mapping) else {}
        return OpsgenieAlert.from_api(alert if isinstance(alert, Mapping) else {})


@dataclass(slots=True)
class OpsgenieAlerter:
    """Maps verdicts to Opsgenie alerts.

    Attributes:
        client: The :class:`OpsgenieClient`.
        p1_rate / p2_rate / p3_rate: Regression-rate thresholds for P1/P2/P3.
        alert_on_p4: Also create (P4) alerts when any dimension regressed.
        tags: Extra tags attached to every alert.
    """

    client: OpsgenieClient
    p1_rate: float = 0.5
    p2_rate: float = 0.2
    p3_rate: float = 0.05
    alert_on_p4: bool = True
    tags: tuple[str, ...] = ("llmdiff", "ci")

    def alert(self, result: Any, *, context: str = "") -> dict[str, Any] | None:
        """Create an alert when the verdict warrants it.

        Returns:
            The API response, or None when the verdict is clean (P5).
        """
        priority = priority_for_result(
            result, p1_rate=self.p1_rate, p2_rate=self.p2_rate, p3_rate=self.p3_rate
        )
        if priority == "P5":
            return None
        if priority == "P4" and not self.alert_on_p4:
            return None
        matrix = getattr(result, "matrix", None)
        verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        message = (
            f"llmdiff regression ({priority}): verdict={verdict} "
            f"loss_rate={rate:.1%} regressed={','.join(regressed) or '-'}"
        )
        return self.client.alert(
            message,
            alias=alias_for_result(result),
            priority=priority,
            description=(f"context: {context}" if context else "llmdiff CI regression"),
            tags=[*self.tags, *regressed],
            details={
                "verdict": verdict,
                "loss_rate": f"{rate:.4f}",
                "regressed_dimensions": regressed,
                "context": context,
            },
        )

    def close(self, result: Any, *, note: str = "regressions resolved") -> dict[str, Any]:
        """Close the alert for ``result``'s alias."""
        return self.client.close_alert(alias_for_result(result), note=note)
