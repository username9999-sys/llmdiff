"""PagerDuty integration: create/manage incidents from verdict regressions.

When a CI gate fails hard (regression rate above a critical threshold),
llmdiff can page the on-call. This module implements:

- :class:`PagerDutyClient` — Events API v2 (``POST /v2/enqueue``) for
  triggering/resolving incidents, plus the REST API v2 for incident lookup,
  with an injectable transport,
- :func:`build_trigger_event` / :func:`build_resolve_event` — payload
  builders with dedup-key support,
- :class:`PagerDutyAlerter` — maps a DiffResult to an incident: severity
  derived from the regression rate, dedup key from the corpus/config hash.

Authentication: Events API uses a 32-char routing key; REST API uses a
``Token token=...`` bearer. Both are supported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "PagerDutyError",
    "HttpResponseLike",
    "HttpClientLike",
    "PagerDutyIncident",
    "PagerDutyClient",
    "PagerDutyAlerter",
    "build_trigger_event",
    "build_resolve_event",
    "severity_for_result",
    "dedup_key_for_result",
]


class PagerDutyError(LLMDiffError):
    """Raised for PagerDuty configuration, parsing or API errors."""


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


SEVERITIES = ("critical", "error", "warning", "info")


def severity_for_result(result: Any, *, critical_rate: float = 0.5, error_rate: float = 0.1) -> str:
    """Map a DiffResult's regression rate to a PagerDuty severity.

    - rate >= ``critical_rate`` -> ``critical``
    - rate >= ``error_rate``    -> ``error``
    - any regressed dimension    -> ``warning``
    - otherwise                  -> ``info``
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        return "info"
    rate = float(getattr(matrix, "regression_rate", 0.0))
    if rate >= critical_rate:
        return "critical"
    if rate >= error_rate:
        return "error"
    if getattr(matrix, "regressed_dimensions", ()):
        return "warning"
    return "info"


def dedup_key_for_result(result: Any, *, prefix: str = "llmdiff") -> str:
    """Stable dedup key for a result: hash of the regressed dimension names.

    Two runs with the same regressed dimensions produce the same key, so
    PagerDuty groups them into one incident instead of spamming.
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    regressed = sorted(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
    digest = hashlib.sha256(",".join(regressed).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def build_trigger_event(
    *,
    routing_key: str,
    summary: str,
    severity: str = "error",
    source: str = "llmdiff",
    dedup_key: str = "",
    custom_details: Mapping[str, Any] | None = None,
    component: str = "ci",
    group: str = "llmdiff",
) -> dict[str, Any]:
    """Build an Events API v2 ``trigger`` payload.

    Raises:
        PagerDutyError: On invalid severity or empty routing key/summary.
    """
    if not routing_key or len(routing_key) != 32:
        raise PagerDutyError("routing_key must be the 32-char Events API key")
    if not summary.strip():
        raise PagerDutyError("summary must be non-empty")
    if severity not in SEVERITIES:
        raise PagerDutyError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    payload: dict[str, Any] = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary[:1024],
            "severity": severity,
            "source": source,
            "component": component,
            "group": group,
        },
    }
    if dedup_key:
        payload["dedup_key"] = dedup_key
    if custom_details:
        payload["payload"]["custom_details"] = dict(custom_details)
    return payload


def build_resolve_event(*, routing_key: str, dedup_key: str) -> dict[str, Any]:
    """Build an Events API v2 ``resolve`` payload for ``dedup_key``."""
    if not routing_key or len(routing_key) != 32:
        raise PagerDutyError("routing_key must be the 32-char Events API key")
    if not dedup_key:
        raise PagerDutyError("dedup_key is required to resolve")
    return {
        "routing_key": routing_key,
        "event_action": "resolve",
        "dedup_key": dedup_key,
    }


@dataclass(frozen=True, slots=True)
class PagerDutyIncident:
    """A PagerDuty REST incident (subset of fields).

    Attributes:
        id: Incident id.
        incident_number: Human number.
        title / status: Title and status (``triggered``, ``acknowledged``, ...).
        severity: Severity name.
        urgency: ``high``/``low``.
        html_url: Link to the incident.
    """

    id: str
    incident_number: int = 0
    title: str = ""
    status: str = ""
    severity: str = ""
    urgency: str = ""
    html_url: str = ""

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> PagerDutyIncident:
        """Parse one incident object from the REST API."""
        severity = data.get("severity") or {}
        return cls(
            id=str(data.get("id", "")),
            incident_number=int(data.get("incident_number", 0) or 0),
            title=str(data.get("title", "")),
            status=str(data.get("status", "")),
            severity=str(severity.get("name", "")) if isinstance(severity, Mapping) else "",
            urgency=str(data.get("urgency", "")),
            html_url=str(data.get("html_url", "")),
        )


@dataclass(slots=True)
class PagerDutyClient:
    """PagerDuty client for Events API v2 and REST API v2.

    Attributes:
        routing_key: 32-char Events API v2 integration key.
        api_token: Optional REST API token (``Token token=...`` auth).
        client: Injectable transport.
        api_base: REST base URL.
        events_base: Events API base URL.
        timeout: Per-request timeout.
    """

    routing_key: str
    client: HttpClientLike
    api_token: str = ""
    api_base: str = "https://api.pagerduty.com"
    events_base: str = "https://events.pagerduty.com"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.routing_key or len(self.routing_key) != 32:
            raise PagerDutyError("routing_key must be the 32-char Events API key")

    def _events_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _rest_headers(self) -> dict[str, str]:
        if not self.api_token:
            raise PagerDutyError("api_token is required for REST API calls")
        return {
            "Authorization": f"Token token={self.api_token}",
            "Accept": "application/vnd.pagerduty+json;version=2",
            "Content-Type": "application/json",
        }

    def _check(self, response: HttpResponseLike, *, what: str) -> Any:
        if 200 <= response.status_code < 300:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise PagerDutyError(
            f"PagerDuty error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # Events API v2
    # ------------------------------------------------------------------ #

    def send_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """POST an Events API v2 payload to ``/v2/enqueue``."""
        response = self.client.post(
            f"{self.events_base}/v2/enqueue",
            headers=self._events_headers(),
            json=dict(event),
            timeout=self.timeout,
        )
        result = self._check(response, what="enqueue event")
        return result if isinstance(result, dict) else {}

    def trigger(
        self,
        summary: str,
        *,
        severity: str = "error",
        dedup_key: str = "",
        custom_details: Mapping[str, Any] | None = None,
        source: str = "llmdiff",
    ) -> dict[str, Any]:
        """Trigger an incident."""
        event = build_trigger_event(
            routing_key=self.routing_key,
            summary=summary,
            severity=severity,
            dedup_key=dedup_key,
            custom_details=custom_details,
            source=source,
        )
        return self.send_event(event)

    def resolve(self, dedup_key: str) -> dict[str, Any]:
        """Resolve the incident identified by ``dedup_key``."""
        return self.send_event(build_resolve_event(routing_key=self.routing_key, dedup_key=dedup_key))

    # ------------------------------------------------------------------ #
    # REST API v2
    # ------------------------------------------------------------------ #

    def list_incidents(
        self,
        *,
        statuses: tuple[str, ...] = ("triggered", "acknowledged"),
        limit: int = 25,
    ) -> list[PagerDutyIncident]:
        """List open incidents."""
        params: dict[str, str] = {"limit": str(limit)}
        for status in statuses:
            params.setdefault("statuses[]", "")
            params[f"statuses[]"] = status  # httpx repeats the key; our mock keeps last
        # Build repeated params properly for real httpx:
        response = self.client.get(
            f"{self.api_base}/incidents",
            headers=self._rest_headers(),
            params={"limit": str(limit), "statuses[]": ",".join(statuses)},
            timeout=self.timeout,
        )
        data = self._check(response, what="list incidents")
        incidents = data.get("incidents", []) if isinstance(data, Mapping) else []
        return [PagerDutyIncident.from_api(item) for item in incidents if isinstance(item, Mapping)]

    def get_incident(self, incident_id: str) -> PagerDutyIncident:
        """Fetch one incident by id."""
        if not incident_id:
            raise PagerDutyError("incident_id is required")
        response = self.client.get(
            f"{self.api_base}/incidents/{incident_id}",
            headers=self._rest_headers(),
            timeout=self.timeout,
        )
        data = self._check(response, what=f"get incident {incident_id}")
        incident = data.get("incident", {}) if isinstance(data, Mapping) else {}
        return PagerDutyIncident.from_api(incident if isinstance(incident, Mapping) else {})


@dataclass(slots=True)
class PagerDutyAlerter:
    """Maps verdicts to PagerDuty incidents.

    Attributes:
        client: The :class:`PagerDutyClient`.
        critical_rate: Regression rate that maps to ``critical`` severity.
        page_on_warning: Also trigger (as ``warning``) when any dimension
            regressed even below ``error_rate``.
    """

    client: PagerDutyClient
    critical_rate: float = 0.5
    error_rate: float = 0.1
    page_on_warning: bool = True

    def alert(self, result: Any, *, context: str = "") -> dict[str, Any] | None:
        """Trigger an incident when the verdict warrants paging.

        Returns:
            The Events API response, or None when no page was needed.
        """
        severity = severity_for_result(
            result, critical_rate=self.critical_rate, error_rate=self.error_rate
        )
        if severity == "info":
            return None
        if severity == "warning" and not self.page_on_warning:
            return None
        matrix = getattr(result, "matrix", None)
        verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        summary = (
            f"llmdiff regression: verdict={verdict} loss_rate={rate:.1%} "
            f"regressed={','.join(regressed) or '-'}"
            + (f" ({context})" if context else "")
        )
        return self.client.trigger(
            summary,
            severity=severity,
            dedup_key=dedup_key_for_result(result),
            custom_details={
                "verdict": verdict,
                "loss_rate": f"{rate:.4f}",
                "regressed_dimensions": regressed,
                "context": context,
            },
        )

    def resolve(self, result: Any) -> dict[str, Any]:
        """Resolve the incident for ``result``'s dedup key."""
        return self.client.resolve(dedup_key_for_result(result))
