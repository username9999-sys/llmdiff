"""Datadog integration: push verdict metrics to the Datadog API.

llmdiff can emit one metric per dimension (win/tie/loss counts, mean scores,
delta) plus an overall regression-rate gauge. This module implements:

- :func:`build_metrics_payload` — build a ``POST /api/v1/series`` (v1) or
  ``/api/v2/series`` (v2) payload from a DiffResult,
- :class:`DatadogClient` — submit metrics (v1 and v2), query events, with an
  injectable transport,
- :class:`DatadogMetricsReporter` — turns a DiffResult into a list of metric
  dicts ready to submit (tags, timestamps, points).

Auth: ``DD-API-KEY`` header (and optional ``DD-APPLICATION-KEY``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "DatadogError",
    "HttpResponseLike",
    "HttpClientLike",
    "MetricPoint",
    "DatadogMetric",
    "DatadogClient",
    "DatadogMetricsReporter",
    "build_metrics_payload",
    "metrics_from_result",
]


class DatadogError(LLMDiffError):
    """Raised for Datadog configuration, parsing or API errors."""


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
class MetricPoint:
    """One (timestamp, value) point of a time series."""

    timestamp: int
    value: float

    def as_list(self) -> list[Any]:
        """Datadog v1 series point format: ``[timestamp, value]``."""
        return [self.timestamp, self.value]


@dataclass(frozen=True, slots=True)
class DatadogMetric:
    """A single Datadog metric.

    Attributes:
        metric: Metric name, e.g. ``llmdiff.dimension.wins``.
        points: The data points.
        tags: Tag list (``key:value`` strings).
        type: ``gauge``, ``count`` or ``rate``.
        host: Optional host tag.
        interval: Optional interval for rates.
    """

    metric: str
    points: tuple[MetricPoint, ...]
    tags: tuple[str, ...] = ()
    type: str = "gauge"
    host: str = ""
    interval: int | None = None

    def to_v1_series(self) -> dict[str, Any]:
        """Serialise for the v1 ``/api/v1/series`` API."""
        series: dict[str, Any] = {
            "metric": self.metric,
            "points": [p.as_list() for p in self.points],
            "type": self.type,
            "tags": list(self.tags),
        }
        if self.host:
            series["host"] = self.host
        if self.interval:
            series["interval"] = self.interval
        return series

    def to_v2_payload(self) -> dict[str, Any]:
        """Serialise for the v2 ``/api/v2/series`` API (one resource)."""
        resource: dict[str, Any] = {
            "type": "timeseries",
            "attributes": {
                "metric": self.metric,
                "points": [
                    {"timestamp": p.timestamp, "value": p.value} for p in self.points
                ],
                "tags": list(self.tags),
                "type": {"gauge": "gauge", "count": "count", "rate": "rate"}.get(self.type, "gauge"),
            },
        }
        if self.host:
            resource["attributes"]["host"] = self.host
        return {"data": resource}


def metrics_from_result(
    result: Any,
    *,
    namespace: str = "llmdiff",
    tags: Sequence[str] = (),
    timestamp: int | None = None,
) -> list[DatadogMetric]:
    """Turn a DiffResult-like object into a list of :class:`DatadogMetric`.

    Emits, per dimension ``d``:

    - ``<ns>.dimension.<d>.wins`` (count)
    - ``<ns>.dimension.<d>.ties`` (count)
    - ``<ns>.dimension.<d>.losses`` (count)
    - ``<ns>.dimension.<d>.baseline_mean`` (gauge)
    - ``<ns>.dimension.<d>.candidate_mean`` (gauge)
    - ``<ns>.dimension.<d>.delta`` (gauge)

    plus overall ``<ns>.wins`` / ``ties`` / ``losses`` / ``regression_rate``
    and ``<ns>.total_cases``.
    """
    ts = int(timestamp if timestamp is not None else time.time())
    base_tags = tuple(f"llmdiff:{t}" if ":" not in t else t for t in tags)
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        raise DatadogError("result has no .matrix; pass a DiffResult or {'matrix': ...}")
    metrics: list[DatadogMetric] = []

    def add(name: str, value: float, *, type: str = "gauge", extra_tags: Sequence[str] = ()) -> None:
        metrics.append(
            DatadogMetric(
                metric=f"{namespace}.{name}",
                points=(MetricPoint(timestamp=ts, value=float(value)),),
                tags=base_tags + tuple(extra_tags),
                type=type,
            )
        )

    for summary in getattr(matrix, "dimensions", ()) or []:
        dim = str(summary.dimension).replace(" ", "_")
        dim_tag = f"dimension:{summary.dimension}"
        add(f"dimension.{dim}.wins", summary.wins, type="count", extra_tags=(dim_tag,))
        add(f"dimension.{dim}.ties", summary.ties, type="count", extra_tags=(dim_tag,))
        add(f"dimension.{dim}.losses", summary.losses, type="count", extra_tags=(dim_tag,))
        add(f"dimension.{dim}.baseline_mean", summary.baseline_mean, extra_tags=(dim_tag,))
        add(f"dimension.{dim}.candidate_mean", summary.candidate_mean, extra_tags=(dim_tag,))
        add(f"dimension.{dim}.delta", summary.delta, extra_tags=(dim_tag,))
    add("wins", getattr(matrix, "wins", 0), type="count")
    add("ties", getattr(matrix, "ties", 0), type="count")
    add("losses", getattr(matrix, "losses", 0), type="count")
    add("regression_rate", getattr(matrix, "regression_rate", 0.0))
    add("total_cases", getattr(matrix, "total_cases", 0), type="count")
    verdict = str(getattr(matrix, "verdict", "unknown"))
    add("verdict_code", {"win": 1.0, "tie": 0.0, "loss": -1.0}.get(verdict, 0.0), extra_tags=(f"verdict:{verdict}",))
    return metrics


def build_metrics_payload(
    metrics: Sequence[DatadogMetric],
    *,
    api_version: int = 1,
) -> dict[str, Any]:
    """Build the request body for the series API.

    Args:
        metrics: The metrics to submit.
        api_version: 1 for ``/api/v1/series`` (``{"series": [...]}``),
            2 for ``/api/v2/series`` (``{"data": [...]}``).
    """
    if api_version == 1:
        return {"series": [m.to_v1_series() for m in metrics]}
    if api_version == 2:
        return {"data": [m.to_v2_payload()["data"] for m in metrics]}
    raise DatadogError(f"unsupported api_version {api_version!r}; expected 1 or 2")


@dataclass(slots=True)
class DatadogClient:
    """Datadog API client for metrics and events.

    Attributes:
        api_key: The Datadog API key.
        app_key: Optional application key (needed for some endpoints).
        client: Injectable transport.
        site: Datadog site (``datadoghq.com``, ``datadoghq.eu``, ...).
        timeout: Per-request timeout.
    """

    api_key: str
    client: HttpClientLike
    app_key: str = ""
    site: str = "datadoghq.com"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise DatadogError("api_key must be non-empty")

    @property
    def api_base(self) -> str:
        return f"https://api.{self.site}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "DD-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        if self.app_key:
            headers["DD-APPLICATION-KEY"] = self.app_key
        return headers

    def _check(self, response: HttpResponseLike, *, what: str, allow: tuple[int, ...] = (200, 201, 202)) -> Any:
        if response.status_code in allow:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise DatadogError(
            f"Datadog error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #

    def submit_series(self, metrics: Sequence[DatadogMetric], *, api_version: int = 1) -> dict[str, Any]:
        """Submit metrics via ``/api/v1/series`` or ``/api/v2/series``."""
        if not metrics:
            raise DatadogError("no metrics to submit")
        path = "/api/v1/series" if api_version == 1 else "/api/v2/series"
        response = self.client.post(
            f"{self.api_base}{path}",
            headers=self._headers(),
            json=build_metrics_payload(metrics, api_version=api_version),
            timeout=self.timeout,
        )
        result = self._check(response, what="submit series", allow=(200, 202))
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def post_event(
        self,
        *,
        title: str,
        text: str,
        tags: Sequence[str] = (),
        alert_type: str = "info",
        source_type_name: str = "llmdiff",
        aggregation_key: str = "",
    ) -> dict[str, Any]:
        """POST /api/v1/events — create a Datadog event."""
        if alert_type not in ("error", "warning", "info", "success"):
            raise DatadogError(f"alert_type must be error|warning|info|success, got {alert_type!r}")
        if not title.strip():
            raise DatadogError("event title must be non-empty")
        payload: dict[str, Any] = {
            "title": title[:100],
            "text": text,
            "alert_type": alert_type,
            "source_type_name": source_type_name,
        }
        if tags:
            payload["tags"] = [str(t) for t in tags]
        if aggregation_key:
            payload["aggregation_key"] = aggregation_key
        response = self.client.post(
            f"{self.api_base}/api/v1/events",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        result = self._check(response, what="post event", allow=(200, 201, 202))
        return result if isinstance(result, dict) else {}

    def query_events(
        self,
        *,
        start: int,
        end: int,
        query: str = "source:llmdiff",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """GET /api/v1/events — list events in a time window."""
        if end <= start:
            raise DatadogError(f"end ({end}) must be > start ({start})")
        response = self.client.get(
            f"{self.api_base}/api/v1/events",
            headers=self._headers(),
            params={
                "start": str(start),
                "end": str(end),
                "query": query,
                "limit": str(limit),
            },
            timeout=self.timeout,
        )
        data = self._check(response, what="query events")
        events = data.get("events", []) if isinstance(data, Mapping) else []
        return [e for e in events if isinstance(e, dict)]


@dataclass(slots=True)
class DatadogMetricsReporter:
    """High-level reporter: DiffResult -> Datadog submission.

    Attributes:
        client: The :class:`DatadogClient`.
        namespace: Metric namespace prefix.
        base_tags: Tags attached to every metric.
        api_version: 1 or 2.
    """

    client: DatadogClient
    namespace: str = "llmdiff"
    base_tags: tuple[str, ...] = ("source:llmdiff",)
    api_version: int = 1

    def report(self, result: Any, *, extra_tags: Sequence[str] = (), timestamp: int | None = None) -> dict[str, Any]:
        """Build metrics from ``result`` and submit them.

        Also posts a Datadog event when any dimension regressed.
        """
        metrics = metrics_from_result(
            result,
            namespace=self.namespace,
            tags=[*self.base_tags, *extra_tags],
            timestamp=timestamp,
        )
        response = self.client.submit_series(metrics, api_version=self.api_version)
        matrix = getattr(result, "matrix", None)
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        if regressed:
            self.client.post_event(
                title=f"llmdiff regression: {', '.join(regressed)}",
                text=f"Verdict matrix regressed on: {', '.join(regressed)}",
                tags=[*self.base_tags, *extra_tags, *(f"dimension:{d}" for d in regressed)],
                alert_type="error",
                aggregation_key="llmdiff-regression",
            )
        return response
