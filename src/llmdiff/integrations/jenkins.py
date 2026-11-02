"""Jenkins integration: trigger builds, poll results, post build comments.

llmdiff can gate Jenkins pipelines on the verdict matrix. This module
implements:

- :class:`JenkinsClient` — REST wrapper for triggering parameterised builds
  (``POST /job/{name}/buildWithParameters``), polling build status
  (``GET /job/{name}/{number}/api/json``), fetching the console log, and
  adding a build description (``POST .../submitDescription``),
- :func:`build_trigger_payload` / :func:`parse_build_info` — payload helpers,
- :class:`JenkinsGate` — maps a :class:`DiffResult`-like matrix to a Jenkins
  build parameter set (``LLMDIFF_VERDICT``, ``LLMDIFF_LOSS_RATE``, ...), so a
  pipeline can fail itself when regressions exceed a budget,
- :func:`wait_for_build` — blocking poll helper with timeout.

Authentication is HTTP Basic (user + API token). The transport is injectable
(``httpx.Client``-like) so tests run offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "JenkinsError",
    "HttpResponseLike",
    "HttpClientLike",
    "BuildInfo",
    "JenkinsClient",
    "JenkinsGate",
    "build_trigger_payload",
    "parse_build_info",
    "verdict_to_params",
    "wait_for_build",
]


class JenkinsError(LLMDiffError):
    """Raised for Jenkins configuration, parsing or API errors."""


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
        data: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float = 10.0,
    ) -> HttpResponseLike: ...


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """Parsed Jenkins build metadata.

    Attributes:
        number: Build number.
        url: Absolute build URL.
        building: True while the build is running.
        result: ``SUCCESS``/``FAILURE``/... or None while building.
        duration_ms: Duration in milliseconds (0 while building).
        display_name: Human-readable name.
        description: Build description text.
    """

    number: int
    url: str = ""
    building: bool = True
    result: str | None = None
    duration_ms: int = 0
    display_name: str = ""
    description: str = ""

    @property
    def finished(self) -> bool:
        """True when the build is no longer running."""
        return not self.building

    @property
    def successful(self) -> bool:
        """True when the build finished with result SUCCESS."""
        return self.result == "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "url": self.url,
            "building": self.building,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "display_name": self.display_name,
            "description": self.description,
        }


def parse_build_info(data: Mapping[str, Any]) -> BuildInfo:
    """Parse the JSON returned by ``/job/.../api/json`` into :class:`BuildInfo`."""
    if not isinstance(data, Mapping):
        raise JenkinsError(f"build info must be a mapping, got {type(data).__name__}")
    number = data.get("number")
    if not isinstance(number, int):
        raise JenkinsError("build info missing integer 'number'")
    result = data.get("result")
    return BuildInfo(
        number=number,
        url=str(data.get("url", "")),
        building=bool(data.get("building", True)),
        result=str(result) if result is not None else None,
        duration_ms=int(data.get("duration", 0) or 0),
        display_name=str(data.get("displayName", "")),
        description=str(data.get("description", "") or ""),
    )


def build_trigger_payload(parameters: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Build the form-encoded body for ``buildWithParameters``.

    Jenkins expects ``name=value`` pairs; values are stringified.
    """
    payload: dict[str, str] = {}
    for key, value in (parameters or {}).items():
        if isinstance(value, bool):
            payload[key] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            payload[key] = repr(value) if isinstance(value, float) else str(value)
        else:
            payload[key] = str(value)
    return payload


def verdict_to_params(result: Any, *, prefix: str = "LLMDIFF") -> dict[str, str]:
    """Map a DiffResult-like object to Jenkins build parameters.

    Produces ``LLMDIFF_VERDICT``, ``LLMDIFF_WINS``, ``LLMDIFF_TIES``,
    ``LLMDIFF_LOSSES``, ``LLMDIFF_REGRESSION_RATE`` and
    ``LLMDIFF_REGRESSED_DIMENSIONS`` so the pipeline can gate on them.
    """
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        raise JenkinsError("result has no .matrix; pass a DiffResult or {'matrix': ...}")
    wins = int(getattr(matrix, "wins", 0))
    ties = int(getattr(matrix, "ties", 0))
    losses = int(getattr(matrix, "losses", 0))
    rate = float(getattr(matrix, "regression_rate", 0.0))
    regressed = getattr(matrix, "regressed_dimensions", ()) or ()
    return {
        f"{prefix}_VERDICT": str(getattr(matrix, "verdict", "unknown")),
        f"{prefix}_WINS": str(wins),
        f"{prefix}_TIES": str(ties),
        f"{prefix}_LOSSES": str(losses),
        f"{prefix}_REGRESSION_RATE": f"{rate:.6f}",
        f"{prefix}_REGRESSED_DIMENSIONS": ",".join(regressed),
        f"{prefix}_GATE": "FAIL" if regressed else "PASS",
    }


@dataclass(slots=True)
class JenkinsClient:
    """Minimal Jenkins REST client.

    Attributes:
        base_url: Jenkins root URL, e.g. ``https://jenkins.example.com``.
        user: Username for basic auth.
        api_token: API token for basic auth.
        client: Injectable transport (``httpx.Client``-like).
        timeout: Per-request timeout.
    """

    base_url: str
    user: str
    api_token: str
    client: HttpClientLike
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise JenkinsError(f"base_url must be http(s), got {self.base_url!r}")
        if not self.user or not self.api_token:
            raise JenkinsError("user and api_token are required for Jenkins basic auth")
        self.base_url = self.base_url.rstrip("/")

    def _headers(self, *, form: bool = False) -> dict[str, str]:
        import base64

        credentials = base64.b64encode(f"{self.user}:{self.api_token}".encode()).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "User-Agent": "llmdiff/0.1",
        }
        if form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            headers["Accept"] = "application/json"
        return headers

    def _job_path(self, job: str, *extra: str) -> str:
        # Support nested folders via "folder/job" -> "job/folder/job".
        segments = [quote(part, safe="") for part in job.split("/") if part]
        for extra_part in extra:
            segments.append(quote(extra_part, safe=""))
        return self._build_job_url(segments)

    def _build_job_url(self, segments: list[str]) -> str:
        path = "/job/".join(segments)
        return f"{self.base_url}/{path}"

    def _check(self, response: HttpResponseLike, *, what: str, allow: tuple[int, ...] = (200, 201, 202)) -> Any:
        if response.status_code in allow:
            if response.status_code == 204:
                return None
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return None
        raise JenkinsError(
            f"Jenkins API error on {what}: HTTP {response.status_code}: {response.text[:300]}"
        )

    # ------------------------------------------------------------------ #
    # builds
    # ------------------------------------------------------------------ #

    def trigger_build(
        self,
        job: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        delay: int = 0,
    ) -> str | None:
        """Trigger a parameterised build; return the queue item URL/location.

        Jenkins answers 201 with a ``Location`` header pointing at the queue
        item. We return that location (or None when absent).
        """
        url = self._build_job_url([*job.split("/"), "buildWithParameters"])
        data = build_trigger_payload(parameters)
        if delay:
            data["delay"] = str(delay)
        response = self.client.request(
            "POST",
            url,
            headers=self._headers(form=True),
            data=data,
            timeout=self.timeout,
        )
        if response.status_code not in (200, 201, 202):
            raise JenkinsError(
                f"trigger build on {job!r} failed: HTTP {response.status_code}: {response.text[:300]}"
            )
        # httpx exposes headers on the response; our protocol only guarantees
        # status_code/text, so re-request queue item via get_queue when needed.
        return getattr(response, "headers", {}).get("Location") if hasattr(response, "headers") else None

    def get_build(self, job: str, number: int) -> BuildInfo:
        """Fetch build metadata."""
        if number < 1:
            raise JenkinsError(f"invalid build number: {number}")
        url = self._build_job_url([*job.split("/"), str(number), "api", "json"])
        response = self.client.request(
            "GET", url, headers=self._headers(), timeout=self.timeout
        )
        data = self._check(response, what=f"get build {job}#{number}")
        if not isinstance(data, Mapping):
            raise JenkinsError(f"unexpected build payload for {job}#{number}")
        return parse_build_info(data)

    def get_console_log(self, job: str, number: int) -> str:
        """Fetch the (possibly partial) console output of a build."""
        url = self._build_job_url([*job.split("/"), str(number), "consoleText"])
        response = self.client.request(
            "GET", url, headers=self._headers(), timeout=self.timeout
        )
        if 200 <= response.status_code < 300:
            return response.text
        raise JenkinsError(
            f"console log for {job}#{number} failed: HTTP {response.status_code}"
        )

    def set_build_description(self, job: str, number: int, description: str) -> None:
        """Set the build description (shown on the build page)."""
        url = self._build_job_url([*job.split("/"), str(number), "submitDescription"])
        response = self.client.request(
            "POST",
            url,
            headers=self._headers(form=True),
            data={"description": description},
            timeout=self.timeout,
        )
        if response.status_code not in (200, 201, 202):
            raise JenkinsError(
                f"set description on {job}#{number} failed: HTTP {response.status_code}"
            )

    def get_queue_item(self, queue_id: int) -> Mapping[str, Any]:
        """Fetch a queue item by id (to discover the assigned build number)."""
        response = self.client.request(
            "GET",
            f"{self.base_url}/queue/item/{queue_id}/api/json",
            headers=self._headers(),
            timeout=self.timeout,
        )
        data = self._check(response, what=f"queue item {queue_id}")
        return data if isinstance(data, Mapping) else {}

    def cancel_queue_item(self, queue_id: int) -> None:
        """Cancel a queued build."""
        response = self.client.request(
            "POST",
            f"{self.base_url}/queue/cancelItem?id={queue_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code not in (200, 201, 202):
            raise JenkinsError(f"cancel queue item {queue_id} failed: HTTP {response.status_code}")

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #

    def report_result(
        self,
        result: Any,
        *,
        job: str,
        number: int,
        max_loss_rate: float = 0.0,
    ) -> BuildInfo:
        """Annotate a build with the verdict and return fresh build info.

        Sets the build description to a one-line verdict summary.
        """
        matrix = getattr(result, "matrix", None)
        verdict = str(getattr(matrix, "verdict", "unknown")) if matrix else "unknown"
        regressed = getattr(matrix, "regressed_dimensions", ()) if matrix else ()
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        gate = "FAIL" if (regressed or rate > max_loss_rate) else "PASS"
        description = (
            f"llmdiff: verdict={verdict} loss_rate={rate:.3f} "
            f"regressed={','.join(regressed) or '-'} gate={gate}"
        )
        self.set_build_description(job, number, description)
        return self.get_build(job, number)


@dataclass(slots=True)
class JenkinsGate:
    """Maps verdicts to Jenkins gate decisions.

    Attributes:
        max_loss_rate: Maximum tolerated regression rate (0..1).
        fail_on_regressed_dimensions: Fail when any dimension regressed.
    """

    max_loss_rate: float = 0.0
    fail_on_regressed_dimensions: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_loss_rate <= 1.0:
            raise JenkinsError(f"max_loss_rate must be in [0, 1], got {self.max_loss_rate}")

    def evaluate(self, result: Any) -> dict[str, Any]:
        """Return ``{"gate": "PASS"|"FAIL", "params": {...}, "reasons": [...]}``."""
        params = verdict_to_params(result)
        matrix = getattr(result, "matrix", None)
        rate = float(getattr(matrix, "regression_rate", 0.0)) if matrix else 0.0
        regressed = list(getattr(matrix, "regressed_dimensions", ()) or ()) if matrix else []
        reasons: list[str] = []
        if self.fail_on_regressed_dimensions and regressed:
            reasons.append(f"regressed dimensions: {', '.join(regressed)}")
        if rate > self.max_loss_rate:
            reasons.append(
                f"regression rate {rate:.4f} exceeds budget {self.max_loss_rate:.4f}"
            )
        gate = "FAIL" if reasons else "PASS"
        return {"gate": gate, "params": params, "reasons": reasons}

    def trigger_with_verdict(
        self,
        client: JenkinsClient,
        job: str,
        result: Any,
        *,
        extra_parameters: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Trigger ``job`` with the verdict as build parameters."""
        decision = self.evaluate(result)
        params = dict(decision["params"])
        params.update(extra_parameters or {})
        return client.trigger_build(job, parameters=params)


def wait_for_build(
    client: JenkinsClient,
    job: str,
    number: int,
    *,
    timeout: float = 600.0,
    interval: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> BuildInfo:
    """Poll a build until it finishes or ``timeout`` elapses.

    Raises:
        JenkinsError: On timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        info = client.get_build(job, number)
        if info.finished:
            return info
        if time.monotonic() >= deadline:
            raise JenkinsError(
                f"build {job}#{number} did not finish within {timeout:.0f}s"
            )
        sleep(interval)
