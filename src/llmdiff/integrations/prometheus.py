"""Prometheus integration: render the ``/metrics`` text exposition format.

llmdiff can expose verdict metrics in the Prometheus text format so a
Prometheus server can scrape them. This module implements:

- :func:`render_metrics` — render a DiffResult as Prometheus text
  (``# HELP`` / ``# TYPE`` comments + samples),
- :func:`render_metrics_from_values` — render raw ``{name: (type, help, samples)}``
  data (used by tests and by the server),
- :func:`parse_metrics` — a small parser for the text format (round-trip
  checks in tests, and useful for debugging),
- :class:`PrometheusApp` — a FastAPI app exposing ``GET /metrics``.

Metric naming follows Prometheus conventions: ``llmdiff_<...>`` with
``_total`` suffix for counters, labels in ``{...}``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "PrometheusError",
    "Sample",
    "MetricFamily",
    "render_metrics",
    "render_metrics_from_values",
    "parse_metrics",
    "metrics_from_result",
    "PrometheusApp",
    "escape_label_value",
    "sanitize_metric_name",
]


class PrometheusError(LLMDiffError):
    """Raised for invalid metric names/labels or parse errors."""


_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def sanitize_metric_name(name: str) -> str:
    """Sanitise ``name`` into a valid Prometheus metric name.

    Invalid characters become ``_``; a leading digit is prefixed with ``_``.
    """
    if not name:
        raise PrometheusError("metric name must be non-empty")
    cleaned = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    if not _METRIC_NAME_RE.match(cleaned):
        raise PrometheusError(f"cannot sanitise {name!r} into a valid metric name")
    return cleaned


def escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


@dataclass(frozen=True, slots=True)
class Sample:
    """One Prometheus sample line.

    Attributes:
        name: Metric name (without labels).
        labels: Label mapping (may be empty).
        value: Numeric value.
        timestamp_ms: Optional timestamp in milliseconds.
    """

    name: str
    labels: Mapping[str, str] = ()
    value: float = 0.0
    timestamp_ms: int | None = None

    def render(self) -> str:
        """Render this sample as one line of the text format."""
        if self.labels:
            label_str = ",".join(
                f'{key}="{escape_label_value(str(value))}"'
                for key, value in sorted(self.labels.items())
            )
            line = f"{self.name}{{{label_str}}} {_format_value(self.value)}"
        else:
            line = f"{self.name} {_format_value(self.value)}"
        if self.timestamp_ms is not None:
            line += f" {int(self.timestamp_ms)}"
        return line


def _format_value(value: float) -> str:
    """Format a sample value: ints without a decimal point, floats compactly."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


@dataclass(frozen=True, slots=True)
class MetricFamily:
    """A metric family: HELP/TYPE header plus samples.

    Attributes:
        name: Metric name.
        help: HELP text.
        type: ``counter``, ``gauge``, ``histogram``, ``summary`` or ``untyped``.
        samples: The samples belonging to this family.
    """

    name: str
    help: str
    type: str = "gauge"
    samples: tuple[Sample, ...] = ()

    def render(self) -> str:
        """Render the family as HELP/TYPE lines followed by samples."""
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} {self.type}",
        ]
        lines.extend(sample.render() for sample in self.samples)
        return "\n".join(lines)


def render_metrics_from_values(
    families: Sequence[MetricFamily],
) -> str:
    """Render metric families to the Prometheus text exposition format.

    The output ends with a trailing newline, as required by the spec.
    """
    if not families:
        return ""
    return "\n".join(family.render() for family in families) + "\n"


def metrics_from_result(
    result: Any,
    *,
    namespace: str = "llmdiff",
    labels: Mapping[str, str] | None = None,
    timestamp_ms: int | None = None,
) -> list[MetricFamily]:
    """Build Prometheus metric families from a DiffResult-like object.

    Emits:

    - ``<ns>_dimension_verdict_total{dimension=..., verdict=...}`` counter
      (one sample per dimension with value 1 for its dominant verdict),
    - ``<ns>_dimension_wins_total{dimension=...}`` counter,
    - ``<ns>_dimension_ties_total{dimension=...}`` counter,
    - ``<ns>_dimension_losses_total{dimension=...}`` counter,
    - ``<ns>_dimension_baseline_mean{dimension=...}`` gauge,
    - ``<ns>_dimension_candidate_mean{dimension=...}`` gauge,
    - ``<ns>_dimension_delta{dimension=...}`` gauge,
    - ``<ns>_wins_total`` / ``<ns>_ties_total`` / ``<ns>_losses_total`` counters,
    - ``<ns>_regression_rate`` gauge,
    - ``<ns>_total_cases`` gauge,
    - ``<ns>_verdict_info{verdict=...}`` gauge (value 1).
    """
    ns = sanitize_metric_name(namespace)
    base_labels = dict(labels or {})
    matrix = getattr(result, "matrix", None)
    if matrix is None and isinstance(result, Mapping):
        matrix = result.get("matrix")
    if matrix is None:
        raise PrometheusError("result has no .matrix; pass a DiffResult or {'matrix': ...}")

    def labels_with(**extra: str) -> dict[str, str]:
        merged = dict(base_labels)
        merged.update(extra)
        return merged

    dimension_summaries = list(getattr(matrix, "dimensions", ()) or ())

    verdict_samples: list[Sample] = []
    wins_samples: list[Sample] = []
    ties_samples: list[Sample] = []
    losses_samples: list[Sample] = []
    baseline_samples: list[Sample] = []
    candidate_samples: list[Sample] = []
    delta_samples: list[Sample] = []
    for summary in dimension_summaries:
        dim = str(summary.dimension)
        verdict_samples.append(
            Sample(f"{ns}_dimension_verdict_total", labels_with(dimension=dim, verdict=summary.verdict), 1.0, timestamp_ms)
        )
        wins_samples.append(Sample(f"{ns}_dimension_wins_total", labels_with(dimension=dim), summary.wins, timestamp_ms))
        ties_samples.append(Sample(f"{ns}_dimension_ties_total", labels_with(dimension=dim), summary.ties, timestamp_ms))
        losses_samples.append(Sample(f"{ns}_dimension_losses_total", labels_with(dimension=dim), summary.losses, timestamp_ms))
        baseline_samples.append(Sample(f"{ns}_dimension_baseline_mean", labels_with(dimension=dim), summary.baseline_mean, timestamp_ms))
        candidate_samples.append(Sample(f"{ns}_dimension_candidate_mean", labels_with(dimension=dim), summary.candidate_mean, timestamp_ms))
        delta_samples.append(Sample(f"{ns}_dimension_delta", labels_with(dimension=dim), summary.delta, timestamp_ms))

    verdict = str(getattr(matrix, "verdict", "unknown"))
    families = [
        MetricFamily(f"{ns}_dimension_verdict_total", "Dominant verdict per dimension.", "counter", tuple(verdict_samples)),
        MetricFamily(f"{ns}_dimension_wins_total", "Cases won per dimension.", "counter", tuple(wins_samples)),
        MetricFamily(f"{ns}_dimension_ties_total", "Cases tied per dimension.", "counter", tuple(ties_samples)),
        MetricFamily(f"{ns}_dimension_losses_total", "Cases lost per dimension.", "counter", tuple(losses_samples)),
        MetricFamily(f"{ns}_dimension_baseline_mean", "Mean baseline score per dimension.", "gauge", tuple(baseline_samples)),
        MetricFamily(f"{ns}_dimension_candidate_mean", "Mean candidate score per dimension.", "gauge", tuple(candidate_samples)),
        MetricFamily(f"{ns}_dimension_delta", "Mean score delta per dimension.", "gauge", tuple(delta_samples)),
        MetricFamily(
            f"{ns}_wins_total", "Total wins across dimensions.", "counter",
            (Sample(f"{ns}_wins_total", base_labels, getattr(matrix, "wins", 0), timestamp_ms),),
        ),
        MetricFamily(
            f"{ns}_ties_total", "Total ties across dimensions.", "counter",
            (Sample(f"{ns}_ties_total", base_labels, getattr(matrix, "ties", 0), timestamp_ms),),
        ),
        MetricFamily(
            f"{ns}_losses_total", "Total losses across dimensions.", "counter",
            (Sample(f"{ns}_losses_total", base_labels, getattr(matrix, "losses", 0), timestamp_ms),),
        ),
        MetricFamily(
            f"{ns}_regression_rate", "Fraction of all comparisons that regressed.", "gauge",
            (Sample(f"{ns}_regression_rate", base_labels, getattr(matrix, "regression_rate", 0.0), timestamp_ms),),
        ),
        MetricFamily(
            f"{ns}_total_cases", "Number of cases evaluated.", "gauge",
            (Sample(f"{ns}_total_cases", base_labels, getattr(matrix, "total_cases", 0), timestamp_ms),),
        ),
        MetricFamily(
            f"{ns}_verdict_info", "Overall verdict as an info-style gauge.", "gauge",
            (Sample(f"{ns}_verdict_info", labels_with(verdict=verdict), 1.0, timestamp_ms),),
        ),
    ]
    return families


def render_metrics(
    result: Any,
    *,
    namespace: str = "llmdiff",
    labels: Mapping[str, str] | None = None,
    timestamp_ms: int | None = None,
) -> str:
    """Render a DiffResult directly to the Prometheus text format."""
    return render_metrics_from_values(
        metrics_from_result(result, namespace=namespace, labels=labels, timestamp_ms=timestamp_ms)
    )


# ---------------------------------------------------------------------- #
# parser (for round-trip tests and debugging)
# ---------------------------------------------------------------------- #

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[^\s]+)"
    r"(?:\s+(?P<ts>\d+))?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _unescape_label_value(value: str) -> str:
    return value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def parse_metrics(text: str) -> list[MetricFamily]:
    """Parse Prometheus text format back into metric families.

    Supports HELP/TYPE comments and sample lines with optional labels and
    timestamps. Raises :class:`PrometheusError` on malformed lines.
    """
    families: dict[str, MetricFamily] = {}
    order: list[str] = []
    current_help: dict[str, str] = {}
    current_type: dict[str, str] = {}
    samples: dict[str, list[Sample]] = {}

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if line.startswith("# HELP "):
                rest = line[len("# HELP ") :]
                name, _, help_text = rest.partition(" ")
                current_help[name] = help_text
            elif line.startswith("# TYPE "):
                rest = line[len("# TYPE ") :]
                name, _, type_text = rest.partition(" ")
                current_type[name] = type_text.strip()
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            raise PrometheusError(f"line {lineno}: cannot parse sample: {line!r}")
        name = match.group("name")
        labels: dict[str, str] = {}
        if match.group("labels"):
            for label_match in _LABEL_RE.finditer(match.group("labels")):
                labels[label_match.group(1)] = _unescape_label_value(label_match.group(2))
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise PrometheusError(f"line {lineno}: invalid value {match.group('value')!r}") from exc
        ts = int(match.group("ts")) if match.group("ts") else None
        sample = Sample(name=name, labels=labels, value=value, timestamp_ms=ts)
        samples.setdefault(name, []).append(sample)
        if name not in families:
            families[name] = MetricFamily(
                name=name,
                help=current_help.get(name, ""),
                type=current_type.get(name, "untyped"),
                samples=(),
            )
            order.append(name)
    return [
        MetricFamily(
            name=name,
            help=families[name].help,
            type=families[name].type,
            samples=tuple(samples[name]),
        )
        for name in order
    ]


# ---------------------------------------------------------------------- #
# FastAPI app
# ---------------------------------------------------------------------- #


class _ResultProvider(Protocol):
    def __call__(self) -> Any: ...


@dataclass(slots=True)
class PrometheusApp:
    """A FastAPI app exposing ``GET /metrics``.

    Attributes:
        result_provider: Zero-arg callable returning the current DiffResult.
        namespace: Metric namespace.
        labels: Extra labels attached to every metric.
    """

    result_provider: _ResultProvider | None = None
    namespace: str = "llmdiff"
    labels: Mapping[str, str] = ()

    def render(self) -> str:
        """Render the current metrics text (empty string when no provider)."""
        if self.result_provider is None:
            return ""
        return render_metrics(
            self.result_provider(),
            namespace=self.namespace,
            labels=self.labels,
            timestamp_ms=int(time.time() * 1000),
        )

    def create_fastapi_app(self) -> Any:
        """Build a FastAPI app with ``GET /metrics`` returning the text format."""
        try:
            from fastapi import FastAPI  # type: ignore[import-untyped]
            from fastapi.responses import PlainTextResponse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise PrometheusError("fastapi is required for the Prometheus app") from exc

        app = FastAPI(title="llmdiff metrics")

        @app.get("/metrics")
        async def metrics() -> Any:
            return PlainTextResponse(self.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

        return app
