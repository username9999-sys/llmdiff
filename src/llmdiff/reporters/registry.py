"""Built-in reporter registry.

:func:`default_reporters` returns one instance of every built-in reporter and
:func:`get_reporter` looks one up by name. The registry is intentionally a
plain function pair (not a class instance) so the CLI can build a
:class:`~llmdiff.reporters.base.ReporterRegistry` from it in one line.
"""

from __future__ import annotations

from llmdiff.core.errors import ReporterError
from llmdiff.core.result import DiffResult
from llmdiff.reporters.badge import BadgeReporter
from llmdiff.reporters.base import Reporter, ReporterRegistry
from llmdiff.reporters.csv_reporter import CsvReporter
from llmdiff.reporters.diff_reporter import DiffReporter
from llmdiff.reporters.github import GithubReporter
from llmdiff.reporters.html import HtmlReporter
from llmdiff.reporters.json_reporter import JsonReporter
from llmdiff.reporters.junit import JunitReporter
from llmdiff.reporters.markdown import MarkdownReporter
from llmdiff.reporters.sarif import SarifReporter
from llmdiff.reporters.slack import SlackReporter
from llmdiff.reporters.terminal import TerminalReporter
from llmdiff.reporters.trend import TrendReporter

__all__ = ["default_reporters", "get_reporter", "build_registry", "REPORTER_NAMES"]

#: Names of every built-in reporter, in registration order.
REPORTER_NAMES = (
    "markdown",
    "json",
    "html",
    "junit",
    "sarif",
    "github",
    "slack",
    "terminal",
    "csv",
    "badge",
    "diff",
    "trend",
)


def default_reporters() -> tuple[Reporter, ...]:
    """Return one instance of every built-in reporter."""
    return (
        MarkdownReporter(),
        JsonReporter(),
        HtmlReporter(),
        JunitReporter(),
        SarifReporter(),
        GithubReporter(),
        SlackReporter(),
        TerminalReporter(color=False),
        CsvReporter(),
        BadgeReporter(),
        DiffReporter(),
        TrendReporter(),
    )


def build_registry() -> ReporterRegistry:
    """Return a :class:`ReporterRegistry` pre-populated with all built-ins."""
    return ReporterRegistry(default_reporters())


def get_reporter(name: str) -> Reporter:
    """Return the built-in reporter registered under ``name``.

    Raises:
        ReporterError: If no built-in reporter has that name.
    """
    for reporter in default_reporters():
        if reporter.name == name:
            return reporter
    raise ReporterError(
        f"unknown reporter: {name!r} (available: {', '.join(REPORTER_NAMES)})"
    )


def render(name: str, result: DiffResult) -> str:
    """Convenience: render ``result`` with the built-in reporter ``name``."""
    return get_reporter(name).render(result)
