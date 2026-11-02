"""llmdiff.reporters — render a DiffResult in any format.

Public API:

- :class:`Reporter` / :class:`BaseReporter` / :class:`ReporterRegistry` — protocol & helpers
- One reporter class per format: markdown, json, html, junit, sarif, github,
  slack, terminal, csv, badge, diff, trend
- :func:`default_reporters`, :func:`get_reporter`, :func:`build_registry` — lookup
"""

from __future__ import annotations

from llmdiff.reporters.badge import BadgeReporter
from llmdiff.reporters.base import (
    BaseReporter,
    Reporter,
    ReporterRegistry,
    escape_md,
    escape_md_cell,
    format_delta,
    format_pct,
    format_score,
    safe_filename,
    verdict_emoji,
    verdict_label,
)
from llmdiff.reporters.csv_reporter import CsvReporter
from llmdiff.reporters.diff_reporter import DiffReporter
from llmdiff.reporters.github import GithubReporter
from llmdiff.reporters.html import HtmlReporter
from llmdiff.reporters.json_reporter import JsonReporter
from llmdiff.reporters.junit import JunitReporter
from llmdiff.reporters.markdown import MarkdownReporter
from llmdiff.reporters.registry import (
    REPORTER_NAMES,
    build_registry,
    default_reporters,
    get_reporter,
)
from llmdiff.reporters.sarif import SarifReporter
from llmdiff.reporters.slack import SlackReporter
from llmdiff.reporters.terminal import TerminalReporter
from llmdiff.reporters.trend import DimensionTrend, TrendReporter, load_previous_result

__all__ = [
    "REPORTER_NAMES",
    "BadgeReporter",
    "BaseReporter",
    "CsvReporter",
    "DiffReporter",
    "DimensionTrend",
    "GithubReporter",
    "HtmlReporter",
    "JsonReporter",
    "JunitReporter",
    "MarkdownReporter",
    "Reporter",
    "ReporterRegistry",
    "SarifReporter",
    "SlackReporter",
    "TerminalReporter",
    "TrendReporter",
    "build_registry",
    "default_reporters",
    "escape_md",
    "escape_md_cell",
    "format_delta",
    "format_pct",
    "format_score",
    "get_reporter",
    "load_previous_result",
    "safe_filename",
    "verdict_emoji",
    "verdict_label",
]
