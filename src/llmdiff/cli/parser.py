"""argparse parser construction for the llmdiff CLI.

The parser is stdlib-only (``argparse``); ``rich``/``click`` are never
required.  Subcommands live in their own modules and register themselves via
``add_*_parser(sub)``.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.list_cmd import (
    add_checks_parser,
    add_reporters_parser,
    add_version_parser,
)
from llmdiff.cli.compare import add_compare_parser
from llmdiff.cli.config_cmd import add_config_parser
from llmdiff.cli.demo_cmd import add_demo_parser
from llmdiff.cli.diff_cmd import add_diff_parser
from llmdiff.cli.export_cmd import add_export_parser
from llmdiff.cli.gate_cmd import add_gate_parser
from llmdiff.cli.import_cmd import add_import_parser
from llmdiff.cli.init_cmd import add_init_parser
from llmdiff.cli.plugins_cmd import add_plugins_parser
from llmdiff.cli.record_cmd import add_record_parser
from llmdiff.cli.replay_cmd import add_replay_parser
from llmdiff.cli.serve_cmd import add_serve_parser
from llmdiff.cli.stats_cmd import add_stats_parser

__all__ = ["build_parser", "COMMANDS"]

#: Subcommand names in help-display order.
COMMANDS = (
    "compare",
    "diff",
    "init",
    "demo",
    "gate",
    "record",
    "replay",
    "serve",
    "plugins",
    "config",
    "version",
    "stats",
    "export",
    "import",
    "checks",
    "reporters",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="llmdiff",
        description=(
            "llmdiff - diff LLM outputs across a corpus of prompts. "
            "Compare a candidate system against a baseline on multiple dimensions "
            "and gate regressions in CI."
        ),
        epilog="exit codes: 0 = ok, 1 = regression gate failed, 2 = usage/config error",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llmdiff {__import__('llmdiff.cli.list_cmd', fromlist=['package_version']).package_version()}",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    add_compare_parser(sub)
    add_diff_parser(sub)
    add_init_parser(sub)
    add_demo_parser(sub)
    add_gate_parser(sub)
    add_record_parser(sub)
    add_replay_parser(sub)
    add_serve_parser(sub)
    add_plugins_parser(sub)
    add_config_parser(sub)
    add_version_parser(sub)
    add_stats_parser(sub)
    add_export_parser(sub)
    add_import_parser(sub)
    add_checks_parser(sub)
    add_reporters_parser(sub)
    return parser
