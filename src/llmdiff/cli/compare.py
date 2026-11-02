"""``llmdiff compare`` - evaluate a corpus and render the verdict matrix.

Usage::

    llmdiff compare corpus.jsonl [--dimensions a,b] [--tolerance 0.05]
        [--format md|json|html|junit|sarif|csv|terminal] [--output file]
        [--max-loss-rate 0.05] [--fail-fast] [--tags x,y]
"""

from __future__ import annotations

import argparse
from typing import Sequence

from llmdiff.cli.config import CliConfig
from llmdiff.cli.corpus_io import load_corpus
from llmdiff.cli.gate import GateThresholds, gate_from_result
from llmdiff.cli.output import emit, err, style
from llmdiff.cli.renderers import render_result
from llmdiff.core import DiffConfig, DiffEngine
from llmdiff.core.errors import LLMDiffError, RegressionGateError

__all__ = ["run_compare", "add_compare_parser"]

FORMATS = ("terminal", "md", "markdown", "json", "html", "junit", "sarif", "csv")


def add_compare_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``compare`` subcommand on ``sub``."""
    p = sub.add_parser(
        "compare",
        help="evaluate a corpus (jsonl/json/csv) and render the verdict matrix",
        description="Load a corpus, run llmdiff.core.evaluate, render the result "
        "and optionally apply a regression gate.",
    )
    p.add_argument("corpus", help="path to the corpus file (.jsonl/.json/.csv, '-' for stdin)")
    p.add_argument(
        "--dimensions",
        "-d",
        default=None,
        metavar="A,B",
        help="comma-separated dimension names to evaluate (default: all registered)",
    )
    p.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=None,
        help="absolute score difference treated as a tie (default: from config or 0.0)",
    )
    p.add_argument(
        "--format",
        "-f",
        default=None,
        choices=FORMATS,
        help="output format (default: from config or 'terminal')",
    )
    p.add_argument("--output", "-o", default=None, help="write output to this file ('-' for stdout)")
    p.add_argument(
        "--max-loss-rate",
        type=float,
        default=None,
        help="gate: maximum allowed loss rate before exit code 1",
    )
    p.add_argument(
        "--max-regressed-dims",
        type=int,
        default=None,
        help="gate: maximum number of regressed dimensions allowed",
    )
    p.add_argument(
        "--min-win-rate",
        type=float,
        default=None,
        help="gate: minimum required win rate",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="abort the run on the first failing check",
    )
    p.add_argument(
        "--tags",
        default=None,
        metavar="X,Y",
        help="only evaluate cases carrying any of these tags",
    )
    p.add_argument(
        "--config",
        default=None,
        help="path to a llmdiff.toml config file (default: auto-discovered)",
    )
    p.set_defaults(func=run_compare)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def run_compare(args: argparse.Namespace, config: CliConfig | None = None) -> int:
    """Execute the ``compare`` command.  Returns the process exit code."""
    from llmdiff.cli.config import load_config

    if config is None:
        try:
            config = load_config(args.config)
        except LLMDiffError as exc:
            err(f"error: {exc}")
            return 2
    try:
        corpus = load_corpus(args.corpus)
    except LLMDiffError as exc:
        err(f"error: cannot load corpus: {exc}")
        return 2
    tags = _split_csv(args.tags)
    if tags:
        corpus = corpus.filter_by_tags(tags)
        if len(corpus) == 0:
            err(f"error: no cases match tags {tags}")
            return 2
    dimensions = _split_csv(args.dimensions) or (
        list(config.dimensions) if config.dimensions else None
    )
    tolerance = args.tolerance if args.tolerance is not None else config.tolerance
    fail_fast = bool(args.fail_fast) if args.fail_fast is not None else False
    try:
        engine = DiffEngine()
        result = engine.run(
            corpus,
            config=DiffConfig(
                dimensions=tuple(dimensions) if dimensions else None,
                tolerance=tolerance,
                fail_fast=fail_fast,
            ),
        )
    except LLMDiffError as exc:
        err(f"error: evaluation failed: {exc}")
        return 2
    fmt = args.format or config.format or "terminal"
    try:
        text = render_result(result, fmt)
    except KeyError as exc:
        err(f"error: {exc}")
        return 2
    if args.output:
        from llmdiff.cli.output import write_text

        write_text(args.output, text)
        emit(f"wrote {args.output} ({fmt})")
    else:
        emit(text, end="")
    # Gate: only applied when at least one gate flag/config threshold is set.
    thresholds = GateThresholds(
        max_loss_rate=(
            args.max_loss_rate if args.max_loss_rate is not None else config.max_loss_rate
        ),
        max_regressed_dims=(
            args.max_regressed_dims
            if args.max_regressed_dims is not None
            else config.max_regressed_dims
        ),
        min_win_rate=args.min_win_rate if args.min_win_rate is not None else config.min_win_rate,
    )
    if thresholds.enabled:
        try:
            gate_from_result(result, thresholds)
        except RegressionGateError as exc:
            for failure in exc.failures:
                emit(style(f"GATE FAIL: {failure}", "bold_red"))
            return 1
        emit(style("gate: PASS", "bold_green"))
    return 0
