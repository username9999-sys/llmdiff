"""``llmdiff init`` - write an example corpus file.

Usage::

    llmdiff init [--out corpus.jsonl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llmdiff.cli.corpus_io import save_corpus
from llmdiff.cli.examples import INIT_TEMPLATE_TEXT, example_corpus
from llmdiff.cli.output import emit, err, style
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_init", "add_init_parser"]


def add_init_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``init`` subcommand."""
    p = sub.add_parser(
        "init",
        help="write an example corpus with 5 realistic cases",
        description="Create a starter corpus file you can edit and evaluate with "
        "'llmdiff compare'.",
    )
    p.add_argument(
        "--out",
        "-o",
        default="corpus.jsonl",
        help="output path (default: corpus.jsonl; '-' for stdout)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    p.set_defaults(func=run_init)


def run_init(args: argparse.Namespace) -> int:
    """Execute ``init``.  Returns the exit code."""
    if str(args.out) != "-":
        target = Path(args.out)
        if target.exists() and not args.force:
            err(f"error: {target} already exists (use --force to overwrite)")
            return 2
    corpus = example_corpus()
    try:
        written = save_corpus(corpus, args.out)
    except LLMDiffError as exc:
        err(f"error: {exc}")
        return 2
    if str(args.out) != "-":
        emit(style(f"wrote {len(corpus)} example cases to {written}", "bold_green"))
        emit("next: edit the file, then run  llmdiff compare " + str(written))
        emit("hint: " + INIT_TEMPLATE_TEXT.splitlines()[1])
    return 0
