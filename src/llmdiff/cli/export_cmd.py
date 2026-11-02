"""``llmdiff export`` - convert a corpus between jsonl / json / csv.

Usage::

    llmdiff export corpus.jsonl --to csv --out corpus.csv
    llmdiff export corpus.csv --to jsonl -o -

Implemented locally with the stdlib (see :mod:`llmdiff.cli.corpus_io`); it
does not depend on the capture package.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.corpus_io import load_corpus, save_corpus
from llmdiff.cli.output import emit, err, style
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_export", "add_export_parser"]

FORMATS = ("jsonl", "json", "csv")


def add_export_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``export`` subcommand."""
    p = sub.add_parser(
        "export",
        help="convert a corpus between jsonl / json / csv",
        description="Read a corpus in one format and write it in another. "
        "Stdlib-only; no capture dependency.",
    )
    p.add_argument("input", help="input corpus file ('-' for stdin)")
    p.add_argument("--to", "-t", required=True, choices=FORMATS, help="target format")
    p.add_argument("--out", "-o", default="-", help="output file (default: stdout)")
    p.add_argument("--from", dest="from_fmt", default=None, choices=FORMATS, help="force input format")
    p.set_defaults(func=run_export)


def run_export(args: argparse.Namespace) -> int:
    """Execute ``export``.  Returns 0 on success, 2 on error."""
    try:
        corpus = load_corpus(args.input, fmt=args.from_fmt)
        out_path = args.out
        if str(out_path) != "-" and not str(out_path).lower().endswith(f".{args.to}"):
            # Ensure the suffix matches the target format for detect_format.
            out_path = f"{out_path}.{args.to}"
        written = save_corpus(corpus, out_path, fmt=args.to)
    except LLMDiffError as exc:
        err(f"error: {exc}")
        return 2
    if str(args.out) != "-":
        emit(style(f"exported {len(corpus)} cases -> {written} ({args.to})", "bold_green"))
    return 0
