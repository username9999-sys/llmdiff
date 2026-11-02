"""``llmdiff import`` - read an external corpus file into jsonl.

Usage::

    llmdiff import data.csv --out corpus.jsonl
    llmdiff import data.json -o -

This is the inverse of :mod:`llmdiff.cli.export_cmd`; it always writes the
canonical JSONL corpus format unless ``--to`` says otherwise.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.corpus_io import load_corpus, save_corpus
from llmdiff.cli.export_cmd import FORMATS
from llmdiff.cli.output import emit, err, style
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_import", "add_import_parser"]


def add_import_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``import`` subcommand."""
    p = sub.add_parser(
        "import",
        help="import an external corpus file (csv/json) into llmdiff format",
        description="Read a csv/json corpus and write it as JSONL (or another format).",
    )
    p.add_argument("input", help="input file (.csv/.json/.jsonl, '-' for stdin)")
    p.add_argument("--out", "-o", default="corpus.jsonl", help="output file (default: corpus.jsonl)")
    p.add_argument("--to", "-t", default="jsonl", choices=FORMATS, help="output format (default: jsonl)")
    p.add_argument("--from", dest="from_fmt", default=None, choices=FORMATS, help="force input format")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    p.set_defaults(func=run_import)


def run_import(args: argparse.Namespace) -> int:
    """Execute ``import``.  Returns 0 on success, 2 on error."""
    from pathlib import Path

    if str(args.out) != "-":
        target = Path(args.out)
        if target.exists() and not args.force:
            err(f"error: {target} already exists (use --force to overwrite)")
            return 2
    try:
        corpus = load_corpus(args.input, fmt=args.from_fmt)
        out_path = args.out
        if str(out_path) != "-" and not str(out_path).lower().endswith(f".{args.to}"):
            out_path = f"{out_path}.{args.to}"
        written = save_corpus(corpus, out_path, fmt=args.to)
    except LLMDiffError as exc:
        err(f"error: {exc}")
        return 2
    if str(args.out) != "-":
        emit(style(f"imported {len(corpus)} cases -> {written} ({args.to})", "bold_green"))
    return 0
