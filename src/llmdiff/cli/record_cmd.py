"""``llmdiff record`` - thin wrapper around the capture proxy.

Usage::

    llmdiff record --proxy --port 8080 --target https://api.openai.com --out rec.jsonl

When :mod:`llmdiff.capture` is importable this launches its capture proxy;
otherwise it prints instructions and exits 2.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.output import emit, err, style

__all__ = ["run_record", "add_record_parser"]

INSTRUCTIONS = """\
The capture package (llmdiff.capture) is not available in this environment.

To record live traffic install the full llmdiff package and run:

    llmdiff record --proxy --port 8080 --target https://api.openai.com/v1 --out rec.jsonl

or call the capture proxy directly:

    python -m llmdiff.capture.proxy --port 8080 --target https://api.openai.com/v1 --out rec.jsonl

Point your app at http://localhost:8080/v1 (OpenAI-compatible) and every
request/response pair is appended to the output corpus.
"""


def add_record_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``record`` subcommand."""
    p = sub.add_parser(
        "record",
        help="record live LLM traffic via the capture proxy",
        description="Launch the llmdiff capture proxy (requires llmdiff.capture).",
    )
    p.add_argument("--proxy", action="store_true", help="run the capture proxy (default mode)")
    p.add_argument("--port", type=int, default=8080, help="proxy listen port (default: 8080)")
    p.add_argument("--target", required=True, help="upstream base URL to forward to")
    p.add_argument("--out", "-o", required=True, help="output corpus file (.jsonl)")
    p.add_argument("--host", default="127.0.0.1", help="proxy bind host")
    p.set_defaults(func=run_record)


def run_record(args: argparse.Namespace) -> int:
    """Execute ``record``.  Returns 0 on success, 2 when capture is unavailable."""
    try:
        from llmdiff.capture.proxy import main as proxy_main  # type: ignore[attr-defined]
    except Exception:
        err("error: llmdiff.capture is not importable in this environment.")
        err(INSTRUCTIONS)
        return 2
    argv = [
        "--host", args.host,
        "--port", str(args.port),
        "--target", args.target,
        "--out", args.out,
    ]
    emit(style(f"starting capture proxy on {args.host}:{args.port} -> {args.target}", "bold_cyan"))
    emit(f"output corpus: {args.out}")
    code = proxy_main(argv)
    return int(code) if isinstance(code, int) else 0
