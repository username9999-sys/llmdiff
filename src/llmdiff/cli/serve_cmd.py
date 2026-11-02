"""``llmdiff serve`` - launch the llmdiff server package.

Usage::

    llmdiff serve --port 8000 [--host 127.0.0.1]

Thin wrapper: imports :mod:`llmdiff.server` lazily and calls its ``main``;
prints instructions and exits 2 when the package is unavailable.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.output import emit, err, style

__all__ = ["run_serve", "add_serve_parser"]

INSTRUCTIONS = """\
The server package (llmdiff.server) is not available in this environment.

With the full package installed:

    llmdiff serve --port 8000
    # then open http://127.0.0.1:8000/ to browse corpora and results
"""


def add_serve_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``serve`` subcommand."""
    p = sub.add_parser(
        "serve",
        help="launch the llmdiff HTTP server",
        description="Serve corpora/results over HTTP (requires llmdiff.server).",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p.set_defaults(func=run_serve)


def run_serve(args: argparse.Namespace) -> int:
    """Execute ``serve``.  Returns 0 on success, 2 when server is unavailable."""
    try:
        from llmdiff.server import main as server_main  # type: ignore[attr-defined]
    except Exception:
        err("error: llmdiff.server is not importable in this environment.")
        err(INSTRUCTIONS)
        return 2
    emit(style(f"starting llmdiff server on http://{args.host}:{args.port}", "bold_cyan"))
    code = server_main(["--host", args.host, "--port", str(args.port)])
    return int(code) if isinstance(code, int) else 0
