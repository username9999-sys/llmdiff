"""llmdiff server - optional FastAPI dashboard for evaluation runs.

Run with:
    python -m llmdiff.server --port 8000
or:
    llmdiff serve --port 8000
"""

from __future__ import annotations

import argparse

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Start the dashboard server. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="llmdiff.server", description="llmdiff dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="llmdiff-runs.db", help="SQLite database path")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn fastapi")
        return 2

    from llmdiff.server.app import create_app
    from llmdiff.server.storage import RunStorage

    app = create_app(RunStorage(args.db))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
