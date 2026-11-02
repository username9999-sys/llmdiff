"""llmdiff server package."""

from __future__ import annotations

from llmdiff.server.storage import RunRecord, RunStorage

__all__ = ["RunStorage", "RunRecord", "create_app"]


def create_app(storage: RunStorage | None = None):
    """Build the FastAPI app (lazy import so the package imports without fastapi)."""
    from llmdiff.server.app import create_app as _create_app

    return _create_app(storage)
