"""Serve the bundled web dashboard (src/llmdiff/web) as static files.

Used by the FastAPI app so `llmdiff serve` shows the SPA without a separate
static server. Falls back gracefully when the web assets are absent.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["WEB_ROOT", "web_file_response", "mount_static"]

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def web_file_response(rel_path: str):
    """Return a FastAPI response for a path under web/, or None if missing.

    ``rel_path`` must be relative and must not escape WEB_ROOT (no ``..``).
    """
    from fastapi.responses import FileResponse, JSONResponse

    if not WEB_ROOT.is_dir():
        return None
    candidate = (WEB_ROOT / rel_path).resolve()
    try:
        candidate.relative_to(WEB_ROOT.resolve())
    except ValueError:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not candidate.is_file():
        return None
    media = _MIME.get(candidate.suffix.lower(), "application/octet-stream")
    return FileResponse(candidate, media_type=media)


def mount_static(app, *, prefix: str = "/app") -> None:
    """Mount GET ``prefix/{path}`` serving files from the web dashboard dir.

    ``GET /app`` and ``GET /app/`` serve index.html. Missing files return 404.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    @app.get(prefix, include_in_schema=False)
    @app.get(prefix + "/", include_in_schema=False)
    def app_index():
        if not WEB_ROOT.is_dir():
            raise HTTPException(status_code=404, detail="web assets not bundled")
        return FileResponse(WEB_ROOT / "index.html")

    @app.get(prefix + "/{path:path}", include_in_schema=False)
    def app_static(path: str):
        resp = web_file_response(path)
        if resp is None:
            raise HTTPException(status_code=404, detail="not found")
        return resp
