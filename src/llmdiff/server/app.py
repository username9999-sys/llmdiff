"""FastAPI application exposing evaluation runs over HTTP.

Endpoints:
    GET    /                     -> dashboard HTML (static)
    GET    /api/health           -> {"status": "ok"}
    POST   /api/runs             -> evaluate a corpus (JSON body) and store the run
    GET    /api/runs             -> list recent runs
    GET    /api/runs/{id}        -> full run result (DiffResult JSON)
    GET    /api/runs/{id}/matrix -> just the verdict matrix
    DELETE /api/runs/{id}        -> delete a run

The app factory takes a RunStorage so tests can use an in-memory database.
"""

from __future__ import annotations

from typing import Any

from llmdiff.core.corpus import Corpus
from llmdiff.core.engine import DiffConfig, evaluate
from llmdiff.server.storage import RunStorage

__all__ = ["create_app"]

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmdiff dashboard</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 2rem; background: #0f1115; color: #d7dae0; }
  h1 { font-size: 1.2rem; }
  table { border-collapse: collapse; margin-top: 1rem; }
  th, td { border: 1px solid #2a2f3a; padding: 0.35rem 0.7rem; text-align: right; }
  th { background: #171b22; }
  td:first-child, th:first-child { text-align: left; }
  .loss { color: #ff6b6b; } .win { color: #69db7c; } .tie { color: #adb5bd; }
  a { color: #74c0fc; }
</style>
</head>
<body>
<h1>llmdiff &mdash; verdict matrix dashboard</h1>
<p>API: <code>GET /api/runs</code> &middot; <code>POST /api/runs</code> &middot; <code>GET /api/runs/{id}</code></p>
<div id="runs">loading&hellip;</div>
<script>
fetch('/api/runs?limit=20').then(r => r.json()).then(data => {
  const el = document.getElementById('runs');
  if (!data.runs.length) { el.textContent = 'No runs yet. POST a corpus to /api/runs.'; return; }
  let html = '<table><tr><th>run</th><th>created</th><th>verdict</th><th>win</th><th>tie</th><th>loss</th></tr>';
  for (const run of data.runs) {
    const m = run.result.matrix;
    html += `<tr><td><a href="/api/runs/${run.id}">${run.id}</a></td><td>${run.created_at}</td>` +
      `<td class="${m.verdict}">${m.verdict}</td><td>${m.wins}</td><td>${m.ties}</td><td class="loss">${m.losses}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
});
</script>
</body>
</html>
"""


def create_app(storage: RunStorage | None = None):
    """Build the FastAPI app. Returns the app instance.

    Raises:
        ImportError: If fastapi is not installed; the caller should fall back
            to a message telling the user to install fastapi and uvicorn.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    storage = storage if storage is not None else RunStorage()
    app = FastAPI(title="llmdiff server", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs", status_code=201)
    def create_run(body: dict[str, Any]) -> JSONResponse:
        records = body.get("records") or []
        dimensions = body.get("dimensions")
        tolerance = float(body.get("tolerance", 0.0))
        corpus = Corpus.from_records(records)
        config = DiffConfig(
            dimensions=tuple(dimensions) if dimensions else None,
            tolerance=tolerance,
        )
        result = evaluate(corpus, dimensions=config.dimensions, tolerance=config.tolerance)
        record = storage.save(result.to_dict(), config.to_dict())
        return JSONResponse(record.to_dict(), status_code=201)

    @app.get("/api/runs")
    def list_runs(limit: int = 100, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        runs = storage.list(limit=limit, offset=offset)
        return {"runs": [r.to_dict() for r in runs], "count": storage.count()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = storage.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return record.to_dict()

    @app.get("/api/runs/{run_id}/matrix")
    def get_matrix(run_id: str) -> dict[str, Any]:
        record = storage.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return record.result["matrix"]

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, bool]:
        if not storage.delete(run_id):
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return {"deleted": True}

    return app
