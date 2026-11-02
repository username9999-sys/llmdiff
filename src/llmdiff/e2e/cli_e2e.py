"""Helpers to run the llmdiff CLI as a subprocess for e2e tests (offline)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["run_cli", "write_corpus", "sample_records", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_cli(
    args: list[str], *, cwd: Path | None = None, timeout: float = 60.0
) -> subprocess.CompletedProcess:
    """Run ``llmdiff.cli:main`` in a subprocess with PYTHONPATH=src."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", "from llmdiff.cli import main; raise SystemExit(main())", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        timeout=timeout,
    )


def write_corpus(path: Path, records: list[dict]) -> Path:
    """Write corpus records as JSONL and return the path."""
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def sample_records() -> list[dict]:
    """A small corpus with one regression (dropped JSON field on c2)."""
    return [
        {
            "id": "c1",
            "prompt": "extract",
            "baseline_output": json.dumps({"a": 1, "b": 2}),
            "candidate_output": json.dumps({"a": 1, "b": 2}),
            "metadata": {"model": "m", "tags": ["json"], "timestamp": "2026-11-01T12:00:00Z"},
        },
        {
            "id": "c2",
            "prompt": "extract",
            "baseline_output": json.dumps({"a": 1, "b": 2}),
            "candidate_output": json.dumps({"a": 1}),
            "metadata": {"model": "m", "tags": ["json"], "timestamp": "2026-11-01T12:00:00Z"},
        },
    ]
