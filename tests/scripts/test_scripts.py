"""Tests for the scripts/ helpers (run as subprocesses)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _run_py(name: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=60,
    )


def test_gen_example_corpus(tmp_path):
    out = tmp_path / "c.jsonl"
    proc = _run_py("gen_example_corpus.py", "20", str(out))
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 20
    rec = json.loads(lines[0])
    assert {"id", "prompt", "baseline_output", "candidate_output"} <= set(rec)


def test_check_docs_links():
    proc = _run_py("check_docs_links.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all local doc links resolve" in proc.stdout


def test_count_loc():
    proc = _run_py("count_loc.py")
    assert proc.returncode == 0
    assert "TOTAL" in proc.stdout
    assert "src" in proc.stdout


def test_shell_scripts_parse():
    for name in ("dev.sh", "lint.sh", "bench.sh", "release.sh"):
        path = SCRIPTS / name
        assert path.is_file(), name
        proc = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, timeout=15
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr}"
