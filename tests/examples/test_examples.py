"""Tests that run every example script as a subprocess (offline)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


def _run(name: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=120,
    )


OFFLINE_EXAMPLES = [
    "01_quickstart_compare.py",
    "03_replay_model_swap.py",
    "05_custom_check_plugin.py",
    "06_custom_reporter.py",
    "07_trend_analysis.py",
    "08_sweep_temperatures.py",
    "09_json_extraction_regression.py",
    "10_safety_pii_redaction.py",
    "11_sarif_code_scanning.py",
    "12_slack_alert.py",
    "13_benchmark_self.py",
    "15_e2e_mock_llm.py",
]


@pytest.mark.parametrize("name", OFFLINE_EXAMPLES)
def test_example_runs_offline(name):
    proc = _run(name)
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"


def test_example_04_ci_gate_exits_1_on_regression():
    # the CI-gate example is *supposed* to fail the gate (exit 1)
    proc = _run("04_ci_gate.py")
    assert proc.returncode == 1
    assert "GATE FAIL" in proc.stdout
