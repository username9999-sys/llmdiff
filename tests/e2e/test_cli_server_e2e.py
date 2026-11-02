"""End-to-end tests: CLI subprocess + server API, all offline."""

from __future__ import annotations

import json

from llmdiff.e2e.cli_e2e import run_cli, sample_records, write_corpus
from llmdiff.e2e.server_e2e import RECORDS, make_client


def test_cli_demo_exit_0():
    proc = run_cli(["demo"])
    assert proc.returncode == 0
    assert "verdict" in proc.stdout.lower() or "matrix" in proc.stdout.lower()


def test_cli_init_writes_parseable_jsonl(tmp_path):
    out = tmp_path / "corpus.jsonl"
    proc = run_cli(["init", "--out", str(out)])
    assert proc.returncode == 0
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) >= 3
    for line in lines:
        rec = json.loads(line)
        assert {"id", "prompt", "baseline_output", "candidate_output"} <= set(rec)


def test_cli_compare_exit_0_and_gate_exit_1(tmp_path):
    corpus = write_corpus(tmp_path / "c.jsonl", sample_records())
    ok = run_cli(["compare", str(corpus)])
    assert ok.returncode == 0
    gated = run_cli(["compare", str(corpus), "--max-loss-rate", "0.0"])
    assert gated.returncode == 1
    assert "GATE FAIL" in gated.stdout


def test_cli_compare_json_output(tmp_path):
    corpus = write_corpus(tmp_path / "c.jsonl", sample_records())
    out = tmp_path / "report.json"
    proc = run_cli(["compare", str(corpus), "--format", "json", "--output", str(out)])
    assert proc.returncode == 0
    payload = json.loads(out.read_text())
    assert "matrix" in payload
    assert payload["matrix"]["total_cases"] == 2


def test_cli_stats(tmp_path):
    corpus = write_corpus(tmp_path / "c.jsonl", sample_records())
    proc = run_cli(["stats", str(corpus)])
    assert proc.returncode == 0
    assert "2" in proc.stdout


def test_cli_export_import_roundtrip(tmp_path):
    corpus = write_corpus(tmp_path / "c.jsonl", sample_records())
    csv_out = tmp_path / "c.csv"
    proc = run_cli(["export", str(corpus), "--to", "csv", "--output", str(csv_out)])
    assert proc.returncode == 0
    back = tmp_path / "back.jsonl"
    proc2 = run_cli(["import", str(csv_out), "--from", "csv", "--out", str(back)])
    assert proc2.returncode == 0
    records = [json.loads(line) for line in back.read_text().splitlines() if line.strip()]
    assert len(records) == 2


def test_cli_version():
    proc = run_cli(["version"])
    assert proc.returncode == 0
    assert "0.1.0" in proc.stdout


def test_server_full_api_flow():
    client = make_client()
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200

    resp = client.post("/api/runs", json={"records": RECORDS})
    assert resp.status_code == 201
    run = resp.json()
    run_id = run["id"]
    assert run["result"]["matrix"]["total_cases"] == 2

    listing = client.get("/api/runs").json()
    assert listing["count"] == 1

    matrix = client.get(f"/api/runs/{run_id}/matrix").json()
    dims = {d["dimension"] for d in matrix["dimensions"]}
    assert "structure" in dims
    # c1 dropped a field -> structure must regress
    assert any(d["dimension"] == "structure" and d["losses"] > 0 for d in matrix["dimensions"])

    assert client.get("/api/runs/nope").status_code == 404
    assert client.delete(f"/api/runs/{run_id}").json() == {"deleted": True}
    assert client.get(f"/api/runs/{run_id}").status_code == 404
