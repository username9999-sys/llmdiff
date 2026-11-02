"""Fixture corpora for e2e tests: small JSONL strings plus a big generated one."""

from __future__ import annotations

import json

from llmdiff.benchmark.corpus_gen import gen_corpus

__all__ = [
    "FIXTURE_SMALL",
    "FIXTURE_REGRESSION",
    "FIXTURE_JSONL_TEXT",
    "big_corpus_records",
    "fixture_jsonl_text",
]

FIXTURE_SMALL: list[dict] = [
    {
        "id": "f1",
        "prompt": "say hi",
        "baseline_output": "hello there",
        "candidate_output": "hello there",
        "metadata": {"model": "m", "tags": ["greet"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "f2",
        "prompt": "2+2?",
        "baseline_output": "4",
        "candidate_output": "4",
        "metadata": {"model": "m", "tags": ["math"], "timestamp": "2026-11-01T12:00:00Z"},
    },
]

FIXTURE_REGRESSION: list[dict] = [
    {
        "id": "r1",
        "prompt": "extract json",
        "baseline_output": json.dumps({"a": 1, "b": 2, "c": 3}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m", "tags": ["json"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "r2",
        "prompt": "extract json",
        "baseline_output": json.dumps({"x": 10, "y": 20}),
        "candidate_output": json.dumps({"x": 10}),
        "metadata": {"model": "m", "tags": ["json"], "timestamp": "2026-11-01T12:00:00Z"},
    },
]


def fixture_jsonl_text(records: list[dict]) -> str:
    """Serialise records to JSONL text."""
    return "".join(json.dumps(r) + "\n" for r in records)


FIXTURE_JSONL_TEXT = fixture_jsonl_text(FIXTURE_SMALL)


def big_corpus_records(n: int = 10_000, *, seed: int = 0, kind: str = "summarization") -> list[dict]:
    """Generate a large corpus for perf tests (deterministic)."""
    return gen_corpus(kind, n, seed=seed)
