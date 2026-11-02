"""Predefined benchmark scenarios that inject known regressions.

Each scenario builds a (baseline, candidate) corpus pair from a generator and
applies a specific defect, so the benchmark can measure whether llmdiff's
verdict matrix actually detects it.
"""

from __future__ import annotations

import json

from llmdiff.benchmark.corpus_gen import GENERATORS, gen_corpus

__all__ = [
    "Scenario",
    "SCENARIOS",
    "build_scenario",
    "inject_truncation",
    "inject_field_drop",
    "inject_refusal",
    "inject_latency_regression",
    "inject_token_bloat",
]


class Scenario:
    """A named benchmark scenario with its corpus records."""

    __slots__ = ("name", "kind", "n", "noise", "seed", "inject")

    def __init__(
        self,
        name: str,
        kind: str,
        n: int,
        *,
        noise: float = 0.0,
        seed: int = 0,
        inject: str | None = None,
    ) -> None:
        if kind not in GENERATORS:
            raise ValueError(f"unknown corpus kind {kind!r}")
        self.name = name
        self.kind = kind
        self.n = n
        self.noise = noise
        self.seed = seed
        self.inject = inject

    def records(self) -> list[dict]:
        """Generate the corpus records with the scenario's injection applied."""
        recs = gen_corpus(self.kind, self.n, seed=self.seed, noise=self.noise)
        if self.inject:
            recs = _INJECTORS[self.inject](recs)
        return recs


def inject_truncation(records: list[dict]) -> list[dict]:
    """Truncate every candidate output mid-sentence."""
    out = []
    for r in records:
        cand = r["candidate_output"]
        r = dict(r)
        r["candidate_output"] = cand[: max(1, len(cand) // 3)] + "..."
        out.append(r)
    return out


def inject_field_drop(records: list[dict]) -> list[dict]:
    """Drop one JSON field from every candidate that is JSON."""
    out = []
    for r in records:
        cand = r["candidate_output"]
        r = dict(r)
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and data:
                data.pop(next(iter(data)))
                r["candidate_output"] = json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            pass
        out.append(r)
    return out


def inject_refusal(records: list[dict]) -> list[dict]:
    """Replace every candidate output with a refusal phrase."""
    out = []
    for r in records:
        r = dict(r)
        r["candidate_output"] = "I'm sorry, I cannot help with that."
        out.append(r)
    return out


def inject_latency_regression(records: list[dict]) -> list[dict]:
    """Add latency metadata showing the candidate is 5x slower."""
    out = []
    for r in records:
        r = dict(r)
        meta = dict(r.get("metadata") or {})
        meta["baseline_ms"] = 100.0
        meta["candidate_ms"] = 500.0
        r["metadata"] = meta
        out.append(r)
    return out


def inject_token_bloat(records: list[dict]) -> list[dict]:
    """Add token metadata showing the candidate uses 10x more tokens."""
    out = []
    for r in records:
        r = dict(r)
        meta = dict(r.get("metadata") or {})
        meta["baseline_tokens_out"] = 50
        meta["candidate_tokens_out"] = 500
        r["metadata"] = meta
        out.append(r)
    return out


_INJECTORS = {
    "truncation": inject_truncation,
    "field_drop": inject_field_drop,
    "refusal": inject_refusal,
    "latency": inject_latency_regression,
    "token_bloat": inject_token_bloat,
}

SCENARIOS: dict[str, Scenario] = {
    "clean_json": Scenario("clean_json", "json_extraction", 50, noise=0.0, seed=1),
    "field_drop_json": Scenario("field_drop_json", "json_extraction", 50, inject="field_drop", seed=2),
    "truncated_summary": Scenario("truncated_summary", "summarization", 50, inject="truncation", seed=3),
    "refusal_all": Scenario("refusal_all", "classification", 40, inject="refusal", seed=4),
    "broken_code": Scenario("broken_code", "code_gen", 40, noise=0.6, seed=5),
    "latency_regression": Scenario("latency_regression", "summarization", 30, inject="latency", seed=6),
    "token_bloat": Scenario("token_bloat", "summarization", 30, inject="token_bloat", seed=7),
}


def build_scenario(name: str) -> list[dict]:
    """Build the corpus records for a named scenario."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name].records()
