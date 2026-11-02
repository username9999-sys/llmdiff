"""Example 15 — end-to-end pipeline against a mock LLM server.

Run:  PYTHONPATH=src python3 examples/15_e2e_mock_llm.py
Fully offline.
"""

from __future__ import annotations

from llmdiff.e2e.harness import run_e2e


def main() -> int:
    res = run_e2e(
        ["p1", "p2", "p3", "p4"],
        responses={"p1": "alpha", "p2": "beta", "p3": "gamma", "p4": "delta"},
        candidate_responses={"p1": "alpha", "p2": "WRONG", "p3": "WRONG", "p4": "delta"},
        dimensions=("exact", "similarity"),
    )
    m = res.result.matrix
    print(f"e2e cases={res.n_cases} verdict={m.verdict} W/T/L={m.wins}/{m.ties}/{m.losses}")
    assert m.get("exact").losses == 2
    print("e2e regression detected: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
