"""Example 04 — CI gate: exit 1 when loss rate exceeds the budget.

Run:  PYTHONPATH=src python3 examples/04_ci_gate.py
Fully offline.
"""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.cli.gate import GateThresholds, gate_from_result
from llmdiff.core.errors import RegressionGateError


def main() -> int:
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": '{"a": 1, "b": 2}',
         "candidate_output": '{"a": 1, "b": 2}'},
        {"id": "b", "prompt": "p", "baseline_output": '{"a": 1, "b": 2}',
         "candidate_output": '{"a": 1}'},
    ])
    result = evaluate(corpus, dimensions=("structure",))
    thresholds = GateThresholds(max_loss_rate=0.0)
    try:
        gate_from_result(result, thresholds)
    except RegressionGateError as exc:
        print("GATE FAIL:")
        for f in exc.failures:
            print(" -", f)
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
