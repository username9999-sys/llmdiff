"""Example 09 — JSON extraction regression detection.

Run:  PYTHONPATH=src python3 examples/09_json_extraction_regression.py
Fully offline.
"""

from __future__ import annotations

import json

from llmdiff import Corpus, evaluate


def main() -> int:
    baseline = {"order_id": "A-1", "total": 59.99, "currency": "USD", "items": 2}
    candidate_dropped = {"order_id": "A-1", "total": 59.99, "currency": "USD"}  # items dropped
    corpus = Corpus.from_records([
        {"id": "j1", "prompt": "extract", "baseline_output": json.dumps(baseline),
         "candidate_output": json.dumps(candidate_dropped)},
        {"id": "j2", "prompt": "extract", "baseline_output": json.dumps(baseline),
         "candidate_output": json.dumps(baseline)},
    ])
    result = evaluate(corpus, dimensions=("structure", "json_valid"))
    for d in result.matrix.dimensions:
        print(f"{d.dimension:<12} W={d.wins} T={d.ties} L={d.losses}")
    assert result.matrix.get("structure").losses == 1
    print("detected the dropped field: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
