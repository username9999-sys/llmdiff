"""Example 13 — run the self-benchmark suite.

Run:  PYTHONPATH=src python3 examples/13_benchmark_self.py
Fully offline.
"""

from __future__ import annotations

from llmdiff.benchmark.cli_hooks import bench


def main() -> int:
    out = bench(dimensions=("exact", "structure", "content", "latency", "tokens"))
    m = out["metrics"]
    print(
        f"scenarios={m['n_scenarios']} detection={m['detection_rate']:.0%} "
        f"fpr={m['false_positive_rate']:.0%} cases={m['total_cases']}"
    )
    assert m["detection_rate"] > 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
