"""Example 07 — trend analysis across several runs.

Run:  PYTHONPATH=src python3 examples/07_trend_analysis.py
Fully offline.
"""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.analytics.report import build_report, render_markdown
from llmdiff.analytics.trends import linear_regression, r_squared


def _corpus(noise_rate: float):
    return Corpus.from_records([
        {
            "id": f"c{i}",
            "prompt": "p",
            "baseline_output": '{"a": 1, "b": 2}',
            "candidate_output": '{"a": 1}' if i % 10 < noise_rate * 10 else '{"a": 1, "b": 2}',
        }
        for i in range(40)
    ])


def main() -> int:
    results = [evaluate(_corpus(n), dimensions=("structure",)) for n in (0.0, 0.2, 0.5, 0.8)]
    series = [r.matrix.get("structure").loss_rate for r in results]
    intercept, sl = linear_regression(series)
    print(f"structure loss-rate series: {[round(s, 3) for s in series]}")
    print(f"slope={sl:+.4f} r^2={r_squared(series):.3f}")
    report = build_report(results)
    md = render_markdown(report)
    assert "trend: structure" in md
    print("report rendered,", len(md), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
