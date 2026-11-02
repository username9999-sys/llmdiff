"""Example 06 — custom reporter plugin.

Run:  PYTHONPATH=src python3 examples/06_custom_reporter.py
Fully offline.
"""

from __future__ import annotations

from llmdiff import Corpus, evaluate


class OneLineReporter:
    @property
    def name(self) -> str:
        return "oneline"

    def render(self, result) -> str:
        m = result.matrix
        return f"verdict={m.verdict} W={m.wins} T={m.ties} L={m.losses} rate={m.regression_rate:.3f}"


def main() -> int:
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello", "candidate_output": "hello"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world", "candidate_output": "hi"},
    ])
    result = evaluate(corpus, dimensions=("exact", "similarity"))
    print(OneLineReporter().render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
