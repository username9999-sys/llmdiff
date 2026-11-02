"""Example 01 — quickstart: evaluate the bundled example corpus and print the matrix.

Run:  PYTHONPATH=src python3 examples/01_quickstart_compare.py
Fully offline.
"""

from __future__ import annotations

from pathlib import Path

from llmdiff import Corpus, evaluate

HERE = Path(__file__).resolve().parent


def main() -> int:
    corpus = Corpus.from_jsonl(HERE / "corpus.example.jsonl")
    result = evaluate(corpus)
    m = result.matrix
    print(f"cases: {m.total_cases}  verdict: {m.verdict}  W/T/L: {m.wins}/{m.ties}/{m.losses}")
    print(f"{'dimension':<18} {'win':>4} {'tie':>4} {'loss':>4}  {'delta':>7}")
    for d in m.dimensions:
        print(f"{d.dimension:<18} {d.wins:>4} {d.ties:>4} {d.losses:>4}  {d.delta:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
