"""Example 03 — replay a corpus against a mock provider and compare.

Run:  PYTHONPATH=src python3 examples/03_replay_model_swap.py
Fully offline: uses ScriptedProvider, no network.
"""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.replay.mock_provider import ScriptedProvider
from llmdiff.replay.runner import replay


def main() -> int:
    corpus = Corpus.from_records([
        {"id": "c1", "prompt": "summarize: the sky is blue", "baseline_output": "", "candidate_output": ""},
        {"id": "c2", "prompt": "2+2?", "baseline_output": "", "candidate_output": ""},
        {"id": "c3", "prompt": "capital of France?", "baseline_output": "", "candidate_output": ""},
    ])
    new_provider = ScriptedProvider(["the sky is blue", "4", "Paris"])
    old_provider = ScriptedProvider(["the sky is blue", "5", "Paris"])
    replayed, _stats = replay(corpus, new_provider)
    baseline, _stats2 = replay(corpus, old_provider)
    merged = baseline.merge_candidates(replayed)
    result = evaluate(merged, dimensions=("exact", "similarity", "content"))
    m = result.matrix
    print(f"verdict: {m.verdict}  W/T/L: {m.wins}/{m.ties}/{m.losses}")
    for d in m.dimensions:
        print(f"  {d.dimension:<12} W={d.wins} T={d.ties} L={d.losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
