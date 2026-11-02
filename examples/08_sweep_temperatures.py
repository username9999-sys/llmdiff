"""Example 08 — parameter sweep with a mock provider.

Run:  PYTHONPATH=src python3 examples/08_sweep_temperatures.py
Fully offline.
"""

from __future__ import annotations

from llmdiff import Corpus
from llmdiff.replay.mock_provider import ScriptedProvider
from llmdiff.replay.params import expand_sweep, sweep


def main() -> int:
    corpus = Corpus.from_records([
        {"id": "c1", "prompt": "hi", "baseline_output": "", "candidate_output": ""},
        {"id": "c2", "prompt": "yo", "baseline_output": "", "candidate_output": ""},
    ])
    spec = {"temperature": [0.0, 0.7, 1.0]}
    configs = expand_sweep({}, spec)
    print(f"{len(configs)} configs: {[c['temperature'] for c in configs]}")
    results = sweep(corpus, ScriptedProvider(["ok", "ok"]), spec)
    for name, c in results.items():
        print(f"  {name}: {len(c)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
