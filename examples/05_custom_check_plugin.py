"""Example 05 — custom check plugin via the registry.

Run:  PYTHONPATH=src python3 examples/05_custom_check_plugin.py
Fully offline.
"""

from __future__ import annotations

from llmdiff import CheckRegistry, Corpus, DiffConfig, DiffEngine
from llmdiff.core.checks import CheckContext


class NoShoutingCheck:
    """1.0 when the output has no ALL-CAPS words longer than 4 chars."""

    @property
    def name(self) -> str:
        return "no_shouting"

    def score(self, output: str, context: CheckContext) -> float:
        words = output.split()
        shouting = [w for w in words if len(w) > 4 and w.isupper() and w.isalpha()]
        return 1.0 if not shouting else max(0.0, 1.0 - len(shouting) / 5.0)


def main() -> int:
    registry = CheckRegistry([NoShoutingCheck()])
    engine = DiffEngine(registry=registry, config=DiffConfig(dimensions=("no_shouting",)))
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "calm answer", "candidate_output": "calm answer"},
        {"id": "b", "prompt": "p", "baseline_output": "calm answer", "candidate_output": "STOP NOW PLEASE"},
    ])
    result = engine.run(corpus)
    d = result.matrix.get("no_shouting")
    print(f"no_shouting: W={d.wins} T={d.ties} L={d.losses}")
    assert d.losses == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
