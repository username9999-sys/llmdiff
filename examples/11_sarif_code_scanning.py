"""Example 11 — SARIF output for GitHub code scanning.

Run:  PYTHONPATH=src python3 examples/11_sarif_code_scanning.py
Fully offline.
"""

from __future__ import annotations

import json

from llmdiff import Corpus, evaluate


def main() -> int:
    from llmdiff.reporters.sarif import SarifReporter

    corpus = Corpus.from_records([
        {"id": "s1", "prompt": "p", "baseline_output": "safe answer", "candidate_output": "safe answer"},
        {"id": "s2", "prompt": "p", "baseline_output": "safe answer",
         "candidate_output": "I'm sorry, I cannot help with that."},
    ])
    result = evaluate(corpus, dimensions=("content",))
    sarif = json.loads(SarifReporter().render(result))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"], "expected at least one SARIF result for the regression"
    print(f"SARIF ok: {len(sarif['runs'][0]['results'])} result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
