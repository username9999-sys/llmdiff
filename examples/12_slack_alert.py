"""Example 12 — Slack Block Kit alert payload for a regressed run.

Run:  PYTHONPATH=src python3 examples/12_slack_alert.py
Fully offline.
"""

from __future__ import annotations

import json

from llmdiff import Corpus, evaluate


def main() -> int:
    from llmdiff.reporters.slack import SlackReporter

    corpus = Corpus.from_records([
        {"id": "k1", "prompt": "p", "baseline_output": "good answer", "candidate_output": "good answer"},
        {"id": "k2", "prompt": "p", "baseline_output": "good answer", "candidate_output": "bad"},
    ])
    result = evaluate(corpus, dimensions=("exact", "similarity"))
    payload = json.loads(SlackReporter().render(result))
    assert payload["blocks"], "expected Slack blocks"
    assert "loss" in json.dumps(payload)
    print("Slack payload ok:", len(payload["blocks"]), "blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
