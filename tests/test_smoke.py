"""Smoke test: the core package imports and evaluates a corpus end to end."""

import json

from llmdiff import Corpus, evaluate

CORPUS = [
    {
        "id": "c1",
        "prompt": "hi",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m1", "tags": ["x"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "c2",
        "prompt": "yo",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
]


def test_evaluate_end_to_end():
    corpus = Corpus.from_records(CORPUS)
    result = evaluate(corpus)
    assert len(result.cases) == 2
    dims = {d.dimension for d in result.matrix.dimensions}
    assert {"exact", "similarity", "length", "content", "json_valid", "structure"} <= dims
    c1 = result.case("c1")
    assert c1 is not None
    structure = next(r for r in c1.results if r.dimension == "structure")
    assert structure.verdict.value == "loss"
    c2 = result.case("c2")
    assert c2 is not None
    exact = next(r for r in c2.results if r.dimension == "exact")
    assert exact.verdict.value == "loss"


if __name__ == "__main__":
    test_evaluate_end_to_end()
    print("SMOKE OK")
