"""Tests for parameter sweeps: expand_sweep, config_name, sweep()."""

from __future__ import annotations

from llmdiff.core.corpus import Corpus
from llmdiff.replay.mock_provider import MockProvider
from llmdiff.replay.params import config_name, expand_sweep, sweep


def make_corpus(n: int = 3) -> Corpus:
    return Corpus.from_records(
        [
            {"id": f"c{i}", "prompt": f"p{i}", "baseline_output": "b", "candidate_output": ""}
            for i in range(n)
        ]
    )


def test_expand_sweep_cartesian_product():
    combos = expand_sweep({"seed": 1}, {"temperature": [0.0, 0.7], "model": ["a", "b"]})
    assert len(combos) == 4
    assert {"seed": 1, "temperature": 0.0, "model": "a"} in combos
    assert {"seed": 1, "temperature": 0.7, "model": "b"} in combos


def test_expand_sweep_empty_spec_returns_base():
    assert expand_sweep({"x": 1}, {}) == [{"x": 1}]


def test_config_name_stable_and_sorted():
    assert config_name({"temperature": 0.0, "model": "m"}) == "model='m'|temperature=0.0"
    assert config_name({}) == "default"


def test_sweep_produces_one_corpus_per_config():
    corpus = make_corpus(3)
    provider = MockProvider({"*": "out"})
    results = sweep(
        corpus,
        provider,
        {"temperature": [0.0, 0.7, 1.0]},
        base_params={"seed": 42},
        progress=False,
    )
    assert len(results) == 3
    for name, out_corpus in results.items():
        assert "temperature=" in name
        assert len(out_corpus) == 3
        assert all(c.candidate_output == "out" for c in out_corpus)


def test_sweep_two_dimensions():
    corpus = make_corpus(2)
    results = sweep(
        corpus,
        MockProvider({"*": "x"}),
        {"temperature": [0.0, 1.0], "top_p": [0.9, 1.0]},
        progress=False,
    )
    assert len(results) == 4


def test_sweep_writes_out_dir(tmp_path):
    corpus = make_corpus(2)
    results = sweep(
        corpus,
        MockProvider({"*": "x"}),
        {"temperature": [0.0, 1.0]},
        out_dir=str(tmp_path),
        progress=False,
    )
    files = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert len(files) == 2
    for name in results:
        safe = name.replace("/", "_").replace(" ", "")
        assert (tmp_path / f"{safe}.jsonl").exists()


def test_sweep_params_reach_provider():
    seen: list[dict] = []

    def fn(prompt, **params):
        seen.append(dict(params))
        return f"t={params.get('temperature')}"

    corpus = make_corpus(1)
    results = sweep(
        corpus, MockProvider(fn), {"temperature": [0.0, 0.5]}, progress=False
    )
    temps = sorted(str(s.get("temperature")) for s in seen)
    assert temps == ["0.0", "0.5"]
    outputs = sorted(c.candidate_output for c in results.values() for c in c)
    assert outputs == ["t=0.0", "t=0.5"]
