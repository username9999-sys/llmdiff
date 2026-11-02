"""Tests for benchmark.corpus_gen — determinism and noise behavior."""

from __future__ import annotations

import pytest

from llmdiff import Corpus, evaluate
from llmdiff.benchmark.corpus_gen import GENERATORS, gen_corpus, gen_json_extraction


def test_deterministic_same_seed():
    assert gen_json_extraction(20, seed=42) == gen_json_extraction(20, seed=42)


def test_different_seed_differs():
    assert gen_json_extraction(20, seed=1) != gen_json_extraction(20, seed=2)


def test_noise_zero_means_identical_outputs():
    for rec in gen_json_extraction(50, seed=0, noise=0.0):
        assert rec["baseline_output"] == rec["candidate_output"]


def test_noise_one_always_differs():
    recs = gen_json_extraction(50, seed=0, noise=1.0)
    assert all(r["baseline_output"] != r["candidate_output"] for r in recs)


def test_all_generators_produce_valid_corpus():
    for kind in GENERATORS:
        corpus = Corpus.from_records(gen_corpus(kind, 10, seed=0))
        assert len(corpus) == 10


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        gen_corpus("nope", 5)


def test_noise_injection_detected_by_engine():
    clean = Corpus.from_records(gen_corpus("json_extraction", 40, seed=1, noise=0.0))
    noisy = Corpus.from_records(gen_corpus("json_extraction", 40, seed=1, noise=0.5))
    r_clean = evaluate(clean, dimensions=("structure",))
    r_noisy = evaluate(noisy, dimensions=("structure",))
    assert r_clean.matrix.get("structure").losses == 0
    assert r_noisy.matrix.get("structure").losses > 0
