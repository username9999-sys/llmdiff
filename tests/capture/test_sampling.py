"""Tests for llmdiff.capture.sampling — deterministic with seeds."""

from __future__ import annotations

import pytest

from llmdiff.capture.errors import SamplingError
from llmdiff.capture.sampling import (
    AlwaysSampler,
    CompositeSampler,
    HeadSampler,
    NeverSampler,
    RandomSampler,
    RateLimitSampler,
    ReservoirSampler,
    StratifiedSampler,
    make_sampler,
)


def _rec(i, tags=()):
    return {"prompt": f"p{i}", "output": "o", "metadata": {"tags": list(tags)}}


def test_always_never():
    assert AlwaysSampler().should_keep(_rec(0)) is True
    assert NeverSampler().should_keep(_rec(0)) is False


def test_rate_limit_keeps_one_in_n():
    s = RateLimitSampler(3)
    results = [s.should_keep(_rec(i)) for i in range(9)]
    assert results == [True, False, False, True, False, False, True, False, False]


def test_rate_limit_invalid():
    with pytest.raises(SamplingError):
        RateLimitSampler(0)


def test_random_sampler_deterministic_with_seed():
    s1 = RandomSampler(0.5, seed=42)
    s2 = RandomSampler(0.5, seed=42)
    r1 = [s1.should_keep(_rec(i)) for i in range(100)]
    r2 = [s2.should_keep(_rec(i)) for i in range(100)]
    assert r1 == r2
    assert any(r1) and not all(r1)  # actually samples


def test_random_sampler_p1_keeps_all():
    s = RandomSampler(1.0)
    assert all(s.should_keep(_rec(i)) for i in range(10))


def test_random_sampler_invalid_p():
    with pytest.raises(SamplingError):
        RandomSampler(0.0)
    with pytest.raises(SamplingError):
        RandomSampler(1.5)


def test_reservoir_keeps_capacity():
    s = ReservoirSampler(10, seed=1)
    for i in range(1000):
        s.should_keep(_rec(i))
    assert s.seen == 1000
    assert len(s.kept) == 10
    # all kept records are distinct prompts
    prompts = {r["prompt"] for r in s.kept}
    assert len(prompts) == 10


def test_reservoir_deterministic():
    def run():
        s = ReservoirSampler(5, seed=7)
        for i in range(50):
            s.should_keep(_rec(i))
        return [r["prompt"] for r in s.kept]

    assert run() == run()


def test_reservoir_invalid_capacity():
    with pytest.raises(SamplingError):
        ReservoirSampler(0)


def test_stratified_per_tag_counters():
    s = StratifiedSampler(2)
    # tag "a" records: keep 1st, 3rd
    a = [s.should_keep(_rec(i, ["a"])) for i in range(4)]
    assert a == [True, False, True, False]
    # tag "b" has its own counter
    b = [s.should_keep(_rec(i, ["b"])) for i in range(2)]
    assert b == [True, False]
    # untagged uses default_n
    u = [s.should_keep(_rec(i)) for i in range(2)]
    assert u == [True, False]
    assert set(s.counts()) == {"a", "b", "__untagged__"}


def test_head_sampler():
    s = HeadSampler(3)
    results = [s.should_keep(_rec(i)) for i in range(6)]
    assert results == [True, True, True, False, False, False]


def test_composite_all_and_any():
    head = HeadSampler(2)
    rate = RateLimitSampler(2)
    both = CompositeSampler([head, rate], mode="all")
    # head keeps 1,2; rate keeps 1,3 -> all keeps only 1
    results = [both.should_keep(_rec(i)) for i in range(4)]
    assert results == [True, False, False, False]

    head2 = HeadSampler(2)
    rate2 = RateLimitSampler(2)
    either = CompositeSampler([head2, rate2], mode="any")
    results2 = [either.should_keep(_rec(i)) for i in range(4)]
    assert results2 == [True, True, True, False]


def test_composite_invalid():
    with pytest.raises(SamplingError):
        CompositeSampler([], mode="all")
    with pytest.raises(SamplingError):
        CompositeSampler([AlwaysSampler()], mode="xor")


def test_make_sampler_from_spec():
    assert isinstance(make_sampler(None), AlwaysSampler)
    assert isinstance(make_sampler({"type": "rate", "n": 5}), RateLimitSampler)
    assert isinstance(make_sampler({"type": "random", "p": 0.5, "seed": 1}), RandomSampler)
    assert isinstance(make_sampler({"type": "reservoir", "capacity": 3}), ReservoirSampler)
    assert isinstance(make_sampler({"type": "stratified", "n": 2}), StratifiedSampler)
    assert isinstance(make_sampler({"type": "head", "n": 2}), HeadSampler)
    assert isinstance(make_sampler({"type": "never"}), NeverSampler)
    comp = make_sampler({"type": "all", "samplers": [{"type": "always"}, {"type": "never"}]})
    assert isinstance(comp, CompositeSampler)
    assert comp.should_keep(_rec(0)) is False
    with pytest.raises(SamplingError):
        make_sampler({"type": "wat"})
    with pytest.raises(SamplingError):
        make_sampler(42)  # type: ignore[arg-type]


def test_make_sampler_passthrough():
    s = RateLimitSampler(2)
    assert make_sampler(s) is s
