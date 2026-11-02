"""Tests for llmdiff.capture.filters."""

from __future__ import annotations

import pytest

from llmdiff.capture.errors import FilterError
from llmdiff.capture.filters import (
    FilterChain,
    InvertFilter,
    ModelFilter,
    OutputLengthFilter,
    PredicateFilter,
    PromptLengthFilter,
    TagFilter,
    apply_filters,
)


def _rec(prompt="hello", output="world", model="m1", tags=("a",)):
    return {
        "id": "x",
        "prompt": prompt,
        "baseline_output": "",
        "candidate_output": output,
        "metadata": {"model": model, "tags": list(tags)},
    }


def test_tag_filter_any_and_all():
    recs = [_rec(tags=("a", "b")), _rec(tags=("b",)), _rec(tags=("c",))]
    any_f = TagFilter(["a", "c"])
    assert [r["metadata"]["tags"] for r in apply_filters(recs, any_f)] == [["a", "b"], ["c"]]
    all_f = TagFilter(["a", "b"], match_all=True)
    assert len(apply_filters(recs, all_f)) == 1


def test_tag_filter_empty_rejected():
    with pytest.raises(FilterError):
        TagFilter([])


def test_model_filter():
    recs = [_rec(model="m1"), _rec(model="m2"), _rec(model="m1")]
    out = apply_filters(recs, ModelFilter(["m1"]))
    assert len(out) == 2


def test_prompt_length_filter():
    recs = [_rec(prompt="ab"), _rec(prompt="abcdef"), _rec(prompt="abcdefgh")]
    f = PromptLengthFilter(3, 6)
    out = apply_filters(recs, f)
    assert [r["prompt"] for r in out] == ["abcdef"]


def test_output_length_filter():
    recs = [_rec(output="x"), _rec(output="yyyy")]
    out = apply_filters(recs, OutputLengthFilter(min_len=2))
    assert len(out) == 1 and out[0]["candidate_output"] == "yyyy"


def test_length_filter_validation():
    with pytest.raises(FilterError):
        PromptLengthFilter(10, 5)
    with pytest.raises(FilterError):
        OutputLengthFilter(-1)


def test_predicate_filter():
    f = PredicateFilter(lambda r: r["prompt"].startswith("h"), name="starts_h")
    assert f.keep(_rec(prompt="hi")) is True
    assert f.keep(_rec(prompt="bye")) is False
    with pytest.raises(FilterError):
        PredicateFilter(42)  # type: ignore[arg-type]


def test_invert_filter():
    f = InvertFilter(TagFilter(["a"]))
    assert f.keep(_rec(tags=("a",))) is False
    assert f.keep(_rec(tags=("b",))) is True


def test_filter_chain_and_short_circuit():
    chain = FilterChain([TagFilter(["a"]), ModelFilter(["m1"]), PromptLengthFilter(1, 100)])
    assert chain.keep(_rec()) is True
    assert chain.keep(_rec(model="m2")) is False
    assert chain.keep(_rec(tags=("z",))) is False


def test_filter_chain_rejects_non_filter():
    with pytest.raises(FilterError):
        FilterChain([object()])


def test_apply_filters_none_and_sequence():
    recs = [_rec(), _rec(model="m2")]
    assert len(apply_filters(recs, None)) == 2
    out = apply_filters(recs, [TagFilter(["a"]), ModelFilter(["m2"])])
    assert len(out) == 1 and out[0]["metadata"]["model"] == "m2"


def test_record_without_metadata():
    rec = {"id": "1", "prompt": "p", "baseline_output": "", "candidate_output": "o", "metadata": {}}
    assert TagFilter(["a"]).keep(rec) is False
    assert ModelFilter(["m"]).keep(rec) is False
    assert PromptLengthFilter(0, 10).keep(rec) is True
