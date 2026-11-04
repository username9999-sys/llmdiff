"""Tests for structure_data_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.structure_data_deep import (
    JsonArrayLeafCountCheck,
    JsonAvgArrayLenCheck,
    JsonBoolLeafCountCheck,
    JsonDuplicateKeyRiskCheck,
    JsonEmptyArrayCountCheck,
    JsonEmptyObjectCountCheck,
    JsonKeyLengthStatsCheck,
    JsonKeyOrderMatchCheck,
    JsonLeafCountCheck,
    JsonMaxArrayLenCheck,
    JsonMinArrayLenCheck,
    JsonNestedDepthStatsCheck,
    JsonNullLeafCountCheck,
    JsonNumberLeafCountCheck,
    JsonObjectLeafCountCheck,
    JsonPathSetCheck,
    JsonPrettyPrintCheck,
    JsonStringLeafCountCheck,
    JsonTopLevelKeySetCheck,
    JsonUnicodeEscapeCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

J = '{"a": 1, "b": {"c": [1, 2, 3], "d": "x"}, "e": null, "f": true}'


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_json_path_set():
    ctx = _ctx(J, J)
    assert JsonPathSetCheck().score(J, ctx) == 1.0
    assert JsonPathSetCheck().score('{"a": 1}', ctx) < 1.0


def test_json_leaf_count():
    ctx = _ctx(J, J)
    assert JsonLeafCountCheck().score(J, ctx) == 1.0


def test_array_len_stats():
    ctx = _ctx(J, J)
    assert JsonMaxArrayLenCheck().score(J, ctx) == 1.0
    assert JsonMinArrayLenCheck().score(J, ctx) == 1.0
    assert JsonAvgArrayLenCheck().score(J, ctx) == 1.0


def test_empty_counts():
    ctx = _ctx('{"a": [], "b": {}}', '{"a": [], "b": {}}')
    assert JsonEmptyArrayCountCheck().score('{"a": []}', ctx) == 1.0
    assert JsonEmptyObjectCountCheck().score('{"b": {}}', ctx) == 1.0


def test_key_order_and_top_keys():
    ctx = _ctx('{"a": 1, "b": 2}', '{"a": 1, "b": 2}')
    assert JsonKeyOrderMatchCheck().score('{"a": 1, "b": 2}', ctx) == 1.0
    assert JsonKeyOrderMatchCheck().score('{"b": 2, "a": 1}', ctx) < 1.0
    assert JsonTopLevelKeySetCheck().score('{"a": 1, "b": 2}', ctx) == 1.0


def test_depth_stats():
    ctx = _ctx(J, J)
    assert JsonNestedDepthStatsCheck().score(J, ctx) == 1.0


def test_leaf_type_counts():
    ctx = _ctx(J, J)
    assert JsonStringLeafCountCheck().score(J, ctx) == 1.0
    assert JsonNumberLeafCountCheck().score(J, ctx) == 1.0
    assert JsonBoolLeafCountCheck().score(J, ctx) == 1.0
    assert JsonNullLeafCountCheck().score(J, ctx) == 1.0
    assert JsonArrayLeafCountCheck().score(J, ctx) == 1.0
    assert JsonObjectLeafCountCheck().score(J, ctx) == 1.0


def test_dup_key_risk():
    ctx = _ctx('{"a": 1, "a": 2}', '{"a": 1, "a": 2}')
    assert JsonDuplicateKeyRiskCheck().score('{"a": 1, "a": 2}', ctx) == 1.0
    assert JsonDuplicateKeyRiskCheck().score('{"a": 1}', ctx) < 1.0


def test_unicode_escapes():
    ctx = _ctx('{"a": "\\u00e9"}', '{"a": "\\u00e9"}')
    assert JsonUnicodeEscapeCountCheck().score('{"a": "\\u00e9"}', ctx) == 1.0


def test_pretty_print():
    pretty = '{\n  "a": 1\n}'
    ctx = _ctx(pretty, pretty)
    assert JsonPrettyPrintCheck().score(pretty, ctx) == 1.0
    assert JsonPrettyPrintCheck().score('{"a": 1}', ctx) == 0.0


def test_key_len_avg():
    ctx = _ctx('{"aa": 1, "bb": 2}', '{"aa": 1, "bb": 2}')
    assert JsonKeyLengthStatsCheck().score('{"aa": 1, "bb": 2}', ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": J,
         "candidate_output": J},
        {"id": "b", "prompt": "p", "baseline_output": J,
         "candidate_output": '{"a": 1}'},
    ])
    result = evaluate(corpus, dimensions=("json_path_set", "json_leaf_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["json_path_set"].losses == 1
    assert dims["json_leaf_count"].losses == 1
