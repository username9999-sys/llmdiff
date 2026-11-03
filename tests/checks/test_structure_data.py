"""Tests for structure_data dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.structure_data import (
    CsvCellTypeRatioCheck,
    CsvColumnCountCheck,
    CsvDelimiterMatchCheck,
    CsvHeaderMatchCheck,
    CsvQuotedCellRatioCheck,
    CsvRowCountCheck,
    JsonArrayLengthCheck,
    JsonBoolCountCheck,
    JsonDepthCheck,
    JsonKeySetCheck,
    JsonNestedKeyCountCheck,
    JsonNullCountCheck,
    JsonNumberSumCheck,
    JsonStringLengthSumCheck,
    JsonTypeSignatureCheck,
    JsonValueTypeMatchCheck,
    KeyOrderPreservedCheck,
    YamlDepthCheck,
    YamlKeyCountCheck,
    YamlListItemCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_json_key_set():
    ctx = _ctx('{"a": 1, "b": {"c": 2}}', '{"a": 1, "b": {"c": 2}}')
    assert JsonKeySetCheck().score('{"a": 1, "b": {"c": 2}}', ctx) == 1.0
    assert JsonKeySetCheck().score('{"a": 1}', ctx) < 1.0


def test_json_depth():
    ctx = _ctx('{"a": {"b": {"c": 1}}}', '{"a": {"b": {"c": 1}}}')
    assert JsonDepthCheck().score('{"a": {"b": {"c": 1}}}', ctx) == 1.0
    assert JsonDepthCheck().score('{"a": 1}', ctx) < 1.0


def test_json_array_len():
    ctx = _ctx('{"items": [1, 2, 3]}', '{"items": [1, 2, 3]}')
    assert JsonArrayLengthCheck().score('{"items": [1, 2, 3]}', ctx) == 1.0
    assert JsonArrayLengthCheck().score('{"items": [1]}', ctx) < 1.0


def test_json_type_sig():
    ctx = _ctx('{"a": 1, "b": "x"}', '{"a": 1, "b": "x"}')
    assert JsonTypeSignatureCheck().score('{"a": 1, "b": "x"}', ctx) == 1.0
    assert JsonTypeSignatureCheck().score('{"a": "1", "b": "x"}', ctx) == 0.0


def test_json_value_types():
    ctx = _ctx('{"a": 1}', '{"a": 1}')
    assert JsonValueTypeMatchCheck().score('{"a": 1}', ctx) == 1.0
    assert JsonValueTypeMatchCheck().score('{"a": "1"}', ctx) < 1.0


def test_json_nested_keys():
    ctx = _ctx('{"a": {"b": 1}, "c": 2}', '{"a": {"b": 1}, "c": 2}')
    assert JsonNestedKeyCountCheck().score('{"a": {"b": 1}, "c": 2}', ctx) == 1.0


def test_json_num_sum():
    ctx = _ctx('{"a": 1, "b": 2}', '{"a": 1, "b": 2}')
    assert JsonNumberSumCheck().score('{"a": 1, "b": 2}', ctx) == 1.0
    assert JsonNumberSumCheck().score('{"a": 1, "b": 5}', ctx) < 1.0


def test_json_str_len_sum():
    ctx = _ctx('{"a": "hello"}', '{"a": "hello"}')
    assert JsonStringLengthSumCheck().score('{"a": "hello"}', ctx) == 1.0


def test_json_bool_count():
    ctx = _ctx('{"a": true, "b": false}', '{"a": true, "b": false}')
    assert JsonBoolCountCheck().score('{"a": true, "b": false}', ctx) == 1.0
    assert JsonBoolCountCheck().score('{"a": true}', ctx) < 1.0


def test_json_null_count():
    ctx = _ctx('{"a": null}', '{"a": null}')
    assert JsonNullCountCheck().score('{"a": null}', ctx) == 1.0
    assert JsonNullCountCheck().score('{"a": 1}', ctx) < 1.0


def test_yaml_key_count():
    ctx = _ctx("a: 1\nb: 2", "a: 1\nb: 2")
    assert YamlKeyCountCheck().score("a: 1\nb: 2", ctx) == 1.0
    assert YamlKeyCountCheck().score("a: 1", ctx) < 1.0


def test_yaml_depth():
    ctx = _ctx("a:\n  b:\n    c: 1", "a:\n  b:\n    c: 1")
    assert YamlDepthCheck().score("a:\n  b:\n    c: 1", ctx) == 1.0


def test_yaml_list_items():
    ctx = _ctx("- a\n- b\n- c", "- a\n- b\n- c")
    assert YamlListItemCountCheck().score("- a\n- b\n- c", ctx) == 1.0
    assert YamlListItemCountCheck().score("- a", ctx) < 1.0


def test_csv_col_count():
    ctx = _ctx("a,b,c\n1,2,3", "a,b,c\n1,2,3")
    assert CsvColumnCountCheck().score("a,b,c\n1,2,3", ctx) == 1.0
    assert CsvColumnCountCheck().score("a,b\n1,2", ctx) < 1.0


def test_csv_row_count():
    ctx = _ctx("a,b\n1,2\n3,4", "a,b\n1,2\n3,4")
    assert CsvRowCountCheck().score("a,b\n1,2\n3,4", ctx) == 1.0
    assert CsvRowCountCheck().score("a,b\n1,2", ctx) < 1.0


def test_csv_header():
    ctx = _ctx("name,age\nann,30", "name,age\nann,30")
    assert CsvHeaderMatchCheck().score("name,age\nann,30", ctx) == 1.0
    assert CsvHeaderMatchCheck().score("name,score\nann,30", ctx) < 1.0


def test_csv_numeric_ratio():
    ctx = _ctx("a,b\n1,2", "a,b\n1,2")
    assert CsvCellTypeRatioCheck().score("a,b\n1,2", ctx) == 1.0


def test_csv_delimiter():
    ctx = _ctx("a,b\n1,2", "a,b\n1,2")
    assert CsvDelimiterMatchCheck().score("a,b\n1,2", ctx) == 1.0
    assert CsvDelimiterMatchCheck().score("a;b\n1;2", ctx) == 0.0


def test_csv_quoted_ratio():
    ctx = _ctx('"a","b"\n"1","2"', '"a","b"\n"1","2"')
    assert CsvQuotedCellRatioCheck().score('"a","b"\n"1","2"', ctx) == 1.0


def test_key_order():
    ctx = _ctx('{"a": 1, "b": 2}', '{"a": 1, "b": 2}')
    assert KeyOrderPreservedCheck().score('{"a": 1, "b": 2}', ctx) == 1.0
    assert KeyOrderPreservedCheck().score('{"b": 2, "a": 1}', ctx) < 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": '{"a": 1, "b": [1, 2]}',
         "candidate_output": '{"a": 1, "b": [1, 2]}'},
        {"id": "b", "prompt": "p", "baseline_output": '{"a": 1, "b": [1, 2]}',
         "candidate_output": '{"a": 1}'},
    ])
    result = evaluate(corpus, dimensions=("json_key_set", "json_array_len"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["json_key_set"].losses == 1
    assert dims["json_array_len"].losses == 1
