"""Tests for yaml_deep dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.yaml_deep import (
    YamlAliasCountCheck,
    YamlAnchorCountCheck,
    YamlBoolValueCountCheck,
    YamlCommentCountCheck,
    YamlDocMarkerCountCheck,
    YamlFlowStyleCountCheck,
    YamlFoldedBlockCountCheck,
    YamlIndentWidthCheck,
    YamlKeyCountDeepCheck,
    YamlKeyWithDotCountCheck,
    YamlLiteralBlockCountCheck,
    YamlMaxDepthCheck,
    YamlMergeKeyCountCheck,
    YamlMultilineBlockCountCheck,
    YamlNestedMapCountCheck,
    YamlNestedSeqCountCheck,
    YamlNullValueCountCheck,
    YamlNumberValueCountCheck,
    YamlQuotedStringCountCheck,
    YamlTagCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

YAML = (
    "---\n"
    "# a comment\n"
    "base: &base\n"
    "  x: 1\n"
    "derived:\n"
    "  <<: *base\n"
    "  y: true\n"
    "  z: null\n"
    "  n: 42\n"
    '  s: "quoted"\n'
    "  text: |\n"
    "    line1\n"
    "  folded: >\n"
    "    f1\n"
    "list: [1, 2]\n"
    "items:\n"
    "  - a\n"
    "  - b\n"
    "a.b: 1\n"
    "!!str tagged: x\n"
)


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_anchors_aliases_merge():
    ctx = _ctx(YAML, YAML)
    assert YamlAnchorCountCheck().score(YAML, ctx) == 1.0
    assert YamlAliasCountCheck().score(YAML, ctx) == 1.0
    assert YamlMergeKeyCountCheck().score(YAML, ctx) == 1.0
    assert YamlAnchorCountCheck().score("no anchors", ctx) < 1.0


def test_blocks_comments_docs_tags():
    ctx = _ctx(YAML, YAML)
    assert YamlMultilineBlockCountCheck().score(YAML, ctx) == 1.0
    assert YamlLiteralBlockCountCheck().score(YAML, ctx) == 1.0
    assert YamlFoldedBlockCountCheck().score(YAML, ctx) == 1.0
    assert YamlCommentCountCheck().score(YAML, ctx) == 1.0
    assert YamlDocMarkerCountCheck().score(YAML, ctx) == 1.0
    assert YamlTagCountCheck().score(YAML, ctx) == 1.0


def test_quoted_flow_nested():
    ctx = _ctx(YAML, YAML)
    assert YamlQuotedStringCountCheck().score(YAML, ctx) == 1.0
    assert YamlFlowStyleCountCheck().score(YAML, ctx) == 1.0
    assert YamlNestedMapCountCheck().score(YAML, ctx) == 1.0
    assert YamlNestedSeqCountCheck().score(YAML, ctx) == 1.0


def test_values_and_keys():
    ctx = _ctx(YAML, YAML)
    assert YamlKeyWithDotCountCheck().score(YAML, ctx) == 1.0
    assert YamlBoolValueCountCheck().score(YAML, ctx) == 1.0
    assert YamlNullValueCountCheck().score(YAML, ctx) == 1.0
    assert YamlNumberValueCountCheck().score(YAML, ctx) == 1.0
    assert YamlKeyCountDeepCheck().score(YAML, ctx) == 1.0


def test_indent_depth():
    ctx = _ctx(YAML, YAML)
    assert YamlIndentWidthCheck().score(YAML, ctx) == 1.0
    assert YamlMaxDepthCheck().score(YAML, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": YAML,
         "candidate_output": YAML},
        {"id": "b", "prompt": "p", "baseline_output": YAML,
         "candidate_output": "key: value\n"},
    ])
    result = evaluate(corpus, dimensions=("yaml_key_count_deep", "yaml_anchor_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["yaml_key_count_deep"].losses == 1
    assert dims["yaml_anchor_count"].losses == 1
