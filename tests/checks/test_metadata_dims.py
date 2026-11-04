"""Tests for metadata_dims dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.metadata_dims import (
    AvgLineLengthCheck,
    BlankLineRatioCheck,
    CharsPerTokenCheck,
    CodeToTextRatioCheck,
    CsvLikelihoodCheck,
    DiffLikelihoodCheck,
    HtmlLikelihoodCheck,
    JsonLikelihoodCheck,
    LengthBandCheck,
    LinesPerParagraphCheck,
    LogLikelihoodCheck,
    MarkdownDensityCheck,
    MaxLineLengthBandCheck,
    StackTraceLikelihoodCheck,
    TableLikelihoodCheck,
    TokenEstimateCheck,
    TokenEstimateRatioCheck,
    WordsPerSentenceBandCheck,
    XmlLikelihoodCheck,
    YamlLikelihoodCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_length_band():
    ctx = _ctx("hello", "hello")
    assert LengthBandCheck().score("hello", ctx) == 1.0
    assert LengthBandCheck().score("x" * 100000, ctx) < 1.0


def test_token_estimate():
    ctx = _ctx("a b c", "a b c")
    assert TokenEstimateCheck().score("a b c", ctx) == 1.0
    assert TokenEstimateCheck().score("a", ctx) < 1.0


def test_token_ratio():
    ctx = _ctx("a b c d", "a b c d")
    assert TokenEstimateRatioCheck().score("a b c d", ctx) == 1.0


def test_chars_per_token():
    ctx = _ctx("aa bb", "aa bb")
    assert CharsPerTokenCheck().score("aa bb", ctx) == 1.0


def test_wps_band():
    ctx = _ctx("Short one. Another short.", "Short one. Another short.")
    assert WordsPerSentenceBandCheck().score("Short one. Another short.", ctx) == 1.0


def test_lines_per_para():
    ctx = _ctx("a\nb\n\nc\nd", "a\nb\n\nc\nd")
    assert LinesPerParagraphCheck().score("a\nb\n\nc\nd", ctx) == 1.0


def test_avg_line_len():
    ctx = _ctx("aa\nbb", "aa\nbb")
    assert AvgLineLengthCheck().score("aa\nbb", ctx) == 1.0


def test_max_line_band():
    ctx = _ctx("short", "short")
    assert MaxLineLengthBandCheck().score("short", ctx) == 1.0
    assert MaxLineLengthBandCheck().score("x" * 600, ctx) < 1.0


def test_blank_line_ratio():
    ctx = _ctx("a\n\nb", "a\n\nb")
    assert BlankLineRatioCheck().score("a\n\nb", ctx) == 1.0


def test_code_text_ratio():
    ctx = _ctx("```\nx\n```", "```\nx\n```")
    assert CodeToTextRatioCheck().score("```\nx\n```", ctx) == 1.0


def test_md_density():
    ctx = _ctx("# Title\n- item", "# Title\n- item")
    assert MarkdownDensityCheck().score("# Title\n- item", ctx) == 1.0


def test_format_likelihoods():
    jctx = _ctx('{"a": 1}', '{"a": 1}')
    assert JsonLikelihoodCheck().score('{"a": 1}', jctx) == 1.0
    yctx = _ctx("a: 1", "a: 1")
    assert YamlLikelihoodCheck().score("a: 1", yctx) == 1.0
    cctx = _ctx("a,b\n1,2", "a,b\n1,2")
    assert CsvLikelihoodCheck().score("a,b\n1,2", cctx) == 1.0
    hctx = _ctx("<div>x</div>", "<div>x</div>")
    assert HtmlLikelihoodCheck().score("<div>x</div>", hctx) == 1.0
    xctx = _ctx("<?xml version='1.0'?><a/>", "<?xml version='1.0'?><a/>")
    assert XmlLikelihoodCheck().score("<?xml version='1.0'?><a/>", xctx) == 1.0


def test_log_stacktrace_diff_table_likelihood():
    lctx = _ctx("2026-01-01 ERROR x", "2026-01-01 ERROR x")
    assert LogLikelihoodCheck().score("2026-01-01 ERROR x", lctx) == 1.0
    sctx = _ctx('Traceback:\n  File "x.py"', 'Traceback:\n  File "x.py"')
    assert StackTraceLikelihoodCheck().score('Traceback:\n  File "x.py"', sctx) == 1.0
    dctx = _ctx("--- a\n+++ b\n@@ -1 +1 @@", "--- a\n+++ b\n@@ -1 +1 @@")
    assert DiffLikelihoodCheck().score("--- a\n+++ b\n@@ -1 +1 @@", dctx) == 1.0
    tctx = _ctx("| a | b |\n|---|---|", "| a | b |\n|---|---|")
    assert TableLikelihoodCheck().score("| a | b |\n|---|---|", tctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "one two three",
         "candidate_output": "one two three"},
        {"id": "b", "prompt": "p", "baseline_output": "one two three",
         "candidate_output": "one"},
    ])
    result = evaluate(corpus, dimensions=("token_estimate", "length_band"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["token_estimate"].losses == 1
    assert dims["length_band"].losses == 1