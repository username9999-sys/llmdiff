"""Tests for unicode, custom_regex, semantic, and judge dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.custom_regex import RegexForbiddenCheck, RegexMustMatchCheck, compile_patterns
from llmdiff.checks.judge import JudgeCheck, heuristic_judge_score, parse_judge_score
from llmdiff.checks.semantic import EmbeddingCache, SemanticSimilarityCheck, hashed_ngram_cosine
from llmdiff.checks.unicode import (
    ControlCharCheck,
    MojibakeCheck,
    RtlMarkCheck,
    UnicodeScriptCheck,
    script_distribution,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext
from llmdiff.core.errors import CheckError


def _ctx(
    baseline: str, candidate: str, metadata: dict | None = None, prompt: str = "p"
) -> CheckContext:
    case = Case(
        id="t", prompt=prompt, baseline_output=baseline, candidate_output=candidate,
        metadata=metadata or {},
    )
    return CheckContext(case=case, metadata=case.metadata)


def test_script_distribution():
    d = script_distribution("hello world")
    assert d.get("LATIN", 0) == 1.0
    assert script_distribution("") == {}


def test_unicode_script_same():
    ctx = _ctx("hello", "world")
    assert UnicodeScriptCheck().score("world", ctx) == 1.0


def test_unicode_script_diff():
    ctx = _ctx("hello", "你好")
    assert UnicodeScriptCheck().score("你好", ctx) < 1.0


def test_mojibake():
    ctx = _ctx("ok", "ok")
    assert MojibakeCheck().score("ok", ctx) == 1.0
    assert MojibakeCheck().score("cafÃ©", ctx) < 1.0


def test_control_chars():
    ctx = _ctx("ok", "ok")
    assert ControlCharCheck().score("ok\t\n", ctx) == 1.0
    assert ControlCharCheck().score("ok\x07", ctx) < 1.0


def test_bidi_marks():
    ctx = _ctx("ok", "ok")
    assert RtlMarkCheck().score("ok", ctx) == 1.0
    assert RtlMarkCheck().score("ok\u202e", ctx) == 0.0


def test_compile_patterns_bad():
    try:
        compile_patterns(["("], field="regex_must_match")
        raise AssertionError("should raise")
    except CheckError:
        pass


def test_regex_must_match():
    ctx = _ctx("x", "x", {"regex_must_match": [r"\d+", r"[A-Z]+"]})
    assert RegexMustMatchCheck().score("ABC 123", ctx) == 1.0
    assert RegexMustMatchCheck().score("abc 123", ctx) == 0.5
    assert RegexMustMatchCheck().score("x", _ctx("x", "x")) == 1.0


def test_regex_forbidden():
    ctx = _ctx("x", "x", {"regex_forbidden": [r"\bTODO\b"]})
    assert RegexForbiddenCheck().score("clean", ctx) == 1.0
    assert RegexForbiddenCheck().score("TODO: fix", ctx) == 0.0


def test_hashed_ngram_cosine():
    assert hashed_ngram_cosine("a b c", "a b c") == 1.0
    assert hashed_ngram_cosine("a b c", "x y z") < 0.5
    assert hashed_ngram_cosine("", "") == 1.0


def test_embedding_cache():
    cache = EmbeddingCache()
    assert cache.get("hello") is None
    cache.put("hello", [1.0, 2.0])
    assert cache.get("hello") == [1.0, 2.0]


def test_semantic_fallback():
    check = SemanticSimilarityCheck()
    check._model = None
    check._model_loaded = True  # force fallback without loading any model
    ctx = _ctx("the cat sat on the mat", "the cat sat on the mat")
    assert check.score("the cat sat on the mat", ctx) == 1.0
    assert check.score("completely different words here", ctx) < 1.0


def test_parse_judge_score():
    assert parse_judge_score("8/10") == 0.8
    assert parse_judge_score("Score: 7 out of 10") == 0.7
    assert parse_judge_score("no number") is None


def test_heuristic_judge():
    assert heuristic_judge_score("", "prompt") == 0.0
    s = heuristic_judge_score(
        "a helpful answer about the refund policy", "what is the refund policy?"
    )
    assert 0.5 < s <= 1.0


def test_judge_offline():
    ctx = _ctx("x", "a good answer", prompt="say something good")
    assert 0.0 < JudgeCheck().score("a good answer", ctx) <= 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "hello world foo"},
        {"id": "b", "prompt": "p", "baseline_output": "hello world foo",
         "candidate_output": "zzz yyy xxx"},
    ])
    result = evaluate(corpus, dimensions=("semantic", "regex_must_match"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["semantic"].losses == 1
