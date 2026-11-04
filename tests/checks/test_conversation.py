"""Tests for conversation dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.conversation import (
    AcknowledgmentCountCheck,
    AssistantTurnCountCheck,
    BulletTurnCountCheck,
    ClarificationCountCheck,
    CodeBlockPerTurnCheck,
    ConversationDepthCheck,
    EmptyTurnCheck,
    FollowUpQuestionCountCheck,
    LongTurnRatioCheck,
    MarkdownTurnCountCheck,
    NumberedTurnCountCheck,
    QuestionPerTurnCheck,
    QuoteTurnCountCheck,
    RoleMarkerCountCheck,
    ShortTurnRatioCheck,
    SummarySectionCountCheck,
    SystemTurnCountCheck,
    TurnAlternationCheck,
    TurnCountMatchCheck,
    UserTurnCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

CHAT = "User: hi\nAssistant: hello!\nUser: what is 2+2?\nAssistant: 4"
CHAT2 = "User: hi\nAssistant: hello there friend"


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_role_marker_count():
    ctx = _ctx(CHAT, CHAT)
    assert RoleMarkerCountCheck().score(CHAT, ctx) == 1.0
    assert RoleMarkerCountCheck().score(CHAT2, ctx) < 1.0


def test_user_assistant_turns():
    ctx = _ctx(CHAT, CHAT)
    assert UserTurnCountCheck().score(CHAT, ctx) == 1.0
    assert AssistantTurnCountCheck().score(CHAT, ctx) == 1.0
    assert UserTurnCountCheck().score(CHAT2, ctx) < 1.0


def test_system_turn_count():
    ctx = _ctx("system: be nice\nUser: hi", "system: be nice\nUser: hi")
    assert SystemTurnCountCheck().score("system: be nice\nUser: hi", ctx) == 1.0


def test_turn_alternation():
    ctx = _ctx(CHAT, CHAT)
    assert TurnAlternationCheck().score(CHAT, ctx) == 1.0


def test_codeblock_per_turn():
    chat = "User: code?\nAssistant: ```python\nx=1\n```"
    ctx = _ctx(chat, chat)
    assert CodeBlockPerTurnCheck().score(chat, ctx) == 1.0


def test_question_per_turn():
    ctx = _ctx(CHAT, CHAT)
    assert QuestionPerTurnCheck().score(CHAT, ctx) == 1.0


def test_ack_count():
    ctx = _ctx("Assistant: got it, thanks!", "Assistant: got it, thanks!")
    assert AcknowledgmentCountCheck().score("Assistant: got it, thanks!", ctx) == 1.0


def test_clarify_count():
    ctx = _ctx("Assistant: could you clarify?", "Assistant: could you clarify?")
    assert ClarificationCountCheck().score("Assistant: could you clarify?", ctx) == 1.0


def test_followup_count():
    ctx = _ctx("Assistant: what about X?", "Assistant: what about X?")
    assert FollowUpQuestionCountCheck().score("Assistant: what about X?", ctx) == 1.0


def test_summary_section_count():
    ctx = _ctx("## Summary\nall done", "## Summary\nall done")
    assert SummarySectionCountCheck().score("## Summary\nall done", ctx) == 1.0


def test_bullet_numbered_turns():
    chat = "Assistant: - a\n- b"
    ctx = _ctx(chat, chat)
    assert BulletTurnCountCheck().score(chat, ctx) == 1.0
    chat2 = "Assistant: 1. a\n2. b"
    ctx2 = _ctx(chat2, chat2)
    assert NumberedTurnCountCheck().score(chat2, ctx2) == 1.0


def test_quote_turn_count():
    chat = "Assistant: > quoted text"
    ctx = _ctx(chat, chat)
    assert QuoteTurnCountCheck().score(chat, ctx) == 1.0


def test_md_turn_count():
    chat = "Assistant: **bold** and `code`"
    ctx = _ctx(chat, chat)
    assert MarkdownTurnCountCheck().score(chat, ctx) == 1.0


def test_empty_turn_count():
    chat = "User: hi\nAssistant:"
    ctx = _ctx(chat, chat)
    assert EmptyTurnCheck().score(chat, ctx) == 1.0


def test_long_short_turn_ratio():
    long_chat = "User: hi\nAssistant: " + "x" * 600
    ctx = _ctx(long_chat, long_chat)
    assert LongTurnRatioCheck().score(long_chat, ctx) == 1.0
    short_chat = "User: hi\nAssistant: ok"
    sctx = _ctx(short_chat, short_chat)
    assert ShortTurnRatioCheck().score(short_chat, sctx) == 1.0


def test_turn_count_and_depth():
    ctx = _ctx(CHAT, CHAT)
    assert TurnCountMatchCheck().score(CHAT, ctx) == 1.0
    assert ConversationDepthCheck().score(CHAT, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": CHAT,
         "candidate_output": CHAT},
        {"id": "b", "prompt": "p", "baseline_output": CHAT,
         "candidate_output": CHAT2},
    ])
    result = evaluate(corpus, dimensions=("turn_count", "user_turn_count"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["turn_count"].losses == 1
    assert dims["user_turn_count"].losses == 1
