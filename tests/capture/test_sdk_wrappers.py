"""Tests for the duck-typed SDK wrappers (fake clients, no network, no SDK import)."""

from __future__ import annotations

import pytest

from llmdiff.capture.record import TrafficRecorder
from llmdiff.capture.sdk_anthropic import CapturingAnthropic
from llmdiff.capture.sdk_openai import CapturingOpenAI


# ------------------------------------------------------------- fakes
class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 3
        self.completion_tokens = 2

    def model_dump(self):
        return {"prompt_tokens": 3, "completion_tokens": 2}


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse("pong")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()
        self.models = "passthrough-models"


class FakeAnthropicBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeAnthropicUsage:
    def model_dump(self):
        return {"input_tokens": 5, "output_tokens": 1}


class FakeAnthropicResponse:
    def __init__(self, text):
        self.content = [FakeAnthropicBlock(text)]
        self.usage = FakeAnthropicUsage()


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeAnthropicResponse("pong")


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessages()


# ------------------------------------------------------------- openai wrapper
def test_capturing_openai_records():
    recorder = TrafficRecorder()
    client = CapturingOpenAI(FakeOpenAIClient(), recorder)
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "ping"}]
    )
    assert resp.choices[0].message.content == "pong"  # original returned
    assert len(recorder) == 1
    case = recorder.cases[0]
    assert "ping" in case.prompt
    assert case.candidate_output == "pong"
    assert case.metadata["model"] == "gpt-4o"
    assert case.metadata["endpoint"] == "/v1/chat/completions"
    assert case.metadata["usage"]["prompt_tokens"] == 3
    assert "latency_ms" in case.metadata and "timestamp" in case.metadata


def test_capturing_openai_passthrough_attributes():
    recorder = TrafficRecorder()
    inner = FakeOpenAIClient()
    client = CapturingOpenAI(inner, recorder)
    assert client.models == "passthrough-models"


def test_capturing_openai_error_still_recorded_and_raised():
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()

    recorder = TrafficRecorder()
    client = CapturingOpenAI(FailingClient(), recorder)
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert len(recorder) == 1
    assert recorder.cases[0].metadata.get("error") is True
    assert recorder.cases[0].candidate_output == ""


def test_capturing_openai_dict_response():
    class DictCompletions:
        def create(self, **kwargs):
            return {"choices": [{"message": {"content": "dict-out"}}]}

    class DictChat:
        completions = DictCompletions()

    class DictClient:
        chat = DictChat()

    recorder = TrafficRecorder()
    client = CapturingOpenAI(DictClient(), recorder)
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert recorder.cases[0].candidate_output == "dict-out"


def test_capturing_openai_requires_chat_attr():
    with pytest.raises(TypeError):
        CapturingOpenAI(object(), TrafficRecorder())


# ------------------------------------------------------------- anthropic wrapper
def test_capturing_anthropic_records():
    recorder = TrafficRecorder()
    client = CapturingAnthropic(FakeAnthropicClient(), recorder)
    resp = client.messages.create(
        model="claude-x",
        system="be nice",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=16,
    )
    assert resp.content[0].text == "pong"
    case = recorder.cases[0]
    assert "hello" in case.prompt
    assert "be nice" in case.prompt
    assert case.candidate_output == "pong"
    assert case.metadata["model"] == "claude-x"
    assert case.metadata["endpoint"] == "/v1/messages"
    assert case.metadata["usage"]["input_tokens"] == 5


def test_capturing_anthropic_error_recorded():
    class FailingMessages:
        def create(self, **kwargs):
            raise ValueError("nope")

    class FailingClient:
        messages = FailingMessages()

    recorder = TrafficRecorder()
    client = CapturingAnthropic(FailingClient(), recorder)
    with pytest.raises(ValueError):
        client.messages.create(model="m", messages=[])
    assert recorder.cases[0].metadata.get("error") is True


def test_capturing_anthropic_requires_messages_attr():
    with pytest.raises(TypeError):
        CapturingAnthropic(object(), TrafficRecorder())


def test_wrappers_work_without_sdk_installed():
    # The modules must import fine even if openai/anthropic are absent;
    # verify no top-level import statements reference the SDK packages
    # (docstring examples don't count).
    import re

    import llmdiff.capture.sdk_anthropic as sa
    import llmdiff.capture.sdk_openai as so

    stmt_re = re.compile(r"^(?:import|from)\s+(openai|anthropic)\b", re.MULTILINE)
    for mod in (sa, so):
        with open(mod.__file__, encoding="utf-8") as handle:
            src = handle.read()
        assert stmt_re.search(src) is None, f"{mod.__name__} imports an SDK at top level"
