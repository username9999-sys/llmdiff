"""Tests for prompt templating and TemplateProvider."""

from __future__ import annotations

from llmdiff.replay.mock_provider import MockProvider
from llmdiff.replay.prompt_templates import TemplateProvider, render_template


def test_render_template_format_fallback_vars():
    out = render_template("Hello {name}, you are {age} years old.", {"name": "Ada", "age": 36})
    assert out == "Hello Ada, you are 36 years old."


def test_render_template_jinja_syntax_when_available():
    out = render_template("Hi {{ user }}!", {"user": "Bob"})
    assert out == "Hi Bob!"


def test_render_template_missing_key_raises_for_format():
    import pytest

    with pytest.raises(KeyError):
        render_template("value: {missing}", {})


def test_template_provider_renders_before_call():
    inner = MockProvider({"*": "ok"})
    provider = TemplateProvider(
        inner, "Summarise: {prompt} (lang={lang})", default_vars={"lang": "en"}
    )
    response = provider.complete("some text", template_vars={"lang": "id"})
    assert response.text == "ok"
    sent_prompt = inner.calls[0][0]
    assert sent_prompt == "Summarise: some text (lang=id)"


def test_template_provider_prompt_default_var():
    inner = MockProvider({"*": "ok"})
    provider = TemplateProvider(inner, "Q: {prompt}")
    provider.complete("what is 2+2?")
    assert inner.calls[0][0] == "Q: what is 2+2?"


def test_template_provider_name():
    inner = MockProvider()
    provider = TemplateProvider(inner, "{prompt}")
    assert provider.name.startswith("template(")
