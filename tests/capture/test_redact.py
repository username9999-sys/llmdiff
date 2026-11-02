"""Tests for llmdiff.capture.redact — synthetic strings only, no real secrets."""

from __future__ import annotations

import pytest

from llmdiff.capture.errors import RedactionError
from llmdiff.capture.redact import (
    AWSKeyRedactor,
    BearerTokenRedactor,
    CreditCardRedactor,
    CustomRedactor,
    EmailRedactor,
    JWTRedactor,
    PhoneRedactor,
    RedactionPipeline,
    SecretRedactor,
    default_pipeline,
    load_pipeline_from_config,
    redact_text,
)


def test_email_redacted():
    out = EmailRedactor().redact("contact alice@example.com please")
    assert "alice@example.com" not in out
    assert "[EMAIL]" in out


def test_phone_redacted():
    out = PhoneRedactor().redact("call +1-555-123-4567 now")
    assert "+1-555-123-4567" not in out
    assert "[PHONE]" in out


def test_phone_ignores_plain_digit_ids():
    out = PhoneRedactor().redact("order id 1234567890 shipped")
    assert "1234567890" in out  # bare digit run, not a phone


def test_credit_card_luhn():
    # 4111 1111 1111 1111 passes Luhn
    out = CreditCardRedactor().redact("card 4111 1111 1111 1111 ok")
    assert "4111 1111 1111 1111" not in out
    assert "[CARD]" in out


def test_credit_card_rejects_non_luhn():
    out = CreditCardRedactor().redact("number 4111 1111 1111 1112 stays")
    assert "4111 1111 1111 1112" in out  # fails Luhn, left alone


def test_aws_key():
    out = AWSKeyRedactor().redact("key AKIAIOSFODNN7EXAMPLE here")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[AWS_KEY]" in out


def test_bearer_token_preserves_prefix():
    out = BearerTokenRedactor().redact("Authorization: Bearer abcdef1234567890xyz")
    assert "abcdef1234567890xyz" not in out
    assert "Bearer [BEARER]" in out


def test_jwt():
    # synthetic JWT-shaped string (header.payload.signature, base64url)
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c"
    )
    out = JWTRedactor().redact(f"token {jwt} end")
    assert jwt not in out
    assert "[JWT]" in out


def test_secret_assignment():
    out = SecretRedactor().redact('{"api_key": "abcdef123456"}')
    assert "abcdef123456" not in out
    assert "[SECRET]" in out
    assert "api_key" in out  # key name preserved


def test_custom_redactor():
    red = CustomRedactor("caps", str.upper)
    assert red.redact("abc") == "ABC"
    with pytest.raises(RedactionError):
        CustomRedactor("bad", 42)  # type: ignore[arg-type]


def test_pipeline_composes_and_preserves_structure():
    pipe = default_pipeline()
    text = "mail bob@corp.io, key AKIAIOSFODNN7EXAMPLE, card 4111 1111 1111 1111"
    out = pipe.redact(text)
    assert "[EMAIL]" in out and "[AWS_KEY]" in out and "[CARD]" in out
    assert "mail" in out and "key" in out and "card" in out  # structure kept


def test_pipeline_rejects_non_redactor():
    with pytest.raises(RedactionError):
        RedactionPipeline([object()])


def test_reversible_mapping_roundtrip():
    pipe = RedactionPipeline([EmailRedactor("[EMAIL#{hash}]")], reversible=True)
    out = pipe.redact("reach alice@example.com now")
    assert "alice@example.com" not in out
    mapping = pipe.mapping
    assert mapping, "reversible mapping should be populated"
    restored = pipe.restore(out, mapping)
    assert "alice@example.com" in restored


def test_save_and_load_mapping(tmp_path):
    pipe = RedactionPipeline([EmailRedactor("[EMAIL#{hash}]")], reversible=True)
    pipe.redact("mail alice@example.com")
    path = pipe.save_mapping(tmp_path / "map.json")
    loaded = RedactionPipeline.load_mapping(path)
    assert loaded == dict(pipe.mapping)


def test_load_mapping_invalid(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(RedactionError):
        RedactionPipeline.load_mapping(bad)


def test_redact_text_convenience():
    out = redact_text("email a@b.co")
    assert "[EMAIL]" in out


def test_load_pipeline_from_config():
    pipe = load_pipeline_from_config({"redactors": ["email", "jwt"], "reversible": False})
    assert pipe.names == ["email", "jwt"]
    with pytest.raises(RedactionError):
        load_pipeline_from_config({"redactors": ["nope"]})
    with pytest.raises(RedactionError):
        load_pipeline_from_config({"redactors": "email"})  # type: ignore[arg-type]


def test_regex_redactor_bad_group():
    with pytest.raises(RedactionError):
        from llmdiff.capture.redact import RegexRedactor

        RegexRedactor("bad", r"(a)", "[X]", group=5)


def test_idempotent():
    pipe = default_pipeline()
    once = pipe.redact("mail a@b.co")
    twice = pipe.redact(once)
    assert once == twice
