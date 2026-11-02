"""Example 10 — PII redaction before capture.

Run:  PYTHONPATH=src python3 examples/10_safety_pii_redaction.py
Fully offline.
"""

from __future__ import annotations

from llmdiff.capture.redact import RedactionPipeline


def main() -> int:
    pipeline = RedactionPipeline()
    text = "Contact john.doe@example.com or call +1 555 123 4567. Key: AKIAIOSFODNN7EXAMPLE"
    redacted = pipeline.redact(text)
    print("original:", text)
    print("redacted:", redacted)
    assert "john.doe@example.com" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "[EMAIL]" in redacted
    print("PII redacted: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
