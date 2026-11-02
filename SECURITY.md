# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Contact the maintainer
(username9999-sys on GitHub) with a description and, if possible, a
reproduction. You will get a reply within 72 hours.

## Scope

`llmdiff` processes LLM traffic, which may contain sensitive data. The
`capture` subpackage redacts PII/secrets by default, but redaction is
best-effort: treat corpora as sensitive and never commit real production
traffic to the repository.
