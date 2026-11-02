"""PII and secret redaction pipeline for captured traffic.

Captured LLM traffic routinely contains emails, phone numbers, credit card
numbers, cloud API keys, bearer tokens and JWTs.  This module provides a
pluggable, regex-based redaction pipeline that replaces each match with a
stable placeholder (``[EMAIL]``, ``[AWS_KEY]`` ...) while preserving the
overall structure of the text, so diffs and checks downstream still behave
sensibly on redacted corpora.

Design:

* :class:`Redactor` is a ``Protocol`` — any object with a
  ``redact(text) -> str`` method (and optionally ``name``) can be plugged in.
* The built-in redactors are small classes, one per secret family, each with
  a compiled regex and a placeholder.
* :class:`RedactionPipeline` composes redactors and applies them in order.
* Optionally the pipeline can keep a *reversible mapping*: instead of a
  fixed placeholder it stores ``[EMAIL#<sha256-8>]`` and remembers the
  original value keyed by that hash, so an operator with access to the
  mapping file can restore the original text.  The mapping is stored
  separately from the corpus and is opt-in.

Everything here is stdlib-only (``re``, ``hashlib``, ``json``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llmdiff.capture.errors import RedactionError

__all__ = [
    "Redactor",
    "RegexRedactor",
    "EmailRedactor",
    "PhoneRedactor",
    "CreditCardRedactor",
    "AWSKeyRedactor",
    "BearerTokenRedactor",
    "JWTRedactor",
    "SecretRedactor",
    "CustomRedactor",
    "RedactionPipeline",
    "DEFAULT_REDACTORS",
    "default_pipeline",
    "redact_text",
]


@runtime_checkable
class Redactor(Protocol):
    """Protocol for a single redaction step.

    A redactor transforms text, replacing sensitive spans with placeholders.
    Implementations must be idempotent: redacting already-redacted text must
    not change it further.
    """

    name: str

    def redact(self, text: str) -> str:  # pragma: no cover - protocol
        """Return ``text`` with sensitive spans replaced by placeholders."""
        ...


def _hash8(value: str) -> str:
    """Short stable hash used for reversible placeholders."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


class RegexRedactor:
    """A redactor driven by a single compiled regex.

    Args:
        name: Human-readable name, also used in error messages.
        pattern: Regex with at least one group; group 1 (or the whole match
            when there is no group) is the sensitive span that gets replaced.
        placeholder: Replacement template.  May contain ``{hash}`` which is
            replaced by a short hash of the matched value (useful for
            reversible mappings), or be a plain string.
        group: Which capture group to replace (default 1; 0 = whole match).
        flags: Extra ``re`` flags OR-ed into the pattern.
    """

    def __init__(
        self,
        name: str,
        pattern: str | re.Pattern[str],
        placeholder: str,
        *,
        group: int = 1,
        flags: int = 0,
    ) -> None:
        if isinstance(pattern, re.Pattern):
            self._regex = pattern
        else:
            self._regex = re.compile(pattern, flags)
        if group < 0:
            raise RedactionError(f"redactor {name!r}: group must be >= 0, got {group}")
        if group != 0 and group > (self._regex.groups or 0):
            if self._regex.groups == 0:
                group = 0  # no capture groups: replace the whole match
            else:
                raise RedactionError(
                    f"redactor {name!r}: group {group} out of range for pattern with "
                    f"{self._regex.groups} group(s)"
                )
        self.name = name
        self.placeholder = placeholder
        self.group = group

    def redact(self, text: str) -> str:
        if not isinstance(text, str):
            raise RedactionError(f"redactor {self.name!r}: expected str, got {type(text).__name__}")

        def _sub(match: re.Match[str]) -> str:
            try:
                value = match.group(self.group)
            except IndexError:  # pragma: no cover - guarded in __init__
                value = match.group(0)
            if value is None:
                value = match.group(0)
            return self.placeholder.format(hash=_hash8(value))

        return self._regex.sub(_sub, text)

    def find(self, text: str) -> list[str]:
        """Return all sensitive spans found in ``text`` (without replacing)."""
        found: list[str] = []
        for match in self._regex.finditer(text):
            try:
                value = match.group(self.group)
            except IndexError:  # pragma: no cover
                value = match.group(0)
            found.append(value if value is not None else match.group(0))
        return found

    def __repr__(self) -> str:
        return f"RegexRedactor(name={self.name!r}, pattern={self._regex.pattern!r})"


class EmailRedactor(RegexRedactor):
    """Redacts email addresses as ``[EMAIL]`` (or ``[EMAIL#hash]``)."""

    _PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    def __init__(self, placeholder: str = "[EMAIL]") -> None:
        super().__init__("email", self._PATTERN, placeholder)


class PhoneRedactor(RegexRedactor):
    """Redacts common international phone number shapes as ``[PHONE]``.

    Matches numbers with optional country code and separators, requiring at
    least 7 digits total so that short numeric strings are left alone.
    """

    _PATTERN = (
        r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
        r"(?:\d[\s.-]?){7,12}\d(?!\w)"
    )

    def __init__(self, placeholder: str = "[PHONE]") -> None:
        super().__init__("phone", self._PATTERN, placeholder)

    def redact(self, text: str) -> str:
        # Guard against matching plain long digit runs that are not phones
        # (e.g. ids) by requiring at least one separator or a leading +/paren.
        def _sub(match: re.Match[str]) -> str:
            value = match.group(0)
            digits = re.sub(r"\D", "", value)
            if len(digits) < 7:
                return value
            has_sep = any(c in value for c in "+(- .") or value.lstrip().startswith("+")
            if not has_sep:
                # A bare run of 7+ digits with no formatting is likely an id, not a phone.
                return value
            return self.placeholder.format(hash=_hash8(value))

        return self._regex.sub(_sub, text)


class CreditCardRedactor(RegexRedactor):
    """Redacts credit-card-like numbers (Luhn-checked) as ``[CARD]``."""

    _PATTERN = r"\b(?:\d[ -]?){13,19}\b"

    def __init__(self, placeholder: str = "[CARD]") -> None:
        super().__init__("credit_card", self._PATTERN, placeholder)

    @staticmethod
    def _luhn_ok(digits: str) -> bool:
        total = 0
        parity = len(digits) % 2
        for index, ch in enumerate(digits):
            d = ord(ch) - 48
            if index % 2 == parity:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    def redact(self, text: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            value = match.group(0)
            digits = re.sub(r"\D", "", value)
            if not (13 <= len(digits) <= 19) or not self._luhn_ok(digits):
                return value  # not a plausible card number; leave untouched
            return self.placeholder.format(hash=_hash8(digits))

        return self._regex.sub(_sub, text)


class AWSKeyRedactor(RegexRedactor):
    """Redacts AWS access key ids (``AKIA...`` / ``ASIA...``) as ``[AWS_KEY]``."""

    _PATTERN = r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"

    def __init__(self, placeholder: str = "[AWS_KEY]") -> None:
        super().__init__("aws_key", self._PATTERN, placeholder)


class BearerTokenRedactor(RegexRedactor):
    """Redacts ``Authorization: Bearer <token>`` values and ``Bearer <token>``
    inline mentions as ``[BEARER]``."""

    _PATTERN = r"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]{8,})"

    def __init__(self, placeholder: str = "[BEARER]") -> None:
        super().__init__("bearer_token", self._PATTERN, placeholder, group=2)

    def redact(self, text: str) -> str:
        # Preserve the "Bearer " prefix, replace only the token itself.
        return self._regex.sub(
            lambda m: m.group(1) + self.placeholder.format(hash=_hash8(m.group(2))),
            text,
        )


class JWTRedactor(RegexRedactor):
    """Redacts JSON Web Tokens (three base64url segments) as ``[JWT]``."""

    _PATTERN = r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b"

    def __init__(self, placeholder: str = "[JWT]") -> None:
        super().__init__("jwt", self._PATTERN, placeholder, group=0)


class SecretRedactor(RegexRedactor):
    """Redacts generic ``key=value`` / ``key: value`` secret assignments.

    Covers common key names such as ``api_key``, ``secret``, ``password``,
    ``token``, ``private_key`` in JSON-ish or env-ish text.
    """

    _PATTERN = (
        r"(?i)\b(api[_-]?key|secret|secret[_-]?key|access[_-]?token|auth[_-]?token|"
        r"password|passwd|private[_-]?key|client[_-]?secret)\b"
        r"(\"?\s*[=:]\"?\s*|\s+)([\"']?)([^\"'\n\s]{6,})\3?"
    )

    def __init__(self, placeholder: str = "[SECRET]") -> None:
        super().__init__("secret_assignment", self._PATTERN, placeholder, group=4)

    def redact(self, text: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            prefix = match.group(1) + match.group(2) + match.group(3)
            return prefix + self.placeholder.format(hash=_hash8(match.group(4)))

        return self._regex.sub(_sub, text)


class CustomRedactor:
    """Wrap an arbitrary callable as a :class:`Redactor`.

    Args:
        name: Name of the redactor.
        func: ``callable(text: str) -> str`` performing the replacement.
    """

    def __init__(self, name: str, func: Callable[[str], str]) -> None:
        if not callable(func):
            raise RedactionError(f"CustomRedactor {name!r}: func must be callable")
        self.name = name
        self._func = func

    def redact(self, text: str) -> str:
        return self._func(text)

    def __repr__(self) -> str:
        return f"CustomRedactor(name={self.name!r})"


class RedactionPipeline:
    """An ordered composition of redactors.

    Args:
        redactors: Redactors applied in order.  When ``None``, the default
            set (:data:`DEFAULT_REDACTORS`) is used.
        reversible: When true, placeholders embed a short hash of the original
            value and the original values are recorded in an internal mapping
            retrievable via :meth:`mapping` / :meth:`save_mapping`.  The
            mapping is *not* written anywhere unless explicitly saved.
    """

    def __init__(
        self,
        redactors: Iterable[Redactor] | None = None,
        *,
        reversible: bool = False,
    ) -> None:
        self.reversible = reversible
        if redactors is None:
            redactors = DEFAULT_REDACTORS
        self.redactors: list[Redactor] = list(redactors)
        for redactor in self.redactors:
            if not hasattr(redactor, "redact") or not callable(redactor.redact):
                raise RedactionError(
                    f"redactor {redactor!r} does not implement the Redactor protocol"
                )
        self._mapping: dict[str, str] = {}

    # ------------------------------------------------------------------ API
    def redact(self, text: str) -> str:
        """Apply every redactor in order and return the redacted text."""
        if not isinstance(text, str):
            raise RedactionError(f"expected str, got {type(text).__name__}")
        result = text
        for redactor in self.redactors:
            before = result
            result = redactor.redact(result)
            if self.reversible:
                self._collect_mapping(redactor, before, result)
        return result

    def redact_record(self, prompt: str, output: str) -> tuple[str, str]:
        """Redact both sides of a captured exchange."""
        return self.redact(prompt), self.redact(output)

    @property
    def names(self) -> list[str]:
        """Names of the composed redactors, in order."""
        return [getattr(r, "name", type(r).__name__) for r in self.redactors]

    # ------------------------------------------------------- reversible map
    def _collect_mapping(self, redactor: Redactor, before: str, after: str) -> None:
        """Best-effort extraction of original values for the reversible map.

        For regex redactors we re-scan ``before`` and pair each match with the
        placeholder that replaced it.  For custom redactors we cannot reliably
        recover originals, so they are skipped.
        """
        if not isinstance(redactor, RegexRedactor):
            return
        matches = list(redactor._regex.finditer(before))
        if not matches:
            return
        # Walk through the after-text finding placeholders in order.
        # Turn the placeholder template "[EMAIL#{hash}]" into a regex:
        # escape everything except the {hash} slot, which becomes a capture group.
        template = redactor.placeholder
        parts = template.split("{hash}")
        if len(parts) == 2:
            pattern = re.escape(parts[0]) + r"([0-9a-f]{8})" + re.escape(parts[1])
        else:
            pattern = re.escape(template)
        placeholder_re = re.compile(pattern)
        hashes = [m.group(1) for m in placeholder_re.finditer(after)]
        for match, digest in zip(matches, hashes, strict=False):
            try:
                value = match.group(redactor.group)
            except IndexError:
                value = match.group(0)
            if value is None:
                value = match.group(0)
            if _hash8(value) == digest:
                self._mapping[digest] = value

    @property
    def mapping(self) -> Mapping[str, str]:
        """Hash -> original value mapping (only populated when reversible)."""
        return dict(self._mapping)

    def save_mapping(self, path: str | Path) -> Path:
        """Persist the reversible mapping as JSON.  Handle with care: it
        contains the original secrets."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._mapping, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load_mapping(cls, path: str | Path) -> dict[str, str]:
        """Load a mapping previously written by :meth:`save_mapping`."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RedactionError(f"cannot load redaction mapping {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise RedactionError(f"redaction mapping {source} must be a JSON object")
        return {str(k): str(v) for k, v in data.items()}

    def restore(self, text: str, mapping: Mapping[str, str] | None = None) -> str:
        """Reverse placeholders using ``mapping`` (defaults to the internal one)."""
        mapping = self._mapping if mapping is None else mapping
        if not mapping:
            return text

        def _sub(match: re.Match[str]) -> str:
            digest = match.group(2)
            if digest and digest in mapping:
                return mapping[digest]
            return match.group(0)

        # Match any [NAME#hash] placeholder.
        return re.sub(r"\[([A-Z_]+)(?:#([0-9a-f]{8}))?\]", _sub, text)

    def __repr__(self) -> str:
        return f"RedactionPipeline(redactors={self.names!r}, reversible={self.reversible})"


DEFAULT_REDACTORS: tuple[Redactor, ...] = (
    EmailRedactor(),
    CreditCardRedactor(),  # before PhoneRedactor: card numbers contain spaces
    AWSKeyRedactor(),
    BearerTokenRedactor(),
    JWTRedactor(),
    PhoneRedactor(),
    SecretRedactor(),
)
"""The default redactor set applied by :func:`default_pipeline`."""


def default_pipeline(*, reversible: bool = False) -> RedactionPipeline:
    """Return a pipeline with the default redactors."""
    return RedactionPipeline(DEFAULT_REDACTORS, reversible=reversible)


def redact_text(
    text: str,
    pipeline: RedactionPipeline | Sequence[Redactor] | None = None,
) -> str:
    """Convenience one-shot redaction.

    Args:
        text: Text to redact.
        pipeline: An existing pipeline, a sequence of redactors, or ``None``
            to build the default pipeline.
    """
    if pipeline is None:
        pipe = default_pipeline()
    elif isinstance(pipeline, RedactionPipeline):
        pipe = pipeline
    else:
        pipe = RedactionPipeline(pipeline)
    return pipe.redact(text)


# --------------------------------------------------------------------- YAML
def load_pipeline_from_config(config: Mapping[str, Any]) -> RedactionPipeline:
    """Build a pipeline from a plain mapping (e.g. parsed YAML).

    Expected shape::

        redactors:
          - email
          - phone
          - credit_card
          - aws_key
          - bearer
          - jwt
          - secret
          - custom: my_name        # requires "custom_funcs" registry
        reversible: true

    Unknown names raise :class:`RedactionError`.
    """
    if not isinstance(config, Mapping):
        raise RedactionError("redaction config must be a mapping")
    registry: dict[str, Callable[[], Redactor]] = {
        "email": EmailRedactor,
        "phone": PhoneRedactor,
        "credit_card": CreditCardRedactor,
        "aws_key": AWSKeyRedactor,
        "bearer": BearerTokenRedactor,
        "jwt": JWTRedactor,
        "secret": SecretRedactor,
    }
    names = config.get("redactors")
    if names is None:
        redactors: list[Redactor] = list(DEFAULT_REDACTORS)
    else:
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise RedactionError("'redactors' must be a list of names")
        redactors = []
        for entry in names:
            if isinstance(entry, Mapping):
                entry = entry.get("custom") or entry.get("name")
            if not isinstance(entry, str):
                raise RedactionError(f"invalid redactor entry: {entry!r}")
            factory = registry.get(entry.lower())
            if factory is None:
                raise RedactionError(
                    f"unknown redactor {entry!r}; known: {sorted(registry)}"
                )
            redactors.append(factory())
    reversible = bool(config.get("reversible", False))
    return RedactionPipeline(redactors, reversible=reversible)
