"""Capture-local exception hierarchy.

Every error raised by :mod:`llmdiff.capture` derives from
:class:`CaptureError`, which itself derives from
:class:`llmdiff.core.errors.LLMDiffError`, so callers can catch the whole
llmdiff family with a single ``except LLMDiffError`` clause while still being
able to distinguish capture failures from core failures.

The core package is intentionally not modified; these errors live here.
"""

from __future__ import annotations

from llmdiff.core.errors import LLMDiffError

__all__ = [
    "CaptureError",
    "RecordError",
    "RedactionError",
    "MiddlewareError",
    "ProxyError",
    "FormatExportError",
    "SamplingError",
    "FilterError",
    "DedupError",
    "SessionError",
]


class CaptureError(LLMDiffError):
    """Base class for every error raised by the capture subpackage."""


class RecordError(CaptureError):
    """Raised when a traffic record cannot be created, stored or loaded."""


class RedactionError(CaptureError):
    """Raised when the redaction pipeline is misconfigured or fails."""


class MiddlewareError(CaptureError):
    """Raised when the ASGI capture middleware cannot process a request."""


class ProxyError(CaptureError):
    """Raised when the local capture proxy cannot forward a request."""


class FormatExportError(CaptureError):
    """Raised when a corpus cannot be imported from or exported to a format."""


class SamplingError(CaptureError):
    """Raised when a sampler is misconfigured (bad rate, capacity, ...)."""


class FilterError(CaptureError):
    """Raised when a record filter is misconfigured."""


class DedupError(CaptureError):
    """Raised when deduplication is misconfigured (bad threshold, ...)."""


class SessionError(CaptureError):
    """Raised when session management is used incorrectly (e.g. double end)."""
