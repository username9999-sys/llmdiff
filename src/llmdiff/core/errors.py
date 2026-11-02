"""Exception hierarchy for llmdiff.

Every error raised by the package derives from :class:`LLMDiffError` so callers
can catch the whole family with one ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "LLMDiffError",
    "CorpusError",
    "FormatError",
    "CheckError",
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ReporterError",
    "ConfigError",
    "PluginError",
    "RegressionGateError",
]


class LLMDiffError(Exception):
    """Base class for every error raised by llmdiff."""


class CorpusError(LLMDiffError):
    """Raised when a corpus is malformed, unreadable, or internally inconsistent."""


class FormatError(CorpusError):
    """Raised when a corpus file cannot be parsed in the requested format."""


class CheckError(LLMDiffError):
    """Raised when a check cannot produce a score for a case."""


class ProviderError(LLMDiffError):
    """Raised when an LLM provider call fails."""


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects the supplied credentials."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider signals rate limiting.

    Attributes:
        retry_after: Seconds to wait before retrying, when the provider said so.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds its deadline."""


class ReporterError(LLMDiffError):
    """Raised when a reporter cannot render a result."""


class ConfigError(LLMDiffError):
    """Raised when configuration is missing, invalid, or contradictory."""


class PluginError(LLMDiffError):
    """Raised when a plugin cannot be loaded or violates the plugin contract."""


class RegressionGateError(LLMDiffError):
    """Raised when a regression gate fails.

    Attributes:
        exit_code: Process exit code that should be used for this failure.
        failures: Human-readable descriptions of every failed gate condition.
    """

    def __init__(self, message: str, *, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = list(failures or [])
        self.exit_code = 1
