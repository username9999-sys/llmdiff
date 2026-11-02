"""llmdiff - git diff for LLM outputs.

Replay real production traffic against a new prompt or model and get a
win/tie/loss verdict matrix instead of a vague "looks fine".
"""

from __future__ import annotations

from llmdiff.core import (
    Case,
    CaseDiff,
    Check,
    CheckContext,
    CheckError,
    CheckRegistry,
    CheckResult,
    ConfigError,
    Corpus,
    CorpusError,
    CorpusStats,
    DEFAULT_TOLERANCE,
    DiffConfig,
    DiffEngine,
    DiffResult,
    DimensionSummary,
    FormatError,
    LLMDiffError,
    PluginError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RegressionGateError,
    ReporterError,
    Verdict,
    VerdictMatrix,
    compare_scores,
    default_registry,
    evaluate,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Case",
    "CaseDiff",
    "Check",
    "CheckContext",
    "CheckError",
    "CheckRegistry",
    "CheckResult",
    "ConfigError",
    "Corpus",
    "CorpusError",
    "CorpusStats",
    "DEFAULT_TOLERANCE",
    "DiffConfig",
    "DiffEngine",
    "DiffResult",
    "DimensionSummary",
    "FormatError",
    "LLMDiffError",
    "PluginError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RegressionGateError",
    "ReporterError",
    "Verdict",
    "VerdictMatrix",
    "compare_scores",
    "default_registry",
    "evaluate",
]
