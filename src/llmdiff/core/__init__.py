"""Stable public API of the core layer.

Everything re-exported here is what the rest of the package - and third-party
plugins - are expected to build against.
"""

from __future__ import annotations

from llmdiff.core.case import Case
from llmdiff.core.checks import (
    DEFAULT_TOLERANCE,
    Check,
    CheckContext,
    CheckResult,
    Verdict,
    compare_scores,
)
from llmdiff.core.corpus import Corpus, CorpusStats
from llmdiff.core.engine import DiffConfig, DiffEngine, evaluate
from llmdiff.core.errors import (
    CheckError,
    ConfigError,
    CorpusError,
    FormatError,
    LLMDiffError,
    PluginError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RegressionGateError,
    ReporterError,
)
from llmdiff.core.matrix import DimensionSummary, VerdictMatrix
from llmdiff.core.registry import CheckRegistry, default_registry
from llmdiff.core.result import CaseDiff, DiffResult

__all__ = [
    "Case",
    "Corpus",
    "CorpusStats",
    "Check",
    "CheckContext",
    "CheckResult",
    "Verdict",
    "compare_scores",
    "DEFAULT_TOLERANCE",
    "DimensionSummary",
    "VerdictMatrix",
    "CheckRegistry",
    "default_registry",
    "DiffConfig",
    "DiffEngine",
    "DiffResult",
    "CaseDiff",
    "evaluate",
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
