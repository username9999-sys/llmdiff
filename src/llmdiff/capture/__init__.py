"""llmdiff.capture — record real LLM traffic into corpora.

Public API:

* :class:`TrafficRecorder` — thread-safe recorder; ``record()``, ``save()``,
  ``load()``.
* :class:`RedactionPipeline` + built-in redactors — PII/secret redaction.
* :class:`CaptureMiddleware` — ASGI middleware capturing
  ``/v1/chat/completions`` and ``/v1/messages``.
* :class:`CapturingOpenAI` / :class:`CapturingAnthropic` — duck-typed SDK
  wrappers.
* :func:`run_proxy` / ``python -m llmdiff.capture.proxy`` — local capture
  proxy.
* Samplers (:mod:`llmdiff.capture.sampling`), filters
  (:mod:`llmdiff.capture.filters`), dedup (:mod:`llmdiff.capture.dedup`),
  formats (:mod:`llmdiff.capture.formats`), sessions
  (:mod:`llmdiff.capture.session`).

Capture-local errors live in :mod:`llmdiff.capture.errors` and all subclass
:class:`llmdiff.core.errors.LLMDiffError`.
"""

from __future__ import annotations

from llmdiff.capture.dedup import (
    DedupPipeline,
    ExactDeduper,
    SimHashDeduper,
    dedup_records,
    hamming,
    prompt_hash,
    simhash,
)
from llmdiff.capture.errors import (
    CaptureError,
    DedupError,
    FilterError,
    FormatExportError,
    MiddlewareError,
    ProxyError,
    RecordError,
    RedactionError,
    SamplingError,
    SessionError,
)
from llmdiff.capture.filters import (
    FilterChain,
    InvertFilter,
    ModelFilter,
    OutputLengthFilter,
    PredicateFilter,
    PromptLengthFilter,
    RecordFilter,
    TagFilter,
    apply_filters,
)
from llmdiff.capture.formats import (
    export_corpus,
    export_csv,
    export_json_array,
    export_parquet,
    export_sqlite,
    import_corpus,
    import_csv,
    import_json_array,
    import_sqlite,
)
from llmdiff.capture.middleware import (
    CaptureMiddleware,
    extract_output,
    extract_prompt,
    parse_sse_chunks,
)
from llmdiff.capture.proxy import build_arg_parser, main, make_server, run_proxy
from llmdiff.capture.record import TrafficRecorder
from llmdiff.capture.redact import (
    AWSKeyRedactor,
    BearerTokenRedactor,
    CreditCardRedactor,
    CustomRedactor,
    EmailRedactor,
    JWTRedactor,
    PhoneRedactor,
    RedactionPipeline,
    RegexRedactor,
    SecretRedactor,
    default_pipeline,
    load_pipeline_from_config,
    redact_text,
)
from llmdiff.capture.sampling import (
    AlwaysSampler,
    CompositeSampler,
    HeadSampler,
    NeverSampler,
    RandomSampler,
    RateLimitSampler,
    ReservoirSampler,
    Sampler,
    StratifiedSampler,
    make_sampler,
)
from llmdiff.capture.sdk_anthropic import CapturingAnthropic, CapturingMessages
from llmdiff.capture.sdk_openai import CapturingChat, CapturingCompletions, CapturingOpenAI
from llmdiff.capture.session import CaptureSession, SessionSummary, list_sessions

__all__ = [
    # record
    "TrafficRecorder",
    # redact
    "RedactionPipeline",
    "RegexRedactor",
    "EmailRedactor",
    "PhoneRedactor",
    "CreditCardRedactor",
    "AWSKeyRedactor",
    "BearerTokenRedactor",
    "JWTRedactor",
    "SecretRedactor",
    "CustomRedactor",
    "default_pipeline",
    "redact_text",
    "load_pipeline_from_config",
    # middleware
    "CaptureMiddleware",
    "extract_prompt",
    "extract_output",
    "parse_sse_chunks",
    # sdk wrappers
    "CapturingOpenAI",
    "CapturingChat",
    "CapturingCompletions",
    "CapturingAnthropic",
    "CapturingMessages",
    # proxy
    "run_proxy",
    "make_server",
    "build_arg_parser",
    "main",
    # sampling
    "Sampler",
    "AlwaysSampler",
    "NeverSampler",
    "RandomSampler",
    "RateLimitSampler",
    "ReservoirSampler",
    "StratifiedSampler",
    "HeadSampler",
    "CompositeSampler",
    "make_sampler",
    # filters
    "RecordFilter",
    "TagFilter",
    "ModelFilter",
    "PromptLengthFilter",
    "OutputLengthFilter",
    "PredicateFilter",
    "InvertFilter",
    "FilterChain",
    "apply_filters",
    # dedup
    "ExactDeduper",
    "SimHashDeduper",
    "DedupPipeline",
    "dedup_records",
    "simhash",
    "hamming",
    "prompt_hash",
    # formats
    "export_corpus",
    "import_corpus",
    "export_csv",
    "import_csv",
    "export_json_array",
    "import_json_array",
    "export_parquet",
    "export_sqlite",
    "import_sqlite",
    # session
    "CaptureSession",
    "SessionSummary",
    "list_sessions",
    # errors
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
