"""End-to-end harness: run the whole llmdiff pipeline offline against a mock LLM.

Pipeline: mock LLM server -> capture baseline outputs -> replay with candidate
responses -> build corpus -> evaluate -> verdict matrix. All local.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from llmdiff import Corpus, evaluate
from llmdiff.core.result import DiffResult
from llmdiff.e2e.mock_llm_server import MockLLMServer

__all__ = ["E2EResult", "run_e2e", "chat_once"]


def chat_once(base_url: str, prompt: str, *, model: str = "mock-model", timeout: float = 10.0) -> str:
    """Send one chat completion request to an OpenAI-compatible endpoint."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


@dataclass(frozen=True, slots=True)
class E2EResult:
    """Outcome of one end-to-end pipeline run."""

    n_cases: int
    server_url: str
    result: DiffResult
    captured: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def run_e2e(
    prompts: list[str],
    *,
    responses: dict[str, str] | None = None,
    candidate_responses: dict[str, str] | None = None,
    dimensions: tuple[str, ...] | None = None,
    model: str = "mock-model",
) -> E2EResult:
    """Run the full pipeline offline.

    1. Start a mock LLM server with ``responses`` (baseline behavior).
    2. "Capture" baseline outputs by calling it once per prompt.
    3. Start a second server with ``candidate_responses`` and replay to get
       candidate outputs.
    4. Build a corpus, evaluate, return the verdict matrix.
    """
    if not prompts:
        raise ValueError("at least one prompt is required")

    with MockLLMServer(responses or {}, default="baseline reply") as srv:
        baseline_outputs = [chat_once(srv.base_url, p, model=model) for p in prompts]
        url = srv.base_url

    cand = candidate_responses if candidate_responses is not None else responses
    with MockLLMServer(cand or {}, default="candidate reply") as srv2:
        candidate_outputs = [chat_once(srv2.base_url, p, model=model) for p in prompts]

    records = [
        {
            "id": f"e2e-{i}",
            "prompt": p,
            "baseline_output": b,
            "candidate_output": c,
            "metadata": {
                "model": model,
                "tags": ["e2e"],
                "timestamp": "2026-11-01T12:00:00Z",
            },
        }
        for i, (p, b, c) in enumerate(
            zip(prompts, baseline_outputs, candidate_outputs, strict=True)
        )
    ]
    captured = tuple(records)
    corpus = Corpus.from_records(records)
    result = evaluate(corpus, dimensions=dimensions)
    return E2EResult(n_cases=len(corpus), server_url=url, result=result, captured=captured)
