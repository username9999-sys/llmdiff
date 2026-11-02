"""LLM-as-judge dimension with a deterministic offline fallback.

When an HTTP judge endpoint is configured (metadata ``judge_endpoint`` plus
optional ``judge_api_key``), the check calls it via httpx and parses a 0-10
score. Without configuration it uses a deterministic heuristic judge so tests
never hit the network.
"""

from __future__ import annotations

import json
import re

from llmdiff.core.checks import CheckContext

__all__ = ["JudgeCheck", "heuristic_judge_score", "parse_judge_score"]

_MAX = 100_000
_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)?")


def parse_judge_score(text: str) -> float | None:
    """Parse a 0-10 score from judge text; returns None when absent."""
    m = _SCORE_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    if value > 10:
        return None
    return value / 10.0


def heuristic_judge_score(output: str, prompt: str) -> float:
    """Deterministic offline judge: length sanity + prompt coverage + no refusal."""
    sample = output[:_MAX].strip()
    if not sample:
        return 0.0
    score = 0.4
    words = {w for w in re.findall(r"[a-z]{3,}", prompt.lower())}
    if words:
        hit = sum(1 for w in words if w in sample.lower())
        score += 0.3 * (hit / len(words))
    if not sample.lower().startswith(("i'm sorry", "i cannot", "as an ai")):
        score += 0.2
    lines = sample.count("\n")
    if 0 < len(sample) < 20_000 and lines < 500:
        score += 0.1
    return max(0.0, min(1.0, score))


class JudgeCheck:
    """Score an output with an LLM judge, or the heuristic fallback."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None) -> None:
        self._endpoint = endpoint
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "judge"

    def score(self, output: str, context: CheckContext) -> float:
        endpoint = self._endpoint or context.metadata.get("judge_endpoint")
        if endpoint:
            try:
                import httpx

                headers = {"Content-Type": "application/json"}
                api_key = self._api_key or context.metadata.get("judge_api_key")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "prompt": context.prompt[:_MAX],
                    "output": output[:_MAX],
                    "instruction": "Rate the output quality 0-10. Reply with just the number.",
                }
                resp = httpx.post(str(endpoint), json=payload, headers=headers, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                text = data.get("content") or data.get("text") or json.dumps(data)
                parsed = parse_judge_score(str(text))
                if parsed is not None:
                    return parsed
            except Exception:
                pass  # fall through to heuristic
        return heuristic_judge_score(output, context.prompt)
