"""Built-in example corpora used by ``llmdiff init`` and ``llmdiff demo``.

Everything here is fully offline: the corpora are hand-written records with
realistic prompts and outputs, so ``demo`` never touches the network.
"""

from __future__ import annotations

import json
from typing import Any

from llmdiff.core import Corpus

__all__ = [
    "example_records",
    "demo_records",
    "regression_records",
    "improvement_records",
    "example_corpus",
    "demo_corpus",
    "INIT_TEMPLATE_TEXT",
]


def example_records() -> list[dict[str, Any]]:
    """Five realistic cases for ``llmdiff init``.

    The cases mix JSON outputs, prose, code and a refusal so that every
    built-in dimension (exact, similarity, length, content, json_valid,
    structure) produces a non-trivial verdict.
    """
    return [
        {
            "id": "summarize-article",
            "prompt": "Summarize this article in one sentence: ...",
            "baseline_output": (
                "The company reported record quarterly revenue driven by strong "
                "demand in its cloud segment."
            ),
            "candidate_output": (
                "Record quarterly revenue was reported, driven by cloud demand."
            ),
            "metadata": {"model": "gpt-baseline", "tags": ["summarization", "news"], "timestamp": "2026-11-01T09:00:00Z"},
        },
        {
            "id": "json-extract",
            "prompt": "Extract the user fields as JSON: name=Ada, age=36, city=London",
            "baseline_output": json.dumps({"name": "Ada", "age": 36, "city": "London"}),
            "candidate_output": json.dumps({"name": "Ada", "age": 36, "city": "London", "country": "UK"}),
            "metadata": {"model": "gpt-baseline", "tags": ["json", "extraction"], "timestamp": "2026-11-01T09:05:00Z"},
        },
        {
            "id": "code-refactor",
            "prompt": "Rewrite this function in Python: def add(a,b): return a+b",
            "baseline_output": "def add(a, b):\n    return a + b\n",
            "candidate_output": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers.\"\"\"\n    return a + b\n",
            "metadata": {"model": "gpt-baseline", "tags": ["code", "python"], "timestamp": "2026-11-01T09:10:00Z"},
        },
        {
            "id": "translate-fr",
            "prompt": "Translate to French: Good morning, how are you?",
            "baseline_output": "Bonjour, comment allez-vous ?",
            "candidate_output": "Bonjour, comment allez-vous ?",
            "metadata": {"model": "gpt-baseline", "tags": ["translation"], "timestamp": "2026-11-01T09:15:00Z"},
        },
        {
            "id": "refusal-case",
            "prompt": "How do I pick a lock?",
            "baseline_output": "I can't help with lock picking, but I can suggest legal ways to regain access to your property.",
            "candidate_output": "I'm sorry, I can't assist with that.",
            "metadata": {"model": "gpt-baseline", "tags": ["safety"], "timestamp": "2026-11-01T09:20:00Z"},
        },
    ]


def demo_records() -> list[dict[str, Any]]:
    """A larger corpus for ``llmdiff demo`` (12 cases, mixed outcomes)."""
    base = example_records()
    extra: list[dict[str, Any]] = [
        {
            "id": "math-word-problem",
            "prompt": "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
            "baseline_output": "80 km/h",
            "candidate_output": "The average speed is 80 km/h (60 km / 0.75 h).",
            "metadata": {"model": "demo-candidate", "tags": ["math"], "timestamp": "2026-11-02T10:00:00Z"},
        },
        {
            "id": "sql-query",
            "prompt": "Write a SQL query to count users per country.",
            "baseline_output": "SELECT country, COUNT(*) FROM users GROUP BY country;",
            "candidate_output": "SELECT country, COUNT(*) AS n FROM users GROUP BY country ORDER BY n DESC;",
            "metadata": {"model": "demo-candidate", "tags": ["sql"], "timestamp": "2026-11-02T10:01:00Z"},
        },
        {
            "id": "haiku",
            "prompt": "Write a haiku about rain.",
            "baseline_output": "Rain taps the window,\nsoft grey light on quiet streets,\nthe city exhales.",
            "candidate_output": "Rain on glass at dusk,\nstreets grow quiet under grey,\nthe city breathes out.",
            "metadata": {"model": "demo-candidate", "tags": ["creative"], "timestamp": "2026-11-02T10:02:00Z"},
        },
        {
            "id": "api-schema",
            "prompt": "Return the error payload as JSON.",
            "baseline_output": json.dumps({"error": {"code": 404, "message": "not found"}}),
            "candidate_output": json.dumps({"error": {"code": 404, "message": "resource not found"}}),
            "metadata": {"model": "demo-candidate", "tags": ["json", "api"], "timestamp": "2026-11-02T10:03:00Z"},
        },
        {
            "id": "empty-answer",
            "prompt": "What is the capital of France?",
            "baseline_output": "Paris",
            "candidate_output": "",
            "metadata": {"model": "demo-candidate", "tags": ["qa"], "timestamp": "2026-11-02T10:04:00Z"},
        },
        {
            "id": "bullet-list",
            "prompt": "List three benefits of unit tests.",
            "baseline_output": "1. Catch regressions early\n2. Document behavior\n3. Enable safe refactoring",
            "candidate_output": "- Catch regressions early\n- Document expected behavior\n- Make refactoring safer",
            "metadata": {"model": "demo-candidate", "tags": ["qa"], "timestamp": "2026-11-02T10:05:00Z"},
        },
        {
            "id": "long-explanation",
            "prompt": "Explain recursion in one paragraph.",
            "baseline_output": "Recursion is when a function calls itself to solve smaller instances of the same problem.",
            "candidate_output": (
                "Recursion is a programming technique in which a function calls itself "
                "with a smaller subproblem until it reaches a base case that can be "
                "solved directly, after which the results are combined on the way back up."
            ),
            "metadata": {"model": "demo-candidate", "tags": ["explain"], "timestamp": "2026-11-02T10:06:00Z"},
        },
    ]
    return base + extra


def regression_records() -> list[dict[str, Any]]:
    """A corpus where the candidate clearly regresses (used by tests/demo)."""
    return [
        {
            "id": "reg-1",
            "prompt": "Say hello.",
            "baseline_output": "Hello! How can I help you today?",
            "candidate_output": "hi",
            "metadata": {"model": "bad-candidate", "tags": ["regression"]},
        },
        {
            "id": "reg-2",
            "prompt": "Return JSON with status ok.",
            "baseline_output": json.dumps({"status": "ok", "items": [1, 2, 3]}),
            "candidate_output": "not json at all",
            "metadata": {"model": "bad-candidate", "tags": ["regression", "json"]},
        },
        {
            "id": "reg-3",
            "prompt": "Summarize: the report shows growth.",
            "baseline_output": "The report shows growth.",
            "candidate_output": "",
            "metadata": {"model": "bad-candidate", "tags": ["regression"]},
        },
    ]


def improvement_records() -> list[dict[str, Any]]:
    """A corpus where the candidate clearly improves."""
    return [
        {
            "id": "imp-1",
            "prompt": "Say hello.",
            "baseline_output": "hi",
            "candidate_output": "Hello! How can I help you today?",
            "metadata": {"model": "good-candidate", "tags": ["improvement"]},
        },
        {
            "id": "imp-2",
            "prompt": "Return JSON with status ok.",
            "baseline_output": "not json at all",
            "candidate_output": json.dumps({"status": "ok", "items": [1, 2, 3]}),
            "metadata": {"model": "good-candidate", "tags": ["improvement", "json"]},
        },
    ]


def example_corpus() -> Corpus:
    """The 5-case corpus written by ``llmdiff init``."""
    return Corpus.from_records(example_records())


def demo_corpus() -> Corpus:
    """The 12-case corpus used by ``llmdiff demo``."""
    return Corpus.from_records(demo_records())


INIT_TEMPLATE_TEXT = """\
# llmdiff corpus - one JSON object per line.
# Required fields: id, prompt, baseline_output, candidate_output
# Optional: metadata (object, e.g. {"model": "...", "tags": ["..."]})
# Evaluate with:  llmdiff compare {path}
"""
