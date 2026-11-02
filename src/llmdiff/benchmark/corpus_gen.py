"""Synthetic corpus generators for benchmarking llmdiff itself (dogfooding).

All generators are seeded and deterministic: same seed -> same corpus.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence

__all__ = [
    "gen_json_extraction",
    "gen_summarization",
    "gen_translation",
    "gen_classification",
    "gen_code_gen",
    "gen_corpus",
    "GENERATORS",
]


def _record(cid: str, prompt: str, baseline: str, candidate: str, tags: Sequence[str]) -> dict:
    return {
        "id": cid,
        "prompt": prompt,
        "baseline_output": baseline,
        "candidate_output": candidate,
        "metadata": {
            "model": "bench",
            "tags": list(tags),
            "timestamp": "2026-11-01T12:00:00Z",
        },
    }


def gen_json_extraction(n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """JSON extraction task: candidate may drop fields when noise > 0."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        order = {
            "order_id": f"A-{1000 + i}",
            "total": round(rng.uniform(10, 500), 2),
            "currency": rng.choice(["USD", "EUR", "IDR"]),
            "items": rng.randint(1, 9),
        }
        prompt = f"Extract the order details as JSON from: {json.dumps(order)}"
        baseline = json.dumps(order)
        candidate = baseline
        if rng.random() < noise:
            dropped = dict(order)
            dropped.pop(rng.choice(list(dropped)))
            candidate = json.dumps(dropped)
        out.append(_record(f"json-{i}", prompt, baseline, candidate, ["json", "extraction"]))
    return out


def gen_summarization(n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Summarization task: candidate may truncate when noise > 0."""
    rng = random.Random(seed)
    topics = ["refund policy", "shipping times", "warranty terms", "privacy policy", "pricing"]
    out = []
    for i in range(n):
        topic = rng.choice(topics)
        text = "The " + topic + " document says: " + " ".join(
            rng.choice(["customers", "may", "request", "within", "30", "days", "for", "a", "full", "refund"])
            for _ in range(20)
        )
        prompt = f"Summarize in one sentence: {text}"
        baseline = f"Summary of the {topic}: customers may request within 30 days for a full refund."
        candidate = baseline
        if rng.random() < noise:
            candidate = baseline[: len(baseline) // 2] + "..."
        out.append(_record(f"sum-{i}", prompt, baseline, candidate, ["summary"]))
    return out


def gen_translation(n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Translation task: candidate may stay untranslated when noise > 0."""
    rng = random.Random(seed)
    pairs = [
        ("good morning", "bonjour"),
        ("thank you", "merci"),
        ("hello", "hallo"),
        ("goodbye", "au revoir"),
        ("please", "s'il vous plait"),
    ]
    out = []
    for i in range(n):
        src, dst = rng.choice(pairs)
        prompt = f"Translate to French/German: {src}"
        candidate = dst if rng.random() >= noise else src
        out.append(_record(f"tr-{i}", prompt, dst, candidate, ["translation"]))
    return out


def gen_classification(n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Sentiment classification: candidate may mislabel when noise > 0."""
    rng = random.Random(seed)
    samples = [
        ("The app crashes every time I open settings.", "negative"),
        ("I love this product, it works perfectly!", "positive"),
        ("It is okay, nothing special.", "neutral"),
        ("Terrible support, nobody replied for a week.", "negative"),
        ("Best purchase I have made this year.", "positive"),
    ]
    out = []
    for i in range(n):
        text, label = rng.choice(samples)
        prompt = f"Classify the sentiment: {text}"
        candidate = label if rng.random() >= noise else rng.choice(["positive", "negative", "neutral"])
        out.append(_record(f"cls-{i}", prompt, label, candidate, ["classification"]))
    return out


def gen_code_gen(n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Code generation: candidate may emit broken syntax when noise > 0."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name = f"func_{i}"
        prompt = f"Write a Python function {name} that returns the square of its argument."
        baseline = f"def {name}(x):\n    return x * x\n"
        candidate = baseline
        if rng.random() < noise:
            candidate = f"def {name}(x:\n    return x * x\n"  # syntax error
        out.append(_record(f"code-{i}", prompt, baseline, candidate, ["code"]))
    return out


GENERATORS = {
    "json_extraction": gen_json_extraction,
    "summarization": gen_summarization,
    "translation": gen_translation,
    "classification": gen_classification,
    "code_gen": gen_code_gen,
}


def gen_corpus(kind: str, n: int, *, seed: int = 0, noise: float = 0.0) -> list[dict]:
    """Dispatch to a named generator. Raises ValueError for unknown kinds."""
    if kind not in GENERATORS:
        raise ValueError(f"unknown corpus kind {kind!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[kind](n, seed=seed, noise=noise)
