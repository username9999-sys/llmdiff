"""Semantic similarity dimension with a zero-dependency fallback.

Uses sentence-transformers embeddings when available; otherwise falls back to
hashed-ngram cosine similarity, which is fully deterministic and offline.
Embeddings are cached by content hash so repeated scoring is cheap.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from llmdiff.core.checks import CheckContext

__all__ = ["SemanticSimilarityCheck", "hashed_ngram_cosine", "EmbeddingCache"]

_MAX = 100_000
_NGRAM = 3
_BUCKETS = 2048


def _tokens(text: str) -> list[str]:
    return text.lower().split()[:_MAX]


def hashed_ngram_cosine(a: str, b: str) -> float:
    """Cosine similarity over hashed word-ngram count vectors."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if not ta and not tb else 0.0

    def vec(tokens: list[str]) -> dict[int, float]:
        counts: Counter[int] = Counter()
        for n in (1, 2, _NGRAM):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                # Stable hash (md5) — builtin hash() is salted per process.
                bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % _BUCKETS
                counts[bucket] += 1.0
        return dict(counts)

    va, vb = vec(ta), vec(tb)
    dot = sum(v * vb.get(k, 0.0) for k, v in va.items())
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return min(1.0, max(0.0, dot / (na * nb)))


class EmbeddingCache:
    """Simple in-memory cache mapping content hash -> embedding vector."""

    def __init__(self, max_entries: int = 4096) -> None:
        self._data: dict[str, list[float]] = {}
        self._max = max_entries

    def key(self, text: str) -> str:
        return hashlib.sha256(text[:_MAX].encode("utf-8", "replace")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self._data.get(self.key(text))

    def put(self, text: str, vector: list[float]) -> None:
        if len(self._data) >= self._max:
            self._data.clear()
        self._data[self.key(text)] = vector


class SemanticSimilarityCheck:
    """Cosine similarity between output and baseline.

    Uses sentence-transformers when installed (cached); otherwise the hashed
    n-gram fallback. Always deterministic for the same inputs.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        use_embeddings: bool | None = None,
    ) -> None:
        """Create the semantic check.

        Args:
            model_name: sentence-transformers model name (only used when
                embeddings are enabled).
            use_embeddings: None reads the ``LLMDIFF_EMBEDDINGS`` env var
                (``1``/``true``/``yes``/``on`` enables). Default is **off**, so
                the check uses the fast hashed-ngram fallback unless the user
                explicitly opts in. This keeps ``llmdiff demo`` and CI fast
                and fully offline.
        """
        if use_embeddings is None:
            import os

            use_embeddings = os.environ.get("LLMDIFF_EMBEDDINGS", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self._model_name = model_name
        self._use_embeddings = use_embeddings
        self._model: object | None = None
        self._model_loaded = False
        self._cache = EmbeddingCache()

    def _ensure_model(self) -> None:
        """Load the sentence-transformers model once, only when embeddings are on."""
        if self._model_loaded or not self._use_embeddings:
            self._model_loaded = True
            return
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    @property
    def name(self) -> str:
        return "semantic"

    def _embed(self, text: str) -> list[float] | None:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        self._ensure_model()
        if self._model is None:
            return None
        vec = self._model.encode(text[:_MAX]).tolist()
        self._cache.put(text, vec)
        return vec

    def score(self, output: str, context: CheckContext) -> float:
        base = context.baseline_output
        va, vb = self._embed(base), self._embed(output)
        if va is not None and vb is not None:
            dot = sum(x * y for x, y in zip(va, vb, strict=True))
            na = math.sqrt(sum(x * x for x in va))
            nb = math.sqrt(sum(x * x for x in vb))
            if na == 0 or nb == 0:
                return 1.0 if na == nb else 0.0
            return max(0.0, min(1.0, dot / (na * nb)))
        return hashed_ngram_cosine(base, output)
