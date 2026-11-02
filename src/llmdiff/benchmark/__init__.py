"""llmdiff.benchmark — self-benchmarking of llmdiff (dogfooding)."""

from __future__ import annotations

from llmdiff.benchmark.corpus_gen import GENERATORS, gen_corpus

__all__ = ["GENERATORS", "gen_corpus"]
