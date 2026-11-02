"""Compare two replayed corpora via :meth:`Corpus.merge_candidates`.

Given a baseline corpus (candidate outputs from the old system) and a
candidate corpus (outputs from the new system, produced by
:func:`llmdiff.replay.runner.replay`), :func:`compare_runs` returns a corpus
ready for ``llmdiff.evaluate()``: baselines from the baseline corpus, candidate
outputs from the candidate corpus, matched by id.
"""

from __future__ import annotations

from pathlib import Path

from llmdiff.core.corpus import Corpus
from llmdiff.core.errors import CorpusError

__all__ = ["compare_runs", "load_pair"]


def compare_runs(baseline: Corpus, candidate: Corpus) -> Corpus:
    """Merge a baseline corpus with a candidate corpus for evaluation.

    The returned corpus keeps prompts and baseline outputs from ``baseline``
    and takes candidate outputs (plus metadata) from ``candidate``. This is
    exactly :meth:`Corpus.merge_candidates` on the baseline, with an extra
    guard that reports how many candidate cases were unused.

    Raises:
        CorpusError: If the candidate corpus is missing ids present in the
            baseline (via ``merge_candidates``), or if the candidate corpus
            has extra ids not in the baseline.
    """
    baseline_ids = {case.id for case in baseline}
    candidate_ids = {case.id for case in candidate}
    extra = candidate_ids - baseline_ids
    if extra:
        raise CorpusError(
            f"candidate corpus has {len(extra)} id(s) absent from baseline: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )
    return baseline.merge_candidates(candidate)


def load_pair(baseline_path: str | Path, candidate_path: str | Path) -> Corpus:
    """Load two JSONL corpora from disk and return the merged comparison corpus."""
    baseline = Corpus.from_jsonl(baseline_path)
    candidate = Corpus.from_jsonl(candidate_path)
    return compare_runs(baseline, candidate)
