"""``llmdiff stats`` - print corpus statistics.

Usage::

    llmdiff stats corpus.jsonl

Prints totals, tag/model breakdown, size distribution and an ASCII histogram
of prompt lengths.
"""

from __future__ import annotations

import argparse
from collections import Counter

from llmdiff.cli.corpus_io import load_corpus
from llmdiff.cli.output import emit, err, print_kv, print_table, rule
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_stats", "add_stats_parser", "prompt_length_histogram", "corpus_report"]

_HIST_BUCKETS = (0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 100000)


def prompt_length_histogram(lengths: list[int], *, width: int = 40) -> list[str]:
    """Render an ASCII histogram of prompt lengths.

    Buckets are defined by :data:`_HIST_BUCKETS`; each line shows the bucket
    range, a bar proportional to the bucket count, and the count.
    """
    if not lengths:
        return ["(no cases)"]
    counts = Counter()
    for length in lengths:
        for i in range(len(_HIST_BUCKETS) - 1):
            if _HIST_BUCKETS[i] <= length < _HIST_BUCKETS[i + 1]:
                counts[i] += 1
                break
        else:
            counts[len(_HIST_BUCKETS) - 2] += 1
    peak = max(counts.values()) if counts else 1
    lines = []
    for i in range(len(_HIST_BUCKETS) - 1):
        lo, hi = _HIST_BUCKETS[i], _HIST_BUCKETS[i + 1]
        count = counts.get(i, 0)
        bar = "█" * round(width * count / peak) if peak else ""
        label = f"{lo:>6}-{hi if hi < 100000 else '∞':>6}"
        lines.append(f"{label}  {bar} {count}")
    return lines


def corpus_report(corpus) -> str:
    """Build the full stats report text for ``corpus``."""
    stats = corpus.stats()
    lines: list[str] = []
    lines.append(f"corpus stats: {len(corpus)} cases")
    lines.append(f"  with tags : {stats.with_tags}")
    lines.append(f"  models    : {', '.join(stats.models) if stats.models else '(none)'}")
    tag_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    baseline_vs_candidate: list[tuple[int, int]] = []
    for case in corpus:
        meta = case.metadata or {}
        for tag in meta.get("tags", []) or []:
            tag_counter[str(tag)] += 1
        if meta.get("model"):
            model_counter[str(meta["model"])] += 1
        prompt_lengths.append(len(case.prompt))
        output_lengths.append(len(case.candidate_output))
        baseline_vs_candidate.append((len(case.baseline_output), len(case.candidate_output)))
    lines.append("")
    lines.append("tags:")
    for tag, count in tag_counter.most_common():
        lines.append(f"  {tag:<24} {count}")
    if not tag_counter:
        lines.append("  (none)")
    lines.append("")
    lines.append("models:")
    for model, count in model_counter.most_common():
        lines.append(f"  {model:<24} {count}")
    if not model_counter:
        lines.append("  (none)")
    lines.append("")
    if prompt_lengths:
        total = sum(prompt_lengths)
        lines.append(
            f"prompt length: min={min(prompt_lengths)} max={max(prompt_lengths)} "
            f"mean={total / len(prompt_lengths):.1f} total={total}"
        )
        lines.append("")
        lines.append("prompt length histogram (chars):")
        lines.extend("  " + line for line in prompt_length_histogram(prompt_lengths))
        lines.append("")
        out_total = sum(output_lengths)
        lines.append(
            f"candidate output length: min={min(output_lengths)} max={max(output_lengths)} "
            f"mean={out_total / len(output_lengths):.1f} total={out_total}"
        )
        deltas = [cand - base for base, cand in baseline_vs_candidate]
        if deltas:
            lines.append(
                f"output length delta (candidate - baseline): min={min(deltas)} "
                f"max={max(deltas)} mean={sum(deltas) / len(deltas):+.1f}"
            )
    return "\n".join(lines) + "\n"


def add_stats_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``stats`` subcommand."""
    p = sub.add_parser(
        "stats",
        help="print corpus statistics (totals, tags, models, histograms)",
    )
    p.add_argument("corpus", help="path to the corpus file ('-' for stdin)")
    p.add_argument("--tags", action="store_true", help="also print the tag table")
    p.set_defaults(func=run_stats)


def run_stats(args: argparse.Namespace) -> int:
    """Execute ``stats``.  Returns 0 on success, 2 on error."""
    try:
        corpus = load_corpus(args.corpus)
    except LLMDiffError as exc:
        err(f"error: cannot load corpus: {exc}")
        return 2
    emit(corpus_report(corpus), end="")
    return 0
