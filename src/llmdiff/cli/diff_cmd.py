"""``llmdiff diff`` - compare two corpus files case-by-case.

Usage::

    llmdiff diff baseline.jsonl candidate.jsonl [--format ...] [--output ...]

Cases are matched by ``id``: the baseline corpus supplies ``baseline_output``
and the candidate corpus supplies ``candidate_output``; prompts and metadata
are taken from the baseline (falling back to the candidate when an id only
exists there).
"""

from __future__ import annotations

import argparse

from llmdiff.cli.config import CliConfig, load_config
from llmdiff.cli.corpus_io import load_corpus, save_corpus
from llmdiff.cli.gate import GateThresholds, gate_from_result
from llmdiff.cli.output import emit, err, style
from llmdiff.cli.renderers import render_result
from llmdiff.core import Case, Corpus, evaluate
from llmdiff.core.errors import CorpusError, LLMDiffError, RegressionGateError

__all__ = ["run_diff", "add_diff_parser", "merge_corpora"]


def merge_corpora(baseline: Corpus, candidate: Corpus) -> Corpus:
    """Merge two corpora into one comparison corpus matched by case id.

    Raises:
        CorpusError: when no ids overlap.
    """
    candidate_by_id = {case.id: case for case in candidate}
    merged: list[Case] = []
    for base in baseline:
        cand = candidate_by_id.get(base.id)
        if cand is None:
            continue
        metadata = dict(base.metadata)
        metadata.setdefault("baseline_model", base.metadata.get("model"))
        metadata.setdefault("candidate_model", cand.metadata.get("model"))
        merged.append(
            Case(
                id=base.id,
                prompt=base.prompt or cand.prompt,
                baseline_output=base.baseline_output,
                candidate_output=cand.candidate_output,
                metadata=metadata,
            )
        )
    if not merged:
        raise CorpusError("no overlapping case ids between baseline and candidate corpora")
    return Corpus(merged)


def add_diff_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``diff`` subcommand."""
    p = sub.add_parser(
        "diff",
        help="diff two corpus files matched by case id",
        description="Match cases by id across a baseline and a candidate corpus, "
        "evaluate the merged corpus and render the result.",
    )
    p.add_argument("baseline", help="baseline corpus file (.jsonl/.json/.csv)")
    p.add_argument("candidate", help="candidate corpus file")
    p.add_argument("--dimensions", "-d", default=None, metavar="A,B", help="dimensions to evaluate")
    p.add_argument("--tolerance", "-t", type=float, default=None, help="tie tolerance")
    p.add_argument("--format", "-f", default=None, help="output format (default: terminal)")
    p.add_argument("--output", "-o", default=None, help="write rendered output to this file")
    p.add_argument(
        "--merged-out",
        default=None,
        help="optionally write the merged comparison corpus to this file",
    )
    p.add_argument("--max-loss-rate", type=float, default=None, help="gate: max loss rate")
    p.add_argument("--config", default=None, help="path to llmdiff.toml")
    p.set_defaults(func=run_diff)


def run_diff(args: argparse.Namespace, config: CliConfig | None = None) -> int:
    """Execute ``diff``.  Returns the exit code."""
    if config is None:
        try:
            config = load_config(args.config)
        except LLMDiffError as exc:
            err(f"error: {exc}")
            return 2
    try:
        baseline = load_corpus(args.baseline)
        candidate = load_corpus(args.candidate)
        merged = merge_corpora(baseline, candidate)
    except LLMDiffError as exc:
        err(f"error: {exc}")
        return 2
    if args.merged_out:
        save_corpus(merged, args.merged_out)
        emit(f"wrote merged corpus ({len(merged)} cases) to {args.merged_out}")
    dims = [d.strip() for d in args.dimensions.split(",") if d.strip()] if args.dimensions else (
        list(config.dimensions) if config.dimensions else None
    )
    tolerance = args.tolerance if args.tolerance is not None else config.tolerance
    try:
        result = evaluate(merged, dimensions=dims, tolerance=tolerance)
    except LLMDiffError as exc:
        err(f"error: evaluation failed: {exc}")
        return 2
    fmt = args.format or config.format or "terminal"
    try:
        text = render_result(result, fmt)
    except KeyError as exc:
        err(f"error: {exc}")
        return 2
    if args.output:
        from llmdiff.cli.output import write_text

        write_text(args.output, text)
        emit(f"wrote {args.output} ({fmt})")
    else:
        emit(text, end="")
    thresholds = GateThresholds(
        max_loss_rate=args.max_loss_rate if args.max_loss_rate is not None else config.max_loss_rate,
        max_regressed_dims=config.max_regressed_dims,
        min_win_rate=config.min_win_rate,
    )
    try:
        gate_from_result(result, thresholds)
    except RegressionGateError as exc:
        for failure in exc.failures:
            emit(style(f"GATE FAIL: {failure}", "bold_red"))
        return 1
    emit(style("gate: PASS", "bold_green"))
    return 0
