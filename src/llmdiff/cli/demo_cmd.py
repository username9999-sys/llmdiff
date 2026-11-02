"""``llmdiff demo`` - fully offline end-to-end demo.

Usage::

    llmdiff demo [--format terminal|md|json] [--dimensions a,b] [--tolerance 0.05]

Builds an example corpus in memory, runs :func:`llmdiff.core.evaluate` and
prints the verdict matrix.  Never touches the network.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.examples import demo_corpus
from llmdiff.cli.output import emit, err, rule, style
from llmdiff.cli.renderers import render_result
from llmdiff.core import evaluate
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_demo", "add_demo_parser"]


def add_demo_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``demo`` subcommand."""
    p = sub.add_parser(
        "demo",
        help="run a fully offline demo (no network)",
        description="Evaluate a built-in example corpus and print the verdict matrix.",
    )
    p.add_argument("--format", "-f", default="terminal", help="output format (default: terminal)")
    p.add_argument("--dimensions", "-d", default=None, metavar="A,B", help="dimensions to evaluate")
    p.add_argument("--tolerance", "-t", type=float, default=0.0, help="tie tolerance")
    p.set_defaults(func=run_demo)


def run_demo(args: argparse.Namespace) -> int:
    """Execute ``demo``.  Returns 0 on success, 2 on error."""
    corpus = demo_corpus()
    dims = (
        [d.strip() for d in args.dimensions.split(",") if d.strip()]
        if args.dimensions
        else None
    )
    try:
        result = evaluate(corpus, dimensions=dims, tolerance=args.tolerance)
    except LLMDiffError as exc:
        err(f"error: demo evaluation failed: {exc}")
        return 2
    rule("llmdiff demo - offline evaluation")
    emit(f"corpus: {len(corpus)} built-in cases, dimensions: "
         f"{', '.join(s.dimension for s in result.matrix.dimensions)}")
    emit("")
    try:
        emit(render_result(result, args.format), end="")
    except KeyError as exc:
        err(f"error: {exc}")
        return 2
    m = result.matrix
    verdict_style = {"win": "bold_green", "loss": "bold_red", "tie": "yellow"}.get(m.verdict, "bold")
    emit("")
    emit(style(f"overall verdict: {m.verdict}  (wins={m.wins} ties={m.ties} losses={m.losses}, "
               f"regression_rate={m.regression_rate:.3f})", verdict_style))
    emit("demo finished - no network was used.")
    return 0
