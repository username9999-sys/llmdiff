"""``llmdiff gate`` - decide pass/fail for a saved result JSON.

Usage::

    llmdiff gate result.json --max-loss-rate 0.05 --max-regressed-dims 0 --min-win-rate 0.0

Exit code 0 when the gate passes, 1 when it fails (with every violation
printed), 2 on usage/IO errors.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.config import CliConfig, load_config
from llmdiff.cli.gate import GateThresholds, evaluate_gate, load_result_json
from llmdiff.cli.output import emit, err, style
from llmdiff.core.errors import LLMDiffError, RegressionGateError

__all__ = ["run_gate", "add_gate_parser"]


def add_gate_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``gate`` subcommand."""
    p = sub.add_parser(
        "gate",
        help="apply a regression gate to a saved result JSON",
        description="Load a JSON DiffResult and decide pass/fail against thresholds. "
        "Exit 1 on failure.",
    )
    p.add_argument("result", help="path to a result JSON produced by 'compare --format json' ('-' for stdin)")
    p.add_argument("--max-loss-rate", type=float, default=None, help="max allowed loss rate")
    p.add_argument("--max-regressed-dims", type=int, default=None, help="max regressed dimensions")
    p.add_argument("--min-win-rate", type=float, default=None, help="min required win rate")
    p.add_argument("--config", default=None, help="path to llmdiff.toml for default thresholds")
    p.add_argument("--quiet", "-q", action="store_true", help="only print failures")
    p.set_defaults(func=run_gate)


def run_gate(args: argparse.Namespace, config: CliConfig | None = None) -> int:
    """Execute ``gate``.  Returns 0 (pass), 1 (fail) or 2 (error)."""
    if config is None:
        try:
            config = load_config(args.config)
        except LLMDiffError as exc:
            err(f"error: {exc}")
            return 2
    try:
        result = load_result_json(args.result)
    except LLMDiffError as exc:
        err(f"error: cannot load result: {exc}")
        return 2
    thresholds = GateThresholds(
        max_loss_rate=args.max_loss_rate if args.max_loss_rate is not None else config.max_loss_rate,
        max_regressed_dims=(
            args.max_regressed_dims
            if args.max_regressed_dims is not None
            else config.max_regressed_dims
        ),
        min_win_rate=args.min_win_rate if args.min_win_rate is not None else config.min_win_rate,
    ).with_defaults()
    failures = evaluate_gate(result, thresholds)
    matrix = result.matrix
    if not args.quiet:
        emit(
            f"result: cases={matrix.total_cases} wins={matrix.wins} ties={matrix.ties} "
            f"losses={matrix.losses} regression_rate={matrix.regression_rate:.4f}"
        )
        emit(f"thresholds: {thresholds.to_dict()}")
    if failures:
        for failure in failures:
            emit(style(f"GATE FAIL: {failure}", "bold_red"))
        emit(style(f"gate: FAIL ({len(failures)} violation(s))", "bold_red"))
        return 1
    if not args.quiet:
        emit(style("gate: PASS", "bold_green"))
    return 0


def gate_or_raise(result, thresholds: GateThresholds) -> None:
    """Convenience wrapper raising :class:`RegressionGateError` on failure."""
    failures = evaluate_gate(result, thresholds)
    if failures:
        raise RegressionGateError("regression gate failed", failures=failures)
