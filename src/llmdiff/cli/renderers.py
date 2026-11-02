"""Built-in result renderers for the CLI.

The CLI prefers the optional :mod:`llmdiff.reporters` package when it is
importable; these stdlib-only renderers are the guaranteed fallback and also
implement the formats the reporters package may not cover (``terminal``,
``csv``, ``junit``, ``sarif``, ``html``).

Every renderer takes a :class:`llmdiff.core.DiffResult` and returns a string.
"""

from __future__ import annotations

import csv
import html
import io
import json
from collections.abc import Mapping
from typing import Any

from llmdiff.core import DiffResult
from llmdiff.core.checks import Verdict

__all__ = [
    "RENDERERS",
    "render_result",
    "render_terminal",
    "render_markdown",
    "render_json",
    "render_csv",
    "render_junit",
    "render_sarif",
    "render_html",
    "render_matrix_table",
]

#: Registry of format name -> renderer callable.
RENDERERS: Mapping[str, Any] = {}


def _register(name: str):
    def deco(fn):
        RENDERERS[name] = fn
        return fn

    return deco


def render_result(result: DiffResult, fmt: str) -> str:
    """Render ``result`` in format ``fmt``.

    Falls back to the reporters package for formats not built in here.
    Raises:
        KeyError: when the format is unknown.
    """
    fn = RENDERERS.get(fmt)
    if fn is not None:
        return fn(result)
    try:
        from llmdiff.reporters import render  # type: ignore[attr-defined]
    except Exception:
        raise KeyError(f"unknown format {fmt!r} (built-ins: {sorted(RENDERERS)})") from None
    return render(result, fmt)


def _verdict_symbol(verdict: str) -> str:
    return {"win": "W", "tie": "=", "loss": "L"}.get(verdict, "?")


def render_matrix_table(result: DiffResult) -> str:
    """Plain-text matrix table (dimension x win/tie/loss + means)."""
    lines = [
        f"{'dimension':<14} {'wins':>5} {'ties':>5} {'losses':>6} {'base':>7} {'cand':>7} {'delta':>7}  verdict",
        "-" * 72,
    ]
    for summary in result.matrix.dimensions:
        delta = summary.candidate_mean - summary.baseline_mean
        lines.append(
            f"{summary.dimension:<14} {summary.wins:>5} {summary.ties:>5} {summary.losses:>6} "
            f"{summary.baseline_mean:>7.3f} {summary.candidate_mean:>7.3f} {delta:>+7.3f}  {summary.verdict}"
        )
    m = result.matrix
    lines.append("-" * 72)
    lines.append(
        f"{'TOTAL':<14} {m.wins:>5} {m.ties:>5} {m.losses:>6} "
        f"{'':>7} {'':>7} {'':>7}  {m.verdict} (regression_rate={m.regression_rate:.3f})"
    )
    return "\n".join(lines) + "\n"


@_register("terminal")
def render_terminal(result: DiffResult) -> str:
    """Human-friendly terminal report: matrix table plus per-case regressions."""
    out = ["== llmdiff verdict matrix ==", render_matrix_table(result), ""]
    out.append("== per-case summary ==")
    out.append(f"{'case':<24} {'W':>3} {'=':>3} {'L':>3}  verdict")
    out.append("-" * 44)
    for case in result.cases:
        out.append(
            f"{case.case_id:<24} {case.wins:>3} {case.ties:>3} {case.losses:>3}  {case.verdict}"
        )
    regressions = result.regressions
    if regressions:
        out.append("")
        out.append(f"== regressions ({len(regressions)}) ==")
        for case in regressions:
            out.append(f"  {case.case_id}:")
            for r in case.regressions():
                out.append(
                    f"    L {r.dimension}: {r.baseline_score:.3f} -> {r.candidate_score:.3f}"
                    + (f"  ({r.detail})" if r.detail else "")
                )
    return "\n".join(out) + "\n"


@_register("md")
@_register("markdown")
def render_markdown(result: DiffResult) -> str:
    """Markdown report with a matrix table and regression list."""
    m = result.matrix
    lines = [
        "# llmdiff report",
        "",
        f"- cases: {m.total_cases}",
        f"- wins: {m.wins}  ties: {m.ties}  losses: {m.losses}",
        f"- overall verdict: **{m.verdict}** (regression rate {m.regression_rate:.1%})",
        "",
        "## Verdict matrix",
        "",
        "| dimension | wins | ties | losses | baseline mean | candidate mean | delta | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in m.dimensions:
        delta = s.candidate_mean - s.baseline_mean
        lines.append(
            f"| {s.dimension} | {s.wins} | {s.ties} | {s.losses} | {s.baseline_mean:.3f} "
            f"| {s.candidate_mean:.3f} | {delta:+.3f} | {s.verdict} |"
        )
    lines += ["", "## Cases", "", "| case | W | = | L | verdict |", "|---|---:|---:|---:|---|"]
    for case in result.cases:
        lines.append(f"| {case.case_id} | {case.wins} | {case.ties} | {case.losses} | {case.verdict} |")
    if result.regressions:
        lines += ["", "## Regressions", ""]
        for case in result.regressions:
            lines.append(f"### {case.case_id}")
            lines.append("")
            for r in case.regressions():
                detail = f" — {r.detail}" if r.detail else ""
                lines.append(
                    f"- **{r.dimension}**: {r.baseline_score:.3f} -> {r.candidate_score:.3f}{detail}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


@_register("json")
def render_json(result: DiffResult) -> str:
    """Full JSON serialisation of the DiffResult."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n"


@_register("csv")
def render_csv(result: DiffResult) -> str:
    """One row per (case, dimension) with scores and verdict."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["case_id", "dimension", "verdict", "baseline_score", "candidate_score", "delta", "detail"]
    )
    for case in result.cases:
        for r in case.results:
            writer.writerow(
                [case.case_id, r.dimension, r.verdict.value, f"{r.baseline_score:.6f}",
                 f"{r.candidate_score:.6f}", f"{r.delta:.6f}", r.detail]
            )
    return buffer.getvalue()


@_register("junit")
def render_junit(result: DiffResult) -> str:
    """JUnit XML where each case is a testsuite and each loss is a failure."""
    m = result.matrix
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<testsuites tests="{m.total_cases}" failures="{len(result.regressions)}" '
        f'time="0">',
    ]
    for case in result.cases:
        failures = case.losses
        lines.append(
            f'  <testsuite name="{html.escape(case.case_id)}" tests="{len(case.results)}" '
            f'failures="{failures}">'
        )
        for r in case.results:
            if r.verdict is Verdict.LOSS:
                lines.append(
                    f'    <testcase name="{html.escape(r.dimension)}" classname="{html.escape(case.case_id)}">'
                )
                lines.append(
                    f'      <failure message="{html.escape(r.detail or "regression")}">'
                    f"{r.baseline_score:.6f} -&gt; {r.candidate_score:.6f}</failure>"
                )
                lines.append("    </testcase>")
            else:
                lines.append(
                    f'    <testcase name="{html.escape(r.dimension)}" classname="{html.escape(case.case_id)}"/>'
                )
        lines.append("  </testsuite>")
    lines.append("</testsuites>")
    return "\n".join(lines) + "\n"


@_register("sarif")
def render_sarif(result: DiffResult) -> str:
    """SARIF 2.1.0 log with one result per regressed (case, dimension)."""
    rules = [
        {
            "id": f"LLMDIFF/{s.dimension.upper()}",
            "shortDescription": {"text": f"llmdiff dimension {s.dimension}"},
        }
        for s in result.matrix.dimensions
    ]
    results: list[dict[str, Any]] = []
    for case in result.cases:
        for r in case.results:
            if r.verdict is Verdict.LOSS:
                results.append(
                    {
                        "ruleId": f"LLMDIFF/{r.dimension.upper()}",
                        "level": "error",
                        "message": {
                            "text": f"{case.case_id}: {r.dimension} regressed "
                            f"{r.baseline_score:.4f} -> {r.candidate_score:.4f}"
                            + (f" ({r.detail})" if r.detail else "")
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "corpus.jsonl"},
                                    "region": {"snippet": {"text": case.case_id}},
                                }
                            }
                        ],
                    }
                )
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "llmdiff", "version": "0.1.0", "rules": rules}},
                "results": results,
            }
        ],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


@_register("html")
def render_html(result: DiffResult) -> str:
    """Standalone HTML report (inline CSS, no external assets)."""
    m = result.matrix
    rows = []
    for s in m.dimensions:
        delta = s.candidate_mean - s.baseline_mean
        cls = {"win": "win", "loss": "loss", "tie": "tie"}.get(s.verdict, "tie")
        rows.append(
            f"<tr class='{cls}'><td>{html.escape(s.dimension)}</td><td>{s.wins}</td><td>{s.ties}</td>"
            f"<td>{s.losses}</td><td>{s.baseline_mean:.3f}</td><td>{s.candidate_mean:.3f}</td>"
            f"<td>{delta:+.3f}</td><td>{s.verdict}</td></tr>"
        )
    case_rows = []
    for case in result.cases:
        cls = "loss" if case.is_regression else ("win" if case.wins > case.losses else "tie")
        case_rows.append(
            f"<tr class='{cls}'><td>{html.escape(case.case_id)}</td><td>{case.wins}</td>"
            f"<td>{case.ties}</td><td>{case.losses}</td><td>{html.escape(case.verdict)}</td></tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>llmdiff report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#222}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:4px 10px}"
        ".win td{background:#e6f6e6}.loss td{background:#fbe4e4}.tie td{background:#fdf6e0}"
        "h1{font-size:1.4rem}</style></head><body>"
        f"<h1>llmdiff report</h1><p>cases: {m.total_cases} · wins: {m.wins} · ties: {m.ties} · "
        f"losses: {m.losses} · verdict: <b>{m.verdict}</b> · regression rate: {m.regression_rate:.1%}</p>"
        "<h2>Verdict matrix</h2><table><tr><th>dimension</th><th>wins</th><th>ties</th><th>losses</th>"
        "<th>baseline</th><th>candidate</th><th>delta</th><th>verdict</th></tr>"
        + "".join(rows)
        + "</table><h2>Cases</h2><table><tr><th>case</th><th>W</th><th>=</th><th>L</th><th>verdict</th></tr>"
        + "".join(case_rows)
        + "</table></body></html>\n"
    )
