"""Analytics report builder: combine stats/significance/trends into one report.

Takes a DiffResult (or a list of them) and produces a markdown or JSON report
with per-dimension stats, significance tests vs baseline, and trend info when
multiple runs are given.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from llmdiff.analytics.significance import cohens_d, paired_t_test, sign_test
from llmdiff.analytics.stats import describe
from llmdiff.analytics.trends import linear_regression, r_squared

__all__ = ["dimension_report", "build_report", "render_markdown", "render_json"]


def _case_scores(result: Any, dimension: str) -> tuple[list[float], list[float]]:
    """Return (baseline_scores, candidate_scores) for one dimension across cases."""
    base: list[float] = []
    cand: list[float] = []
    for case in result.cases:
        for r in case.results:
            if r.dimension == dimension:
                base.append(r.baseline_score)
                cand.append(r.candidate_score)
                break
    return base, cand


def dimension_report(result: Any, dimension: str) -> dict[str, Any]:
    """Full per-dimension analytics for one DiffResult."""
    summary = result.matrix.get(dimension)
    base, cand = _case_scores(result, dimension)
    out: dict[str, Any] = {
        "dimension": dimension,
        "wins": summary.wins if summary else 0,
        "ties": summary.ties if summary else 0,
        "losses": summary.losses if summary else 0,
        "baseline_mean": summary.baseline_mean if summary else 0.0,
        "candidate_mean": summary.candidate_mean if summary else 0.0,
        "delta": summary.delta if summary else 0.0,
        "verdict": summary.verdict if summary else "tie",
    }
    if base and cand and len(base) == len(cand) and len(base) >= 2:
        out["sign_test_p"] = sign_test(base, cand)
        out["paired_t_p"] = paired_t_test(base, cand)
        out["cohens_d"] = cohens_d(base, cand)
        out["baseline_scores"] = describe(base)
        out["candidate_scores"] = describe(cand)
    return out


def build_report(
    results: Sequence[Any], *, dimensions: Sequence[str] | None = None
) -> dict[str, Any]:
    """Build a full analytics report over one or more DiffResult objects.

    With a single result this is per-dimension analytics. With several results
    (ordered by time) it also includes per-dimension trend (slope, r^2).
    """
    if not results:
        raise ValueError("at least one result is required")
    first = results[0]
    all_dims = [d.dimension for d in first.matrix.dimensions]
    dims = list(dimensions) if dimensions is not None else all_dims

    per_dim: dict[str, Any] = {}
    for dim in dims:
        entry = dimension_report(first, dim)
        if len(results) > 1:
            series = []
            for res in results:
                s = res.matrix.get(dim)
                series.append(s.loss_rate if s else 0.0)
            intercept, sl = linear_regression(series)
            entry["trend"] = {
                "loss_rate_series": series,
                "slope": sl,
                "intercept": intercept,
                "r_squared": r_squared(series),
                "n_runs": len(results),
            }
        per_dim[dim] = entry

    matrix = first.matrix
    return {
        "schema_version": 1,
        "n_runs": len(results),
        "overall": {
            "verdict": matrix.verdict,
            "wins": matrix.wins,
            "ties": matrix.ties,
            "losses": matrix.losses,
            "regression_rate": matrix.regression_rate,
            "regressed_dimensions": list(matrix.regressed_dimensions),
        },
        "dimensions": per_dim,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report dict as markdown."""
    lines: list[str] = ["# llmdiff analytics report", ""]
    overall = report["overall"]
    lines.append(f"- runs analyzed: {report['n_runs']}")
    lines.append(f"- overall verdict: **{overall['verdict']}**")
    lines.append(
        f"- wins/ties/losses: {overall['wins']} / {overall['ties']} / {overall['losses']}"
        f" (regression rate {overall['regression_rate']:.3f})"
    )
    if overall["regressed_dimensions"]:
        lines.append(f"- regressed dimensions: {', '.join(overall['regressed_dimensions'])}")
    lines.append("")
    lines.append("| dimension | win | tie | loss | baseline | candidate | delta | p(t) | d |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, d in report["dimensions"].items():
        p_t = d.get("paired_t_p")
        p_str = f"{p_t:.4f}" if p_t is not None else "-"
        d_str = f"{d.get('cohens_d', 0.0):.3f}"
        lines.append(
            f"| {name} | {d['wins']} | {d['ties']} | {d['losses']} | "
            f"{d['baseline_mean']:.3f} | {d['candidate_mean']:.3f} | {d['delta']:+.3f} | "
            f"{p_str} | {d_str} |"
        )
    for name, d in report["dimensions"].items():
        trend = d.get("trend")
        if trend:
            lines.append("")
            lines.append(f"### trend: {name}")
            lines.append(
                f"- loss-rate slope over {trend['n_runs']} runs: {trend['slope']:+.5f}"
                f" (r^2 = {trend['r_squared']:.3f})"
            )
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    """Render a report dict as pretty JSON."""
    return json.dumps(report, indent=2, sort_keys=True)
