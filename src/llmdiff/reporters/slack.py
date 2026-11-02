"""Slack Block Kit report renderer.

Builds a Slack ``POST /chat.postMessage`` payload (blocks + text fallback)
summarising the verdict matrix. Pure JSON — no network calls.
"""

from __future__ import annotations

import json
from typing import Any

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, format_delta, format_pct, format_score

__all__ = ["SlackReporter"]

_VERDICT_EMOJI = {"win": ":white_check_mark:", "tie": ":heavy_minus_sign:", "loss": ":x:"}
_VERDICT_COLOR = {"win": "#2eb67d", "tie": "#ecb22e", "loss": "#e01e5a"}


class SlackReporter(BaseReporter):
    """Render a :class:`DiffResult` as a Slack Block Kit payload."""

    name = "slack"

    def __init__(self, *, pretty: bool = False, channel: str | None = None) -> None:
        self.pretty = pretty
        self.channel = channel

    def render(self, result: DiffResult) -> str:
        """Render the Slack message payload as a JSON string."""
        payload = self.to_payload(result)
        if self.pretty:
            return json.dumps(payload, indent=2, ensure_ascii=False)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def to_payload(self, result: DiffResult) -> dict[str, Any]:
        """Build the ``chat.postMessage`` payload for ``result``."""
        matrix = result.matrix
        emoji = _VERDICT_EMOJI.get(matrix.verdict, ":grey_question:")
        color = _VERDICT_COLOR.get(matrix.verdict, "#808080")
        text = (
            f"{emoji} llmdiff {matrix.verdict.upper()}: {matrix.wins}W/{matrix.ties}T/"
            f"{matrix.losses}L ({format_pct(matrix.regression_rate)} regressions)"
        )
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"llmdiff: {matrix.verdict.upper()}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{matrix.wins}* wins · *{matrix.ties}* ties · *{matrix.losses}* losses "
                        f"across *{matrix.total_cases}* case(s) — regression rate "
                        f"*{format_pct(matrix.regression_rate)}*"
                    ),
                },
            },
            {"type": "divider"},
        ]
        if matrix.dimensions:
            rows = ["*Dimension* | *Verdict* | *W/T/L* | *Δ*"]
            for summary in matrix.dimensions:
                rows.append(
                    "`{dim}` | {emoji} {v} | {w}/{t}/{l} | {d}".format(
                        dim=summary.dimension,
                        emoji=_VERDICT_EMOJI.get(summary.verdict, ""),
                        v=summary.verdict,
                        w=summary.wins,
                        t=summary.ties,
                        l=summary.losses,
                        d=format_delta(summary.delta),
                    )
                )
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(rows)},
                }
            )
        regressed = result.regressions
        if regressed:
            lines = [f"• `{c.case_id}` ({c.losses} losses)" for c in regressed[:10]]
            if len(regressed) > 10:
                lines.append(f"• …and {len(regressed) - 10} more")
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": ":x: *Regressed cases:*\n" + "\n".join(lines),
                    },
                }
            )
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "baseline vs candidate means per dimension shown with delta; "
                            "worst delta "
                            + format_delta(min((s.delta for s in matrix.dimensions), default=0.0))
                        ),
                    }
                ],
            }
        )
        payload: dict[str, Any] = {"text": text, "blocks": blocks}
        if self.channel:
            payload["channel"] = self.channel
        # Attach the colour as an attachment for clients that render it.
        payload["attachments"] = [{"color": color, "blocks": []}]
        return payload

    @staticmethod
    def format_score_unused(value: float) -> str:  # pragma: no cover - helper kept for API symmetry
        return format_score(value)
