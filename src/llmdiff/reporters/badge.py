"""shields.io-style badge renderer.

Produces a badge for the overall verdict in three flavours:

* ``json``  — shields.io endpoint JSON
  (``{"schemaVersion": 1, "label": ..., "message": ..., "color": ...}``)
* ``markdown`` — ``[![label](https://img.shields.io/badge/label-message-color)](url)``
* ``url`` — just the shields.io image URL
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from llmdiff.core.result import DiffResult
from llmdiff.reporters.base import BaseReporter, format_pct

__all__ = ["BadgeReporter", "BADGE_COLORS"]

#: shields.io colour per verdict.
BADGE_COLORS = {
    "win": "brightgreen",
    "tie": "yellow",
    "loss": "red",
}


class BadgeReporter(BaseReporter):
    """Render a shields.io-style badge for the overall verdict."""

    name = "badge"

    def __init__(self, *, style: str = "markdown", url: str = "") -> None:
        if style not in ("json", "markdown", "url"):
            raise ValueError(f"unknown badge style: {style!r} (json|markdown|url)")
        self.style = style
        self.url = url

    def render(self, result: DiffResult) -> str:
        """Render the badge in the configured style."""
        info = self.badge_info(result)
        if self.style == "json":
            return json.dumps(info, separators=(",", ":")) + "\n"
        image_url = self.image_url(info)
        if self.style == "url":
            return image_url + "\n"
        label = quote(info["label"])
        message = quote(info["message"])
        color = info["color"]
        badge = f"https://img.shields.io/badge/{label}-{message}-{color}"
        if self.url:
            return f"[![llmdiff]({badge})]({self.url})\n"
        return f"![llmdiff]({badge})\n"

    def badge_info(self, result: DiffResult) -> dict[str, Any]:
        """Return the shields.io endpoint JSON fields for ``result``."""
        matrix = result.matrix
        verdict = matrix.verdict
        return {
            "schemaVersion": 1,
            "label": "llmdiff",
            "message": f"{verdict} {format_pct(matrix.regression_rate, digits=0)} regress",
            "color": BADGE_COLORS.get(verdict, "lightgrey"),
            "namedLogo": "git",
        }

    @staticmethod
    def image_url(info: dict[str, Any]) -> str:
        """Build the shields.io static badge URL from ``info`` fields."""
        label = quote(str(info["label"]))
        message = quote(str(info["message"]))
        color = quote(str(info["color"]))
        logo = quote(str(info.get("namedLogo", "")))
        url = f"https://img.shields.io/badge/{label}-{message}-{color}"
        if logo:
            url += f"?logo={logo}"
        return url
