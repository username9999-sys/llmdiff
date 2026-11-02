"""Output helpers for the llmdiff CLI.

Everything here is stdlib-only.  Colour is applied only when the output stream
is a TTY and the ``NO_COLOR`` environment variable is not set (see
https://no-color.org/).  When the optional ``rich`` package is importable the
table printer upgrades to a rich table, but plain-text output is always
available as a fallback so the CLI keeps its zero-dependency promise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "supports_color",
    "colorize",
    "style",
    "print_table",
    "print_kv",
    "write_text",
    "write_json",
    "dump_json",
    "emit",
    "err",
    "rule",
    "STATUS_COLORS",
]

# ANSI SGR codes keyed by logical style name.
_STYLES: Mapping[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bold_red": "\033[1;31m",
    "bold_green": "\033[1;32m",
    "bold_yellow": "\033[1;33m",
    "bold_blue": "\033[1;34m",
    "bold_cyan": "\033[1;36m",
}

# Verdict/status -> style mapping used across commands.
STATUS_COLORS: Mapping[str, str] = {
    "win": "bold_green",
    "improvement": "bold_green",
    "pass": "bold_green",
    "ok": "bold_green",
    "tie": "yellow",
    "loss": "bold_red",
    "regression": "bold_red",
    "fail": "bold_red",
    "error": "bold_red",
    "warn": "bold_yellow",
    "info": "cyan",
}


def supports_color(stream: Any = None) -> bool:
    """Return True when colourised output is appropriate for ``stream``.

    Colour is disabled when ``NO_COLOR`` is set (any value), when the stream
    is not a TTY, or when ``TERM`` is ``dumb``.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream if stream is not None else sys.stdout
    isatty = getattr(stream, "isatty", None)
    if isatty is None or not isatty():
        return False
    term = os.environ.get("TERM", "")
    return term != "dumb"


def style(text: str, name: str, *, stream: Any = None, enabled: bool | None = None) -> str:
    """Wrap ``text`` in the ANSI codes for style ``name`` when colour is on."""
    code = _STYLES.get(name)
    if code is None:
        return text
    if enabled is None:
        enabled = supports_color(stream)
    if not enabled:
        return text
    return f"{code}{text}{_STYLES['reset']}"


def colorize(text: str, name: str, *, stream: Any = None, enabled: bool | None = None) -> str:
    """Alias for :func:`style` kept for readability at call sites."""
    return style(text, name, stream=stream, enabled=enabled)


def emit(message: str = "", *, stream: Any = None, end: str = "\n") -> None:
    """Print ``message`` to ``stream`` (defaults to stdout)."""
    stream = stream if stream is not None else sys.stdout
    stream.write(message + end)


def err(message: str = "", *, end: str = "\n") -> None:
    """Print ``message`` to stderr."""
    sys.stderr.write(message + end)


def rule(title: str = "", *, stream: Any = None, width: int = 72) -> None:
    """Print a horizontal rule, optionally with a centred title."""
    stream = stream if stream is not None else sys.stdout
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        line = f"{'─' * left} {title} {'─' * right}"
    else:
        line = "─" * width
    stream.write(line[:width] + "\n")


def _cell_width(value: Any) -> int:
    return len(str(value))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    stream: Any = None,
    align: Sequence[str] | None = None,
    colorize_row: Any = None,
) -> None:
    """Print an ASCII table with padded columns.

    Args:
        headers: Column header labels.
        rows: Row values; each row must have the same arity as ``headers``.
        stream: Output stream (defaults to stdout).
        align: Per-column alignment, ``"l"`` (left, default) or ``"r"`` (right).
        colorize_row: Optional callable ``(row_index, row) -> style_name | None``
            used to colour an entire row.
    """
    stream = stream if stream is not None else sys.stdout
    headers = [str(h) for h in headers]
    materialised = [[_stringify(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in materialised:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    if align is None:
        align = ["l"] * len(headers)

    def _format_row(cells: Sequence[str], header: bool = False) -> str:
        parts = []
        for i, cell in enumerate(cells):
            width = widths[i] if i < len(widths) else 0
            if i < len(align) and align[i] == "r":
                parts.append(cell.rjust(width))
            else:
                parts.append(cell.ljust(width))
        line = "  ".join(parts).rstrip()
        if header:
            line = style(line, "bold", stream=stream)
        return line

    emit(_format_row(headers, header=True), stream=stream)
    emit("  ".join("─" * w for w in widths).rstrip(), stream=stream)
    for index, row in enumerate(materialised):
        line = _format_row(row)
        if colorize_row is not None:
            name = colorize_row(index, row)
            if name:
                line = style(line, name, stream=stream)
        emit(line, stream=stream)


def print_kv(
    pairs: Iterable[tuple[str, Any]],
    *,
    stream: Any = None,
    key_style: str = "bold_cyan",
    value_style: str | None = None,
) -> None:
    """Print ``key: value`` lines with aligned keys."""
    stream = stream if stream is not None else sys.stdout
    items = [(str(k), _stringify(v)) for k, v in pairs]
    width = max((len(k) for k, _ in items), default=0)
    for key, value in items:
        key_text = style(key.ljust(width), key_style, stream=stream)
        if value_style:
            value = style(value, value_style, stream=stream)
        emit(f"{key_text}  {value}", stream=stream)


def write_text(path: str | Path, text: str, *, stream: Any = None) -> Path:
    """Write ``text`` to ``path`` (UTF-8), creating parent directories.

    Returns the resolved :class:`~pathlib.Path`.  When ``path`` is ``"-"`` the
    text is written to ``stream`` (stdout) instead and ``Path("-")`` is
    returned.
    """
    if str(path) == "-":
        emit(text, stream=stream, end="")
        return Path("-")
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def dump_json(data: Any, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    """Serialise ``data`` to a JSON string."""
    return json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)


def write_json(path: str | Path, data: Any, *, indent: int | None = 2, stream: Any = None) -> Path:
    """Write ``data`` as JSON to ``path`` (or stdout when ``path`` is ``"-"``)."""
    return write_text(path, dump_json(data, indent=indent) + "\n", stream=stream)
