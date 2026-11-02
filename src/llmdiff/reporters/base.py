"""Reporter protocol, registry and shared rendering helpers.

A *reporter* turns a :class:`~llmdiff.core.result.DiffResult` into a string in
some target format (markdown, HTML, JUnit XML, SARIF, ...). Every reporter in
this package implements the :class:`Reporter` protocol so callers can swap
formats without changing call sites.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llmdiff.core.errors import ReporterError
from llmdiff.core.result import DiffResult

__all__ = [
    "Reporter",
    "BaseReporter",
    "ReporterRegistry",
    "VERDICT_EMOJI",
    "VERDICT_LABEL",
    "safe_filename",
    "escape_md",
    "escape_md_cell",
    "verdict_emoji",
    "verdict_label",
    "format_delta",
    "format_pct",
    "format_score",
]

#: Emoji shown next to each verdict in rich (markdown/html) reporters.
VERDICT_EMOJI: dict[str, str] = {
    "win": "\u2705",  # ✅
    "tie": "\u2796",  # ➖
    "loss": "\u274c",  # ❌
}

#: Plain-text label for each verdict, used when emoji are not appropriate.
VERDICT_LABEL: dict[str, str] = {
    "win": "WIN",
    "tie": "TIE",
    "loss": "LOSS",
}

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, *, default: str = "report", extension: str = "") -> str:
    """Return ``name`` sanitised for use as a file name.

    Anything that is not a letter, digit, dot, dash or underscore becomes an
    underscore; leading dots are stripped so the result is never hidden; an
    empty result falls back to ``default``. ``extension`` (with or without a
    leading dot) is appended when given.

    >>> safe_filename("case/one: two", extension="md")
    'case_one_two.md'
    """
    cleaned = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    if not cleaned:
        cleaned = default
    if extension:
        ext = extension if extension.startswith(".") else f".{extension}"
        if not cleaned.endswith(ext):
            cleaned = f"{cleaned}{ext}"
    return cleaned


def escape_md(text: str) -> str:
    """Escape markdown special characters in ``text`` for inline rendering."""
    # Backslash-escape the ASCII punctuation that markdown treats specially.
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", text)


def escape_md_cell(text: str) -> str:
    """Escape ``text`` so it is safe inside a markdown table cell.

    Pipes are replaced (they would break the table) and newlines become
    spaces so one logical value stays on one row.
    """
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def verdict_emoji(verdict: str) -> str:
    """Return the emoji for ``verdict``, falling back to a question mark."""
    return VERDICT_EMOJI.get(verdict, "\u2753")


def verdict_label(verdict: str) -> str:
    """Return the upper-case label for ``verdict``."""
    return VERDICT_LABEL.get(verdict, verdict.upper())


def format_delta(value: float, *, digits: int = 3, sign: bool = True) -> str:
    """Format a signed score delta, e.g. ``+0.250`` / ``-0.125`` / ``0.000``."""
    if sign:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def format_pct(rate: float, *, digits: int = 1) -> str:
    """Format a 0..1 rate as a percentage string, e.g. ``42.0%``."""
    return f"{rate * 100:.{digits}f}%"


def format_score(value: float, *, digits: int = 3) -> str:
    """Format a 0..1 score without a sign."""
    return f"{value:.{digits}f}"


@runtime_checkable
class Reporter(Protocol):
    """Protocol every reporter implements.

    A reporter is stateless with respect to a single run: ``render`` takes a
    :class:`DiffResult` and returns the rendered string. Reporters may hold
    construction-time options (pretty/compact, theme, ...) but must not
    mutate the result.
    """

    @property
    def name(self) -> str:
        """Stable identifier of this reporter, e.g. ``"markdown"``."""
        ...

    def render(self, result: DiffResult) -> str:
        """Render ``result`` to a string in this reporter's format."""
        ...


class BaseReporter:
    """Convenience base class implementing :meth:`render_to_file`.

    Subclasses set :attr:`name` and implement :meth:`render`.
    """

    name: str = "base"

    def render(self, result: DiffResult) -> str:  # pragma: no cover - abstract
        raise ReporterError(f"reporter {self.name!r} does not implement render()")

    def render_to_file(self, result: DiffResult, path: str | Path) -> Path:
        """Render ``result`` and write it to ``path``, returning the path.

        Parent directories are created. The file is written as UTF-8.

        Raises:
            ReporterError: If rendering fails or the path is not writable.
        """
        try:
            text = self.render(result)
        except ReporterError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as ReporterError
            raise ReporterError(f"reporter {self.name!r} failed to render: {exc}") from exc
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ReporterError(f"cannot write report to {target}: {exc}") from exc
        return target

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class ReporterRegistry:
    """A name-to-reporter mapping with lookup helpers.

    Mirrors :class:`~llmdiff.core.registry.CheckRegistry` so the CLI can do
    ``registry.get("markdown")`` without caring which module provides it.
    """

    __slots__ = ("_reporters",)

    def __init__(self, reporters: list[Reporter] | tuple[Reporter, ...] = ()) -> None:
        self._reporters: dict[str, Reporter] = {}
        for reporter in reporters:
            self.register(reporter)

    def register(self, reporter: Reporter) -> None:
        """Register ``reporter`` under its ``name``, replacing any existing one.

        Raises:
            ReporterError: If the reporter has no usable name.
        """
        name = getattr(reporter, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ReporterError(f"reporter {reporter!r} has no usable name")
        self._reporters[name] = reporter

    def unregister(self, name: str) -> None:
        """Remove the reporter registered under ``name``. Missing names are ignored."""
        self._reporters.pop(name, None)

    def get(self, name: str) -> Reporter:
        """Return the reporter registered under ``name``.

        Raises:
            ReporterError: If no reporter is registered under that name.
        """
        try:
            return self._reporters[name]
        except KeyError:
            raise ReporterError(
                f"unknown reporter: {name!r} (available: {', '.join(self.names()) or 'none'})"
            ) from None

    def names(self) -> tuple[str, ...]:
        """Return the registered reporter names in registration order."""
        return tuple(self._reporters)

    def all(self) -> tuple[Reporter, ...]:
        """Return every registered reporter in registration order."""
        return tuple(self._reporters.values())

    def __contains__(self, name: object) -> bool:
        return name in self._reporters

    def __iter__(self):
        return iter(self._reporters.values())

    def __len__(self) -> int:
        return len(self._reporters)

    def render(self, name: str, result: DiffResult) -> str:
        """Convenience: look up ``name`` and render ``result`` with it."""
        return self.get(name).render(result)

    def render_to_file(self, name: str, result: DiffResult, path: str | Path) -> Path:
        """Convenience: look up ``name`` and write the report to ``path``."""
        reporter = self.get(name)
        if not isinstance(reporter, BaseReporter):
            raise ReporterError(f"reporter {name!r} does not support render_to_file")
        return reporter.render_to_file(result, path)


def config_to_dict(config: Any) -> Any:
    """Return a JSON-serialisable form of a run config.

    ``DiffConfig`` and similar objects expose ``to_dict``; anything else is
    returned unchanged so reporters can embed it verbatim.
    """
    if config is None:
        return None
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return config
