"""Plugin scaffolding templates.

:func:`scaffold` writes a ready-to-edit plugin project into a target
directory: a manifest, an entry file with a working ``Check`` example, a
config defaults block, a hook example and a test stub. Templates are plain
strings rendered with :class:`string.Template` so they stay dependency-free.

Available templates:

- ``check`` — a new comparison dimension (default),
- ``reporter`` — a reporter plugin,
- ``hooks`` — a lifecycle-hooks-only plugin,
- ``full`` — check + hooks + config in one plugin.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llmdiff.core.errors import PluginError

__all__ = [
    "TEMPLATES",
    "TemplateVars",
    "render_template",
    "scaffold",
    "scaffold_files",
]

TEMPLATES: tuple[str, ...] = ("check", "reporter", "hooks", "full")
"""Names of the available scaffolding templates."""


@dataclass(frozen=True, slots=True)
class TemplateVars:
    """Variables substituted into the templates.

    Attributes:
        name: Plugin name (slug, lowercase).
        class_name: CamelCase class name derived from ``name``.
        dimension: Check dimension name (defaults to ``name`` with ``-`` -> ``_``).
        version: Version string.
        author: Author name.
        description: One-line description.
        year: Copyright year.
    """

    name: str
    class_name: str
    dimension: str
    version: str = "0.1.0"
    author: str = ""
    description: str = ""
    year: str = "2026"

    @classmethod
    def from_name(cls, name: str, **overrides: Any) -> TemplateVars:
        """Build vars from a plugin name, deriving class/dimension names."""
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise PluginError(f"invalid plugin name for scaffolding: {name!r}")
        parts = [part for part in name.replace("-", "_").split("_") if part]
        class_name = "".join(part.capitalize() for part in parts) or "Plugin"
        dimension = name.replace("-", "_")
        return cls(
            name=name,
            class_name=class_name,
            dimension=dimension,
            **overrides,
        )


_MANIFEST_YAML = """\
# llmdiff plugin manifest. See docs/plugins.md for the full schema.
name: ${name}
version: ${version}
author: ${author}
description: ${description}
entry: plugin.py
hooks: [${hooks}]
dependencies: []
permissions: []
min_llmdiff_version: "0.1.0"
license: MIT
"""

_CHECK_ENTRY = '''\
"""${description}

Plugin: ${name} v${version}
Author: ${author}
"""

from __future__ import annotations

from llmdiff.core.checks import CheckContext

CONFIG_DEFAULTS = {
    "threshold": 0.5,
    "verbose": False,
}


class ${class_name}Check:
    """Scores outputs on the ``${dimension}`` dimension."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "${dimension}"

    def score(self, output: str, context: CheckContext) -> float:
        """Return a score in [0.0, 1.0]; higher is better.

        Replace this body with the real comparison logic. Keep it
        deterministic: same inputs must give the same score.
        """
        if not output.strip():
            return 0.0
        # TODO: implement the real scoring.
        return 1.0 if len(output) > 0 else 0.0


PLUGIN = ${class_name}Check()
'''

_REPORTER_ENTRY = '''\
"""${description}

Plugin: ${name} v${version} (reporter)
Author: ${author}
"""

from __future__ import annotations

from typing import Any


class ${class_name}Reporter:
    """Renders a DiffResult as ``${name}`` output."""

    @property
    def name(self) -> str:
        return "${name}"

    def render(self, result: Any) -> str:
        """Return the rendered report as a string."""
        matrix = result.matrix
        lines = [f"# ${name} report", ""]
        for summary in matrix.dimensions:
            lines.append(
                f"- {summary.dimension}: "
                f"W{summary.wins} T{summary.ties} L{summary.losses} "
                f"(delta {summary.delta:+.3f})"
            )
        return "\\n".join(lines) + "\\n"


PLUGIN = ${class_name}Reporter()
'''

_HOOKS_ENTRY = '''\
"""${description}

Plugin: ${name} v${version} (lifecycle hooks)
Author: ${author}
"""

from __future__ import annotations

from llmdiff.plugins.hooks import hook


@hook("before_eval", plugin="${name}")
def log_corpus_size(*, corpus, config) -> None:
    """Runs before every evaluation."""
    print(f"[${name}] evaluating {len(corpus.cases)} case(s)")


@hook("after_eval", plugin="${name}")
def log_verdict(*, result, config) -> None:
    """Runs after every evaluation."""
    print(f"[${name}] verdict: {result.matrix.verdict}")
'''

_FULL_ENTRY = '''\
"""${description}

Plugin: ${name} v${version}
Author: ${author}
"""

from __future__ import annotations

from llmdiff.core.checks import CheckContext
from llmdiff.plugins.hooks import hook

CONFIG_DEFAULTS = {
    "threshold": 0.5,
    "announce": True,
}


class ${class_name}Check:
    """Scores outputs on the ``${dimension}`` dimension."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "${dimension}"

    def score(self, output: str, context: CheckContext) -> float:
        if not output.strip():
            return 0.0
        # TODO: implement the real scoring.
        return 1.0


@hook("after_eval", plugin="${name}")
def announce(*, result, config) -> None:
    if config.get("announce", True):
        print(f"[${name}] verdict: {result.matrix.verdict}")


PLUGIN = ${class_name}Check()
'''

_TEST_STUB = '''\
"""Tests for the ${name} plugin."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PLUGIN_FILE = Path(__file__).resolve().parent.parent / "${name}" / "plugin.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("${name}_plugin", PLUGIN_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_exposes_check():
    module = _load_module()
    check = module.PLUGIN
    assert check.name == "${dimension}"
    assert 0.0 <= check.score("hello", context=None) <= 1.0  # type: ignore[arg-type]
'''


def render_template(template: str, variables: Mapping[str, Any]) -> str:
    """Render ``template`` with ``variables`` using :class:`string.Template`.

    Raises:
        PluginError: On missing substitution keys.
    """
    try:
        return string.Template(template).substitute(variables)
    except KeyError as exc:
        raise PluginError(f"template variable missing: {exc}") from exc


def scaffold_files(
    template: str,
    variables: TemplateVars,
) -> dict[str, str]:
    """Return the ``{relative_path: content}`` mapping for a scaffold.

    Args:
        template: One of :data:`TEMPLATES`.
        variables: Substitution variables.

    Raises:
        PluginError: For an unknown template name.
    """
    if template not in TEMPLATES:
        raise PluginError(
            f"unknown template {template!r}; available: {', '.join(TEMPLATES)}"
        )
    mapping = {
        "name": variables.name,
        "version": variables.version,
        "author": variables.author,
        "description": variables.description or f"llmdiff plugin {variables.name}",
        "class_name": variables.class_name,
        "dimension": variables.dimension,
        "hooks": "after_eval" if template in ("hooks", "full") else "",
    }
    files: dict[str, str] = {}
    if template == "check":
        files["llmdiff-plugin.yaml"] = render_template(_MANIFEST_YAML, mapping)
        files["plugin.py"] = render_template(_CHECK_ENTRY, mapping)
    elif template == "reporter":
        mapping["hooks"] = ""
        files["llmdiff-plugin.yaml"] = render_template(_MANIFEST_YAML, mapping)
        files["plugin.py"] = render_template(_REPORTER_ENTRY, mapping)
    elif template == "hooks":
        files["llmdiff-plugin.yaml"] = render_template(_MANIFEST_YAML, mapping)
        files["plugin.py"] = render_template(_HOOKS_ENTRY, mapping)
    else:  # full
        files["llmdiff-plugin.yaml"] = render_template(_MANIFEST_YAML, mapping)
        files["plugin.py"] = render_template(_FULL_ENTRY, mapping)
    files["tests/test_plugin.py"] = render_template(_TEST_STUB, mapping)
    return files


def scaffold(
    target_dir: str | Path,
    name: str,
    *,
    template: str = "check",
    version: str = "0.1.0",
    author: str = "",
    description: str = "",
    overwrite: bool = False,
) -> Path:
    """Write a plugin scaffold into ``target_dir/name``.

    Args:
        target_dir: Parent directory; the plugin lives in ``<target>/<name>``.
        name: Plugin name (slug).
        template: One of :data:`TEMPLATES`.
        version: Version string for the manifest.
        author: Author name.
        description: One-line description.
        overwrite: Allow replacing existing files.

    Returns:
        The plugin directory path.

    Raises:
        PluginError: On invalid names, unknown templates, or existing files
            when ``overwrite`` is False.
    """
    variables = TemplateVars.from_name(
        name, version=version, author=author, description=description
    )
    files = scaffold_files(template, variables)
    plugin_dir = Path(target_dir) / name
    for relative, content in files.items():
        destination = plugin_dir / relative
        if destination.exists() and not overwrite:
            raise PluginError(
                f"refusing to overwrite existing file: {destination} "
                "(pass overwrite=True to replace it)"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return plugin_dir
