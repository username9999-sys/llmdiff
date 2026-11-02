"""``llmdiff plugins`` - manage the plugin directory ``~/.llmdiff/plugins``.

Usage::

    llmdiff plugins list
    llmdiff plugins add path/to/plugin.py
    llmdiff plugins remove name

A plugin is a plain ``.py`` file.  ``list`` imports each plugin module (in an
isolated namespace) to verify it is syntactically valid and reports the
``__all__``/top-level callables it exposes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from llmdiff.cli.output import emit, err, print_table, style
from llmdiff.core.errors import PluginError

__all__ = ["run_plugins", "add_plugins_parser", "plugin_dir", "list_plugins", "import_check"]


def plugin_dir() -> Path:
    """Return the plugin directory (``~/.llmdiff/plugins``), creating it."""
    import os

    base = Path(os.environ.get("LLMDIFF_PLUGIN_DIR", Path.home() / ".llmdiff" / "plugins"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def list_plugins() -> list[tuple[str, str, str]]:
    """Return ``(name, path, status)`` for every ``.py`` file in the plugin dir."""
    rows: list[tuple[str, str, str]] = []
    for file in sorted(plugin_dir().glob("*.py")):
        if file.name.startswith("_"):
            continue
        name = file.stem
        try:
            module = import_check(file)
            exposed = ", ".join(
                n for n in getattr(module, "__all__", []) or dir(module) if not n.startswith("_")
            )[:60]
            rows.append((name, str(file), f"ok ({exposed or 'no __all__'})"))
        except PluginError as exc:
            rows.append((name, str(file), f"ERROR: {exc}"))
    return rows


def import_check(path: Path) -> ModuleType:
    """Import a plugin file in isolation, raising :class:`PluginError` on failure."""
    if not path.exists():
        raise PluginError(f"plugin file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"llmdiff_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - plugin errors are user-facing
        sys.modules.pop(spec.name, None)
        raise PluginError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    return module


def add_plugins_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``plugins`` subcommand."""
    p = sub.add_parser("plugins", help="manage ~/.llmdiff/plugins")
    psub = p.add_subparsers(dest="action", required=True)
    psub.add_parser("list", help="list installed plugins and import-check them")
    padd = psub.add_parser("add", help="copy a .py file into the plugin dir")
    padd.add_argument("source", help="path to the plugin .py file")
    prem = psub.add_parser("remove", help="remove a plugin by name")
    prem.add_argument("name", help="plugin name (file stem)")
    p.set_defaults(func=run_plugins)


def run_plugins(args: argparse.Namespace) -> int:
    """Execute ``plugins``.  Returns 0 on success, 2 on error."""
    action = args.action
    if action == "list":
        rows = list_plugins()
        if not rows:
            emit(f"no plugins in {plugin_dir()}")
            return 0
        print_table(["name", "path", "status"], rows)
        return 0
    if action == "add":
        source = Path(args.source).expanduser()
        if not source.exists():
            err(f"error: {source} not found")
            return 2
        if source.suffix != ".py":
            err("error: plugin must be a .py file")
            return 2
        try:
            import_check(source)
        except PluginError as exc:
            err(f"error: plugin failed import check: {exc}")
            return 2
        dest = plugin_dir() / source.name
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        emit(style(f"added plugin {source.stem} -> {dest}", "bold_green"))
        return 0
    if action == "remove":
        target = plugin_dir() / f"{args.name}.py"
        if not target.exists():
            err(f"error: plugin {args.name!r} not found in {plugin_dir()}")
            return 2
        target.unlink()
        emit(style(f"removed plugin {args.name}", "bold_green"))
        return 0
    err(f"error: unknown plugins action {action!r}")
    return 2
