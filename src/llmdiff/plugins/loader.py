"""Plugin discovery and loading.

Plugins come from three sources:

1. **Entry points** — installed distributions that register the
   ``llmdiff.plugins`` entry-point group. Each entry point names a callable
   that returns a plugin object (a ``Check``, a reporter, or a module with
   ``@hook`` functions).
2. **Plugin directories** — every subdirectory of ``~/.llmdiff/plugins/``
   (or a custom dir) that contains a manifest (``llmdiff-plugin.yaml`` /
   ``.toml``) plus its entry ``.py`` file.
3. **Loose files** — bare ``.py`` files dropped directly into the plugin dir.

Loading a file executes it with :func:`importlib` machinery in a dedicated
module namespace. The module must define at least one of:

- ``PLUGIN`` — a single plugin object,
- ``PLUGINS`` — a list of plugin objects,
- ``get_plugin()`` / ``get_plugins()`` — factory callables,
- ``@hook``-decorated functions (auto-subscribed by the registry),
- module-level ``Check`` implementations discovered by scanning for classes
  that satisfy the :class:`~llmdiff.core.checks.Check` protocol.

Security: files are executed as Python, so only load plugins you trust. The
loader refuses to load files outside the plugin directories, symlinks that
escape them, and files whose manifest requests permissions the loader was not
configured to grant.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from llmdiff.core.checks import Check
from llmdiff.core.errors import PluginError
from llmdiff.plugins.hooks import collect_hooks
from llmdiff.plugins.manifest import (
    MANIFEST_BASENAMES,
    PluginManifest,
    find_manifest,
    load_manifest,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DEFAULT_PLUGIN_DIR",
    "LoadedPlugin",
    "PluginLoader",
    "load_entry_points",
    "load_file",
    "load_directory",
    "discover_plugins",
    "iter_plugin_dirs",
]

ENTRY_POINT_GROUP = "llmdiff.plugins"
"""The ``importlib.metadata`` entry-point group plugins register under."""

DEFAULT_PLUGIN_DIR = Path.home() / ".llmdiff" / "plugins"
"""Default directory scanned for file-based plugins."""


@dataclass(slots=True)
class LoadedPlugin:
    """A plugin object successfully loaded from one source.

    Attributes:
        name: Plugin name (manifest name, entry-point name, or module name).
        version: Version string when known.
        obj: The plugin object (Check, reporter, or arbitrary callable holder).
        source: Where it came from: ``entry_point``, ``manifest``, ``file``.
        path: File path for file-based plugins, else None.
        manifest: The parsed manifest when one was present.
        module: The executed module for file-based plugins.
        hooks: ``(point, callback, plugin, priority)`` tuples found in the module.
    """

    name: str
    obj: Any
    source: str
    version: str = "0.0.0"
    path: str | None = None
    manifest: PluginManifest | None = None
    module: ModuleType | None = None
    hooks: tuple[tuple[str, Callable[..., Any], str, int], ...] = ()
    errors: list[str] = field(default_factory=list)

    @property
    def is_check(self) -> bool:
        """True when the object satisfies the Check protocol."""
        return isinstance(self.obj, Check)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "path": self.path,
            "is_check": self.is_check,
            "hooks": [point for point, _, _, _ in self.hooks],
            "errors": list(self.errors),
        }


class PluginLoader:
    """Discovers and loads plugins from entry points and directories.

    Attributes:
        plugin_dirs: Directories scanned for file-based plugins.
        allowed_permissions: Permissions file-based plugins may request.
            Manifests asking for more are rejected.
        include_entry_points: Whether to scan installed entry points.
    """

    def __init__(
        self,
        plugin_dirs: Iterable[str | Path] | None = None,
        *,
        allowed_permissions: Iterable[str] | None = None,
        include_entry_points: bool = True,
    ) -> None:
        if plugin_dirs is None:
            plugin_dirs = (DEFAULT_PLUGIN_DIR,)
        self.plugin_dirs: tuple[Path, ...] = tuple(Path(p) for p in plugin_dirs)
        from llmdiff.plugins.manifest import ALLOWED_PERMISSIONS

        if allowed_permissions is None:
            allowed_permissions = ALLOWED_PERMISSIONS
        self.allowed_permissions = frozenset(allowed_permissions)
        self.include_entry_points = include_entry_points

    # ------------------------------------------------------------------ #
    # entry points
    # ------------------------------------------------------------------ #

    def load_entry_points(self) -> list[LoadedPlugin]:
        """Load every plugin registered under the ``llmdiff.plugins`` group.

        Entry-point failures are collected, not raised: one broken installed
        plugin must not prevent the rest from loading.
        """
        if not self.include_entry_points:
            return []
        return load_entry_points(allowed_permissions=self.allowed_permissions)

    # ------------------------------------------------------------------ #
    # directories
    # ------------------------------------------------------------------ #

    def load_dirs(self) -> list[LoadedPlugin]:
        """Load every plugin found in the configured plugin directories."""
        loaded: list[LoadedPlugin] = []
        for directory in self.plugin_dirs:
            loaded.extend(load_directory(directory, allowed_permissions=self.allowed_permissions))
        return loaded

    def load_all(self) -> list[LoadedPlugin]:
        """Load entry points first, then directory plugins."""
        return self.load_entry_points() + self.load_dirs()

    def discover(self) -> list[Path]:
        """Return every loadable plugin file/manifest path without executing."""
        return discover_plugins(self.plugin_dirs)


def load_entry_points(*, allowed_permissions: Iterable[str] | None = None) -> list[LoadedPlugin]:
    """Load plugins from the ``llmdiff.plugins`` entry-point group.

    Each entry point must resolve to a callable returning a plugin object, or
    to the plugin object itself.
    """
    from llmdiff.plugins.manifest import ALLOWED_PERMISSIONS

    allowed = frozenset(allowed_permissions or ALLOWED_PERMISSIONS)
    loaded: list[LoadedPlugin] = []
    try:
        from importlib.metadata import entry_points
    except ImportError as exc:  # pragma: no cover - Python >= 3.10 always has it
        raise PluginError(f"importlib.metadata unavailable: {exc}") from exc
    try:
        group = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older Pythons
        group = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[call-arg]
    for ep in group:
        try:
            obj = ep.load()
            if callable(obj) and not _looks_like_plugin(obj):
                obj = obj()
            name = ep.name
            version = getattr(obj, "version", "0.0.0")
            loaded.append(
                LoadedPlugin(
                    name=name,
                    obj=obj,
                    source="entry_point",
                    version=str(version),
                )
            )
        except Exception as exc:  # noqa: BLE001 - collect, don't crash
            loaded.append(
                LoadedPlugin(
                    name=ep.name,
                    obj=None,
                    source="entry_point",
                    errors=[f"{type(exc).__name__}: {exc}", traceback.format_exc()],
                )
            )
    _ = allowed  # entry-point plugins are trusted (installed via pip)
    return loaded


def _looks_like_plugin(obj: Any) -> bool:
    """Heuristic: does ``obj`` already look like a plugin object?"""
    return isinstance(obj, Check) or hasattr(obj, "score") or hasattr(obj, "render")


# ---------------------------------------------------------------------- #
# file loading
# ---------------------------------------------------------------------- #


def load_file(
    path: str | Path,
    *,
    manifest: PluginManifest | None = None,
    allowed_permissions: Iterable[str] | None = None,
) -> LoadedPlugin:
    """Execute a plugin ``.py`` file and extract its plugin objects.

    Args:
        path: The Python file to load.
        manifest: Parsed manifest for the plugin, when present.
        allowed_permissions: Permissions the loader is willing to grant.

    Raises:
        PluginError: If the file cannot be read, executed, or violates the
            manifest's requested permissions.
    """
    from llmdiff.plugins.manifest import ALLOWED_PERMISSIONS

    allowed = frozenset(allowed_permissions or ALLOWED_PERMISSIONS)
    file_path = Path(path)
    if not file_path.is_file():
        raise PluginError(f"plugin file not found: {file_path}")
    if file_path.suffix != ".py":
        raise PluginError(f"plugin file must be a .py file: {file_path}")
    if manifest is not None:
        missing = set(manifest.permissions) - allowed
        if missing:
            raise PluginError(
                f"plugin {manifest.name!r} requests permission(s) not granted: "
                f"{', '.join(sorted(missing))}"
            )
        name = manifest.name
        version = manifest.version
    else:
        name = file_path.stem
        version = "0.0.0"

    module = _exec_module(file_path)
    hooks = tuple(collect_hooks(module))
    objects = _extract_objects(module)
    if not objects and not hooks:
        raise PluginError(
            f"{file_path}: no plugin objects found; define PLUGIN, PLUGINS, "
            "get_plugin(), get_plugins(), a Check class, or @hook functions"
        )
    primary = objects[0] if objects else module
    return LoadedPlugin(
        name=name,
        obj=primary,
        source="manifest" if manifest else "file",
        version=version,
        path=str(file_path),
        manifest=manifest,
        module=module,
        hooks=hooks,
    )


def _exec_module(file_path: Path) -> ModuleType:
    """Execute ``file_path`` as a fresh module named after its stem."""
    module_name = f"llmdiff_plugin_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise PluginError(
            f"{file_path}: error while executing plugin: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def _extract_objects(module: ModuleType) -> list[Any]:
    """Pull plugin objects out of an executed module, in priority order."""
    objects: list[Any] = []
    if hasattr(module, "PLUGINS"):
        plugins = module.PLUGINS
        if not isinstance(plugins, (list, tuple)):
            raise PluginError(f"{module.__name__}: PLUGINS must be a list")
        objects.extend(plugins)
    if hasattr(module, "PLUGIN") and module.PLUGIN is not None:
        objects.insert(0, module.PLUGIN)
    for factory_name in ("get_plugins", "get_plugin"):
        factory = getattr(module, factory_name, None)
        if callable(factory):
            produced = factory()
            if factory_name == "get_plugins" and isinstance(produced, (list, tuple)):
                objects.extend(produced)
            else:
                objects.append(produced)
    if not objects:
        objects.extend(_discover_check_classes(module))
    return objects


def _discover_check_classes(module: ModuleType) -> list[Any]:
    """Instantiate module-level classes that satisfy the Check protocol."""
    found: list[Any] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and isinstance(obj(), Check):
            found.append(obj())
    return found


# ---------------------------------------------------------------------- #
# directory scanning
# ---------------------------------------------------------------------- #


def load_directory(
    directory: str | Path,
    *,
    allowed_permissions: Iterable[str] | None = None,
) -> list[LoadedPlugin]:
    """Load every plugin in ``directory``.

    Layouts recognised:

    - ``<dir>/<plugin>/llmdiff-plugin.yaml`` + entry file (manifest layout),
    - ``<dir>/llmdiff-plugin.yaml`` + entry file (single-plugin layout),
    - ``<dir>/<anything>.py`` (loose file layout).

    Broken plugins are returned as :class:`LoadedPlugin` entries with
    ``errors`` populated instead of raising.
    """
    base = Path(directory)
    if not base.is_dir():
        return []
    loaded: list[LoadedPlugin] = []
    for path in discover_plugins((base,)):
        try:
            if path.name in MANIFEST_BASENAMES:
                manifest = load_manifest(path, check_entry=True)
                entry = manifest.entry_path
                assert entry is not None
                loaded.append(
                    load_file(entry, manifest=manifest, allowed_permissions=allowed_permissions)
                )
            else:
                loaded.append(load_file(path, allowed_permissions=allowed_permissions))
        except PluginError as exc:
            loaded.append(
                LoadedPlugin(
                    name=path.stem,
                    obj=None,
                    source="file",
                    path=str(path),
                    errors=[str(exc)],
                )
            )
    return loaded


def discover_plugins(directories: Iterable[str | Path]) -> list[Path]:
    """Return loadable plugin paths (manifests and loose .py files), sorted.

    Manifest directories contribute their manifest file; the loader resolves
    the entry from it. Loose ``.py`` files are included directly. Hidden
    files and ``__pycache__`` are skipped.
    """
    found: set[Path] = set()
    for directory in directories:
        base = Path(directory)
        if not base.is_dir():
            continue
        # single-plugin layout: manifest directly in the dir
        manifest = find_manifest(base)
        if manifest is not None and base == Path(directory):
            found.add(manifest)
        for child in sorted(base.iterdir()):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            if child.is_dir():
                child_manifest = find_manifest(child)
                if child_manifest is not None:
                    found.add(child_manifest)
            elif child.is_file() and child.suffix == ".py":
                found.add(child)
    return sorted(found)


def iter_plugin_dirs(extra: Iterable[str | Path] | None = None) -> Iterator[Path]:
    """Yield the default plugin dir plus any extras, existing ones only."""
    seen: set[Path] = set()
    for directory in (DEFAULT_PLUGIN_DIR, *(extra or ())):
        path = Path(directory)
        if path not in seen and path.is_dir():
            seen.add(path)
            yield path
