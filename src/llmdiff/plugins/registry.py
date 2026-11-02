"""The plugin registry: load, validate, order and expose plugins.

:class:`PluginRegistry` ties together the loader, manifest validation, hook
manager, config store and sandbox. It resolves inter-plugin **dependencies**
(topological order) and detects **conflicts** (two plugins claiming the same
check name, or a dependency cycle).

Typical use::

    registry = PluginRegistry(plugin_dirs=["./my_plugins"])
    registry.discover_all()
    registry.load_all()
    checks = registry.checks()          # Check objects, dependency order
    registry.hooks.emit("before_eval", corpus=corpus, config=config)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmdiff.core.checks import Check
from llmdiff.core.errors import PluginError
from llmdiff.plugins.config import ConfigStore, PluginConfig
from llmdiff.plugins.hooks import HookManager
from llmdiff.plugins.loader import LoadedPlugin, PluginLoader
from llmdiff.plugins.manifest import PluginManifest
from llmdiff.plugins.sandbox import Sandbox, SandboxLimits

__all__ = [
    "PluginConflict",
    "PluginEntry",
    "PluginRegistry",
    "resolve_dependencies",
]


@dataclass(frozen=True, slots=True)
class PluginConflict:
    """A detected conflict between plugins.

    Attributes:
        kind: ``"duplicate_check"``, ``"dependency_cycle"`` or
            ``"missing_dependency"``.
        plugins: Names of the plugins involved.
        detail: Human-readable explanation.
    """

    kind: str
    plugins: tuple[str, ...]
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {', '.join(self.plugins)}: {self.detail}"


@dataclass(slots=True)
class PluginEntry:
    """One loaded plugin inside the registry.

    Attributes:
        loaded: The :class:`LoadedPlugin` from the loader.
        config: Resolved per-plugin configuration.
        sandbox: The sandbox used to wrap ``score`` calls.
        enabled: Whether the plugin participates in evaluation.
    """

    loaded: LoadedPlugin
    config: PluginConfig | None = None
    sandbox: Sandbox | None = None
    enabled: bool = True

    @property
    def name(self) -> str:
        return self.loaded.name

    @property
    def checks(self) -> tuple[Check, ...]:
        """All Check objects this plugin provides."""
        obj = self.loaded.obj
        if isinstance(obj, Check):
            return (obj,)
        module = self.loaded.module
        if module is not None and hasattr(module, "PLUGINS"):
            return tuple(p for p in module.PLUGINS if isinstance(p, Check))
        return ()

    def to_dict(self) -> dict[str, Any]:
        data = self.loaded.to_dict()
        data["enabled"] = self.enabled
        if self.config is not None:
            data["config"] = self.config.as_dict()
        return data


def resolve_dependencies(
    manifests: Mapping[str, PluginManifest],
) -> tuple[tuple[str, ...], list[PluginConflict]]:
    """Topologically order plugin names by their declared dependencies.

    Args:
        manifests: Mapping of plugin name to its manifest.

    Returns:
        ``(ordered_names, conflicts)``. ``ordered_names`` lists every plugin
        such that dependencies come before dependents. Cycles and missing
        dependencies are reported as conflicts and the involved plugins are
        appended at the end in name order so nothing is silently dropped.
    """
    conflicts: list[PluginConflict] = []
    ordered: list[str] = []
    state: dict[str, int] = {}  # 0=unvisited, 1=in-stack, 2=done

    def visit(name: str, stack: list[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = stack[stack.index(name) :] + [name]
            conflicts.append(
                PluginConflict(
                    kind="dependency_cycle",
                    plugins=tuple(cycle[:-1]),
                    detail=f"dependency cycle: {' -> '.join(cycle)}",
                )
            )
            return
        manifest = manifests.get(name)
        if manifest is None:
            conflicts.append(
                PluginConflict(
                    kind="missing_dependency",
                    plugins=(name,),
                    detail=f"plugin {name!r} is not loaded but is required",
                )
            )
            state[name] = 2
            ordered.append(name)
            return
        state[name] = 1
        stack.append(name)
        for dep in manifest.dependencies:
            if dep.name not in manifests:
                conflicts.append(
                    PluginConflict(
                        kind="missing_dependency",
                        plugins=(name, dep.name),
                        detail=(
                            f"plugin {name!r} depends on {dep.name!r} "
                            f"({dep}), which is not loaded"
                        ),
                    )
                )
                continue
                # note: missing deps do not block ordering of the dependent
            visit(dep.name, stack)
        stack.pop()
        state[name] = 2
        ordered.append(name)

    for name in sorted(manifests):
        visit(name, [])
    return tuple(ordered), conflicts


@dataclass(slots=True)
class PluginRegistry:
    """Holds every loaded plugin and provides lookup helpers.

    Attributes:
        loader: The :class:`PluginLoader` used for discovery.
        hooks: The shared :class:`HookManager`.
        config_store: The shared :class:`ConfigStore`.
        default_limits: Sandbox limits applied to every plugin check.
        entries: Loaded plugins keyed by name.
        conflicts: Conflicts detected during the last load.
    """

    loader: PluginLoader = field(default_factory=PluginLoader)
    hooks: HookManager = field(default_factory=HookManager)
    config_store: ConfigStore = field(default_factory=ConfigStore)
    default_limits: SandboxLimits = field(default_factory=SandboxLimits)
    entries: dict[str, PluginEntry] = field(default_factory=dict)
    conflicts: list[PluginConflict] = field(default_factory=list)
    load_order: tuple[str, ...] = ()

    def __init__(
        self,
        plugin_dirs: Iterable[str | Path] | None = None,
        *,
        include_entry_points: bool = True,
        allowed_permissions: Iterable[str] | None = None,
        config_store: ConfigStore | None = None,
        hooks: HookManager | None = None,
        default_limits: SandboxLimits | None = None,
    ) -> None:
        self.loader = PluginLoader(
            plugin_dirs,
            include_entry_points=include_entry_points,
            allowed_permissions=allowed_permissions,
        )
        self.hooks = hooks if hooks is not None else HookManager()
        self.config_store = config_store or ConfigStore()
        self.default_limits = default_limits or SandboxLimits()
        self.entries = {}
        self.conflicts = []
        self.load_order = ()

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def discover_all(self) -> list[Path]:
        """Return every discoverable plugin path without executing them."""
        return self.loader.discover()

    def load_all(self, *, strict: bool = False) -> list[LoadedPlugin]:
        """Load entry points and directory plugins into the registry.

        Args:
            strict: When True, the first load error raises; otherwise errors
                are recorded on the entry and in ``self.conflicts``.

        Returns:
            The list of :class:`LoadedPlugin` objects (including failed ones).

        Raises:
            PluginError: In strict mode, on the first load failure.
        """
        loaded = self.loader.load_all()
        manifests: dict[str, PluginManifest] = {}
        for item in loaded:
            if item.manifest is not None:
                manifests[item.name] = item.manifest
        order, conflicts = resolve_dependencies(manifests)
        self.conflicts = list(conflicts)

        # Register in dependency order; fall back to load order for the rest.
        by_name = {item.name: item for item in loaded}
        sequence = [name for name in order if name in by_name]
        sequence += [item.name for item in loaded if item.name not in sequence]
        self.load_order = tuple(sequence)

        for name in self.load_order:
            item = by_name[name]
            if item.errors and strict:
                raise PluginError(f"failed to load plugin {name!r}: {item.errors[0]}")
            entry = PluginEntry(
                loaded=item,
                config=self._resolve_config(item),
                sandbox=Sandbox(limits=self.default_limits),
            )
            self.entries[name] = entry
            self._subscribe_hooks(entry)
        self._detect_check_conflicts()
        return loaded

    def register_loaded(self, item: LoadedPlugin, *, strict: bool = False) -> PluginEntry:
        """Register a single already-loaded plugin (e.g. from an entry point)."""
        if item.errors and strict:
            raise PluginError(f"failed to load plugin {item.name!r}: {item.errors[0]}")
        entry = PluginEntry(
            loaded=item,
            config=self._resolve_config(item),
            sandbox=Sandbox(limits=self.default_limits),
        )
        self.entries[item.name] = entry
        self._subscribe_hooks(entry)
        self._detect_check_conflicts()
        return entry

    def _resolve_config(self, item: LoadedPlugin) -> PluginConfig | None:
        manifest = item.manifest
        defaults: dict[str, Any] = {}
        if manifest is not None and isinstance(manifest.extra.get("config_defaults"), Mapping):
            defaults = dict(manifest.extra["config_defaults"])
        module = item.module
        if module is not None and hasattr(module, "CONFIG_DEFAULTS"):
            raw = module.CONFIG_DEFAULTS
            if isinstance(raw, Mapping):
                defaults.update(raw)
        try:
            return self.config_store.for_plugin(item.name, defaults)
        except Exception:  # noqa: BLE001 - config must never block loading
            return PluginConfig(plugin=item.name, defaults=defaults)

    def _subscribe_hooks(self, entry: PluginEntry) -> None:
        for point, callback, plugin, priority in entry.loaded.hooks:
            self.hooks.subscribe(point, callback, plugin=plugin or entry.name, priority=priority)

    def _detect_check_conflicts(self) -> None:
        seen: dict[str, str] = {}
        for name in self.load_order:
            entry = self.entries.get(name)
            if entry is None:
                continue
            for check in entry.checks:
                check_name = getattr(check, "name", "")
                if check_name in seen and seen[check_name] != name:
                    self.conflicts.append(
                        PluginConflict(
                            kind="duplicate_check",
                            plugins=(seen[check_name], name),
                            detail=(
                                f"both provide a check named {check_name!r}; "
                                f"the later one ({name}) wins"
                            ),
                        )
                    )
                seen[check_name] = name

    # ------------------------------------------------------------------ #
    # lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> PluginEntry:
        """Return the entry for ``name``.

        Raises:
            PluginError: When no plugin with that name is registered.
        """
        try:
            return self.entries[name]
        except KeyError:
            raise PluginError(f"unknown plugin: {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> tuple[str, ...]:
        """Registered plugin names in load (dependency) order."""
        return tuple(self.load_order) if self.load_order else tuple(self.entries)

    def all(self) -> tuple[PluginEntry, ...]:
        """All entries in load order."""
        return tuple(self.entries[name] for name in self.names())

    def enabled(self) -> tuple[PluginEntry, ...]:
        """Entries that are enabled."""
        return tuple(entry for entry in self.all() if entry.enabled)

    def checks(self) -> tuple[Check, ...]:
        """Every Check provided by enabled plugins, in load order.

        Later plugins override earlier ones with the same check name, which
        is exactly what the conflict detector reports.
        """
        by_name: dict[str, Check] = {}
        for entry in self.enabled():
            for check in entry.checks:
                by_name[check.name] = check
        return tuple(by_name.values())

    def check(self, name: str) -> Check:
        """Return the check registered under ``name`` from plugins.

        Raises:
            PluginError: When no enabled plugin provides that check.
        """
        for check in self.checks():
            if check.name == name:
                return check
        raise PluginError(f"no plugin provides a check named {name!r}")

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a plugin by name."""
        self.get(name).enabled = enabled

    def score_sandboxed(self, plugin_name: str, check: Check, output: str, context: Any) -> float:
        """Run a plugin check's ``score`` inside its sandbox.

        Raises:
            PluginError: On sandbox violations or check errors.
        """
        entry = self.get(plugin_name)
        assert entry.sandbox is not None
        result = entry.sandbox.run(check.score, output, context)
        value = result.value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PluginError(
                f"plugin {plugin_name!r} check {check.name!r} returned "
                f"{type(value).__name__}, expected a float"
            )
        return float(min(1.0, max(0.0, value)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable registry summary."""
        return {
            "plugins": [entry.to_dict() for entry in self.all()],
            "load_order": list(self.names()),
            "conflicts": [
                {"kind": c.kind, "plugins": list(c.plugins), "detail": c.detail}
                for c in self.conflicts
            ],
            "hooks": {point: len(self.hooks.subscribers(point)) for point in self.hooks.points()},
        }
