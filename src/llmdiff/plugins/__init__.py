"""llmdiff plugin system.

This package implements everything a third-party plugin needs:

- :mod:`llmdiff.plugins.manifest` — YAML/TOML manifest schema + validation,
- :mod:`llmdiff.plugins.loader` — entry-point and file-based discovery,
- :mod:`llmdiff.plugins.registry` — dependency resolution + conflict detection,
- :mod:`llmdiff.plugins.hooks` — before/after eval, replay and report hooks,
- :mod:`llmdiff.plugins.config` — per-plugin config with env overrides,
- :mod:`llmdiff.plugins.sandbox` — time/memory limits around ``score`` calls,
- :mod:`llmdiff.plugins.marketplace` — JSON index search/install/uninstall,
- :mod:`llmdiff.plugins.templates` — plugin scaffolding.

Quick example::

    from llmdiff.plugins import PluginRegistry

    registry = PluginRegistry(plugin_dirs=["./plugins"])
    registry.load_all()
    for check in registry.checks():
        print(check.name)
"""

from __future__ import annotations

from llmdiff.plugins.config import (
    DEFAULT_CONFIG_DIR,
    ENV_PREFIX,
    ConfigStore,
    PluginConfig,
    coerce_value,
    env_overrides,
    load_config_file,
)
from llmdiff.plugins.hooks import (
    HOOK_POINTS,
    Hook,
    HookError,
    HookFailure,
    HookInvocation,
    HookManager,
    HookReport,
    collect_hooks,
    hook,
    make_hook_manager,
    subscribe_module,
)
from llmdiff.plugins.loader import (
    DEFAULT_PLUGIN_DIR,
    ENTRY_POINT_GROUP,
    LoadedPlugin,
    PluginLoader,
    discover_plugins,
    iter_plugin_dirs,
    load_directory,
    load_entry_points,
    load_file,
)
from llmdiff.plugins.manifest import (
    ALLOWED_PERMISSIONS,
    MANIFEST_BASENAMES,
    PluginDependency,
    PluginManifest,
    find_manifest,
    load_manifest,
    manifest_to_dict,
    parse_manifest,
    validate_manifest,
)
from llmdiff.plugins.marketplace import (
    INDEX_FILENAME,
    Marketplace,
    MarketplaceEntry,
    MarketplaceIndex,
    build_index,
    fetch_url,
    sha256_of,
)
from llmdiff.plugins.registry import (
    PluginConflict,
    PluginEntry,
    PluginRegistry,
    resolve_dependencies,
)
from llmdiff.plugins.sandbox import (
    Sandbox,
    SandboxLimits,
    SandboxMemoryError,
    SandboxResult,
    SandboxTimeoutError,
    run_sandboxed,
)
from llmdiff.plugins.templates import (
    TEMPLATES,
    TemplateVars,
    render_template,
    scaffold,
    scaffold_files,
)

__all__ = [
    # manifest
    "ALLOWED_PERMISSIONS",
    "MANIFEST_BASENAMES",
    "PluginDependency",
    "PluginManifest",
    "find_manifest",
    "load_manifest",
    "manifest_to_dict",
    "parse_manifest",
    "validate_manifest",
    # loader
    "DEFAULT_PLUGIN_DIR",
    "ENTRY_POINT_GROUP",
    "LoadedPlugin",
    "PluginLoader",
    "discover_plugins",
    "iter_plugin_dirs",
    "load_directory",
    "load_entry_points",
    "load_file",
    # registry
    "PluginConflict",
    "PluginEntry",
    "PluginRegistry",
    "resolve_dependencies",
    # hooks
    "HOOK_POINTS",
    "Hook",
    "HookError",
    "HookFailure",
    "HookInvocation",
    "HookManager",
    "HookReport",
    "collect_hooks",
    "hook",
    "make_hook_manager",
    "subscribe_module",
    # config
    "DEFAULT_CONFIG_DIR",
    "ENV_PREFIX",
    "ConfigStore",
    "PluginConfig",
    "coerce_value",
    "env_overrides",
    "load_config_file",
    # sandbox
    "Sandbox",
    "SandboxLimits",
    "SandboxMemoryError",
    "SandboxResult",
    "SandboxTimeoutError",
    "run_sandboxed",
    # marketplace
    "INDEX_FILENAME",
    "Marketplace",
    "MarketplaceEntry",
    "MarketplaceIndex",
    "build_index",
    "fetch_url",
    "sha256_of",
    # templates
    "TEMPLATES",
    "TemplateVars",
    "render_template",
    "scaffold",
    "scaffold_files",
]
