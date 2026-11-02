"""Per-plugin configuration with environment overrides.

Each plugin gets an isolated configuration namespace. Values are resolved in
this order (later wins):

1. built-in defaults declared by the plugin,
2. the global config file (``~/.llmdiff/config.yaml`` or ``config.toml``),
3. the per-plugin config file (``~/.llmdiff/plugins/<name>.yaml``),
4. environment variables of the form ``LLMDIFF_PLUGIN_<NAME>__<KEY>``.

Environment overrides are always strings; they are coerced to the type of the
default value when possible (bool/int/float/list), otherwise kept as strings.

Example::

    schema = {"timeout": 30, "retries": 3, "verbose": False, "tags": ["a"]}
    cfg = PluginConfig("my-plugin", schema, env=os.environ)
    cfg.get("timeout")          # 30 unless overridden
    cfg.get("missing", 42)      # 42
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmdiff.core.errors import ConfigError, PluginError

__all__ = [
    "DEFAULT_CONFIG_DIR",
    "ENV_PREFIX",
    "PluginConfig",
    "ConfigStore",
    "coerce_value",
    "load_config_file",
    "env_overrides",
]

DEFAULT_CONFIG_DIR = Path.home() / ".llmdiff"
"""Default directory for global and per-plugin configuration."""

ENV_PREFIX = "LLMDIFF_PLUGIN_"
"""Prefix for environment overrides: ``LLMDIFF_PLUGIN_<NAME>__<KEY>``."""

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off", "n", "f"})


def coerce_value(raw: Any, template: Any) -> Any:
    """Coerce ``raw`` to the type of ``template`` when possible.

    - ``bool`` templates accept the usual truthy/falsy strings.
    - ``int``/``float`` templates parse numeric strings.
    - ``list``/``tuple`` templates split comma-separated strings.
    - ``dict`` templates accept JSON-ish strings via ``json.loads``.
    - anything else is returned as-is (strings stay strings).
    """
    if template is None:
        return raw
    if isinstance(template, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in _TRUE_STRINGS:
                return True
            if lowered in _FALSE_STRINGS:
                return False
            raise ConfigError(f"cannot coerce {raw!r} to bool")
        return bool(raw)
    if isinstance(template, int) and not isinstance(template, bool):
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip(), 10)
            except ValueError as exc:
                raise ConfigError(f"cannot coerce {raw!r} to int") from exc
        raise ConfigError(f"cannot coerce {raw!r} to int")
    if isinstance(template, float):
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError as exc:
                raise ConfigError(f"cannot coerce {raw!r} to float") from exc
        raise ConfigError(f"cannot coerce {raw!r} to float")
    if isinstance(template, (list, tuple)):
        if isinstance(raw, (list, tuple)):
            return type(template)(raw)
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.split(",")]
            return type(template)(part for part in parts if part != "")
        raise ConfigError(f"cannot coerce {raw!r} to {type(template).__name__}")
    if isinstance(template, Mapping):
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, str):
            import json

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"cannot coerce {raw!r} to dict: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ConfigError(f"cannot coerce {raw!r} to dict: JSON value is not an object")
            return parsed
        raise ConfigError(f"cannot coerce {raw!r} to dict")
    return raw


def env_overrides(plugin_name: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Collect ``LLMDIFF_PLUGIN_<NAME>__<KEY>`` variables from ``env``.

    The plugin name is uppercased and ``-`` becomes ``_``. Keys are lowercased.
    """
    env = os.environ if env is None else env
    prefix = ENV_PREFIX + plugin_name.upper().replace("-", "_") + "__"
    found: dict[str, str] = {}
    for key, value in env.items():
        if key.startswith(prefix):
            found[key[len(prefix) :].lower()] = value
    return found


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML or TOML config file into a dict.

    Returns an empty dict when the file does not exist. Raises
    :class:`ConfigError` for unreadable or malformed files.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {}
    if not config_path.is_file():
        raise ConfigError(f"config path is not a file: {config_path}")
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    if config_path.suffix == ".toml":
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
        return dict(data)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ConfigError(
            f"PyYAML is required to read {config_path}; install pyyaml or use .toml"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{config_path}: top level must be a mapping, got {type(data).__name__}")
    return dict(data)


@dataclass(slots=True)
class PluginConfig:
    """Resolved configuration for one plugin.

    Attributes:
        plugin: Plugin name this config belongs to.
        defaults: The plugin-declared default values.
        values: The fully resolved values (defaults < files < env).
    """

    plugin: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    values: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    """Maps each resolved key to where it came from (``default``/``file``/``env``)."""

    def get(self, key: str, default: Any = None) -> Any:
        """Return the resolved value for ``key``, or ``default`` when unset."""
        if key in self.values:
            return self.values[key]
        if key in self.defaults:
            return self.defaults[key]
        return default

    def __getitem__(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        if key in self.defaults:
            return self.defaults[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self.values or key in self.defaults

    def as_dict(self) -> dict[str, Any]:
        """Return a merged dict of defaults overlaid with resolved values."""
        merged = dict(self.defaults)
        merged.update(self.values)
        return merged

    def source_of(self, key: str) -> str | None:
        """Return where ``key`` was resolved from, or None for defaults."""
        return self.sources.get(key)

    @classmethod
    def resolve(
        cls,
        plugin: str,
        defaults: Mapping[str, Any] | None = None,
        *,
        global_config: Mapping[str, Any] | None = None,
        plugin_config: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> PluginConfig:
        """Resolve a config from defaults, file layers and environment.

        Args:
            plugin: Plugin name (used for the env prefix and file sections).
            defaults: Plugin-declared defaults; their types drive coercion.
            global_config: Parsed global config file; the ``plugins.<name>``
                section (or a top-level ``<name>`` section) applies.
            plugin_config: Parsed per-plugin config file.
            env: Environment mapping; defaults to ``os.environ``.
        """
        defaults = dict(defaults or {})
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}

        def _apply(layer: Mapping[str, Any], source: str) -> None:
            for key, raw in layer.items():
                if key in defaults:
                    values[key] = coerce_value(raw, defaults[key])
                else:
                    values[key] = raw
                sources[key] = source

        if global_config:
            plugins_section = global_config.get("plugins")
            if isinstance(plugins_section, Mapping) and plugin in plugins_section:
                section = plugins_section[plugin]
                if not isinstance(section, Mapping):
                    raise ConfigError(
                        f"global config: plugins.{plugin} must be a mapping, "
                        f"got {type(section).__name__}"
                    )
                _apply(section, "global-file")
            elif plugin in global_config and isinstance(global_config[plugin], Mapping):
                _apply(global_config[plugin], "global-file")

        if plugin_config:
            _apply(plugin_config, "plugin-file")

        for key, raw in env_overrides(plugin, env).items():
            if key in defaults:
                values[key] = coerce_value(raw, defaults[key])
            else:
                values[key] = raw
            sources[key] = "env"

        return cls(plugin=plugin, defaults=defaults, values=values, sources=sources)


@dataclass(slots=True)
class ConfigStore:
    """Loads and caches configuration for many plugins.

    Attributes:
        config_dir: Root config directory (default ``~/.llmdiff``).
        global_path: Path of the global config file.
        env: Environment mapping used for overrides.
    """

    config_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG_DIR)
    env: Mapping[str, str] = field(default_factory=dict)
    _global_cache: dict[str, Any] | None = None
    _plugin_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.env:
            self.env = dict(os.environ)

    @property
    def global_path(self) -> Path:
        """The global config file, preferring YAML then TOML."""
        for name in ("config.yaml", "config.yml", "config.toml"):
            candidate = self.config_dir / name
            if candidate.is_file():
                return candidate
        return self.config_dir / "config.yaml"

    def global_config(self) -> dict[str, Any]:
        """Return the parsed global config (cached)."""
        if self._global_cache is None:
            self._global_cache = load_config_file(self.global_path)
        return self._global_cache

    def plugin_file(self, plugin: str) -> Path:
        """Return the per-plugin config file path (may not exist)."""
        safe = plugin.replace("/", "_").replace("\\", "_")
        for ext in (".yaml", ".yml", ".toml"):
            candidate = self.config_dir / "plugins" / f"{safe}{ext}"
            if candidate.is_file():
                return candidate
        return self.config_dir / "plugins" / f"{safe}.yaml"

    def plugin_config(self, plugin: str) -> dict[str, Any]:
        """Return the parsed per-plugin config file (cached)."""
        if plugin not in self._plugin_cache:
            self._plugin_cache[plugin] = load_config_file(self.plugin_file(plugin))
        return self._plugin_cache[plugin]

    def for_plugin(
        self, plugin: str, defaults: Mapping[str, Any] | None = None
    ) -> PluginConfig:
        """Resolve the :class:`PluginConfig` for ``plugin``."""
        return PluginConfig.resolve(
            plugin,
            defaults,
            global_config=self.global_config(),
            plugin_config=self.plugin_config(plugin),
            env=self.env,
        )

    def set_plugin_value(self, plugin: str, key: str, value: Any) -> None:
        """Write a value into the per-plugin config file (YAML).

        Creates the file and parent directories as needed. Raises
        :class:`PluginError` when PyYAML is unavailable or the write fails.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise PluginError("PyYAML is required to write plugin config files") from exc
        path = self.plugin_file(plugin)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = load_config_file(path)
        data[key] = value
        try:
            path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            raise PluginError(f"cannot write plugin config {path}: {exc}") from exc
        self._plugin_cache[plugin] = data

    def reload(self) -> None:
        """Drop all caches so the next access re-reads from disk."""
        self._global_cache = None
        self._plugin_cache.clear()
