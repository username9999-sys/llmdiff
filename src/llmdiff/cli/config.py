"""User configuration for the llmdiff CLI (``llmdiff.toml``).

The config file is optional.  When present it supplies defaults for the
``compare``/``diff``/``gate`` commands: default dimensions, tolerance, output
format, reporter and gate thresholds.

Reading uses :mod:`tomllib` (stdlib, Python 3.11+).  Writing uses a small
hand-rolled serializer that only supports the flat/nested-table shapes this
package itself produces, which keeps the zero-dependency promise (no
``tomli_w`` required).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from llmdiff.core.errors import ConfigError

__all__ = [
    "CliConfig",
    "DEFAULTS",
    "config_path",
    "load_config",
    "save_config",
    "set_value",
    "get_value",
    "dumps_toml",
]

CONFIG_FILENAME = "llmdiff.toml"

#: Default values applied when no config file exists.
DEFAULTS: Mapping[str, Any] = {
    "dimensions": None,  # None -> all registered dimensions
    "tolerance": 0.0,
    "format": "terminal",
    "reporter": "markdown",
    "max_loss_rate": None,  # None -> gate disabled unless a flag/config sets it
    "max_regressed_dims": None,
    "min_win_rate": None,
    "color": True,
}


@dataclass
class CliConfig:
    """Resolved CLI configuration (file values merged over :data:`DEFAULTS`)."""

    dimensions: tuple[str, ...] | None = None
    tolerance: float = 0.0
    format: str = "terminal"
    reporter: str = "markdown"
    max_loss_rate: float | None = None
    max_regressed_dims: int | None = None
    min_win_rate: float | None = None
    color: bool = True
    path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: Path | None = None) -> CliConfig:
        """Build a config from a parsed TOML mapping, validating known keys."""
        known = {
            "dimensions": None,
            "tolerance": 0.0,
            "format": "terminal",
            "reporter": "markdown",
            "max_loss_rate": None,
            "max_regressed_dims": None,
            "min_win_rate": None,
            "color": True,
        }
        values: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in data.items():
            if key in known:
                values[key] = value
            elif isinstance(value, Mapping):
                extra[key] = dict(value)
            else:
                extra[key] = value
        dims = values.get("dimensions")
        if dims is not None:
            if isinstance(dims, str):
                dims = tuple(d.strip() for d in dims.split(",") if d.strip())
            if not isinstance(dims, (list, tuple)):
                raise ConfigError("config 'dimensions' must be a list of strings or a CSV string")
            dims = tuple(str(d) for d in dims)
        try:
            tolerance = float(values.get("tolerance", 0.0))
            max_loss_rate = values.get("max_loss_rate")
            max_loss_rate = None if max_loss_rate is None else float(max_loss_rate)
            max_regressed_dims = values.get("max_regressed_dims")
            max_regressed_dims = None if max_regressed_dims is None else int(max_regressed_dims)
            min_win_rate = values.get("min_win_rate")
            min_win_rate = None if min_win_rate is None else float(min_win_rate)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid numeric config value: {exc}") from exc
        if not 0.0 <= tolerance <= 1.0:
            raise ConfigError("config 'tolerance' must be within [0, 1]")
        if max_loss_rate is not None and not 0.0 <= max_loss_rate <= 1.0:
            raise ConfigError("config 'max_loss_rate' must be within [0, 1]")
        if min_win_rate is not None and not 0.0 <= min_win_rate <= 1.0:
            raise ConfigError("config 'min_win_rate' must be within [0, 1]")
        if max_regressed_dims is not None and max_regressed_dims < 0:
            raise ConfigError("config 'max_regressed_dims' must be >= 0")
        fmt = str(values.get("format", "terminal"))
        reporter = str(values.get("reporter", "markdown"))
        color = bool(values.get("color", True))
        return cls(
            dimensions=dims,
            tolerance=tolerance,
            format=fmt,
            reporter=reporter,
            max_loss_rate=max_loss_rate,
            max_regressed_dims=max_regressed_dims,
            min_win_rate=min_win_rate,
            color=color,
            path=path,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain mapping (suitable for TOML serialisation)."""
        data: dict[str, Any] = {
            "dimensions": list(self.dimensions) if self.dimensions is not None else None,
            "tolerance": self.tolerance,
            "format": self.format,
            "reporter": self.reporter,
            "max_loss_rate": self.max_loss_rate,
            "max_regressed_dims": self.max_regressed_dims,
            "min_win_rate": self.min_win_rate,
            "color": self.color,
        }
        data.update(self.extra)
        return data


def config_path(*, explicit: str | Path | None = None) -> Path:
    """Resolve the config file location.

    Order: explicit argument, ``LLMDIFF_CONFIG`` env var, ``./llmdiff.toml``,
    ``~/.config/llmdiff/llmdiff.toml``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("LLMDIFF_CONFIG")
    if env:
        return Path(env).expanduser()
    local = Path.cwd() / CONFIG_FILENAME
    if local.exists():
        return local
    return Path.home() / ".config" / "llmdiff" / CONFIG_FILENAME


def load_config(path: str | Path | None = None) -> CliConfig:
    """Load the config file, falling back to defaults when it does not exist."""
    target = config_path(explicit=path)
    if not target.exists():
        return CliConfig(path=target)
    try:
        raw = target.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {target} must contain a TOML table")
    return CliConfig.from_mapping(data, path=target)


def _toml_scalar(value: Any) -> str:
    if value is None:
        raise ConfigError("cannot serialise None to TOML (omit the key instead)")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    raise ConfigError(f"cannot serialise value of type {type(value).__name__} to TOML")


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serialise a flat/nested mapping to TOML text.

    Supports str/int/float/bool/None/list values and one level of nested
    tables, which covers everything :class:`CliConfig` produces.
    """
    lines: list[str] = ["# llmdiff configuration - https://github.com/username9999-sys/llmdiff", ""]
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if value is None:
            continue  # unset keys are simply omitted
        if isinstance(value, Mapping):
            tables.append((str(key), value))
        else:
            scalars.append((str(key), value))
    for key, value in scalars:
        lines.append(f"{key} = {_toml_scalar(value)}")
    for name, table in tables:
        lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
    return "\n".join(lines) + "\n"


def save_config(config: CliConfig, path: str | Path | None = None) -> Path:
    """Write ``config`` to disk as TOML, creating parent directories."""
    target = config_path(explicit=path) if path is not None else (config.path or config_path())
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_toml(config.to_dict()), encoding="utf-8")
    config.path = target
    return target


def get_value(config: CliConfig, key: str) -> Any:
    """Return one config value by dotted key (``a.b`` for nested tables)."""
    parts = key.split(".")
    value: Any = config.to_dict()
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"unknown config key: {key!r}")
        value = value[part]
    return value


def set_value(config: CliConfig, key: str, raw: str) -> Any:
    """Set one config value from its string representation.

    Type is inferred from the current value: bool (``true/false/1/0/yes/no``),
    int, float, CSV string -> list for ``dimensions``, otherwise string.
    """
    parts = key.split(".")
    if len(parts) == 1:
        current = get_value(config, key)
        target: dict[str, Any] = config.to_dict()
        holder = target
    else:
        parent_key = ".".join(parts[:-1])
        parent = get_value(config, parent_key)
        if not isinstance(parent, dict):
            raise ConfigError(f"config key {parent_key!r} is not a table")
        current = parent.get(parts[-1])
        holder = parent
    leaf = parts[-1]
    if isinstance(current, bool) or raw.lower() in ("true", "false", "yes", "no", "on", "off"):
        lowered = raw.lower()
        if lowered in ("true", "yes", "on", "1"):
            holder[leaf] = True
        elif lowered in ("false", "no", "off", "0"):
            holder[leaf] = False
        else:
            holder[leaf] = raw
    elif isinstance(current, int) and not isinstance(current, bool):
        try:
            holder[leaf] = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{key}: expected an integer, got {raw!r}") from exc
    elif isinstance(current, float):
        try:
            holder[leaf] = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{key}: expected a number, got {raw!r}") from exc
    elif leaf == "dimensions":
        holder[leaf] = [d.strip() for d in raw.split(",") if d.strip()] or None
    else:
        holder[leaf] = raw
    # Rebuild the dataclass from the mutated mapping.
    rebuilt = CliConfig.from_mapping(target, path=config.path)
    config.dimensions = rebuilt.dimensions
    config.tolerance = rebuilt.tolerance
    config.format = rebuilt.format
    config.reporter = rebuilt.reporter
    config.max_loss_rate = rebuilt.max_loss_rate
    config.max_regressed_dims = rebuilt.max_regressed_dims
    config.min_win_rate = rebuilt.min_win_rate
    config.color = rebuilt.color
    config.extra = rebuilt.extra
    return holder[leaf]
