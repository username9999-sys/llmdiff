"""Plugin manifest parsing and validation.

A plugin ships with a manifest file (``llmdiff-plugin.yaml`` or
``llmdiff-plugin.toml``) that declares its identity, entry point, hook
subscriptions, dependencies and required permissions. This module turns the
raw YAML/TOML document into a validated :class:`PluginManifest`.

Manifest schema (YAML example)::

    name: my-dimension
    version: 1.2.0
    author: Jane Doe
    description: Scores outputs for politeness.
    entry: plugin.py            # python file next to the manifest
    hooks: [before_eval]         # optional list of hook points
    dependencies:                # optional, other plugin names with versions
      - base-checks>=1.0.0
    permissions: [network]       # optional: network, filesystem, env
    min_llmdiff_version: "0.1.0"
    homepage: https://example.com/my-plugin
    license: MIT

Validation is strict: unknown keys are rejected, names must match
``^[a-z][a-z0-9_-]*$``, versions must be dotted numeric triplets, and the
entry file must exist next to the manifest when ``check_entry=True``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from llmdiff.core.errors import PluginError

__all__ = [
    "MANIFEST_BASENAMES",
    "ALLOWED_PERMISSIONS",
    "PluginDependency",
    "PluginManifest",
    "parse_manifest",
    "load_manifest",
    "validate_manifest",
    "manifest_to_dict",
    "find_manifest",
]

MANIFEST_BASENAMES = ("llmdiff-plugin.yaml", "llmdiff-plugin.yml", "llmdiff-plugin.toml")
"""File names recognised as plugin manifests, in preference order."""

ALLOWED_PERMISSIONS = frozenset({"network", "filesystem", "env", "subprocess"})
"""Permissions a plugin may request. Unknown permissions are rejected."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_DEP_RE = re.compile(r"^([a-z][a-z0-9_-]*)\s*(?:(>=|<=|==|>|<|~=)\s*([0-9][0-9A-Za-z.\-+]*))?$")

_ALLOWED_KEYS = frozenset(
    {
        "name",
        "version",
        "author",
        "description",
        "entry",
        "hooks",
        "dependencies",
        "permissions",
        "min_llmdiff_version",
        "homepage",
        "license",
    }
)


@dataclass(frozen=True, slots=True)
class PluginDependency:
    """A single dependency on another plugin.

    Attributes:
        name: Name of the required plugin.
        operator: Optional version constraint operator (``>=``, ``==``, ...).
        version: Optional version the constraint refers to.
    """

    name: str
    operator: str | None = None
    version: str | None = None

    def __str__(self) -> str:
        if self.operator and self.version:
            return f"{self.name}{self.operator}{self.version}"
        return self.name

    def satisfied_by(self, version: str) -> bool:
        """Return True when ``version`` satisfies this constraint."""
        if self.operator is None or self.version is None:
            return True
        return _compare_versions(version, self.operator, self.version)

    @classmethod
    def parse(cls, spec: str) -> PluginDependency:
        """Parse a dependency string such as ``base-checks>=1.0.0``.

        Raises:
            PluginError: If the spec is malformed.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise PluginError("dependency spec must be a non-empty string")
        match = _DEP_RE.match(spec.strip())
        if match is None:
            raise PluginError(f"malformed dependency spec: {spec!r}")
        name, operator, version = match.groups()
        return cls(name=name, operator=operator, version=version)


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """A validated plugin manifest.

    Attributes:
        name: Unique plugin name.
        version: Semver-like version string.
        author: Author name or handle.
        description: One-paragraph description.
        entry: Python entry file, relative to the manifest directory.
        hooks: Hook points the plugin subscribes to.
        dependencies: Other plugins that must load first.
        permissions: Capabilities the plugin requests.
        min_llmdiff_version: Minimum llmdiff version required.
        homepage: Project URL.
        license: SPDX license identifier.
        source: Path of the file the manifest was loaded from, if any.
    """

    name: str
    version: str
    author: str = ""
    description: str = ""
    entry: str = "plugin.py"
    hooks: tuple[str, ...] = ()
    dependencies: tuple[PluginDependency, ...] = ()
    permissions: tuple[str, ...] = ()
    min_llmdiff_version: str = "0.0.0"
    homepage: str = ""
    license: str = ""
    source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def entry_path(self) -> Path | None:
        """Absolute path of the entry file, when a source is known."""
        if self.source is None:
            return None
        return Path(self.source).parent / self.entry

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the manifest."""
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "entry": self.entry,
            "hooks": list(self.hooks),
            "dependencies": [str(dep) for dep in self.dependencies],
            "permissions": list(self.permissions),
            "min_llmdiff_version": self.min_llmdiff_version,
            "homepage": self.homepage,
            "license": self.license,
        }
        if self.source is not None:
            data["source"] = self.source
        if self.extra:
            data["extra"] = dict(self.extra)
        return data


def _compare_versions(left: str, operator: str, right: str) -> bool:
    """Compare two dotted-numeric versions with a constraint operator."""
    left_parts = _version_key(left)
    right_parts = _version_key(right)
    if operator == ">=":
        return left_parts >= right_parts
    if operator == "<=":
        return left_parts <= right_parts
    if operator == ">":
        return left_parts > right_parts
    if operator == "<":
        return left_parts < right_parts
    if operator == "==":
        return left_parts == right_parts
    if operator == "~=":
        # Compatible release: same major.minor, >= right.
        return left_parts[:2] == right_parts[:2] and left_parts >= right_parts
    raise PluginError(f"unsupported version operator: {operator!r}")


def _version_key(version: str) -> tuple[int, ...]:
    """Turn ``1.2.3`` into ``(1, 2, 3)``; non-numeric parts sort as 0."""
    parts: list[int] = []
    for chunk in re.split(r"[-+.]", version):
        digits = re.match(r"\d+", chunk)
        parts.append(int(digits.group(0)) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _require_str(data: Mapping[str, Any], key: str, *, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PluginError(f"{source}: manifest field {key!r} must be a non-empty string")
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PluginError(f"manifest field {key!r} must be a string, got {type(value).__name__}")
    return value.strip()


def _str_list(data: Mapping[str, Any], key: str, *, source: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise PluginError(
            f"{source}: manifest field {key!r} must be a list, got {type(value).__name__}"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PluginError(f"{source}: manifest field {key!r} entries must be non-empty strings")
        items.append(item.strip())
    return tuple(items)


def validate_manifest(data: Any, *, source: str = "<manifest>", check_entry: bool = False) -> PluginManifest:
    """Validate a raw manifest mapping and return a :class:`PluginManifest`.

    Args:
        data: The parsed YAML/TOML document.
        source: Human-readable origin used in error messages.
        check_entry: When True, require the entry file to exist next to
            ``source`` (only meaningful when ``source`` is a real path).

    Raises:
        PluginError: On any schema violation.
    """
    if not isinstance(data, Mapping):
        raise PluginError(f"{source}: manifest must be a mapping, got {type(data).__name__}")

    unknown = sorted(set(data) - _ALLOWED_KEYS)
    if unknown:
        raise PluginError(f"{source}: unknown manifest field(s): {', '.join(unknown)}")

    name = _require_str(data, "name", source=source)
    if not _NAME_RE.match(name):
        raise PluginError(
            f"{source}: plugin name {name!r} must match {_NAME_RE.pattern} "
            "(lowercase letter first, then lowercase letters, digits, '-' or '_')"
        )

    version = _require_str(data, "version", source=source)
    if not _VERSION_RE.match(version):
        raise PluginError(
            f"{source}: version {version!r} must look like '1.2.3' (optionally with -pre/+build)"
        )

    entry = _optional_str(data, "entry") or "plugin.py"
    if not entry.endswith(".py"):
        raise PluginError(f"{source}: entry {entry!r} must be a .py file")
    if Path(entry).name != entry:
        raise PluginError(f"{source}: entry {entry!r} must be a plain file name, not a path")
    if check_entry and source != "<manifest>":
        entry_path = Path(source).parent / entry
        if not entry_path.is_file():
            raise PluginError(f"{source}: entry file does not exist: {entry_path}")

    hooks = _str_list(data, "hooks", source=source)
    from llmdiff.plugins.hooks import HOOK_POINTS  # local import to avoid a cycle

    for hook in hooks:
        if hook not in HOOK_POINTS:
            raise PluginError(
                f"{source}: unknown hook {hook!r}; allowed: {', '.join(HOOK_POINTS)}"
            )

    raw_deps = data.get("dependencies", [])
    if raw_deps is None:
        raw_deps = []
    if not isinstance(raw_deps, (list, tuple)):
        raise PluginError(f"{source}: 'dependencies' must be a list of strings")
    dependencies = tuple(PluginDependency.parse(str(dep)) for dep in raw_deps)
    dep_names = [dep.name for dep in dependencies]
    if len(set(dep_names)) != len(dep_names):
        raise PluginError(f"{source}: duplicate dependency entries: {dep_names}")
    if name in dep_names:
        raise PluginError(f"{source}: plugin {name!r} cannot depend on itself")

    permissions = _str_list(data, "permissions", source=source)
    for permission in permissions:
        if permission not in ALLOWED_PERMISSIONS:
            raise PluginError(
                f"{source}: unknown permission {permission!r}; "
                f"allowed: {', '.join(sorted(ALLOWED_PERMISSIONS))}"
            )

    min_version = _optional_str(data, "min_llmdiff_version") or "0.0.0"
    if not _VERSION_RE.match(min_version):
        raise PluginError(f"{source}: min_llmdiff_version {min_version!r} must look like '1.2.3'")

    homepage = _optional_str(data, "homepage")
    if homepage and not re.match(r"^https?://", homepage):
        raise PluginError(f"{source}: homepage must start with http:// or https://")

    extra = {key: value for key, value in data.items() if key not in _ALLOWED_KEYS}
    return PluginManifest(
        name=name,
        version=version,
        author=_optional_str(data, "author"),
        description=_optional_str(data, "description"),
        entry=entry,
        hooks=hooks,
        dependencies=dependencies,
        permissions=permissions,
        min_llmdiff_version=min_version,
        homepage=homepage,
        license=_optional_str(data, "license"),
        source=source if source != "<manifest>" else None,
        extra=extra,
    )


def parse_manifest(text: str, *, source: str = "<manifest>", check_entry: bool = False) -> PluginManifest:
    """Parse manifest text in YAML or TOML form.

    The format is chosen from the ``source`` extension when available
    (``.toml`` -> TOML, anything else -> YAML); YAML is a superset that also
    parses simple TOML-looking flow documents poorly, so prefer explicit
    extensions.

    Raises:
        PluginError: If the text cannot be parsed or fails validation.
    """
    if not isinstance(text, str) or not text.strip():
        raise PluginError(f"{source}: manifest text is empty")
    if source.endswith(".toml"):
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise PluginError(f"{source}: invalid TOML: {exc}") from exc
        return validate_manifest(data, source=source, check_entry=check_entry)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PluginError(
            f"{source}: PyYAML is required to parse YAML manifests; "
            "install pyyaml or use a .toml manifest"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PluginError(f"{source}: invalid YAML: {exc}") from exc
    return validate_manifest(data, source=source, check_entry=check_entry)


def load_manifest(path: str | Path, *, check_entry: bool = True) -> PluginManifest:
    """Load and validate a manifest file from disk.

    Args:
        path: Path to a ``.yaml``/``.yml``/``.toml`` manifest file.
        check_entry: Require the entry file to exist next to the manifest.

    Raises:
        PluginError: If the file is missing, unreadable or invalid.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PluginError(f"manifest not found: {manifest_path}")
    if manifest_path.suffix not in (".yaml", ".yml", ".toml"):
        raise PluginError(
            f"unsupported manifest extension {manifest_path.suffix!r}; "
            "expected .yaml, .yml or .toml"
        )
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return parse_manifest(text, source=str(manifest_path), check_entry=check_entry)


def find_manifest(directory: str | Path) -> Path | None:
    """Return the manifest file inside ``directory``, or None.

    Searches the well-known basenames in preference order.
    """
    base = Path(directory)
    if not base.is_dir():
        return None
    for basename in MANIFEST_BASENAMES:
        candidate = base / basename
        if candidate.is_file():
            return candidate
    return None


def manifest_to_dict(manifest: PluginManifest) -> dict[str, Any]:
    """Serialise a manifest back to a plain dict (see ``to_dict``)."""
    return manifest.to_dict()
