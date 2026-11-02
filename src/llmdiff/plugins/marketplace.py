"""A tiny JSON marketplace for llmdiff plugins.

A marketplace is a directory (local) or URL (remote) that contains an
``index.json`` describing available plugins::

    {
      "name": "community",
      "plugins": [
        {
          "name": "politeness",
          "version": "1.0.0",
          "description": "Scores politeness of outputs.",
          "author": "jane",
          "homepage": "https://example.com/politeness",
          "license": "MIT",
          "download": "politeness-1.0.0.zip",        # relative to the index
          "sha256": "abc123...",                     # optional but recommended
          "dependencies": ["base-checks>=1.0.0"],
          "hooks": ["after_eval"],
          "permissions": []
        }
      ]
    }

:class:`Marketplace` can ``search`` the index, ``install`` a plugin into a
local plugin directory (verifying the sha256 when present), ``uninstall`` it,
and ``list_installed`` what is already there.

All network access goes through an injectable ``fetcher`` callable so tests
can run fully offline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from llmdiff.core.errors import PluginError
from llmdiff.plugins.loader import DEFAULT_PLUGIN_DIR
from llmdiff.plugins.manifest import PluginManifest, load_manifest

__all__ = [
    "INDEX_FILENAME",
    "MarketplaceEntry",
    "MarketplaceIndex",
    "Marketplace",
    "fetch_url",
    "sha256_of",
]

INDEX_FILENAME = "index.json"
"""Name of the marketplace index file."""

Fetcher = Callable[[str], bytes]
"""A callable that returns the bytes for a URL."""


def fetch_url(url: str, *, timeout: float = 10.0) -> bytes:
    """Fetch ``url`` with urllib and return the raw bytes.

    Raises:
        PluginError: On any network or HTTP error.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - caller chooses URL
            return response.read()
    except Exception as exc:  # noqa: BLE001 - normalise to PluginError
        raise PluginError(f"failed to fetch {url}: {type(exc).__name__}: {exc}") from exc


def sha256_of(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class MarketplaceEntry:
    """One plugin listed in a marketplace index.

    Attributes:
        name: Plugin name.
        version: Version string.
        description: Short description.
        author: Author handle.
        homepage: Project URL.
        license: SPDX identifier.
        download: Download URL or path relative to the index URL.
        sha256: Expected SHA-256 of the download, when published.
        dependencies: Dependency specs.
        hooks: Hook points the plugin subscribes to.
        permissions: Permissions the plugin requests.
        raw: The raw index record.
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    download: str = ""
    sha256: str = ""
    dependencies: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarketplaceEntry:
        """Build an entry from an index record, validating required keys."""
        if not isinstance(data, Mapping):
            raise PluginError(f"marketplace entry must be a mapping, got {type(data).__name__}")
        name = data.get("name")
        version = data.get("version")
        if not isinstance(name, str) or not name.strip():
            raise PluginError("marketplace entry missing 'name'")
        if not isinstance(version, str) or not version.strip():
            raise PluginError(f"marketplace entry {name!r} missing 'version'")
        deps = data.get("dependencies") or []
        hooks = data.get("hooks") or []
        perms = data.get("permissions") or []
        if not isinstance(deps, (list, tuple)) or not all(isinstance(d, str) for d in deps):
            raise PluginError(f"marketplace entry {name!r}: 'dependencies' must be a list of strings")
        if not isinstance(hooks, (list, tuple)) or not all(isinstance(h, str) for h in hooks):
            raise PluginError(f"marketplace entry {name!r}: 'hooks' must be a list of strings")
        if not isinstance(perms, (list, tuple)) or not all(isinstance(p, str) for p in perms):
            raise PluginError(f"marketplace entry {name!r}: 'permissions' must be a list of strings")
        return cls(
            name=name.strip(),
            version=version.strip(),
            description=str(data.get("description", "")),
            author=str(data.get("author", "")),
            homepage=str(data.get("homepage", "")),
            license=str(data.get("license", "")),
            download=str(data.get("download", "")),
            sha256=str(data.get("sha256", "")),
            dependencies=tuple(deps),
            hooks=tuple(hooks),
            permissions=tuple(perms),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to an index record."""
        return dict(self.raw)


@dataclass(frozen=True, slots=True)
class MarketplaceIndex:
    """A parsed marketplace index.

    Attributes:
        name: Marketplace name.
        entries: Plugin entries keyed by name.
        base_url: URL the index was loaded from (for resolving relative
            download links). Empty for local file indexes.
    """

    name: str
    entries: Mapping[str, MarketplaceEntry]
    base_url: str = ""

    @classmethod
    def from_dict(cls, data: Any, *, base_url: str = "") -> MarketplaceIndex:
        """Parse an index document.

        Raises:
            PluginError: On malformed documents.
        """
        if not isinstance(data, Mapping):
            raise PluginError(f"marketplace index must be a mapping, got {type(data).__name__}")
        plugins = data.get("plugins", [])
        if not isinstance(plugins, (list, tuple)):
            raise PluginError("marketplace index 'plugins' must be a list")
        entries: dict[str, MarketplaceEntry] = {}
        for record in plugins:
            entry = MarketplaceEntry.from_dict(record)
            if entry.name in entries:
                raise PluginError(f"duplicate plugin {entry.name!r} in marketplace index")
            entries[entry.name] = entry
        return cls(name=str(data.get("name", "marketplace")), entries=entries, base_url=base_url)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the index."""
        return {
            "name": self.name,
            "plugins": [entry.to_dict() for entry in self.entries.values()],
        }

    def search(self, query: str = "", *, limit: int = 50) -> tuple[MarketplaceEntry, ...]:
        """Return entries whose name/description match ``query`` (case-insensitive).

        An empty query returns everything, capped at ``limit``.
        """
        if limit <= 0:
            return ()
        needle = query.strip().lower()
        results: list[MarketplaceEntry] = []
        for entry in self.entries.values():
            if not needle:
                results.append(entry)
                continue
            haystack = f"{entry.name} {entry.description} {entry.author}".lower()
            if needle in haystack:
                results.append(entry)
            if len(results) >= limit:
                break
        return tuple(results)

    def get(self, name: str) -> MarketplaceEntry:
        """Return the entry for ``name``.

        Raises:
            PluginError: When absent.
        """
        try:
            return self.entries[name]
        except KeyError:
            raise PluginError(f"plugin {name!r} not found in marketplace {self.name!r}") from None


@dataclass(slots=True)
class Marketplace:
    """A plugin marketplace: search, install, uninstall.

    Attributes:
        index_url: URL or local path of ``index.json``.
        install_dir: Local directory plugins are installed into
            (default ``~/.llmdiff/plugins``).
        fetcher: Callable used to fetch remote bytes; defaults to
            :func:`fetch_url`. Inject a fake for offline tests.
    """

    index_url: str
    install_dir: Path = field(default_factory=lambda: DEFAULT_PLUGIN_DIR)
    fetcher: Fetcher = fetch_url
    _index: MarketplaceIndex | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # index
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> MarketplaceIndex:
        """The parsed index, loaded lazily and cached."""
        if self._index is None:
            self._index = self.load_index()
        return self._index

    def load_index(self) -> MarketplaceIndex:
        """Fetch and parse the index.

        Raises:
            PluginError: On fetch/parse errors.
        """
        raw = self._read(self.index_url)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginError(f"invalid marketplace index at {self.index_url}: {exc}") from exc
        base = self.index_url.rsplit("/", 1)[0] + "/" if "/" in self.index_url else ""
        return MarketplaceIndex.from_dict(data, base_url=base)

    def reload_index(self) -> MarketplaceIndex:
        """Drop the cached index and load it again."""
        self._index = None
        return self.index

    def _read(self, url_or_path: str) -> bytes:
        if url_or_path.startswith(("http://", "https://")):
            return self.fetcher(url_or_path)
        path = Path(url_or_path)
        if not path.is_file():
            raise PluginError(f"marketplace index not found: {path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PluginError(f"cannot read {path}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    def search(self, query: str = "", *, limit: int = 50) -> tuple[MarketplaceEntry, ...]:
        """Search the index (see :meth:`MarketplaceIndex.search`)."""
        return self.index.search(query, limit=limit)

    def list_all(self) -> tuple[MarketplaceEntry, ...]:
        """Return every entry in the index."""
        return tuple(self.index.entries.values())

    # ------------------------------------------------------------------ #
    # install / uninstall
    # ------------------------------------------------------------------ #

    def install(self, name: str, *, version: str | None = None) -> Path:
        """Download and install plugin ``name`` into ``install_dir``.

        The download must be a zip archive containing a plugin directory with
        a manifest. When the entry publishes a ``sha256``, the archive is
        verified before extraction.

        Args:
            name: Plugin name in the index.
            version: Optional version pin; must match the index entry.

        Returns:
            The directory the plugin was installed into.

        Raises:
            PluginError: On unknown plugin, version mismatch, checksum
                mismatch, unsafe archive paths, or IO errors.
        """
        entry = self.index.get(name)
        if version is not None and entry.version != version:
            raise PluginError(
                f"plugin {name!r} version {version!r} not in marketplace "
                f"(available: {entry.version})"
            )
        if not entry.download:
            raise PluginError(f"plugin {name!r} has no download URL")
        url = entry.download
        if not url.startswith(("http://", "https://")) and self.index.base_url:
            url = self.index.base_url + url
        archive = self._read(url)
        if entry.sha256:
            digest = sha256_of(archive)
            if digest.lower() != entry.sha256.lower():
                raise PluginError(
                    f"sha256 mismatch for {name!r}: expected {entry.sha256}, got {digest}"
                )
        target = self.install_dir / name
        self._extract_zip(archive, target)
        # Validate the installed manifest when present.
        manifest_path = None
        for candidate in target.rglob("llmdiff-plugin.*"):
            manifest_path = candidate
            break
        if manifest_path is not None:
            load_manifest(manifest_path, check_entry=True)
        return target

    def _extract_zip(self, archive: bytes, target: Path) -> None:
        """Safely extract a zip archive into ``target``.

        Refuses entries that would escape ``target`` (zip-slip protection).
        """
        import io

        try:
            zf = zipfile.ZipFile(io.BytesIO(archive))
        except zipfile.BadZipFile as exc:
            raise PluginError(f"download is not a valid zip archive: {exc}") from exc
        target.mkdir(parents=True, exist_ok=True)
        resolved_target = target.resolve()
        for info in zf.infolist():
            dest = (target / info.filename).resolve()
            if not str(dest).startswith(str(resolved_target) + "/") and dest != resolved_target:
                raise PluginError(f"unsafe path in archive: {info.filename!r}")
        zf.extractall(target)

    def uninstall(self, name: str) -> bool:
        """Remove an installed plugin directory.

        Returns:
            True when something was removed.

        Raises:
            PluginError: On IO errors.
        """
        target = self.install_dir / name
        if not target.exists():
            return False
        try:
            shutil.rmtree(target)
        except OSError as exc:
            raise PluginError(f"cannot uninstall {name!r} from {target}: {exc}") from exc
        return True

    def list_installed(self) -> tuple[str, ...]:
        """Return names of installed plugins (subdirectories with a manifest)."""
        if not self.install_dir.is_dir():
            return ()
        names: list[str] = []
        for child in sorted(self.install_dir.iterdir()):
            if child.is_dir() and any(
                (child / basename).is_file()
                for basename in ("llmdiff-plugin.yaml", "llmdiff-plugin.yml", "llmdiff-plugin.toml")
            ):
                names.append(child.name)
        return tuple(names)

    def is_installed(self, name: str) -> bool:
        """True when ``name`` appears in :meth:`list_installed`."""
        return name in self.list_installed()

    def upgrade(self, name: str) -> Path:
        """Re-install ``name`` from the marketplace (uninstall + install)."""
        self.uninstall(name)
        return self.install(name)


def build_index(entries: list[Mapping[str, Any]], *, name: str = "marketplace") -> MarketplaceIndex:
    """Build a :class:`MarketplaceIndex` from raw entry dicts (for publishing)."""
    return MarketplaceIndex.from_dict({"name": name, "plugins": entries})
