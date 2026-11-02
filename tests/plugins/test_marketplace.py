"""Tests for llmdiff.plugins.marketplace."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from llmdiff.core.errors import PluginError
from llmdiff.plugins.marketplace import (
    INDEX_FILENAME,
    Marketplace,
    MarketplaceEntry,
    MarketplaceIndex,
    build_index,
    sha256_of,
)


def make_index_dict() -> dict:
    return {
        "name": "test-market",
        "plugins": [
            {
                "name": "alpha",
                "version": "1.0.0",
                "description": "Alpha dimension plugin",
                "author": "alice",
                "license": "MIT",
                "download": "alpha-1.0.0.zip",
                "sha256": "",
                "dependencies": [],
                "hooks": ["after_eval"],
                "permissions": [],
            },
            {
                "name": "beta",
                "version": "2.1.0",
                "description": "Beta reporter plugin for verdicts",
                "author": "bob",
                "license": "Apache-2.0",
                "download": "beta-2.1.0.zip",
            },
        ],
    }


def make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


GOOD_MANIFEST = "name: alpha\nversion: 1.0.0\nentry: plugin.py\n"
GOOD_ENTRY = "PLUGIN = 1\n"


class TestMarketplaceEntry:
    def test_from_dict_full(self):
        entry = MarketplaceEntry.from_dict(make_index_dict()["plugins"][0])
        assert entry.name == "alpha"
        assert entry.version == "1.0.0"
        assert entry.author == "alice"
        assert entry.hooks == ("after_eval",)
        assert entry.download == "alpha-1.0.0.zip"

    def test_from_dict_minimal(self):
        entry = MarketplaceEntry.from_dict({"name": "x", "version": "0.1.0"})
        assert entry.description == ""
        assert entry.dependencies == ()
        assert entry.raw["name"] == "x"

    def test_from_dict_not_mapping(self):
        with pytest.raises(PluginError, match="mapping"):
            MarketplaceEntry.from_dict([1])  # type: ignore[arg-type]

    def test_missing_name(self):
        with pytest.raises(PluginError, match="name"):
            MarketplaceEntry.from_dict({"version": "1.0.0"})

    def test_missing_version(self):
        with pytest.raises(PluginError, match="version"):
            MarketplaceEntry.from_dict({"name": "x"})

    def test_bad_dependencies_type(self):
        with pytest.raises(PluginError, match="dependencies"):
            MarketplaceEntry.from_dict({"name": "x", "version": "1.0.0", "dependencies": "a"})

    def test_bad_hooks_type(self):
        with pytest.raises(PluginError, match="hooks"):
            MarketplaceEntry.from_dict({"name": "x", "version": "1.0.0", "hooks": 5})

    def test_bad_permissions_type(self):
        with pytest.raises(PluginError, match="permissions"):
            MarketplaceEntry.from_dict({"name": "x", "version": "1.0.0", "permissions": [1]})

    def test_to_dict_round_trip(self):
        raw = make_index_dict()["plugins"][0]
        entry = MarketplaceEntry.from_dict(raw)
        assert entry.to_dict() == raw


class TestMarketplaceIndex:
    def test_from_dict(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        assert index.name == "test-market"
        assert set(index.entries) == {"alpha", "beta"}

    def test_from_dict_not_mapping(self):
        with pytest.raises(PluginError, match="mapping"):
            MarketplaceIndex.from_dict([1, 2])

    def test_plugins_not_list(self):
        with pytest.raises(PluginError, match="'plugins' must be a list"):
            MarketplaceIndex.from_dict({"plugins": "x"})

    def test_duplicate_plugin_rejected(self):
        data = make_index_dict()
        data["plugins"].append(dict(data["plugins"][0]))
        with pytest.raises(PluginError, match="duplicate"):
            MarketplaceIndex.from_dict(data)

    def test_search_empty_returns_all(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        results = index.search("")
        assert {e.name for e in results} == {"alpha", "beta"}

    def test_search_by_name(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        results = index.search("alpha")
        assert [e.name for e in results] == ["alpha"]

    def test_search_by_description(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        results = index.search("verdicts")
        assert [e.name for e in results] == ["beta"]

    def test_search_by_author(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        results = index.search("BOB")
        assert [e.name for e in results] == ["beta"]

    def test_search_no_match(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        assert index.search("zzz") == ()

    def test_search_limit(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        assert len(index.search("", limit=1)) == 1
        assert index.search("", limit=0) == ()

    def test_get(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        assert index.get("beta").version == "2.1.0"
        with pytest.raises(PluginError, match="not found"):
            index.get("ghost")

    def test_to_dict_round_trip(self):
        index = MarketplaceIndex.from_dict(make_index_dict())
        data = index.to_dict()
        assert data["name"] == "test-market"
        assert len(data["plugins"]) == 2

    def test_build_index_helper(self):
        index = build_index([{"name": "x", "version": "1.0.0"}], name="m")
        assert index.name == "m"
        assert "x" in index.entries


class TestMarketplaceLocal:
    def test_load_local_index(self, tmp_path):
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text(json.dumps(make_index_dict()), encoding="utf-8")
        market = Marketplace(index_url=str(index_file), install_dir=tmp_path / "installs")
        assert market.index.name == "test-market"
        assert len(market.list_all()) == 2

    def test_missing_index(self, tmp_path):
        market = Marketplace(index_url=str(tmp_path / "nope.json"), install_dir=tmp_path)
        with pytest.raises(PluginError, match="not found"):
            market.load_index()

    def test_invalid_json_index(self, tmp_path):
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text("{not json", encoding="utf-8")
        market = Marketplace(index_url=str(index_file), install_dir=tmp_path)
        with pytest.raises(PluginError, match="invalid marketplace index"):
            market.load_index()

    def test_search_delegates(self, tmp_path):
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text(json.dumps(make_index_dict()), encoding="utf-8")
        market = Marketplace(index_url=str(index_file), install_dir=tmp_path)
        assert [e.name for e in market.search("alpha")] == ["alpha"]

    def test_reload_index(self, tmp_path):
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text(json.dumps(make_index_dict()), encoding="utf-8")
        market = Marketplace(index_url=str(index_file), install_dir=tmp_path)
        assert len(market.index.entries) == 2
        data = make_index_dict()
        data["plugins"] = data["plugins"][:1]
        index_file.write_text(json.dumps(data), encoding="utf-8")
        assert len(market.index.entries) == 2  # cached
        assert len(market.reload_index().entries) == 1


class TestMarketplaceInstall:
    def _market_with_zip(self, tmp_path, *, sha: str = "", zip_bytes: bytes | None = None):
        zip_bytes = zip_bytes or make_zip(
            {"alpha/llmdiff-plugin.yaml": GOOD_MANIFEST, "alpha/plugin.py": GOOD_ENTRY}
        )
        data = make_index_dict()
        data["plugins"][0]["sha256"] = sha or sha256_of(zip_bytes)
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text(json.dumps(data), encoding="utf-8")
        zip_file = tmp_path / "alpha-1.0.0.zip"
        zip_file.write_bytes(zip_bytes)

        def fetcher(url: str) -> bytes:
            assert url.endswith("alpha-1.0.0.zip")
            return zip_file.read_bytes()

        install_dir = tmp_path / "installs"
        market = Marketplace(
            index_url=str(index_file), install_dir=install_dir, fetcher=fetcher
        )
        # point the download at the local file via base_url resolution
        return market, install_dir

    def test_install_success(self, tmp_path):
        market, install_dir = self._market_with_zip(tmp_path)
        target = market.install("alpha")
        assert target == install_dir / "alpha"
        assert (target / "alpha" / "plugin.py").is_file()
        assert market.is_installed("alpha")
        assert market.list_installed() == ("alpha",)

    def test_install_unknown_plugin(self, tmp_path):
        market, _ = self._market_with_zip(tmp_path)
        with pytest.raises(PluginError, match="not found"):
            market.install("ghost")

    def test_install_version_mismatch(self, tmp_path):
        market, _ = self._market_with_zip(tmp_path)
        with pytest.raises(PluginError, match="version"):
            market.install("alpha", version="9.9.9")

    def test_install_version_match(self, tmp_path):
        market, _ = self._market_with_zip(tmp_path)
        assert market.install("alpha", version="1.0.0").name == "alpha"

    def test_install_sha_mismatch(self, tmp_path):
        market, _ = self._market_with_zip(tmp_path, sha="0" * 64)
        with pytest.raises(PluginError, match="sha256 mismatch"):
            market.install("alpha")

    def test_install_no_download(self, tmp_path):
        data = make_index_dict()
        data["plugins"][0]["download"] = ""
        index_file = tmp_path / INDEX_FILENAME
        index_file.write_text(json.dumps(data), encoding="utf-8")
        market = Marketplace(index_url=str(index_file), install_dir=tmp_path)
        with pytest.raises(PluginError, match="no download"):
            market.install("alpha")

    def test_install_bad_zip(self, tmp_path):
        market, _ = self._market_with_zip(tmp_path, zip_bytes=b"not a zip")
        with pytest.raises(PluginError, match="not a valid zip"):
            market.install("alpha")

    def test_zip_slip_rejected(self, tmp_path):
        evil = make_zip({"../evil.py": "x"})
        market, _ = self._market_with_zip(tmp_path, zip_bytes=evil)
        with pytest.raises(PluginError, match="unsafe path"):
            market.install("alpha")

    def test_uninstall(self, tmp_path):
        market, install_dir = self._market_with_zip(tmp_path)
        market.install("alpha")
        assert market.uninstall("alpha") is True
        assert not (install_dir / "alpha").exists()
        assert market.uninstall("alpha") is False

    def test_upgrade(self, tmp_path):
        market, install_dir = self._market_with_zip(tmp_path)
        market.install("alpha")
        (install_dir / "alpha" / "marker.txt").write_text("old", encoding="utf-8")
        market.upgrade("alpha")
        assert not (install_dir / "alpha" / "marker.txt").exists()
        assert (install_dir / "alpha" / "alpha" / "plugin.py").is_file()

    def test_list_installed_empty(self, tmp_path):
        market = Marketplace(index_url=str(tmp_path / "x.json"), install_dir=tmp_path / "nope")
        assert market.list_installed() == ()
        assert market.is_installed("x") is False


class TestMarketplaceRemote:
    def test_remote_index_via_fetcher(self):
        payload = json.dumps(make_index_dict()).encode()

        def fetcher(url: str) -> bytes:
            assert url == "https://example.com/market/index.json"
            return payload

        market = Marketplace(
            index_url="https://example.com/market/index.json",
            install_dir="/tmp/llmdiff-test-installs",
            fetcher=fetcher,
        )
        assert market.index.base_url == "https://example.com/market/"
        assert len(market.index.entries) == 2

    def test_remote_fetch_error(self):
        def fetcher(url: str) -> bytes:
            raise OSError("connection refused")

        market = Marketplace(
            index_url="https://example.com/index.json",
            install_dir="/tmp/x",
            fetcher=fetcher,
        )
        with pytest.raises(PluginError, match="failed to fetch"):
            market.load_index()

    def test_relative_download_resolved_against_base(self):
        index_payload = json.dumps(make_index_dict()).encode()
        zip_bytes = make_zip({"alpha/llmdiff-plugin.yaml": GOOD_MANIFEST, "alpha/plugin.py": GOOD_ENTRY})
        calls: list[str] = []

        def fetcher(url: str) -> bytes:
            calls.append(url)
            if url.endswith("index.json"):
                return index_payload
            return zip_bytes

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            market = Marketplace(
                index_url="https://example.com/market/index.json",
                install_dir=Path(tmp) / "installs",
                fetcher=fetcher,
            )
            # fix sha to match our zip
            market._index = None
            data = make_index_dict()
            data["plugins"][0]["sha256"] = sha256_of(zip_bytes)
            index_payload = json.dumps(data).encode()
            market.install("alpha")
            assert any("alpha-1.0.0.zip" in c for c in calls)
            assert (Path(tmp) / "installs" / "alpha").is_dir()


class TestSha256:
    def test_known_vector(self):
        assert sha256_of(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert sha256_of(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
