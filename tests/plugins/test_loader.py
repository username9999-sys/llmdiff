"""Tests for llmdiff.plugins.loader."""

from __future__ import annotations

import textwrap

import pytest

from llmdiff.core.checks import Check
from llmdiff.core.errors import PluginError
from llmdiff.plugins.loader import (
    DEFAULT_PLUGIN_DIR,
    ENTRY_POINT_GROUP,
    LoadedPlugin,
    PluginLoader,
    discover_plugins,
    iter_plugin_dirs,
    load_directory,
    load_file,
)
from llmdiff.plugins.manifest import PluginManifest

from .conftest import good_entry, good_manifest

CHECK_ENTRY = good_entry("Alpha", "alpha_dim")


class TestLoadFile:
    def test_loads_check_plugin(self, tmp_path):
        path = tmp_path / "alpha.py"
        path.write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_file(path)
        assert loaded.name == "alpha"
        assert loaded.source == "file"
        assert loaded.is_check
        assert loaded.obj.name == "alpha_dim"
        assert loaded.version == "0.0.0"
        assert loaded.path == str(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(PluginError, match="not found"):
            load_file(tmp_path / "nope.py")

    def test_non_py_rejected(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(PluginError, match=".py"):
            load_file(path)

    def test_execution_error_wrapped(self, tmp_path):
        path = tmp_path / "broken.py"
        path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        with pytest.raises(PluginError, match="error while executing"):
            load_file(path)

    def test_no_plugin_objects(self, tmp_path):
        path = tmp_path / "empty.py"
        path.write_text("X = 1\n", encoding="utf-8")
        with pytest.raises(PluginError, match="no plugin objects"):
            load_file(path)

    def test_plugins_list(self, tmp_path):
        path = tmp_path / "multi.py"
        path.write_text(
            textwrap.dedent(
                '''
                class A:
                    @property
                    def name(self): return "a"
                    def score(self, output, context): return 1.0

                class B:
                    @property
                    def name(self): return "b"
                    def score(self, output, context): return 0.5

                PLUGINS = [A(), B()]
                '''
            ),
            encoding="utf-8",
        )
        loaded = load_file(path)
        assert isinstance(loaded.obj, Check)
        assert loaded.obj.name == "a"
        assert len(loaded.module.PLUGINS) == 2  # type: ignore[attr-defined]

    def test_get_plugin_factory(self, tmp_path):
        path = tmp_path / "factory.py"
        path.write_text(
            textwrap.dedent(
                '''
                class C:
                    @property
                    def name(self): return "c"
                    def score(self, output, context): return 1.0

                def get_plugin():
                    return C()
                '''
            ),
            encoding="utf-8",
        )
        loaded = load_file(path)
        assert loaded.obj.name == "c"

    def test_get_plugins_factory(self, tmp_path):
        path = tmp_path / "factories.py"
        path.write_text(
            textwrap.dedent(
                '''
                class C:
                    @property
                    def name(self): return "c"
                    def score(self, output, context): return 1.0

                def get_plugins():
                    return [C(), C()]
                '''
            ),
            encoding="utf-8",
        )
        loaded = load_file(path)
        assert loaded.obj.name == "c"

    def test_check_class_auto_discovery(self, tmp_path):
        path = tmp_path / "auto.py"
        path.write_text(
            textwrap.dedent(
                '''
                class AutoCheck:
                    @property
                    def name(self): return "auto"
                    def score(self, output, context): return 0.0
                '''
            ),
            encoding="utf-8",
        )
        loaded = load_file(path)
        assert loaded.obj.name == "auto"

    def test_hooks_collected(self, tmp_path):
        path = tmp_path / "hooky.py"
        path.write_text(
            textwrap.dedent(
                '''
                from llmdiff.plugins.hooks import hook

                @hook("after_eval", plugin="hooky")
                def on_done(*, result):
                    return "done"
                '''
            ),
            encoding="utf-8",
        )
        loaded = load_file(path)
        assert len(loaded.hooks) == 1
        point, cb, plugin, priority = loaded.hooks[0]
        assert point == "after_eval"
        assert plugin == "hooky"
        assert cb(result="r") == "done"

    def test_with_manifest(self, tmp_path):
        manifest = PluginManifest(name="named", version="2.0.0", entry="plugin.py")
        path = tmp_path / "plugin.py"
        path.write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_file(path, manifest=manifest)
        assert loaded.name == "named"
        assert loaded.version == "2.0.0"
        assert loaded.source == "manifest"

    def test_manifest_permission_not_granted(self, tmp_path):
        manifest = PluginManifest(name="x", version="1.0.0", entry="plugin.py", permissions=("subprocess",))
        path = tmp_path / "plugin.py"
        path.write_text(CHECK_ENTRY, encoding="utf-8")
        with pytest.raises(PluginError, match="not granted"):
            load_file(path, manifest=manifest, allowed_permissions=("network",))

    def test_manifest_permission_granted(self, tmp_path):
        manifest = PluginManifest(name="x", version="1.0.0", entry="plugin.py", permissions=("network",))
        path = tmp_path / "plugin.py"
        path.write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_file(path, manifest=manifest, allowed_permissions=("network", "filesystem"))
        assert loaded.name == "x"


class TestLoadDirectory:
    def test_empty_dir(self, tmp_path):
        assert load_directory(tmp_path) == []

    def test_missing_dir(self, tmp_path):
        assert load_directory(tmp_path / "nope") == []

    def test_manifest_layout(self, tmp_path):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "llmdiff-plugin.yaml").write_text(good_manifest("my-plugin"), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_directory(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].name == "my-plugin"
        assert loaded[0].version == "1.0.0"
        assert loaded[0].is_check

    def test_loose_py_file(self, tmp_path):
        (tmp_path / "loose.py").write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_directory(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].name == "loose"
        assert loaded[0].source == "file"

    def test_single_plugin_layout(self, tmp_path):
        (tmp_path / "llmdiff-plugin.yaml").write_text(good_manifest("single"), encoding="utf-8")
        (tmp_path / "plugin.py").write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_directory(tmp_path)
        names = [item.name for item in loaded]
        assert "single" in names

    def test_broken_plugin_recorded_not_raised(self, tmp_path):
        plugin_dir = tmp_path / "broken"
        plugin_dir.mkdir()
        (plugin_dir / "llmdiff-plugin.yaml").write_text(good_manifest("broken"), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text("raise ValueError('x')\n", encoding="utf-8")
        loaded = load_directory(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].errors
        assert loaded[0].obj is None

    def test_hidden_files_skipped(self, tmp_path):
        (tmp_path / ".hidden.py").write_text(CHECK_ENTRY, encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        assert load_directory(tmp_path) == []

    def test_multiple_plugins_sorted(self, tmp_path):
        for name in ("zeta", "alpha", "mid"):
            d = tmp_path / name
            d.mkdir()
            (d / "llmdiff-plugin.yaml").write_text(good_manifest(name), encoding="utf-8")
            (d / "plugin.py").write_text(good_entry(name.capitalize(), f"{name}_dim"), encoding="utf-8")
        loaded = load_directory(tmp_path)
        assert [item.name for item in loaded] == ["alpha", "mid", "zeta"]


class TestDiscoverPlugins:
    def test_discovers_manifest_and_loose(self, tmp_path):
        d = tmp_path / "p1"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("p1"), encoding="utf-8")
        (d / "plugin.py").write_text(CHECK_ENTRY, encoding="utf-8")
        (tmp_path / "loose.py").write_text(CHECK_ENTRY, encoding="utf-8")
        found = discover_plugins([tmp_path])
        names = [p.name for p in found]
        assert "llmdiff-plugin.yaml" in names
        assert "loose.py" in names

    def test_missing_dir_ok(self, tmp_path):
        assert discover_plugins([tmp_path / "nope"]) == []


class TestIterPluginDirs:
    def test_only_existing(self, tmp_path):
        existing = tmp_path / "exists"
        existing.mkdir()
        dirs = list(iter_plugin_dirs([existing, tmp_path / "missing"]))
        assert dirs == [existing]

    def test_default_dir_may_not_exist(self):
        dirs = list(iter_plugin_dirs([]))
        assert all(d.is_dir() for d in dirs)

    def test_dedup(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        dirs = list(iter_plugin_dirs([d, d]))
        assert dirs == [d]


class TestPluginLoader:
    def test_load_dirs(self, tmp_path):
        (tmp_path / "loose.py").write_text(CHECK_ENTRY, encoding="utf-8")
        loader = PluginLoader([tmp_path], include_entry_points=False)
        loaded = loader.load_dirs()
        assert len(loaded) == 1
        assert loaded[0].name == "loose"

    def test_load_all_skips_entry_points(self, tmp_path):
        loader = PluginLoader([tmp_path], include_entry_points=False)
        assert loader.load_all() == []

    def test_discover(self, tmp_path):
        (tmp_path / "loose.py").write_text(CHECK_ENTRY, encoding="utf-8")
        loader = PluginLoader([tmp_path], include_entry_points=False)
        assert len(loader.discover()) == 1

    def test_default_plugin_dir_constant(self):
        assert DEFAULT_PLUGIN_DIR.name == "plugins"
        assert ENTRY_POINT_GROUP == "llmdiff.plugins"


class TestLoadedPlugin:
    def test_to_dict(self):
        loaded = LoadedPlugin(name="x", obj=None, source="file", path="/tmp/x.py", errors=["e1"])
        data = loaded.to_dict()
        assert data["name"] == "x"
        assert data["source"] == "file"
        assert data["is_check"] is False
        assert data["errors"] == ["e1"]

    def test_is_check_true(self, tmp_path):
        path = tmp_path / "c.py"
        path.write_text(CHECK_ENTRY, encoding="utf-8")
        loaded = load_file(path)
        assert loaded.is_check
        assert loaded.to_dict()["is_check"] is True
