"""Tests for llmdiff.plugins.registry."""

from __future__ import annotations

import textwrap

import pytest

from llmdiff.core.errors import PluginError
from llmdiff.plugins.config import ConfigStore
from llmdiff.plugins.hooks import HookManager
from llmdiff.plugins.loader import LoadedPlugin
from llmdiff.plugins.manifest import PluginDependency, PluginManifest
from llmdiff.plugins.registry import (
    PluginConflict,
    PluginEntry,
    PluginRegistry,
    resolve_dependencies,
)
from llmdiff.plugins.sandbox import SandboxLimits

from .conftest import good_entry, good_manifest


def manifest(name: str, deps: tuple[str, ...] = (), version: str = "1.0.0") -> PluginManifest:
    return PluginManifest(
        name=name,
        version=version,
        entry="plugin.py",
        dependencies=tuple(PluginDependency.parse(d) for d in deps),
    )


class TestResolveDependencies:
    def test_empty(self):
        order, conflicts = resolve_dependencies({})
        assert order == ()
        assert conflicts == []

    def test_single(self):
        order, conflicts = resolve_dependencies({"a": manifest("a")})
        assert order == ("a",)
        assert conflicts == []

    def test_linear_chain(self):
        manifests = {
            "c": manifest("c", ("b",)),
            "b": manifest("b", ("a",)),
            "a": manifest("a"),
        }
        order, conflicts = resolve_dependencies(manifests)
        assert order == ("a", "b", "c")
        assert conflicts == []

    def test_diamond(self):
        manifests = {
            "d": manifest("d", ("b", "c")),
            "b": manifest("b", ("a",)),
            "c": manifest("c", ("a",)),
            "a": manifest("a"),
        }
        order, conflicts = resolve_dependencies(manifests)
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order) == {"a", "b", "c", "d"}
        assert conflicts == []

    def test_cycle_detected(self):
        manifests = {
            "a": manifest("a", ("b",)),
            "b": manifest("b", ("a",)),
        }
        order, conflicts = resolve_dependencies(manifests)
        assert any(c.kind == "dependency_cycle" for c in conflicts)
        assert set(order) == {"a", "b"}

    def test_self_cycle(self):
        manifests = {"a": manifest("a", ("a",))}
        order, conflicts = resolve_dependencies(manifests)
        assert any(c.kind == "dependency_cycle" for c in conflicts)

    def test_missing_dependency_reported(self):
        manifests = {"a": manifest("a", ("ghost",))}
        order, conflicts = resolve_dependencies(manifests)
        assert any(c.kind == "missing_dependency" and "ghost" in c.plugins for c in conflicts)
        assert "a" in order

    def test_deterministic_order(self):
        manifests = {name: manifest(name) for name in ("z", "m", "a", "q")}
        order1, _ = resolve_dependencies(manifests)
        order2, _ = resolve_dependencies(manifests)
        assert order1 == order2 == ("a", "m", "q", "z")

    def test_conflict_str(self):
        conflict = PluginConflict(kind="k", plugins=("a", "b"), detail="d")
        assert str(conflict) == "[k] a, b: d"


class TestPluginRegistryLoading:
    def test_empty_registry(self):
        registry = PluginRegistry(plugin_dirs=[])
        registry.load_all()
        assert len(registry) == 0
        assert registry.names() == ()
        assert registry.checks() == ()

    def test_loads_from_dir(self, tmp_path):
        plugin_dir = tmp_path / "alpha"
        plugin_dir.mkdir()
        (plugin_dir / "llmdiff-plugin.yaml").write_text(good_manifest("alpha"), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(good_entry("Alpha", "alpha_dim"), encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        loaded = registry.load_all()
        assert len(loaded) == 1
        assert "alpha" in registry
        checks = registry.checks()
        assert [c.name for c in checks] == ["alpha_dim"]

    def test_load_order_respects_dependencies(self, tmp_path):
        for name, deps in (("base", ""), ("mid", "base"), ("top", "mid")):
            d = tmp_path / name
            d.mkdir()
            (d / "llmdiff-plugin.yaml").write_text(
                good_manifest(name, deps=deps), encoding="utf-8"
            )
            (d / "plugin.py").write_text(good_entry(name.capitalize(), f"{name}_dim"), encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        assert registry.names() == ("base", "mid", "top")

    def test_strict_mode_raises_on_broken(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("broken"), encoding="utf-8")
        (d / "plugin.py").write_text("raise ValueError('x')\n", encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        with pytest.raises(PluginError, match="broken"):
            registry.load_all(strict=True)

    def test_non_strict_records_error(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("broken"), encoding="utf-8")
        (d / "plugin.py").write_text("raise ValueError('x')\n", encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        entry = registry.get("broken")
        assert entry.loaded.errors

    def test_duplicate_check_conflict_detected(self, tmp_path):
        for name in ("one", "two"):
            d = tmp_path / name
            d.mkdir()
            (d / "llmdiff-plugin.yaml").write_text(good_manifest(name), encoding="utf-8")
            (d / "plugin.py").write_text(good_entry(name.capitalize(), "same_dim"), encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        conflicts = [c for c in registry.conflicts if c.kind == "duplicate_check"]
        assert len(conflicts) == 1
        assert conflicts[0].plugins == ("one", "two")
        assert registry.check("same_dim") is not None

    def test_hooks_subscribed_on_load(self, tmp_path):
        d = tmp_path / "hooky"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("hooky", hooks="after_eval"), encoding="utf-8")
        (d / "plugin.py").write_text(
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
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        assert len(registry.hooks.subscribers("after_eval")) == 1
        report = registry.hooks.emit("after_eval", result="r")
        assert report.results() == ["done"]

    def test_config_resolved_from_module_defaults(self, tmp_path):
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("cfg"), encoding="utf-8")
        (d / "plugin.py").write_text(
            "CONFIG_DEFAULTS = {'timeout': 30}\nPLUGIN = 1\n", encoding="utf-8"
        )
        registry = PluginRegistry(
            plugin_dirs=[tmp_path],
            include_entry_points=False,
            config_store=ConfigStore(config_dir=tmp_path / "cfgdir", env={}),
        )
        registry.load_all()
        entry = registry.get("cfg")
        assert entry.config is not None
        assert entry.config.get("timeout") == 30


class TestPluginRegistryLookup:
    def _registry_with(self, tmp_path, name: str = "p", dim: str = "p_dim") -> PluginRegistry:
        d = tmp_path / name
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest(name), encoding="utf-8")
        (d / "plugin.py").write_text(good_entry(name.capitalize(), dim), encoding="utf-8")
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        return registry

    def test_get_unknown_raises(self, tmp_path):
        registry = self._registry_with(tmp_path)
        with pytest.raises(PluginError, match="unknown plugin"):
            registry.get("ghost")

    def test_contains_and_len(self, tmp_path):
        registry = self._registry_with(tmp_path)
        assert "p" in registry
        assert "x" not in registry
        assert len(registry) == 1

    def test_all_and_enabled(self, tmp_path):
        registry = self._registry_with(tmp_path)
        assert len(registry.all()) == 1
        registry.set_enabled("p", False)
        assert registry.enabled() == ()
        assert registry.checks() == ()
        registry.set_enabled("p", True)
        assert len(registry.enabled()) == 1

    def test_check_lookup(self, tmp_path):
        registry = self._registry_with(tmp_path, dim="my_dim")
        check = registry.check("my_dim")
        assert check.name == "my_dim"
        with pytest.raises(PluginError, match="no plugin provides"):
            registry.check("ghost_dim")

    def test_score_sandboxed(self, tmp_path, make_context):
        registry = self._registry_with(tmp_path, dim="sc")
        check = registry.check("sc")
        ctx = make_context(baseline="a", candidate="b")
        score = registry.score_sandboxed("p", check, "hello", ctx)
        assert score == 1.0
        score0 = registry.score_sandboxed("p", check, "", ctx)
        assert score0 == 0.0

    def test_score_sandboxed_clamps(self, tmp_path, make_context):
        d = tmp_path / "clamp"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("clamp"), encoding="utf-8")
        (d / "plugin.py").write_text(
            textwrap.dedent(
                '''
                class C:
                    @property
                    def name(self): return "clamp_dim"
                    def score(self, output, context): return 99.0
                PLUGIN = C()
                '''
            ),
            encoding="utf-8",
        )
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        check = registry.check("clamp_dim")
        ctx = make_context()
        assert registry.score_sandboxed("clamp", check, "x", ctx) == 1.0

    def test_score_sandboxed_rejects_non_float(self, tmp_path, make_context):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "llmdiff-plugin.yaml").write_text(good_manifest("bad"), encoding="utf-8")
        (d / "plugin.py").write_text(
            textwrap.dedent(
                '''
                class C:
                    @property
                    def name(self): return "bad_dim"
                    def score(self, output, context): return "not-a-float"
                PLUGIN = C()
                '''
            ),
            encoding="utf-8",
        )
        registry = PluginRegistry(plugin_dirs=[tmp_path], include_entry_points=False)
        registry.load_all()
        check = registry.check("bad_dim")
        ctx = make_context()
        with pytest.raises(PluginError, match="expected a float"):
            registry.score_sandboxed("bad", check, "x", ctx)

    def test_to_dict(self, tmp_path):
        registry = self._registry_with(tmp_path)
        data = registry.to_dict()
        assert data["load_order"] == ["p"]
        assert data["plugins"][0]["name"] == "p"
        assert data["conflicts"] == []


class TestRegisterLoaded:
    def test_register_entry_point_style(self):
        class MyCheck:
            @property
            def name(self):
                return "ep_dim"

            def score(self, output, context):
                return 1.0

        registry = PluginRegistry(plugin_dirs=[], include_entry_points=False)
        item = LoadedPlugin(name="ep", obj=MyCheck(), source="entry_point", version="3.0.0")
        entry = registry.register_loaded(item)
        assert entry.name == "ep"
        assert isinstance(entry, PluginEntry)
        assert registry.check("ep_dim").name == "ep_dim"

    def test_register_strict_raises_on_errors(self):
        registry = PluginRegistry(plugin_dirs=[], include_entry_points=False)
        item = LoadedPlugin(name="bad", obj=None, source="file", errors=["boom"])
        with pytest.raises(PluginError, match="boom"):
            registry.register_loaded(item, strict=True)

    def test_custom_sandbox_limits(self):
        class MyCheck:
            @property
            def name(self):
                return "x"

            def score(self, output, context):
                return 0.5

        registry = PluginRegistry(
            plugin_dirs=[],
            include_entry_points=False,
            default_limits=SandboxLimits(time_limit=9.0),
        )
        entry = registry.register_loaded(LoadedPlugin(name="e", obj=MyCheck(), source="entry_point"))
        assert entry.sandbox is not None
        assert entry.sandbox.limits.time_limit == 9.0

    def test_custom_hooks_manager_shared(self):
        hooks = HookManager()
        registry = PluginRegistry(plugin_dirs=[], include_entry_points=False, hooks=hooks)
        assert registry.hooks is hooks
