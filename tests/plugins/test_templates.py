"""Tests for llmdiff.plugins.templates."""

from __future__ import annotations

import importlib.util

import pytest

from llmdiff.core.errors import PluginError
from llmdiff.plugins.templates import (
    TEMPLATES,
    TemplateVars,
    render_template,
    scaffold,
    scaffold_files,
)


class TestTemplateVars:
    def test_from_name_simple(self):
        vars = TemplateVars.from_name("my-plugin")
        assert vars.name == "my-plugin"
        assert vars.class_name == "MyPlugin"
        assert vars.dimension == "my_plugin"

    def test_from_name_multi_part(self):
        vars = TemplateVars.from_name("a-b-c")
        assert vars.class_name == "ABC"
        assert vars.dimension == "a_b_c"

    def test_from_name_with_overrides(self):
        vars = TemplateVars.from_name("x", version="2.0.0", author="me", description="d")
        assert vars.version == "2.0.0"
        assert vars.author == "me"
        assert vars.description == "d"

    def test_from_name_invalid(self):
        with pytest.raises(PluginError, match="invalid plugin name"):
            TemplateVars.from_name("!!!")
        with pytest.raises(PluginError, match="invalid plugin name"):
            TemplateVars.from_name("")

    def test_defaults(self):
        vars = TemplateVars(name="x", class_name="X", dimension="x")
        assert vars.version == "0.1.0"
        assert vars.year == "2026"


class TestRenderTemplate:
    def test_substitutes(self):
        out = render_template("hello ${name}!", {"name": "world"})
        assert out == "hello world!"

    def test_missing_key_raises(self):
        with pytest.raises(PluginError, match="missing"):
            render_template("${a} ${b}", {"a": "1"})


class TestScaffoldFiles:
    def test_all_templates_produce_files(self):
        for template in TEMPLATES:
            vars = TemplateVars.from_name("demo", description="Demo plugin")
            files = scaffold_files(template, vars)
            assert "llmdiff-plugin.yaml" in files
            assert "plugin.py" in files
            assert "tests/test_plugin.py" in files
            assert "demo" in files["llmdiff-plugin.yaml"]

    def test_unknown_template(self):
        with pytest.raises(PluginError, match="unknown template"):
            scaffold_files("nope", TemplateVars.from_name("x"))

    def test_check_template_content(self):
        files = scaffold_files("check", TemplateVars.from_name("politeness"))
        entry = files["plugin.py"]
        assert "class PolitenessCheck" in entry
        assert '"politeness"' in entry
        assert "PLUGIN = PolitenessCheck()" in entry
        assert "CONFIG_DEFAULTS" in entry

    def test_reporter_template_content(self):
        files = scaffold_files("reporter", TemplateVars.from_name("myrep"))
        assert "class MyrepReporter" in files["plugin.py"]
        assert "def render" in files["plugin.py"]

    def test_hooks_template_content(self):
        files = scaffold_files("hooks", TemplateVars.from_name("hooky"))
        assert "@hook(" in files["plugin.py"]
        assert "after_eval" in files["llmdiff-plugin.yaml"]

    def test_full_template_content(self):
        files = scaffold_files("full", TemplateVars.from_name("fullp"))
        entry = files["plugin.py"]
        assert "class FullpCheck" in entry
        assert "@hook(" in entry

    def test_test_stub_references_plugin(self):
        files = scaffold_files("check", TemplateVars.from_name("demo"))
        assert "demo/plugin.py" in files["tests/test_plugin.py"]


class TestScaffold:
    def test_writes_files(self, tmp_path):
        plugin_dir = scaffold(tmp_path, "my-plugin", template="check", author="tester")
        assert (plugin_dir / "llmdiff-plugin.yaml").is_file()
        assert (plugin_dir / "plugin.py").is_file()
        assert (plugin_dir / "tests" / "test_plugin.py").is_file()
        manifest_text = (plugin_dir / "llmdiff-plugin.yaml").read_text(encoding="utf-8")
        assert "name: my-plugin" in manifest_text
        assert "author: tester" in manifest_text

    def test_scaffolded_plugin_loads_and_scores(self, tmp_path):
        plugin_dir = scaffold(tmp_path, "scored", template="check")
        spec = importlib.util.spec_from_file_location("scored_plugin", plugin_dir / "plugin.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check = module.PLUGIN
        assert check.name == "scored"
        assert check.score("hello", context=None) == 1.0  # type: ignore[arg-type]
        assert check.score("", context=None) == 0.0  # type: ignore[arg-type]

    def test_scaffolded_hooks_template_loads(self, tmp_path):
        plugin_dir = scaffold(tmp_path, "hookdemo", template="hooks")
        spec = importlib.util.spec_from_file_location("hookdemo_plugin", plugin_dir / "plugin.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "log_corpus_size")
        assert module.log_corpus_size.__llmdiff_hook__["point"] == "before_eval"

    def test_refuses_overwrite(self, tmp_path):
        scaffold(tmp_path, "p", template="check")
        with pytest.raises(PluginError, match="overwrite"):
            scaffold(tmp_path, "p", template="check")

    def test_overwrite_allowed(self, tmp_path):
        scaffold(tmp_path, "p", template="check", description="one")
        scaffold(tmp_path, "p", template="full", description="two", overwrite=True)
        entry = (tmp_path / "p" / "plugin.py").read_text(encoding="utf-8")
        assert "@hook(" in entry  # full template has hooks

    def test_invalid_name(self, tmp_path):
        with pytest.raises(PluginError):
            scaffold(tmp_path, "!!!")

    def test_creates_nested_target(self, tmp_path):
        target = tmp_path / "deep" / "nested"
        plugin_dir = scaffold(target, "x", template="reporter")
        assert plugin_dir.is_dir()

    def test_all_templates_scaffold(self, tmp_path):
        for template in TEMPLATES:
            scaffold(tmp_path, f"plug-{template}", template=template)
            assert (tmp_path / f"plug-{template}" / "plugin.py").is_file()
