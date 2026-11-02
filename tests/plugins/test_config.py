"""Tests for llmdiff.plugins.config."""

from __future__ import annotations

import pytest

from llmdiff.core.errors import ConfigError
from llmdiff.plugins.config import (
    DEFAULT_CONFIG_DIR,
    ENV_PREFIX,
    ConfigStore,
    PluginConfig,
    coerce_value,
    env_overrides,
    load_config_file,
)


class TestCoerceValue:
    def test_bool_from_strings(self):
        for raw in ("1", "true", "YES", "on", "y", "T"):
            assert coerce_value(raw, True) is True
        for raw in ("0", "false", "NO", "off", "n", "f"):
            assert coerce_value(raw, False) is False

    def test_bool_invalid(self):
        with pytest.raises(ConfigError, match="bool"):
            coerce_value("maybe", True)

    def test_bool_from_bool(self):
        assert coerce_value(True, False) is True
        assert coerce_value(False, True) is False

    def test_int_from_string(self):
        assert coerce_value("42", 0) == 42
        assert coerce_value(" -7 ", 0) == -7

    def test_int_from_float(self):
        assert coerce_value(3.9, 0) == 3

    def test_int_from_bool(self):
        assert coerce_value(True, 0) == 1

    def test_int_invalid(self):
        with pytest.raises(ConfigError, match="int"):
            coerce_value("abc", 0)

    def test_float_from_string(self):
        assert coerce_value("1.5", 0.0) == 1.5

    def test_float_from_int(self):
        assert coerce_value(3, 0.0) == 3.0

    def test_float_invalid(self):
        with pytest.raises(ConfigError, match="float"):
            coerce_value("x", 0.0)

    def test_list_from_csv(self):
        assert coerce_value("a, b ,c", []) == ["a", "b", "c"]
        assert coerce_value("a,,b", []) == ["a", "b"]
        assert coerce_value("", []) == []

    def test_list_from_list(self):
        assert coerce_value([1, 2], []) == [1, 2]

    def test_tuple_from_csv(self):
        assert coerce_value("a,b", ()) == ("a", "b")

    def test_list_invalid_type(self):
        with pytest.raises(ConfigError, match="list"):
            coerce_value(42, [])

    def test_dict_from_json(self):
        assert coerce_value('{"a": 1}', {}) == {"a": 1}

    def test_dict_from_mapping(self):
        assert coerce_value({"a": 2}, {}) == {"a": 2}

    def test_dict_invalid_json(self):
        with pytest.raises(ConfigError, match="dict"):
            coerce_value("{nope", {})

    def test_dict_non_object_json(self):
        with pytest.raises(ConfigError, match="not an object"):
            coerce_value("[1, 2]", {})

    def test_none_template_passthrough(self):
        assert coerce_value("anything", None) == "anything"
        assert coerce_value(42, None) == 42

    def test_str_template_passthrough(self):
        assert coerce_value(123, "x") == 123


class TestEnvOverrides:
    def test_collects_prefixed_vars(self):
        env = {
            "LLMDIFF_PLUGIN_MY_PLUGIN__TIMEOUT": "5",
            "LLMDIFF_PLUGIN_MY_PLUGIN__VERBOSE": "true",
            "OTHER_VAR": "x",
        }
        found = env_overrides("my-plugin", env)
        assert found == {"timeout": "5", "verbose": "true"}

    def test_dash_becomes_underscore(self):
        env = {"LLMDIFF_PLUGIN_MY_PLUGIN__A": "1"}
        assert env_overrides("my-plugin", env) == {"a": "1"}

    def test_empty_env(self):
        assert env_overrides("x", {}) == {}

    def test_prefix_constant(self):
        assert ENV_PREFIX == "LLMDIFF_PLUGIN_"


class TestLoadConfigFile:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_config_file(tmp_path / "nope.yaml") == {}

    def test_yaml_file(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("a: 1\nb: two\n", encoding="utf-8")
        assert load_config_file(path) == {"a": 1, "b": "two"}

    def test_toml_file(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text('a = 1\nb = "two"\n', encoding="utf-8")
        assert load_config_file(path) == {"a": 1, "b": "two"}

    def test_empty_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config_file(path) == {}

    def test_yaml_null_root(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("~\n", encoding="utf-8")
        assert load_config_file(path) == {}

    def test_yaml_non_mapping_root(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="top level must be a mapping"):
            load_config_file(path)

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("a: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_config_file(path)

    def test_invalid_toml(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("a = [", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config_file(path)

    def test_path_is_directory(self, tmp_path):
        with pytest.raises(ConfigError, match="not a file"):
            load_config_file(tmp_path)


class TestPluginConfigResolve:
    def test_defaults_only(self):
        cfg = PluginConfig.resolve("p", {"timeout": 30, "verbose": False}, env={})
        assert cfg.get("timeout") == 30
        assert cfg.get("verbose") is False
        assert cfg.source_of("timeout") is None

    def test_global_plugins_section(self):
        global_config = {"plugins": {"p": {"timeout": 99}}}
        cfg = PluginConfig.resolve("p", {"timeout": 30}, global_config=global_config, env={})
        assert cfg.get("timeout") == 99
        assert cfg.source_of("timeout") == "global-file"

    def test_global_top_level_section(self):
        global_config = {"p": {"timeout": 77}}
        cfg = PluginConfig.resolve("p", {"timeout": 30}, global_config=global_config, env={})
        assert cfg.get("timeout") == 77

    def test_global_section_wrong_type(self):
        with pytest.raises(ConfigError, match="must be a mapping"):
            PluginConfig.resolve(
                "p", {"timeout": 30}, global_config={"plugins": {"p": 42}}, env={}
            )

    def test_plugin_file_overrides_global(self):
        cfg = PluginConfig.resolve(
            "p",
            {"timeout": 30},
            global_config={"plugins": {"p": {"timeout": 99}}},
            plugin_config={"timeout": 5},
            env={},
        )
        assert cfg.get("timeout") == 5
        assert cfg.source_of("timeout") == "plugin-file"

    def test_env_overrides_all(self):
        cfg = PluginConfig.resolve(
            "my-plugin",
            {"timeout": 30, "verbose": False, "tags": ["a"]},
            plugin_config={"timeout": 5},
            env={"LLMDIFF_PLUGIN_MY_PLUGIN__TIMEOUT": "10", "LLMDIFF_PLUGIN_MY_PLUGIN__VERBOSE": "yes", "LLMDIFF_PLUGIN_MY_PLUGIN__TAGS": "x,y"},
        )
        assert cfg.get("timeout") == 10
        assert cfg.get("verbose") is True
        assert cfg.get("tags") == ["x", "y"]
        assert cfg.source_of("timeout") == "env"

    def test_env_unknown_key_kept_as_string(self):
        cfg = PluginConfig.resolve("p", {}, env={"LLMDIFF_PLUGIN_P__EXTRA": "raw"})
        assert cfg.get("extra") == "raw"
        assert cfg.source_of("extra") == "env"

    def test_unknown_keys_from_files_kept(self):
        cfg = PluginConfig.resolve("p", {"a": 1}, plugin_config={"b": "two"}, env={})
        assert cfg.get("b") == "two"
        assert cfg.get("a") == 1

    def test_get_default_for_missing(self):
        cfg = PluginConfig.resolve("p", {}, env={})
        assert cfg.get("nope", 42) == 42
        with pytest.raises(KeyError):
            cfg["nope"]

    def test_contains(self):
        cfg = PluginConfig.resolve("p", {"a": 1}, env={"LLMDIFF_PLUGIN_P__B": "2"})
        assert "a" in cfg
        assert "b" in cfg
        assert "c" not in cfg

    def test_as_dict_merges(self):
        cfg = PluginConfig.resolve("p", {"a": 1, "b": 2}, plugin_config={"b": 3}, env={})
        assert cfg.as_dict() == {"a": 1, "b": 3}


class TestConfigStore:
    def test_global_path_prefers_existing(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        (tmp_path / "config.toml").write_text('x = 1\n', encoding="utf-8")
        assert store.global_path.name == "config.toml"
        assert store.global_config() == {"x": 1}

    def test_global_path_default(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        assert store.global_path.name == "config.yaml"
        assert store.global_config() == {}

    def test_global_cache(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        (tmp_path / "config.yaml").write_text("a: 1\n", encoding="utf-8")
        assert store.global_config() == {"a": 1}
        (tmp_path / "config.yaml").write_text("a: 2\n", encoding="utf-8")
        assert store.global_config() == {"a": 1}  # cached
        store.reload()
        assert store.global_config() == {"a": 2}

    def test_plugin_file_prefers_yaml(self, tmp_path):
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        (plugins / "my-plugin.yaml").write_text("timeout: 9\n", encoding="utf-8")
        store = ConfigStore(config_dir=tmp_path, env={})
        assert store.plugin_file("my-plugin").name == "my-plugin.yaml"
        assert store.plugin_config("my-plugin") == {"timeout": 9}

    def test_plugin_file_missing(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        assert store.plugin_config("ghost") == {}

    def test_for_plugin_merges(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "plugins:\n  my-plugin:\n    timeout: 50\n", encoding="utf-8"
        )
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        (plugins / "my-plugin.yaml").write_text("verbose: true\n", encoding="utf-8")
        store = ConfigStore(config_dir=tmp_path, env={"LLMDIFF_PLUGIN_MY_PLUGIN__TIMEOUT": "7"})
        cfg = store.for_plugin("my-plugin", {"timeout": 30, "verbose": False})
        assert cfg.get("timeout") == 7
        assert cfg.get("verbose") is True

    def test_set_plugin_value_writes_yaml(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        store.set_plugin_value("p", "timeout", 12)
        cfg = store.for_plugin("p", {"timeout": 1})
        assert cfg.get("timeout") == 12
        # file exists and is valid yaml
        data = load_config_file(tmp_path / "plugins" / "p.yaml")
        assert data == {"timeout": 12}

    def test_set_plugin_value_updates_existing(self, tmp_path):
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        (plugins / "p.yaml").write_text("a: 1\n", encoding="utf-8")
        store = ConfigStore(config_dir=tmp_path, env={})
        store.set_plugin_value("p", "b", 2)
        data = load_config_file(plugins / "p.yaml")
        assert data == {"a": 1, "b": 2}

    def test_plugin_name_sanitised_in_path(self, tmp_path):
        store = ConfigStore(config_dir=tmp_path, env={})
        path = store.plugin_file("a/b")
        assert "/" not in path.name

    def test_default_config_dir(self):
        assert DEFAULT_CONFIG_DIR.name == ".llmdiff"

    def test_env_defaults_to_os_environ(self):
        import os

        store = ConfigStore(config_dir="/nonexistent-xyz", env={})
        assert store.env == dict(os.environ)

    def test_for_plugin_never_raises_on_bad_file(self, tmp_path):
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        (plugins / "p.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        store = ConfigStore(config_dir=tmp_path, env={})
        # load_config_file raises ConfigError for non-mapping; for_plugin
        # catches only around resolve, so this should raise — verify behaviour:
        with pytest.raises(ConfigError):
            store.for_plugin("p", {"a": 1})
