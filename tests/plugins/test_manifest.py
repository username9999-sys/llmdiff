"""Tests for llmdiff.plugins.manifest."""

from __future__ import annotations

import pytest

from llmdiff.core.errors import PluginError
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

VALID_YAML = """\
name: my-plugin
version: 1.2.3
author: Jane Doe
description: Does a thing.
entry: plugin.py
hooks: [before_eval, after_eval]
dependencies:
  - base-checks>=1.0.0
  - other-plugin
permissions: [network]
min_llmdiff_version: "0.1.0"
homepage: https://example.com/my-plugin
license: MIT
"""

VALID_TOML = """\
name = "my-plugin"
version = "1.2.3"
author = "Jane Doe"
description = "Does a thing."
entry = "plugin.py"
hooks = ["before_eval"]
dependencies = ["base-checks>=1.0.0"]
permissions = ["network"]
min_llmdiff_version = "0.1.0"
homepage = "https://example.com/my-plugin"
license = "MIT"
"""


class TestValidateManifestHappy:
    def test_valid_yaml(self):
        manifest = parse_manifest(VALID_YAML, source="test.yaml")
        assert manifest.name == "my-plugin"
        assert manifest.version == "1.2.3"
        assert manifest.author == "Jane Doe"
        assert manifest.entry == "plugin.py"
        assert manifest.hooks == ("before_eval", "after_eval")
        assert manifest.permissions == ("network",)
        assert manifest.min_llmdiff_version == "0.1.0"
        assert manifest.homepage == "https://example.com/my-plugin"
        assert manifest.license == "MIT"
        assert len(manifest.dependencies) == 2
        assert manifest.dependencies[0].name == "base-checks"
        assert manifest.dependencies[0].operator == ">="
        assert manifest.dependencies[0].version == "1.0.0"
        assert manifest.dependencies[1].name == "other-plugin"
        assert manifest.dependencies[1].operator is None

    def test_valid_toml(self):
        manifest = parse_manifest(VALID_TOML, source="test.toml")
        assert manifest.name == "my-plugin"
        assert manifest.version == "1.2.3"
        assert manifest.hooks == ("before_eval",)
        assert manifest.dependencies[0].name == "base-checks"

    def test_minimal_manifest_defaults(self):
        manifest = validate_manifest({"name": "x", "version": "0.0.1"})
        assert manifest.entry == "plugin.py"
        assert manifest.hooks == ()
        assert manifest.dependencies == ()
        assert manifest.permissions == ()
        assert manifest.min_llmdiff_version == "0.0.0"
        assert manifest.author == ""
        assert manifest.source is None

    def test_hooks_as_single_string(self):
        manifest = validate_manifest({"name": "x", "version": "1.0.0", "hooks": "after_eval"})
        assert manifest.hooks == ("after_eval",)

    def test_null_lists_become_empty(self):
        manifest = validate_manifest(
            {"name": "x", "version": "1.0.0", "hooks": None, "dependencies": None, "permissions": None}
        )
        assert manifest.hooks == ()
        assert manifest.dependencies == ()
        assert manifest.permissions == ()

    def test_version_with_prerelease(self):
        manifest = validate_manifest({"name": "x", "version": "1.0.0-rc.1"})
        assert manifest.version == "1.0.0-rc.1"

    def test_version_with_build_metadata(self):
        manifest = validate_manifest({"name": "x", "version": "2.0.0+build.5"})
        assert manifest.version == "2.0.0+build.5"

    def test_to_dict_round_trip(self):
        manifest = parse_manifest(VALID_YAML, source="test.yaml")
        data = manifest.to_dict()
        assert data["name"] == "my-plugin"
        assert data["hooks"] == ["before_eval", "after_eval"]
        assert data["dependencies"] == ["base-checks>=1.0.0", "other-plugin"]
        assert data["source"] == "test.yaml"
        assert manifest_to_dict(manifest) == data

    def test_entry_path(self, tmp_path):
        manifest_file = tmp_path / "llmdiff-plugin.yaml"
        manifest_file.write_text(VALID_YAML, encoding="utf-8")
        manifest = load_manifest(manifest_file, check_entry=False)
        assert manifest.entry_path == tmp_path / "plugin.py"


class TestValidateManifestErrors:
    def test_not_a_mapping(self):
        with pytest.raises(PluginError, match="must be a mapping"):
            validate_manifest([1, 2, 3])

    def test_unknown_field_rejected(self):
        with pytest.raises(PluginError, match="unknown manifest field"):
            validate_manifest({"name": "x", "version": "1.0.0", "bogus": 1})

    def test_missing_name(self):
        with pytest.raises(PluginError, match="'name'"):
            validate_manifest({"version": "1.0.0"})

    def test_empty_name(self):
        with pytest.raises(PluginError, match="'name'"):
            validate_manifest({"name": "  ", "version": "1.0.0"})

    def test_bad_name_uppercase(self):
        with pytest.raises(PluginError, match="must match"):
            validate_manifest({"name": "MyPlugin", "version": "1.0.0"})

    def test_bad_name_leading_digit(self):
        with pytest.raises(PluginError, match="must match"):
            validate_manifest({"name": "1plugin", "version": "1.0.0"})

    def test_bad_name_chars(self):
        with pytest.raises(PluginError, match="must match"):
            validate_manifest({"name": "my plugin!", "version": "1.0.0"})

    def test_missing_version(self):
        with pytest.raises(PluginError, match="'version'"):
            validate_manifest({"name": "x"})

    def test_bad_version(self):
        with pytest.raises(PluginError, match="must look like"):
            validate_manifest({"name": "x", "version": "1.0"})

    def test_bad_version_word(self):
        with pytest.raises(PluginError, match="must look like"):
            validate_manifest({"name": "x", "version": "latest"})

    def test_entry_must_be_py(self):
        with pytest.raises(PluginError, match="must be a .py file"):
            validate_manifest({"name": "x", "version": "1.0.0", "entry": "plugin.js"})

    def test_entry_must_be_plain_name(self):
        with pytest.raises(PluginError, match="plain file name"):
            validate_manifest({"name": "x", "version": "1.0.0", "entry": "sub/plugin.py"})

    def test_unknown_hook(self):
        with pytest.raises(PluginError, match="unknown hook"):
            validate_manifest({"name": "x", "version": "1.0.0", "hooks": ["nonsense"]})

    def test_hooks_wrong_type(self):
        with pytest.raises(PluginError, match="must be a list"):
            validate_manifest({"name": "x", "version": "1.0.0", "hooks": 42})

    def test_hook_entry_not_string(self):
        with pytest.raises(PluginError, match="non-empty strings"):
            validate_manifest({"name": "x", "version": "1.0.0", "hooks": [""]})

    def test_dependencies_wrong_type(self):
        with pytest.raises(PluginError, match="must be a list"):
            validate_manifest({"name": "x", "version": "1.0.0", "dependencies": "base"})

    def test_duplicate_dependencies(self):
        with pytest.raises(PluginError, match="duplicate dependency"):
            validate_manifest(
                {"name": "x", "version": "1.0.0", "dependencies": ["a", "a>=1.0.0"]}
            )

    def test_self_dependency(self):
        with pytest.raises(PluginError, match="cannot depend on itself"):
            validate_manifest({"name": "x", "version": "1.0.0", "dependencies": ["x"]})

    def test_unknown_permission(self):
        with pytest.raises(PluginError, match="unknown permission"):
            validate_manifest({"name": "x", "version": "1.0.0", "permissions": ["root"]})

    def test_all_allowed_permissions_ok(self):
        manifest = validate_manifest(
            {"name": "x", "version": "1.0.0", "permissions": sorted(ALLOWED_PERMISSIONS)}
        )
        assert set(manifest.permissions) == set(ALLOWED_PERMISSIONS)

    def test_bad_min_version(self):
        with pytest.raises(PluginError, match="min_llmdiff_version"):
            validate_manifest({"name": "x", "version": "1.0.0", "min_llmdiff_version": "soon"})

    def test_bad_homepage_scheme(self):
        with pytest.raises(PluginError, match="http"):
            validate_manifest({"name": "x", "version": "1.0.0", "homepage": "ftp://x"})

    def test_author_wrong_type(self):
        with pytest.raises(PluginError, match="must be a string"):
            validate_manifest({"name": "x", "version": "1.0.0", "author": 42})


class TestParseManifest:
    def test_empty_text(self):
        with pytest.raises(PluginError, match="empty"):
            parse_manifest("   ")

    def test_non_string(self):
        with pytest.raises(PluginError, match="empty"):
            parse_manifest(None)  # type: ignore[arg-type]

    def test_invalid_yaml(self):
        with pytest.raises(PluginError, match="invalid YAML"):
            parse_manifest("name: [unclosed", source="x.yaml")

    def test_invalid_toml(self):
        with pytest.raises(PluginError, match="invalid TOML"):
            parse_manifest("name = [unclosed", source="x.toml")

    def test_yaml_scalar_root(self):
        with pytest.raises(PluginError, match="must be a mapping"):
            parse_manifest("just a string", source="x.yaml")

    def test_yaml_null_root(self):
        with pytest.raises(PluginError, match="must be a mapping"):
            parse_manifest("~", source="x.yaml")


class TestLoadManifest:
    def test_missing_file(self, tmp_path):
        with pytest.raises(PluginError, match="not found"):
            load_manifest(tmp_path / "nope.yaml")

    def test_bad_extension(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(PluginError, match="unsupported manifest extension"):
            load_manifest(bad)

    def test_check_entry_missing(self, tmp_path):
        manifest_file = tmp_path / "llmdiff-plugin.yaml"
        manifest_file.write_text("name: x\nversion: 1.0.0\n", encoding="utf-8")
        with pytest.raises(PluginError, match="entry file does not exist"):
            load_manifest(manifest_file, check_entry=True)

    def test_check_entry_present(self, tmp_path):
        manifest_file = tmp_path / "llmdiff-plugin.yaml"
        manifest_file.write_text("name: x\nversion: 1.0.0\n", encoding="utf-8")
        (tmp_path / "plugin.py").write_text("PLUGIN = 1\n", encoding="utf-8")
        manifest = load_manifest(manifest_file, check_entry=True)
        assert manifest.name == "x"

    def test_all_extensions_accepted(self, tmp_path):
        for suffix in (".yaml", ".yml", ".toml"):
            path = tmp_path / f"m{suffix}"
            if suffix == ".toml":
                path.write_text('name = "x"\nversion = "1.0.0"\n', encoding="utf-8")
            else:
                path.write_text("name: x\nversion: 1.0.0\n", encoding="utf-8")
            manifest = load_manifest(path, check_entry=False)
            assert manifest.name == "x"


class TestFindManifest:
    def test_finds_yaml(self, tmp_path):
        (tmp_path / "llmdiff-plugin.yaml").write_text("name: x\nversion: 1.0.0\n")
        found = find_manifest(tmp_path)
        assert found is not None and found.name == "llmdiff-plugin.yaml"

    def test_prefers_yaml_over_toml(self, tmp_path):
        (tmp_path / "llmdiff-plugin.toml").write_text('name = "x"\nversion = "1.0.0"\n')
        (tmp_path / "llmdiff-plugin.yaml").write_text("name: x\nversion: 1.0.0\n")
        found = find_manifest(tmp_path)
        assert found is not None and found.suffix == ".yaml"

    def test_yml_fallback(self, tmp_path):
        (tmp_path / "llmdiff-plugin.yml").write_text("name: x\nversion: 1.0.0\n")
        found = find_manifest(tmp_path)
        assert found is not None and found.suffix == ".yml"

    def test_none_when_absent(self, tmp_path):
        assert find_manifest(tmp_path) is None

    def test_none_when_not_dir(self, tmp_path):
        assert find_manifest(tmp_path / "missing") is None

    def test_basenames_constant(self):
        assert "llmdiff-plugin.yaml" in MANIFEST_BASENAMES
        assert "llmdiff-plugin.toml" in MANIFEST_BASENAMES


class TestPluginDependency:
    def test_parse_plain(self):
        dep = PluginDependency.parse("base-checks")
        assert dep.name == "base-checks"
        assert dep.operator is None
        assert str(dep) == "base-checks"

    def test_parse_with_operator(self):
        dep = PluginDependency.parse("base-checks >= 1.2.0")
        assert dep.name == "base-checks"
        assert dep.operator == ">="
        assert dep.version == "1.2.0"
        assert str(dep) == "base-checks>=1.2.0"

    def test_parse_all_operators(self):
        for op in (">=", "<=", "==", ">", "<", "~="):
            dep = PluginDependency.parse(f"a{op}1.0.0")
            assert dep.operator == op

    def test_parse_empty(self):
        with pytest.raises(PluginError, match="non-empty"):
            PluginDependency.parse("   ")

    def test_parse_malformed(self):
        with pytest.raises(PluginError, match="malformed"):
            PluginDependency.parse("!!!")

    def test_satisfied_by_no_constraint(self):
        dep = PluginDependency.parse("a")
        assert dep.satisfied_by("0.0.1")
        assert dep.satisfied_by("99.0.0")

    def test_satisfied_by_operators(self):
        assert PluginDependency.parse("a>=1.0.0").satisfied_by("1.0.0")
        assert PluginDependency.parse("a>=1.0.0").satisfied_by("2.0.0")
        assert not PluginDependency.parse("a>=1.0.0").satisfied_by("0.9.9")
        assert PluginDependency.parse("a<2.0.0").satisfied_by("1.9.9")
        assert not PluginDependency.parse("a<2.0.0").satisfied_by("2.0.0")
        assert PluginDependency.parse("a==1.2.3").satisfied_by("1.2.3")
        assert not PluginDependency.parse("a==1.2.3").satisfied_by("1.2.4")
        assert PluginDependency.parse("a>1.0.0").satisfied_by("1.0.1")
        assert not PluginDependency.parse("a>1.0.0").satisfied_by("1.0.0")
        assert PluginDependency.parse("a<=2.0.0").satisfied_by("2.0.0")
        assert PluginDependency.parse("a~=1.2.0").satisfied_by("1.2.9")
        assert not PluginDependency.parse("a~=1.2.0").satisfied_by("1.3.0")

    def test_unsupported_operator_raises(self):
        dep = PluginDependency(name="a", operator="!=", version="1.0.0")
        with pytest.raises(PluginError, match="unsupported"):
            dep.satisfied_by("1.0.0")


class TestManifestDataclass:
    def test_frozen(self):
        manifest = PluginManifest(name="x", version="1.0.0")
        with pytest.raises(AttributeError):
            manifest.name = "y"  # type: ignore[misc]

    def test_entry_path_none_without_source(self):
        manifest = PluginManifest(name="x", version="1.0.0")
        assert manifest.entry_path is None
