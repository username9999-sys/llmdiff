"""Tests for ``config``, ``plugins``, ``version``, ``checks`` and ``reporters``."""

from __future__ import annotations

from llmdiff.cli import main


def test_version_prints_version(capsys):
    code = main(["version"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("llmdiff ")
    assert "0.1.0" in out


def test_checks_lists_dimensions(capsys):
    code = main(["checks"])
    out = capsys.readouterr().out
    assert code == 0
    for dim in ("exact", "similarity", "length", "content", "json_valid", "structure"):
        assert dim in out


def test_checks_verbose(capsys):
    code = main(["checks", "-v"])
    out = capsys.readouterr().out
    assert code == 0
    assert "dimension" in out  # table header


def test_reporters_lists_formats(capsys):
    code = main(["reporters"])
    out = capsys.readouterr().out
    assert code == 0
    for fmt in ("terminal", "json", "md", "csv", "junit", "sarif", "html"):
        assert fmt in out


def test_config_path(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "my.toml"
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    code = main(["config", "path"])
    out = capsys.readouterr().out
    assert code == 0
    assert str(cfg) in out


def test_config_set_and_show(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "llmdiff.toml"
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    code = main(["config", "set", "tolerance", "0.05"])
    assert code == 0
    assert cfg.exists()
    code = main(["config", "show"])
    out = capsys.readouterr().out
    assert code == 0
    assert "tolerance" in out and "0.05" in out


def test_config_set_bool_and_list(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "llmdiff.toml"
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    assert main(["config", "set", "color", "false"]) == 0
    assert main(["config", "set", "dimensions", "exact,similarity"]) == 0
    code = main(["config", "show"])
    out = capsys.readouterr().out
    assert code == 0
    assert "false" in out
    assert "exact" in out


def test_config_set_unknown_key_exit_2(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "llmdiff.toml"
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    code = main(["config", "set", "no.such.key", "1"])
    assert code == 2
    assert "unknown config key" in capsys.readouterr().err


def test_config_show_defaults_when_missing(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "missing.toml"
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    code = main(["config", "show"])
    out = capsys.readouterr().out
    assert code == 0
    assert "tolerance" in out
    assert "max_loss_rate" in out


def test_config_bad_toml_exit_2(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is [not toml", encoding="utf-8")
    monkeypatch.setenv("LLMDIFF_CONFIG", str(cfg))
    code = main(["config", "show"])
    assert code == 2
    assert "cannot read config" in capsys.readouterr().err


def test_plugins_list_empty(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMDIFF_PLUGIN_DIR", str(tmp_path / "plugins"))
    code = main(["plugins", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no plugins" in out


def test_plugins_add_list_remove(tmp_path, capsys, monkeypatch):
    pdir = tmp_path / "plugins"
    monkeypatch.setenv("LLMDIFF_PLUGIN_DIR", str(pdir))
    src = tmp_path / "myplug.py"
    src.write_text("__all__ = ['run']\n\ndef run():\n    return 42\n", encoding="utf-8")
    code = main(["plugins", "add", str(src)])
    assert code == 0
    assert (pdir / "myplug.py").exists()
    code = main(["plugins", "list"])
    out = capsys.readouterr().out
    assert "myplug" in out and "ok" in out
    code = main(["plugins", "remove", "myplug"])
    assert code == 0
    assert not (pdir / "myplug.py").exists()


def test_plugins_add_broken_plugin_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMDIFF_PLUGIN_DIR", str(tmp_path / "plugins"))
    src = tmp_path / "broken.py"
    src.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    code = main(["plugins", "add", str(src)])
    assert code == 2
    assert "import check" in capsys.readouterr().err


def test_plugins_remove_missing_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMDIFF_PLUGIN_DIR", str(tmp_path / "plugins"))
    code = main(["plugins", "remove", "ghost"])
    assert code == 2


def test_plugins_add_missing_source_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMDIFF_PLUGIN_DIR", str(tmp_path / "plugins"))
    code = main(["plugins", "add", str(tmp_path / "nope.py")])
    assert code == 2
