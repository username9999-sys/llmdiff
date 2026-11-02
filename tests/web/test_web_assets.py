"""Structural tests for the web dashboard assets (no browser needed).

Validates: HTML parses, all local src/href resolve, no external runtime
dependencies, JS files pass node --check when node exists, CSS braces balance,
manifest is valid JSON, service worker exists.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "src" / "llmdiff" / "web"


class _LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.refs.append((tag, value))


def test_web_root_exists():
    assert (WEB / "index.html").is_file()
    assert (WEB / "sw.js").is_file()
    assert (WEB / "manifest.webmanifest").is_file()


def test_html_parses_and_local_refs_resolve():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    parser = _LinkCollector()
    parser.feed(html)
    assert parser.refs, "index.html should reference assets"
    for tag, ref in parser.refs:
        if ref.startswith(("http://", "https://", "#", "data:")):
            continue
        assert (WEB / ref).is_file(), f"missing asset referenced by <{tag}>: {ref}"


def test_no_external_runtime_dependencies():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    parser = _LinkCollector()
    parser.feed(html)
    external = [ref for _, ref in parser.refs if ref.startswith(("http://", "https://"))]
    assert external == [], f"external runtime deps in HTML: {external}"
    for js in (WEB / "static" / "js").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        for m in re.finditer(r'(?:src|href|import)\s*[=(]\s*["\'](https?://[^"\']+)', text):
            raise AssertionError(f"external dep in {js.name}: {m.group(1)}")


def test_js_syntax_with_node_if_available():
    node = shutil.which("node")
    if not node:
        return
    for js in sorted((WEB / "static" / "js").glob("*.js")) + [WEB / "sw.js"]:
        proc = subprocess.run(
            [node, "--check", str(js)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"{js.name}: {proc.stderr}"


def test_css_braces_balanced():
    css = (WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert css.count("{") == css.count("}")


def test_manifest_is_valid_json():
    data = json.loads((WEB / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert data["name"] and data["start_url"]


def test_all_referenced_js_exist():
    for name in (
        "api.js",
        "store.js",
        "router.js",
        "components.js",
        "charts.js",
        "views.js",
        "app.js",
    ):
        assert (WEB / "static" / "js" / name).is_file(), f"missing {name}"
