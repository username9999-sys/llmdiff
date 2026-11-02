"""Shared fixtures for the plugins test suite.

Provides:

- ``make_case``: build a real :class:`~llmdiff.core.case.Case`,
- ``make_context``: build a :class:`~llmdiff.core.checks.CheckContext`,
- ``result_factory``: build a real :class:`DiffResult` via the engine,
- ``tmp_plugin_dir``: a temporary directory for file-based plugin tests,
- ``write_plugin``: helper that writes a manifest + entry file pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from llmdiff import Corpus, evaluate
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

CORPUS_RECORDS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "prompt": "hi",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m1", "tags": ["x"], "timestamp": "2026-11-01T12:00:00Z"},
    },
    {
        "id": "c2",
        "prompt": "yo",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
    {
        "id": "c3",
        "prompt": "summarize",
        "baseline_output": "The quick brown fox jumps over the lazy dog.",
        "candidate_output": "",
        "metadata": {"tags": ["empty"]},
    },
]


@pytest.fixture
def make_case() -> Any:
    def _make(
        id: str = "t1",
        prompt: str = "p",
        baseline: str = "base",
        candidate: str = "cand",
        metadata: Mapping[str, Any] | None = None,
    ) -> Case:
        return Case(
            id=id,
            prompt=prompt,
            baseline_output=baseline,
            candidate_output=candidate,
            metadata=dict(metadata or {}),
        )

    return _make


@pytest.fixture
def make_context(make_case: Any) -> Any:
    def _make(**kwargs: Any) -> CheckContext:
        return CheckContext(case=make_case(**kwargs), metadata=dict(kwargs.get("metadata") or {}))

    return _make


@pytest.fixture
def result_factory() -> Any:
    """Run the real engine over CORPUS_RECORDS and return the DiffResult."""

    def _make(records: list[dict[str, Any]] | None = None) -> Any:
        corpus = Corpus.from_records(records if records is not None else CORPUS_RECORDS)
        return evaluate(corpus)

    return _make


@pytest.fixture
def tmp_plugin_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "plugins"
    directory.mkdir()
    return directory


@pytest.fixture
def write_plugin(tmp_plugin_dir: Path) -> Any:
    """Write a manifest + plugin.py pair into the tmp plugin dir.

    Returns a function ``(name, manifest_text, entry_text, subdir=None) -> Path``.
    """

    def _write(
        name: str,
        manifest_text: str,
        entry_text: str,
        *,
        subdir: str | None = None,
        manifest_name: str = "llmdiff-plugin.yaml",
    ) -> Path:
        base = tmp_plugin_dir / subdir if subdir else tmp_plugin_dir
        base.mkdir(parents=True, exist_ok=True)
        (base / manifest_name).write_text(manifest_text, encoding="utf-8")
        (base / "plugin.py").write_text(entry_text, encoding="utf-8")
        return base

    return _write


GOOD_MANIFEST = """\
name: {name}
version: 1.0.0
author: Tester
description: A test plugin.
entry: plugin.py
hooks: [{hooks}]
dependencies: [{deps}]
permissions: [{perms}]
min_llmdiff_version: "0.1.0"
license: MIT
"""

GOOD_ENTRY = '''\
"""Test plugin entry."""

from llmdiff.core.checks import CheckContext


class {cls}Check:
    @property
    def name(self):
        return "{dim}"

    def score(self, output, context: CheckContext):
        return 1.0 if output.strip() else 0.0


PLUGIN = {cls}Check()
'''


def good_manifest(name: str = "test-plugin", *, hooks: str = "", deps: str = "", perms: str = "") -> str:
    return GOOD_MANIFEST.format(name=name, hooks=hooks, deps=deps, perms=perms)


def good_entry(cls: str = "Test", dim: str = "test_dim") -> str:
    return GOOD_ENTRY.format(cls=cls, dim=dim)
