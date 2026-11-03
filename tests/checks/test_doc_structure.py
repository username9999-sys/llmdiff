"""Tests for doc_structure dimensions."""

from __future__ import annotations

from llmdiff import Corpus, evaluate
from llmdiff.checks.doc_structure import (
    AdmonitionCountCheck,
    AnchorLinkCountCheck,
    ApiMentionCountCheck,
    ChangelogMentionCheck,
    CodeOfConductMentionCheck,
    CodeRefCountCheck,
    ContributingMentionCheck,
    DiagramCountCheck,
    ExampleBlockCountCheck,
    ImageEmbedCountCheck,
    InstallSectionPresenceCheck,
    LicenseMentionCheck,
    MermaidBlockCountCheck,
    NoteBlockCountCheck,
    ReadmeBadgeCountCheck,
    SectionCountCheck,
    SecurityPolicyMentionCheck,
    TocEntryCountCheck,
    UsageSectionPresenceCheck,
    WarningBlockCountCheck,
)
from llmdiff.core.case import Case
from llmdiff.core.checks import CheckContext

README = """# Project

![CI](https://img.shields.io/badge/ci.svg)

## Install

pip install x

## Usage

```
run()
```

## Example

See `main.py` and `GET /api/v1/users`.

> NOTE: be careful

- [Install](#install)

```mermaid
graph TD
```

License: MIT
"""


def _ctx(baseline: str, candidate: str) -> CheckContext:
    case = Case(id="t", prompt="p", baseline_output=baseline, candidate_output=candidate)
    return CheckContext(case=case, metadata=case.metadata)


def test_section_count():
    ctx = _ctx(README, README)
    assert SectionCountCheck().score(README, ctx) == 1.0
    assert SectionCountCheck().score("# Only one", ctx) < 1.0


def test_toc_count():
    ctx = _ctx(README, README)
    assert TocEntryCountCheck().score(README, ctx) == 1.0


def test_code_ref_count():
    ctx = _ctx(README, README)
    assert CodeRefCountCheck().score(README, ctx) == 1.0


def test_api_mention_count():
    ctx = _ctx(README, README)
    assert ApiMentionCountCheck().score(README, ctx) == 1.0
    assert ApiMentionCountCheck().score("no api here", ctx) < 1.0


def test_example_block_count():
    ctx = _ctx(README, README)
    assert ExampleBlockCountCheck().score(README, ctx) == 1.0


def test_warning_note_count():
    ctx = _ctx(README, README)
    assert WarningBlockCountCheck().score(README, ctx) == 1.0
    assert NoteBlockCountCheck().score(README, ctx) == 1.0


def test_admonition_count():
    ctx = _ctx("::: warning\nx\n:::", "::: warning\nx\n:::")
    assert AdmonitionCountCheck().score("::: warning\nx\n:::", ctx) == 1.0


def test_badge_count():
    ctx = _ctx(README, README)
    assert ReadmeBadgeCountCheck().score(README, ctx) == 1.0
    other = "![CI](https://img.shields.io/badge/ci.svg)"
    octx = _ctx(other, other)
    assert ReadmeBadgeCountCheck().score(other, octx) == 1.0
    assert ReadmeBadgeCountCheck().score("no badges", octx) < 1.0


def test_section_presence():
    ctx = _ctx(README, README)
    assert InstallSectionPresenceCheck().score(README, ctx) == 1.0
    assert UsageSectionPresenceCheck().score(README, ctx) == 1.0
    assert InstallSectionPresenceCheck().score("# Nothing", ctx) == 0.0


def test_mentions():
    ctx = _ctx(README, README)
    assert LicenseMentionCheck().score(README, ctx) == 1.0
    assert ChangelogMentionCheck().score(README, ctx) == 1.0
    assert ContributingMentionCheck().score(README, ctx) == 1.0
    assert CodeOfConductMentionCheck().score(README, ctx) == 1.0
    assert SecurityPolicyMentionCheck().score(README, ctx) == 1.0
    with_changelog = README + "\nSee CHANGELOG.md\n"
    cctx = _ctx(with_changelog, with_changelog)
    assert ChangelogMentionCheck().score(with_changelog, cctx) == 1.0
    assert ChangelogMentionCheck().score(README, cctx) == 0.0


def test_diagram_mermaid_count():
    ctx = _ctx(README, README)
    assert DiagramCountCheck().score(README, ctx) == 1.0
    assert MermaidBlockCountCheck().score(README, ctx) == 1.0


def test_image_anchor_count():
    ctx = _ctx(README, README)
    assert ImageEmbedCountCheck().score(README, ctx) == 1.0
    assert AnchorLinkCountCheck().score(README, ctx) == 1.0


def test_engine_integration():
    corpus = Corpus.from_records([
        {"id": "a", "prompt": "p", "baseline_output": README,
         "candidate_output": README},
        {"id": "b", "prompt": "p", "baseline_output": README,
         "candidate_output": "# Minimal doc"},
    ])
    result = evaluate(corpus, dimensions=("doc_section_count", "install_section"))
    dims = {d.dimension: d for d in result.matrix.dimensions}
    assert dims["doc_section_count"].losses == 1
    assert dims["install_section"].losses == 1
