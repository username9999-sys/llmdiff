#!/usr/bin/env python3
"""Verify that every relative markdown link in docs/ and README.md resolves.

Usage: python3 scripts/check_docs_links.py
Exit 1 on any broken link.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def main() -> int:
    broken: list[str] = []
    checked = 0
    for md in MD_FILES:
        if not md.is_file():
            continue
        for m in LINK_RE.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            if not (md.parent / target).is_file():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    print(f"checked {checked} local links in {len(MD_FILES)} files")
    if broken:
        print("BROKEN:")
        for b in broken:
            print(" -", b)
        return 1
    print("all local doc links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
