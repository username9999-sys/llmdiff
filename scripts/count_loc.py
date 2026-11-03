#!/usr/bin/env python3
"""Count lines of code per top-level area of the repo.

Usage: python3 scripts/count_loc.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREAS = ["src", "tests", "examples", "scripts", "site", "docs"]


def count(path: Path) -> int:
    if not path.exists():
        return 0
    out = subprocess.run(
        ["bash", "-c", f"find {path} -type f | xargs wc -l 2>/dev/null | tail -1"],
        capture_output=True,
        text=True,
    )
    try:
        return int(out.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return 0


def main() -> int:
    total = 0
    for area in AREAS:
        n = count(ROOT / area)
        total += n
        print(f"{area:<10} {n:>7}")
    print(f"{'TOTAL':<10} {total:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
