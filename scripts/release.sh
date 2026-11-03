#!/usr/bin/env bash
# Release helper: bump version in pyproject.toml, build, twine check.
# Usage: scripts/release.sh 0.2.0
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:?usage: release.sh <version>}"
python3 - "$VERSION" <<'PY'
import re
import sys
from pathlib import Path

version = sys.argv[1]
p = Path("pyproject.toml")
text = p.read_text()
new = re.sub(r'(?m)^version = "[^"]+"', f'version = "{version}"', text, count=1)
assert new != text, "version line not found"
p.write_text(new)
print("bumped to", version)
PY
python3 -m pip install --upgrade pip build twine >/dev/null
python3 -m build
twine check dist/*
