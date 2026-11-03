#!/usr/bin/env bash
# Lint everything (ruff) and fail on any finding.
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check src tests examples
