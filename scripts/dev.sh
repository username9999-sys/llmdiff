#!/usr/bin/env bash
# Dev setup: create venv, install dev extras, run tests.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest tests/ -q
