# Contributing to llmdiff

Thanks for wanting to help. This project grows through small, well-tested
contributions — that is how it gets stars.

## Setup

```bash
git clone https://github.com/username9999-sys/llmdiff.git
cd llmdiff
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
ruff check src tests
```

Every new feature needs tests. Tests must run offline and deterministically.

## Project layout

```
src/llmdiff/
  core/       Case, Corpus, Check protocol, VerdictMatrix, DiffEngine (stable API)
  checks/     comparison dimensions (each class: name + score(output, context))
  capture/    record production traffic into corpora (proxy, SDK wrappers, redaction)
  replay/     replay a corpus against a candidate provider (retry, cache, batch)
  reporters/  render a DiffResult (markdown, json, html, junit, sarif, github, slack, csv)
  cli/        argparse CLI, entry point llmdiff.cli:main
  server/     optional FastAPI dashboard
tests/        one test dir per subpackage, plus test_smoke.py
site/         GitHub Pages landing page (plain HTML/CSS, no build step)
```

## How to add a comparison dimension (good first issue)

1. Create `src/llmdiff/checks/my_dimension.py` with a class implementing the
   `Check` protocol:

```python
class MyCheck:
    @property
    def name(self) -> str:
        return "my_dimension"

    def score(self, output: str, context) -> float:
        # return a float in [0.0, 1.0], higher = better
        ...
```

2. Register it in `src/llmdiff/checks/__init__.py` (`builtin_checks()`).
3. Add `tests/checks/test_my_dimension.py` covering normal + edge cases.
4. Run the full test suite and ruff. Open a PR.

## Commit style

Conventional commits: `feat(checks): add code_syntax dimension`,
`fix(core): handle empty corpus in VerdictMatrix`, `docs: expand quickstart`.

## Code of conduct

Be direct, be kind, assume good intent. No drive-by negativity in issues or PRs.
