# Reporters

A reporter turns a `DiffResult` into output. All reporters implement:

```python
class Reporter(Protocol):
    @property
    def name(self) -> str: ...
    def render(self, result: DiffResult) -> str: ...
```

## Built-in reporters

| name | output | use for |
|---|---|---|
| `markdown` | markdown report with matrix table | PR descriptions, docs |
| `json` | `DiffResult.to_dict()` | machines, storage |
| `html` | self-contained HTML page (inline CSS/SVG, no external deps) | sharing a report |
| `junit` | JUnit XML | Jenkins / GitLab test reports |
| `sarif` | SARIF 2.1.0 | GitHub code scanning |
| `github` | PR comment body with `<details>` sections + status conclusion | GitHub Actions |
| `slack` | Slack Block Kit JSON | Slack alerts |
| `csv` | one row per (case, dimension) | spreadsheets |
| `terminal` | ANSI-colored table (respects `NO_COLOR`) | humans in a terminal |
| `badge` | shields.io badge markdown | README |
| `diff` | unified diff of baseline vs candidate per regressed case | code review |
| `trend` | compares two runs (win/loss rate deltas) | tracking over time |

## Usage

```bash
llmdiff compare corpus.jsonl --format html --output report.html
llmdiff compare corpus.jsonl --format sarif --output results.sarif
```

## Writing your own

```python
from llmdiff.reporters import Reporter, register_reporter

class MyReporter:
    @property
    def name(self) -> str:
        return "mine"

    def render(self, result) -> str:
        return f"verdict: {result.matrix.verdict}"

register_reporter(MyReporter())
```
