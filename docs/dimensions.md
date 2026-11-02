# Comparison dimensions

Every dimension implements the `Check` protocol:

```python
class Check(Protocol):
    @property
    def name(self) -> str: ...
    def score(self, output: str, context: CheckContext) -> float: ...
```

`score` returns a float in `[0.0, 1.0]`, higher = better. The engine calls the
**same** check on the baseline output and the candidate output, then derives a
verdict with `compare_scores`:

- candidate > baseline + tolerance → `win`
- candidate < baseline - tolerance → `loss`
- otherwise → `tie`

## Built-in dimensions

| name | what it measures |
|---|---|
| `exact` | byte-identical to baseline |
| `similarity` | difflib sequence ratio vs baseline |
| `length` | length within [0.5x, 2x] of baseline |
| `content` | non-empty, not a refusal, not truncated |
| `json_valid` | output parses as JSON |
| `structure` | JSON key-path Jaccard vs baseline |

## Selecting dimensions

```bash
llmdiff compare corpus.jsonl --dimensions structure,json_valid
llmdiff compare corpus.jsonl --dimensions all
```

## Writing your own

See [plugins.md](plugins.md). A dimension is a plain class — no registration
boilerplate beyond adding it to a registry.
