# Plugin API

A plugin is any Python object implementing one of the protocols below. No
registration boilerplate beyond adding it to a registry.

## Check plugin (new comparison dimension)

```python
from llmdiff.core.checks import Check, CheckContext

class MyCheck:
    @property
    def name(self) -> str:
        return "my_dimension"

    def score(self, output: str, context: CheckContext) -> float:
        # return a float in [0.0, 1.0]; higher = better
        return 1.0 if "good" in output else 0.0
```

Register and use:

```python
from llmdiff import CheckRegistry, DiffEngine, DiffConfig, Corpus

registry = CheckRegistry([MyCheck()])
engine = DiffEngine(registry=registry, config=DiffConfig(dimensions=("my_dimension",)))
result = engine.run(Corpus.from_jsonl("corpus.jsonl"))
```

Or from the CLI, drop a `.py` file in `~/.llmdiff/plugins/` and run
`llmdiff plugins list` — the CLI loads them automatically.

## Reporter plugin

```python
class MyReporter:
    @property
    def name(self) -> str:
        return "mine"

    def render(self, result) -> str: ...
```

## Rules for plugins

- `score` must be deterministic for the same inputs (no random, no network
  unless the dimension is explicitly a judge/embedding one).
- Clamp your score to [0.0, 1.0]; the engine clamps too, but don't rely on it.
- Raise `CheckError` for inputs you cannot score; with `fail_fast=False` (the
  default) the engine skips the check for that case instead of crashing.
- Never read secrets from disk inside a plugin; take configuration via the
  constructor.
