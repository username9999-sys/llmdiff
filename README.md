# llmdiff

**Git diff for LLM outputs.**

Replay real production traffic against a new prompt or model, and get a
**win / tie / loss verdict matrix** instead of a vague "looks fine".

[![PyPI](https://img.shields.io/pypi/v/llmdiff)](https://pypi.org/project/llmdiff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

```bash
pip install llmdiff
llmdiff demo        # offline demo, no API key needed
```

## The problem

You change a prompt. Tests pass. You ship it. Two days later support notices
that answers quietly got worse — dropped fields, truncated replies, refusals —
and nobody caught it because there was no test that could.

Existing eval tools (promptfoo, Langfuse, DeepEval, Braintrust) are good, but
they all start from a **hand-written golden dataset**. `llmdiff` starts from
**real production traffic**: capture what users actually sent, replay it
against the candidate, and compare outputs across independent dimensions.

## How it works

```
Capture ──▶ Corpus (JSONL) ──▶ Replay against candidate ──▶ Verdict matrix
(production traffic)          (new prompt / model)            (win/tie/loss per dimension)
```

1. **Capture** — record prompts and outputs from your live system (proxy
   middleware, SDK wrapper, or import a log).
2. **Replay** — send the same prompts to the candidate system (new prompt
   version, different model, different provider).
3. **Compare** — every dimension scores baseline and candidate with the *same*
   definition, then derives a verdict per case:

| verdict | meaning |
|---|---|
| `win` | candidate scored strictly higher |
| `tie` | equal within tolerance |
| `loss` | candidate scored strictly lower (regression) |

## Built-in dimensions

| dimension | what it catches |
|---|---|
| `exact` | byte-identical output |
| `similarity` | rewording (difflib ratio) |
| `length` | truncation / bloat |
| `content` | empty output, refusal phrases, mid-sentence cuts |
| `json_valid` | output no longer parses as JSON |
| `structure` | dropped/renamed JSON fields (key-path Jaccard) |

Plus an extensible plugin system: any class with `name` + `score(output, context)`
is a dimension.

## Quickstart

```bash
# A corpus is JSONL, one object per line:
# {"id": "c1", "prompt": "...", "baseline_output": "...",
#  "candidate_output": "...", "metadata": {"model": "...", "tags": [...],
#  "timestamp": "2026-11-01T12:00:00Z"}}

llmdiff init                      # scaffold an example corpus
llmdiff compare corpus.jsonl      # verdict matrix to stdout (markdown)
llmdiff compare corpus.jsonl --format json
llmdiff compare corpus.jsonl --format html --output report.html
llmdiff compare corpus.jsonl --dimensions structure,json_valid --tolerance 0.05

# CI regression gate (exit 1 when regressions exceed the budget):
llmdiff compare corpus.jsonl --max-loss-rate 0.05
```

Example matrix output:

```
dimension     win  tie  loss   baseline  candidate  delta
-----------   ---  ---  ----   --------  ---------  -----
exact          0   38    2      1.000      0.000   -1.000
similarity     0   40    0      1.000      0.812   -0.188
structure      0   35    5      1.000      0.833   -0.167   <-- regressed
```

## Design principles

- **Zero runtime dependencies** — pure stdlib; optional extras only for
  embeddings/judge dimensions.
- **Deterministic** — same corpus in, same matrix out. No LLM calls unless you
  explicitly enable a judge dimension.
- **CI-native** — JSON, JUnit, SARIF, GitHub-comment, and Slack reporters.
- **Extensible** — plugins are plain Python classes.

## Roadmap

- [x] Core verdict matrix engine
- [x] 6 built-in dimensions + plugin protocol
- [x] Capture (proxy middleware, SDK wrappers, redaction, sampling, dedup)
- [x] Replay (multi-provider, retry, cache, batch, sweeps)
- [x] Reporters (markdown, JSON, HTML, JUnit, SARIF, GitHub, Slack, CSV, terminal)
- [x] CLI (`compare`, `diff`, `gate`, `demo`, `record`, `replay`, `serve`)
- [ ] Hosted dashboard (opt-in)
- [ ] Embedding-based semantic dimension (optional extra)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are labeled
[`good first issue`](https://github.com/username9999-sys/llmdiff/labels/good%20first%20issue).

## License

[MIT](LICENSE) — Copyright (c) 2026 username9999-sys
