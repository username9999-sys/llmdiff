# Architecture

```
                         ┌─────────────┐
   production traffic ──▶ │   capture   │ ──▶ corpus.jsonl
                         └─────────────┘        │
                                                ▼
                                        ┌─────────────┐
   candidate system ───────────────────▶ │    replay   │ ──▶ candidate outputs
                                        └─────────────┘        │
                                                               ▼
                                        ┌─────────────────────────────┐
   corpus.jsonl ──────────────────────▶ │  core: Case/Corpus/Engine   │ ──▶ DiffResult
                                        │  checks: dimensions         │      │
                                        └─────────────────────────────┘      ▼
                                              ▲                    ┌─────────────┐
                                              └────────────────────│  reporters  │ ──▶ md/json/html/junit/sarif/...
                                                                   └─────────────┘
                                                                        ▲
                                        ┌─────────────┐                 │
                                        │    cli      │ ────────────────┘
                                        │  (llmdiff)  │
                                        └─────────────┘
                                              │
                                        ┌─────┴─────┐
                                        │   server  │ ──▶ dashboard + REST API (FastAPI, optional)
                                        └───────────┘
```

## Layers

| layer | responsibility | depends on |
|---|---|---|
| `core` | data model (`Case`, `Corpus`), `Check` protocol, `VerdictMatrix`, `DiffEngine` | stdlib only |
| `checks` | comparison dimensions | `core` |
| `capture` | record production traffic into corpora | `core` |
| `replay` | replay a corpus against a candidate provider | `core` |
| `reporters` | render a `DiffResult` | `core` |
| `cli` | user-facing commands | all of the above (lazy imports) |
| `server` | optional HTTP dashboard | `core`, fastapi (optional) |

## Key invariants

1. **One definition per side.** A dimension scores baseline and candidate with
   the same code path; the verdict is derived from the two scores, never
   assigned directly.
2. **Deterministic by default.** No network calls unless a judge/embedding
   dimension or a replay provider is explicitly used.
3. **Zero runtime dependencies.** Everything except `server` works on the
   standard library alone.
4. **Corpora are data, not code.** JSONL in, JSON out; never `eval`.
