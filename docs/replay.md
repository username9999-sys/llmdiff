# Replay against a candidate

Once you have a corpus of prompts with baseline outputs, replay the prompts
against the candidate system (new prompt version, new model, new provider):

```bash
llmdiff replay corpus.jsonl \
  --provider http --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini --api-key "$OPENAI_API_KEY" \
  --out replayed.jsonl
```

Then compare:

```bash
llmdiff diff corpus.jsonl replayed.jsonl --format markdown
```

## Providers

| provider | how |
|---|---|
| `http` | any OpenAI-compatible endpoint (`--base-url`, `--model`, `--api-key`) |
| `openai` | official `openai` SDK if installed |
| `anthropic` | official `anthropic` SDK if installed |
| `ollama` | local Ollama (`--base-url http://localhost:11434`) |
| `mock` | deterministic, for tests and demos |

## Reliability features

- **Retry** with exponential backoff + jitter; honors `Retry-After` on 429.
- **Failover router**: try providers in order, record which one served each call.
- **Cache**: SQLite or in-memory response cache keyed by
  `sha256(provider + model + prompt + params)` — re-running a replay after a
  crash is free.
- **Checkpointing**: partial results are saved every N cases; `--resume`
  continues where it stopped.
- **Concurrency**: `--concurrency 8` with an asyncio semaphore; one failing
  case never sinks the batch (the error is recorded in the case metadata).

## Parameter sweeps

Compare several configurations in one go:

```bash
llmdiff replay corpus.jsonl --provider http --base-url ... --model gpt-4o-mini \
  --sweep '{"temperature": [0.0, 0.7, 1.0]}' --out-dir sweeps/
# produces sweeps/temperature=0.0.jsonl, sweeps/temperature=0.7.jsonl, ...
```
