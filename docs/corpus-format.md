# Corpus format

A corpus is a **JSON Lines** file: one JSON object per line, blank lines
allowed. Each object is one case:

```json
{
  "id": "ex-001",
  "prompt": "Summarize this refund policy in one sentence.",
  "baseline_output": "Refunds are available within 30 days of purchase if the item is unused.",
  "candidate_output": "You can get a refund within 30 days if the item is unused.",
  "metadata": {
    "model": "gpt-4o-mini",
    "tags": ["support", "refund"],
    "timestamp": "2026-11-01T09:15:00Z"
  }
}
```

## Required fields

| field | type | meaning |
|---|---|---|
| `id` | string, unique in the corpus | stable identifier, used to match baseline and candidate corpora |
| `prompt` | string | the input sent to the model |
| `baseline_output` | string | output of the pre-change system |
| `candidate_output` | string | output of the post-change system |

## Optional fields

| field | type | meaning |
|---|---|---|
| `metadata` | object | free-form annotations |
| `metadata.model` | string | model that produced the candidate output |
| `metadata.tags` | array of strings | used for filtering (`--tags`) |
| `metadata.timestamp` | string, ISO-8601 | when the traffic was captured, e.g. `2026-11-01T09:15:00Z` |
| `metadata.baseline_ms` / `candidate_ms` | number | latency, enables the `latency` dimension |
| `metadata.tokens_in` / `tokens_out` | number | token counts, enables the `tokens` dimension |

## Rules

- Ids must be unique within one corpus file; duplicates raise `FormatError`.
- Lines that are not valid JSON raise `FormatError` with the line number.
- The file is UTF-8.

## Converting other formats

```bash
llmdiff import logs.csv --format csv --out corpus.jsonl
llmdiff export corpus.jsonl --format json --out corpus.json   # JSON array
```
