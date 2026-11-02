# Capture production traffic

`llmdiff.capture` records real LLM traffic into a corpus. Three entry points:

## 1. Local proxy (no code changes in your app)

```bash
llmdiff record --proxy --port 8080 --target https://api.openai.com --out traffic.jsonl
# then point your app at http://127.0.0.1:8080/v1
```

## 2. SDK wrappers

```python
from llmdiff.capture import TrafficRecorder, CapturingOpenAI
import openai

recorder = TrafficRecorder("traffic.jsonl", redact=True)
client = CapturingOpenAI(openai.OpenAI(), recorder)
resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

## 3. Import existing logs

```bash
llmdiff import my-logs.csv --format csv --out corpus.jsonl
```

## Redaction

Redaction is ON by default for the proxy and SDK wrappers. It replaces emails,
phone numbers, credit cards, AWS keys, bearer tokens and JWTs with
placeholders like `[EMAIL]` before anything is written to disk. Treat corpora
as sensitive regardless — see [SECURITY.md](../SECURITY.md).

## Sampling and dedup

Recording 100% of traffic is usually wasteful. Use sampling (random, reservoir,
1-in-N rate limit) and near-duplicate removal (simhash) to keep corpora small
and representative:

```python
from llmdiff.capture import TrafficRecorder, ReservoirSampler, SimHashDedup

recorder = TrafficRecorder(
    "traffic.jsonl",
    sampler=ReservoirSampler(size=10_000),
    dedup=SimHashDedup(threshold=3),
)
```
