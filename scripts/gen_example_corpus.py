#!/usr/bin/env python3
"""Generate an example corpus JSONL of N records (deterministic).

Usage: python3 scripts/gen_example_corpus.py [n] [out]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llmdiff.benchmark.corpus_gen import gen_corpus


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 50
    out = Path(argv[2]) if len(argv) > 2 else Path("corpus.example.jsonl")
    records = gen_corpus("json_extraction", n, seed=0)
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"wrote {len(records)} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
