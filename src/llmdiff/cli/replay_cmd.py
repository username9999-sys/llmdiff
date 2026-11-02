"""``llmdiff replay`` - replay a corpus against a provider.

Usage::

    llmdiff replay corpus.jsonl --provider mock --out replayed.jsonl
    llmdiff replay corpus.jsonl --provider http --base-url http://localhost:11434/v1 \
        --model qwen3 --out replayed.jsonl

Thin wrapper around :mod:`llmdiff.replay`; degrades to a clear error message
when the replay package is not importable.
"""

from __future__ import annotations

import argparse

from llmdiff.cli.corpus_io import load_corpus, save_corpus
from llmdiff.cli.output import emit, err, style
from llmdiff.core.errors import LLMDiffError

__all__ = ["run_replay", "add_replay_parser"]

INSTRUCTIONS = """\
The replay package (llmdiff.replay) is not available in this environment.

With the full package installed:

    llmdiff replay corpus.jsonl --provider mock --out replayed.jsonl
    llmdiff replay corpus.jsonl --provider http --base-url http://localhost:11434/v1 \\
        --model qwen3 --api-key $KEY --out replayed.jsonl
"""


def add_replay_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``replay`` subcommand."""
    p = sub.add_parser(
        "replay",
        help="replay a corpus against a provider (mock or http)",
        description="Send every prompt in a corpus to a provider and write the "
        "resulting candidate outputs as a new corpus.",
    )
    p.add_argument("corpus", help="input corpus file ('-' for stdin)")
    p.add_argument("--provider", default="mock", choices=("mock", "http"), help="provider type")
    p.add_argument("--base-url", default=None, help="http provider base URL")
    p.add_argument("--model", default=None, help="model name sent to the provider")
    p.add_argument("--api-key", default=None, help="bearer API key for the http provider")
    p.add_argument("--out", "-o", required=True, help="output corpus file (.jsonl)")
    p.add_argument("--concurrency", type=int, default=4, help="parallel requests (http)")
    p.set_defaults(func=run_replay)


def run_replay(args: argparse.Namespace) -> int:
    """Execute ``replay``.  Returns 0 on success, 2 on error/unavailable."""
    try:
        corpus = load_corpus(args.corpus)
    except LLMDiffError as exc:
        err(f"error: cannot load corpus: {exc}")
        return 2
    try:
        from llmdiff.replay import HttpProvider, MockProvider, replay  # type: ignore[attr-defined]
    except Exception:
        err("error: llmdiff.replay is not importable in this environment.")
        err(INSTRUCTIONS)
        return 2
    if args.provider == "mock":
        provider = MockProvider(default="mock-response")
    else:
        if not args.base_url:
            err("error: --base-url is required for --provider http")
            return 2
        provider = HttpProvider(
            base_url=args.base_url, api_key=args.api_key, model=args.model or "default"
        )
    try:
        new_corpus, stats = replay(corpus, provider, out_path=None, concurrency=args.concurrency)
    except LLMDiffError as exc:
        err(f"error: replay failed: {exc}")
        return 2
    except Exception as exc:  # provider errors are user-facing
        err(f"error: replay failed: {type(exc).__name__}: {exc}")
        return 2
    save_corpus(new_corpus, args.out)
    emit(style(f"replayed {len(new_corpus)} cases -> {args.out}", "bold_green"))
    emit(f"stats: {stats}")
    return 0
