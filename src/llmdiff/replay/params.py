"""Parameter sweeps: produce one candidate corpus per parameter config.

Given a base parameter dict and a sweep spec mapping parameter names to lists
of values, :func:`sweep` runs the corpus once per combination (cartesian
product) and returns ``{config_name: Corpus}``. This is how model/prompt
comparisons are produced: each config becomes a candidate corpus that
:mod:`llmdiff.replay.compare_runs` can merge against a baseline.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from llmdiff.core.corpus import Corpus
from llmdiff.replay.provider import Provider
from llmdiff.replay.runner import replay

__all__ = ["sweep", "expand_sweep", "config_name"]


def expand_sweep(
    base: Mapping[str, Any], sweep_spec: Mapping[str, list[Any]]
) -> list[dict[str, Any]]:
    """Expand a sweep spec into the cartesian product of parameter dicts.

    Args:
        base: Parameters present in every config.
        sweep_spec: Mapping of parameter name to list of values to try.

    Returns:
        One dict per combination, each including ``base``.
    """
    if not sweep_spec:
        return [dict(base)]
    names = list(sweep_spec)
    value_lists = [list(sweep_spec[name]) for name in names]
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        config = dict(base)
        config.update(dict(zip(names, values, strict=True)))
        combos.append(config)
    return combos


def config_name(config: Mapping[str, Any]) -> str:
    """Stable, readable name for a parameter config, e.g. ``temp=0.0|top_p=1.0``."""
    parts = [f"{key}={config[key]!r}" for key in sorted(config)]
    return "|".join(parts) if parts else "default"


def sweep(
    corpus: Corpus,
    provider: Provider,
    sweep_spec: Mapping[str, list[Any]],
    *,
    base_params: Mapping[str, Any] | None = None,
    out_dir: str | None = None,
    **replay_opts: Any,
) -> dict[str, Corpus]:
    """Run the corpus once per sweep config.

    Args:
        corpus: Input corpus.
        provider: Candidate provider.
        sweep_spec: e.g. ``{"temperature": [0.0, 0.7], "model": ["a", "b"]}``.
        base_params: Extra params merged into every config.
        out_dir: When given, each resulting corpus is written to
            ``{out_dir}/{safe_config_name}.jsonl``.
        **replay_opts: Forwarded to :func:`llmdiff.replay.runner.replay`
            (concurrency, cache, checkpoint_path, ...). ``params`` in
            replay_opts is overridden per config.

    Returns:
        Mapping of config name to replayed corpus.
    """
    from pathlib import Path

    base = dict(base_params or {})
    results: dict[str, Corpus] = {}
    for config in expand_sweep(base, sweep_spec):
        name = config_name(config)
        opts = dict(replay_opts)
        opts["params"] = config
        opts["progress"] = opts.get("progress", True)
        out_path = None
        if out_dir is not None:
            safe = name.replace("/", "_").replace(" ", "")
            out_path = str(Path(out_dir) / f"{safe}.jsonl")
        # The provider receives the config as call params via a thin wrapper.
        wrapped = _ParamsProvider(provider, config)
        corpus_out, _stats = replay(corpus, wrapped, out_path=out_path, **opts)
        results[name] = corpus_out
    return results


class _ParamsProvider:
    """Provider wrapper injecting fixed params into every call."""

    def __init__(self, provider: Provider, params: Mapping[str, Any]) -> None:
        self.provider = provider
        self.params = dict(params)

    @property
    def name(self) -> str:
        suffix = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.provider.name}[{suffix}]" if suffix else self.provider.name

    def complete(self, prompt: str, **params: Any) -> Any:
        merged = dict(self.params)
        merged.update(params)
        return self.provider.complete(prompt, **merged)
