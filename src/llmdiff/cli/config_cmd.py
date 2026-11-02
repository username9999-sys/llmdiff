"""``llmdiff config`` - show / set / locate the llmdiff.toml config.

Usage::

    llmdiff config show
    llmdiff config set tolerance 0.05
    llmdiff config path
"""

from __future__ import annotations

import argparse

from llmdiff.cli.config import (
    config_path,
    load_config,
    save_config,
    set_value,
)
from llmdiff.cli.output import emit, err, print_kv, style
from llmdiff.core.errors import ConfigError, LLMDiffError

__all__ = ["run_config", "add_config_parser"]


def add_config_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``config`` subcommand."""
    p = sub.add_parser(
        "config",
        help="manage the llmdiff.toml configuration",
        description="Show, set or locate the llmdiff configuration file.",
    )
    p.add_argument("--config", default=None, help="explicit config file path")
    csub = p.add_subparsers(dest="action", required=True)
    csub.add_parser("show", help="print the resolved configuration")
    csub.add_parser("path", help="print the config file path in use")
    cset = csub.add_parser("set", help="set a config value, e.g. 'config set tolerance 0.05'")
    cset.add_argument("key", help="config key (dotted for nested tables)")
    cset.add_argument("value", help="new value (string; type inferred from current value)")
    p.set_defaults(func=run_config)


def run_config(args: argparse.Namespace) -> int:
    """Execute ``config``.  Returns 0 on success, 2 on error."""
    try:
        config = load_config(args.config)
    except LLMDiffError as exc:
        err(f"error: {exc}")
        return 2
    action = args.action
    if action == "path":
        emit(str(config.path or config_path(explicit=args.config)))
        return 0
    if action == "show":
        emit(style(f"# {config.path or config_path(explicit=args.config)}", "dim"))
        print_kv(sorted(config.to_dict().items()))
        return 0
    if action == "set":
        try:
            value = set_value(config, args.key, args.value)
            target = save_config(config, args.config)
        except (ConfigError, LLMDiffError) as exc:
            err(f"error: {exc}")
            return 2
        emit(style(f"set {args.key} = {value!r}  ->  {target}", "bold_green"))
        return 0
    err(f"error: unknown config action {action!r}")
    return 2
