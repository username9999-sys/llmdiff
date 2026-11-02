"""``main`` - the llmdiff CLI entry point.

``main(argv=None) -> int`` dispatches to the subcommand registered on the
parsed namespace.  Expected failure modes never raise:

* exit 0 - success;
* exit 1 - regression gate failed;
* exit 2 - usage or configuration error (argparse also uses 2).

Unexpected exceptions are caught, printed to stderr, and mapped to exit 2 so
the console script always returns a well-defined code.
"""

from __future__ import annotations

import sys
from typing import Sequence

from llmdiff.cli.output import err
from llmdiff.cli.parser import build_parser

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llmdiff CLI.

    Args:
        argv: Argument list without the program name; ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 ok, 1 gate failed, 2 usage/config error.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # argparse exits with code 2 on usage errors and 0 for --help/--version;
        # surface those as return values so main() never raises.
        return int(exc.code) if isinstance(exc.code, int) else 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        result = func(args)
    except KeyboardInterrupt:
        err("\ninterrupted")
        return 2
    except Exception as exc:  # noqa: BLE001 - never leak a traceback from the CLI
        err(f"error: {type(exc).__name__}: {exc}")
        return 2
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
