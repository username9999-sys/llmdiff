"""``llmdiff checks`` / ``llmdiff reporters`` / ``llmdiff version`` listers."""

from __future__ import annotations

import argparse

from llmdiff.cli.output import emit, print_table
from llmdiff.core import default_registry

__all__ = [
    "run_checks",
    "run_reporters",
    "run_version",
    "add_checks_parser",
    "add_reporters_parser",
    "add_version_parser",
    "package_version",
]


def package_version() -> str:
    """Return the installed llmdiff version (from importlib.metadata, fallback 0.1.0)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("llmdiff")
        except PackageNotFoundError:
            return "0.1.0"
    except Exception:
        return "0.1.0"


def add_checks_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``checks`` - list registered comparison dimensions."""
    p = sub.add_parser(
        "checks",
        help="list registered comparison dimensions",
        description="List every check dimension available to 'compare --dimensions'.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="include docstrings")
    p.set_defaults(func=run_checks)


def run_checks(args: argparse.Namespace) -> int:
    """Execute ``checks``."""
    registry = default_registry()
    rows = []
    for name in registry.names():
        check = registry.get(name)
        doc = (check.__doc__ or "").strip().splitlines()[0] if check.__doc__ else ""
        rows.append((name, doc if args.verbose else ""))
    if args.verbose:
        print_table(["dimension", "description"], rows)
    else:
        for name, _ in rows:
            emit(name)
    return 0


def _reporter_names() -> list[str]:
    """Built-in renderer formats plus any from the reporters package."""
    from llmdiff.cli.renderers import RENDERERS

    names = sorted(RENDERERS)
    try:
        from llmdiff.reporters import available  # type: ignore[attr-defined]

        for extra in available():
            if extra not in names:
                names.append(extra)
    except Exception:
        pass
    return names


def add_reporters_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``reporters`` - list output formats."""
    p = sub.add_parser(
        "reporters",
        help="list available output formats/reporters",
    )
    p.set_defaults(func=run_reporters)


def run_reporters(args: argparse.Namespace) -> int:
    """Execute ``reporters``."""
    for name in _reporter_names():
        emit(name)
    return 0


def add_version_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``version``."""
    p = sub.add_parser("version", help="print the llmdiff version")
    p.set_defaults(func=run_version)


def run_version(args: argparse.Namespace) -> int:
    """Execute ``version``."""
    emit(f"llmdiff {package_version()}")
    return 0
