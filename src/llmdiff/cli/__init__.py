"""llmdiff.cli - the user-facing command line interface of llmdiff.

The package exposes a single entry point, :func:`main`, wired to the
``llmdiff`` console script declared in ``pyproject.toml``::

    [project.scripts]
    llmdiff = "llmdiff.cli:main"

``main`` accepts an optional argument list (``None`` means ``sys.argv[1:]``),
returns a process exit code, and never raises for expected failure modes:

* ``0`` - success;
* ``1`` - a regression gate failed (see :class:`llmdiff.core.RegressionGateError`);
* ``2`` - usage or configuration error.
"""

from __future__ import annotations

from llmdiff.cli.main import main

__all__ = ["main"]
