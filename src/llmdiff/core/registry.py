"""The registry of available comparison dimensions.

Checks are registered by name so a corpus can be evaluated with an explicit,
reproducible selection of dimensions, and so custom dimensions can be added
without touching the engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from llmdiff.core.checks import Check
from llmdiff.core.errors import CheckError

__all__ = ["CheckRegistry", "default_registry"]


class CheckRegistry:
    """A name-to-check mapping with selection helpers."""

    __slots__ = ("_checks",)

    def __init__(self, checks: Iterable[Check] = ()) -> None:
        self._checks: dict[str, Check] = {}
        for check in checks:
            self.register(check)

    def register(self, check: Check) -> None:
        """Register a check, replacing any check with the same name.

        Raises:
            CheckError: If the check has no usable name.
        """
        name = getattr(check, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise CheckError(f"check {check!r} has no usable name")
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Remove a check by name. Missing names are ignored."""
        self._checks.pop(name, None)

    def get(self, name: str) -> Check:
        """Return the check registered under ``name``.

        Raises:
            CheckError: If no check is registered under that name.
        """
        try:
            return self._checks[name]
        except KeyError:
            raise CheckError(f"unknown check dimension: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        """Return the registered dimension names in registration order."""
        return tuple(self._checks)

    def all(self) -> tuple[Check, ...]:
        """Return every registered check in registration order."""
        return tuple(self._checks.values())

    def select(self, names: Iterable[str] | None = None) -> tuple[Check, ...]:
        """Return the checks matching ``names``.

        Args:
            names: Dimension names to select. ``None`` selects everything.

        Raises:
            CheckError: If any requested name is not registered.
        """
        if names is None:
            return self.all()
        return tuple(self.get(name) for name in names)

    def __contains__(self, name: object) -> bool:
        return name in self._checks

    def __iter__(self) -> Iterator[Check]:
        return iter(self._checks.values())

    def __len__(self) -> int:
        return len(self._checks)

    def copy(self) -> CheckRegistry:
        """Return an independent copy of this registry."""
        return CheckRegistry(self._checks.values())


def default_registry() -> CheckRegistry:
    """Return a registry populated with the built-in comparison dimensions.

    The built-in dimensions are imported lazily so that a broken optional
    dependency in one dimension cannot prevent the rest of the package from
    importing.
    """
    from llmdiff.checks import builtin_checks

    return CheckRegistry(builtin_checks())
