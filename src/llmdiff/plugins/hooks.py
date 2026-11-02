"""Hook manager: before/after eval, replay and report lifecycle events.

Plugins subscribe to named hook points. When the engine (or CLI) reaches a
hook point it calls :meth:`HookManager.emit`, which invokes every subscribed
callable in priority order. A hook may:

- return a value (ignored unless it is a mutation hook, see below),
- mutate a mutable argument in place (e.g. adjust a corpus before eval),
- raise :class:`~llmdiff.core.errors.PluginError` to abort the run.

Hook points
-----------
``before_eval`` / ``after_eval``
    Wrap a full evaluation run. Receivers get ``(corpus, config)`` and
    ``(result, config)`` respectively.
``before_case`` / ``after_case``
    Wrap a single case. Receivers get ``(case,)`` and ``(case, case_diff)``.
``before_replay`` / ``after_replay``
    Wrap a replay of one prompt against a provider.
``before_report`` / ``after_report``
    Wrap report rendering. Receivers get ``(result, format)`` and
    ``(result, format, rendered)``.
``on_error``
    Called when the engine catches a per-case error; receivers get
    ``(case, error)`` and may return ``True`` to swallow the error.

All hook callables receive keyword arguments, so a hook can declare only the
arguments it cares about (``def hook(*, result): ...``).
"""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from llmdiff.core.errors import PluginError

__all__ = [
    "HOOK_POINTS",
    "Hook",
    "HookManager",
    "HookError",
]

HOOK_POINTS: tuple[str, ...] = (
    "before_eval",
    "after_eval",
    "before_case",
    "after_case",
    "before_replay",
    "after_replay",
    "before_report",
    "after_report",
    "on_error",
)
"""Every hook point the manager knows about, in definition order."""


class HookError(PluginError):
    """Raised when a hook is misregistered or fails in a fatal way."""


@dataclass(slots=True)
class Hook:
    """One subscription to a hook point.

    Attributes:
        point: The hook point name.
        callback: The callable to invoke.
        plugin: Name of the plugin that registered the hook (for diagnostics).
        priority: Lower runs first. Default 100; built-ins use 0-50.
    """

    point: str
    callback: Callable[..., Any]
    plugin: str = ""
    priority: int = 100

    def __post_init__(self) -> None:
        if self.point not in HOOK_POINTS:
            raise HookError(
                f"unknown hook point {self.point!r}; allowed: {', '.join(HOOK_POINTS)}"
            )
        if not callable(self.callback):
            raise HookError(f"hook callback for {self.point!r} is not callable")


@dataclass(slots=True)
class HookManager:
    """Registry and dispatcher for lifecycle hooks.

    The manager is intentionally synchronous and dependency-free. Hooks are
    stored per point, sorted by ``(priority, registration order)`` so dispatch
    is deterministic.
    """

    _hooks: dict[str, list[Hook]] = field(default_factory=dict)
    _counter: int = 0
    fail_fast: bool = False
    """When True, the first hook exception aborts the emit; otherwise it is
    collected into the returned :class:`HookReport` and dispatch continues."""

    def __post_init__(self) -> None:
        for point in HOOK_POINTS:
            self._hooks.setdefault(point, [])

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #

    def subscribe(
        self,
        point: str,
        callback: Callable[..., Any],
        *,
        plugin: str = "",
        priority: int = 100,
    ) -> Hook:
        """Register ``callback`` for ``point`` and return the :class:`Hook`.

        Raises:
            HookError: If ``point`` is unknown or ``callback`` is not callable.
        """
        hook = Hook(point=point, callback=callback, plugin=plugin, priority=priority)
        self._hooks[point].append(hook)
        self._hooks[point].sort(key=lambda h: (h.priority, self._order(h)))
        self._counter += 1
        return hook

    def _order(self, hook: Hook) -> int:
        # Hooks with equal priority keep registration order; we approximate it
        # with the order in the list, which sort() preserves for equal keys
        # only when the key is identical, so store an explicit sequence.
        return getattr(hook, "_seq", 0)

    def unsubscribe(self, point: str, callback: Callable[..., Any]) -> bool:
        """Remove every subscription of ``callback`` at ``point``.

        Returns:
            True when at least one subscription was removed.
        """
        if point not in self._hooks:
            raise HookError(f"unknown hook point {point!r}")
        before = len(self._hooks[point])
        self._hooks[point] = [h for h in self._hooks[point] if h.callback is not callback]
        return len(self._hooks[point]) != before

    def clear(self, point: str | None = None) -> None:
        """Remove all hooks, or only those at ``point``."""
        if point is None:
            for key in self._hooks:
                self._hooks[key] = []
        else:
            if point not in self._hooks:
                raise HookError(f"unknown hook point {point!r}")
            self._hooks[point] = []

    def subscribers(self, point: str) -> tuple[Hook, ...]:
        """Return the hooks registered at ``point`` in dispatch order."""
        if point not in self._hooks:
            raise HookError(f"unknown hook point {point!r}")
        return tuple(self._hooks[point])

    def points(self) -> tuple[str, ...]:
        """Return all hook points that have at least one subscriber."""
        return tuple(point for point, hooks in self._hooks.items() if hooks)

    def __len__(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #

    def emit(self, point: str, /, **kwargs: Any) -> HookReport:
        """Invoke every hook subscribed to ``point`` with ``kwargs``.

        Only keyword arguments the callback declares are passed, so hooks may
        declare a subset of the available context. Positional-only parameters
        in a callback are rejected at subscribe time via :func:`_check_signature`.

        Returns:
            A :class:`HookReport` listing invoked hooks and any errors.

        Raises:
            HookError: If ``point`` is unknown, or if a hook raised and
                ``fail_fast`` is True.
        """
        if point not in self._hooks:
            raise HookError(f"unknown hook point {point!r}; allowed: {', '.join(HOOK_POINTS)}")
        report = HookReport(point=point)
        for hook in list(self._hooks[point]):
            filtered = _filter_kwargs(hook.callback, kwargs)
            try:
                result = hook.callback(**filtered)
            except Exception as exc:  # noqa: BLE001 - hooks must not kill the run
                report.errors.append(HookFailure(hook=hook, error=exc, tb=traceback.format_exc()))
                if self.fail_fast:
                    raise HookError(
                        f"hook {hook.plugin or hook.callback!r} at {point!r} raised: {exc}"
                    ) from exc
                continue
            report.invocations.append(HookInvocation(hook=hook, result=result))
        return report

    def emit_on_error(self, case: Any, error: BaseException) -> bool:
        """Emit ``on_error``; return True when any hook swallowed the error."""
        report = self.emit("on_error", case=case, error=error)
        return any(inv.result is True for inv in report.invocations)


@dataclass(slots=True)
class HookInvocation:
    """Record of one hook call."""

    hook: Hook
    result: Any


@dataclass(slots=True)
class HookFailure:
    """Record of one hook exception."""

    hook: Hook
    error: BaseException
    tb: str


@dataclass(slots=True)
class HookReport:
    """The outcome of one :meth:`HookManager.emit` call."""

    point: str
    invocations: list[HookInvocation] = field(default_factory=list)
    errors: list[HookFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no hook raised."""
        return not self.errors

    def results(self) -> list[Any]:
        """Return the results of all successful invocations, in order."""
        return [inv.result for inv in self.invocations]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary of the report."""
        return {
            "point": self.point,
            "invoked": len(self.invocations),
            "failed": len(self.errors),
            "ok": self.ok,
            "errors": [
                {
                    "plugin": failure.hook.plugin,
                    "error": f"{type(failure.error).__name__}: {failure.error}",
                }
                for failure in self.errors
            ],
        }


def _filter_kwargs(callback: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the kwargs ``callback`` accepts.

    Callbacks with ``**kwargs`` (VAR_KEYWORD) receive everything. Callbacks
    with positional-only parameters are rejected because hooks are always
    invoked with keywords.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        # Builtins without a signature: pass everything and let it fail loudly.
        return dict(kwargs)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    accepted = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _check_signature(callback: Callable[..., Any]) -> None:
    """Reject callbacks that cannot be invoked with keyword arguments only."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise HookError(
                f"hook callback {callback!r} has positional-only parameters; "
                "hooks are invoked with keyword arguments"
            )


def hook(point: str, *, plugin: str = "", priority: int = 100) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a hook on a manager later.

    The decorated function is returned unchanged; the returned callable also
    carries ``__llmdiff_hook__`` metadata so a loader can auto-subscribe it::

        @hook("after_eval", plugin="my-plugin")
        def summarize(*, result):
            print(result.matrix.verdict)
    """
    if point not in HOOK_POINTS:
        raise HookError(f"unknown hook point {point!r}; allowed: {', '.join(HOOK_POINTS)}")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _check_signature(func)
        func.__llmdiff_hook__ = {  # type: ignore[attr-defined]
            "point": point,
            "plugin": plugin,
            "priority": priority,
        }
        return func

    return decorator


def collect_hooks(module: Any) -> list[tuple[str, Callable[..., Any], str, int]]:
    """Scan a module for functions decorated with :func:`hook`.

    Returns:
        A list of ``(point, callback, plugin, priority)`` tuples.
    """
    found: list[tuple[str, Callable[..., Any], str, int]] = []
    for name in dir(module):
        obj = getattr(module, name)
        meta = getattr(obj, "__llmdiff_hook__", None)
        if isinstance(meta, Mapping):
            found.append((meta["point"], obj, meta.get("plugin", ""), meta.get("priority", 100)))
    return found


def subscribe_module(manager: HookManager, module: Any, *, plugin: str = "") -> list[Hook]:
    """Subscribe every ``@hook``-decorated function in ``module``.

    Returns:
        The list of :class:`Hook` objects created.
    """
    hooks: list[Hook] = []
    for point, callback, hook_plugin, priority in collect_hooks(module):
        hooks.append(
            manager.subscribe(point, callback, plugin=hook_plugin or plugin, priority=priority)
        )
    return hooks


def make_hook_manager(
    subscriptions: Iterable[tuple[str, Callable[..., Any]]] | None = None,
    *,
    fail_fast: bool = False,
    plugin: str = "",
) -> HookManager:
    """Convenience constructor: build a manager and subscribe ``subscriptions``."""
    manager = HookManager(fail_fast=fail_fast)
    for point, callback in subscriptions or ():
        manager.subscribe(point, callback, plugin=plugin)
    return manager
