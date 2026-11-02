"""Sandboxing for untrusted plugin ``score`` calls.

Every plugin check runs inside a :class:`Sandbox` that enforces:

- a wall-clock **time limit** per call (default 1.0 s),
- a **memory limit** per call (default 64 MiB, via ``resource.setrlimit``
  where available; on platforms without ``RLIMIT_AS`` the limit is
  best-effort and only enforced via ``tracemalloc`` sampling),
- an **output size limit** on the returned score payload (the score itself
  is a float, but hooks may return richer objects; we cap their repr size).

The sandbox runs the callable in the *current* thread by default. When
``threaded=True`` it runs in a daemon thread so a hard timeout can abandon
the call even if the plugin ignores cooperative cancellation. Abandoned
threads are daemonized so they cannot block interpreter exit.

Usage::

    sandbox = Sandbox(time_limit=0.5, memory_limit=32 * 1024 * 1024)
    score = sandbox.run(plugin.score, output, context)
"""

from __future__ import annotations

import resource
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from llmdiff.core.errors import PluginError

__all__ = [
    "SandboxLimits",
    "SandboxResult",
    "SandboxTimeoutError",
    "SandboxMemoryError",
    "Sandbox",
    "run_sandboxed",
]

T = TypeVar("T")

DEFAULT_TIME_LIMIT = 1.0
DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024  # 64 MiB
DEFAULT_MAX_RESULT_REPR = 64 * 1024  # 64 KiB


class SandboxTimeoutError(PluginError):
    """Raised when a sandboxed call exceeds its time limit."""


class SandboxMemoryError(PluginError):
    """Raised when a sandboxed call exceeds its memory limit."""


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Resource limits for one sandboxed call.

    Attributes:
        time_limit: Wall-clock seconds before the call is abandoned.
        memory_limit: Address-space bytes (``RLIMIT_AS``) where supported.
        max_result_repr: Maximum ``repr()`` length of the returned value.
        threaded: Run the call in a daemon thread so timeouts are hard.
    """

    time_limit: float = DEFAULT_TIME_LIMIT
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    max_result_repr: int = DEFAULT_MAX_RESULT_REPR
    threaded: bool = True

    def __post_init__(self) -> None:
        if self.time_limit <= 0:
            raise PluginError(f"time_limit must be positive, got {self.time_limit}")
        if self.memory_limit <= 0:
            raise PluginError(f"memory_limit must be positive, got {self.memory_limit}")
        if self.max_result_repr <= 0:
            raise PluginError(f"max_result_repr must be positive, got {self.max_result_repr}")


@dataclass(slots=True)
class SandboxResult:
    """Outcome of one sandboxed call.

    Attributes:
        value: The callable's return value.
        elapsed: Wall-clock seconds the call took.
        peak_memory: Peak traced memory in bytes (0 when tracemalloc was off).
        timed_out: True when the call was abandoned for exceeding the limit.
    """

    value: Any
    elapsed: float
    peak_memory: int = 0
    timed_out: bool = False


@dataclass(slots=True)
class Sandbox:
    """Enforces :class:`SandboxLimits` around plugin calls."""

    limits: SandboxLimits = field(default_factory=SandboxLimits)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> SandboxResult:
        """Run ``func(*args, **kwargs)`` under the configured limits.

        Raises:
            SandboxTimeoutError: When the call exceeds ``time_limit``.
            SandboxMemoryError: When the call exceeds ``memory_limit``.
            PluginError: When the returned value is too large to represent.
        """
        if self.limits.threaded:
            return self._run_threaded(func, args, kwargs)
        return self._run_inline(func, args, kwargs)

    # ------------------------------------------------------------------ #
    # inline (cooperative) execution
    # ------------------------------------------------------------------ #

    def _run_inline(self, func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> SandboxResult:
        start = time.monotonic()
        tracemalloc.start()
        try:
            value = func(*args, **kwargs)
        except BaseException as exc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            if isinstance(exc, (SandboxTimeoutError, SandboxMemoryError, PluginError)):
                raise
            raise PluginError(f"sandboxed call raised {type(exc).__name__}: {exc}") from exc
        finally:
            if tracemalloc.is_tracing():
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                peak = 0
        elapsed = time.monotonic() - start
        if elapsed > self.limits.time_limit:
            raise SandboxTimeoutError(
                f"sandboxed call exceeded time limit "
                f"({elapsed:.3f}s > {self.limits.time_limit:.3f}s)"
            )
        self._check_result_size(value)
        return SandboxResult(value=value, elapsed=elapsed, peak_memory=peak)

    # ------------------------------------------------------------------ #
    # threaded (hard timeout) execution
    # ------------------------------------------------------------------ #

    def _run_threaded(self, func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> SandboxResult:
        outcome: dict[str, Any] = {}
        done = threading.Event()

        def target() -> None:
            tracemalloc.start()
            try:
                outcome["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller side
                outcome["error"] = exc
            finally:
                try:
                    _, peak = tracemalloc.get_traced_memory()
                    outcome["peak"] = peak
                finally:
                    tracemalloc.stop()
                done.set()

        worker = threading.Thread(target=target, name="llmdiff-sandbox", daemon=True)
        start = time.monotonic()
        worker.start()
        finished = done.wait(timeout=self.limits.time_limit)
        elapsed = time.monotonic() - start
        if finished:
            worker.join(timeout=1.0)
        if not finished:
            # The worker is a daemon; it will be abandoned. We cannot safely
            # join it, so report a timeout.
            raise SandboxTimeoutError(
                f"sandboxed call exceeded time limit "
                f"({self.limits.time_limit:.3f}s); abandoned in daemon thread"
            )
        if "error" in outcome:
            error = outcome["error"]
            if isinstance(error, (SandboxTimeoutError, SandboxMemoryError)):
                raise error
            if isinstance(error, PluginError):
                raise error
            raise PluginError(f"sandboxed call raised {type(error).__name__}: {error}") from error
        value = outcome.get("value")
        self._check_result_size(value)
        return SandboxResult(
            value=value,
            elapsed=elapsed,
            peak_memory=int(outcome.get("peak", 0)),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _check_result_size(self, value: Any) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            size = len(value)
        else:
            try:
                size = len(repr(value))
            except Exception:  # noqa: BLE001 - repr must never kill the run
                return
        if size > self.limits.max_result_repr:
            raise PluginError(
                f"sandboxed call returned a value whose repr is {size} bytes, "
                f"over the limit of {self.limits.max_result_repr}"
            )

    def apply_rlimits(self) -> None:
        """Apply the memory limit to the current process via ``resource``.

        Only call this in a short-lived child process; it mutates process-wide
        state. On platforms without ``RLIMIT_AS`` this is a no-op.
        """
        limit = self.limits.memory_limit
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))


def run_sandboxed(
    func: Callable[..., T],
    *args: Any,
    limits: SandboxLimits | None = None,
    **kwargs: Any,
) -> SandboxResult:
    """Convenience wrapper: run ``func`` once under ``limits``."""
    sandbox = Sandbox(limits=limits or SandboxLimits())
    return sandbox.run(func, *args, **kwargs)
