"""Tests for llmdiff.plugins.sandbox."""

from __future__ import annotations

import threading
import time

import pytest

from llmdiff.core.errors import PluginError
from llmdiff.plugins.sandbox import (
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TIME_LIMIT,
    Sandbox,
    SandboxLimits,
    SandboxMemoryError,
    SandboxResult,
    SandboxTimeoutError,
    run_sandboxed,
)


class TestSandboxLimits:
    def test_defaults(self):
        limits = SandboxLimits()
        assert limits.time_limit == DEFAULT_TIME_LIMIT
        assert limits.memory_limit == DEFAULT_MEMORY_LIMIT
        assert limits.threaded is True

    def test_invalid_time_limit(self):
        with pytest.raises(PluginError, match="time_limit"):
            SandboxLimits(time_limit=0)
        with pytest.raises(PluginError, match="time_limit"):
            SandboxLimits(time_limit=-1)

    def test_invalid_memory_limit(self):
        with pytest.raises(PluginError, match="memory_limit"):
            SandboxLimits(memory_limit=0)

    def test_invalid_result_repr(self):
        with pytest.raises(PluginError, match="max_result_repr"):
            SandboxLimits(max_result_repr=-5)


class TestInlineRun:
    def test_returns_value_and_elapsed(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False, time_limit=5.0))
        result = sandbox.run(lambda x: x * 2, 21)
        assert isinstance(result, SandboxResult)
        assert result.value == 42
        assert result.elapsed >= 0.0
        assert not result.timed_out

    def test_kwargs_passed(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False))
        result = sandbox.run(lambda a, b=0: a + b, 1, b=2)
        assert result.value == 3

    def test_inline_timeout_raises(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False, time_limit=0.05))
        with pytest.raises(SandboxTimeoutError, match="time limit"):
            sandbox.run(time.sleep, 0.5)

    def test_inline_plugin_error_wrapped(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False))

        def bad():
            raise ValueError("nope")

        with pytest.raises(PluginError, match="ValueError"):
            sandbox.run(bad)

    def test_inline_plugin_error_passthrough(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False))

        def bad():
            raise PluginError("already plugin")

        with pytest.raises(PluginError, match="already plugin"):
            sandbox.run(bad)

    def test_result_too_large_rejected(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False, max_result_repr=10))
        with pytest.raises(PluginError, match="repr"):
            sandbox.run(lambda: "x" * 1000)

    def test_peak_memory_reported(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=False))
        result = sandbox.run(lambda: [0] * 10000)
        assert result.peak_memory >= 0


class TestThreadedRun:
    def test_returns_value(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=5.0))
        result = sandbox.run(lambda a, b: a + b, 2, 3)
        assert result.value == 5
        assert result.elapsed < 5.0

    def test_hard_timeout_abandons_thread(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=0.1))
        start = time.monotonic()
        with pytest.raises(SandboxTimeoutError, match="abandoned"):
            sandbox.run(time.sleep, 30.0)
        assert time.monotonic() - start < 5.0

    def test_threaded_error_wrapped(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=5.0))

        def bad():
            raise KeyError("k")

        with pytest.raises(PluginError, match="KeyError"):
            sandbox.run(bad)

    def test_threaded_sandbox_error_passthrough(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=5.0))

        def bad():
            raise SandboxMemoryError("mem")

        with pytest.raises(SandboxMemoryError):
            sandbox.run(bad)

    def test_threaded_result_size_checked(self):
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=5.0, max_result_repr=8))
        with pytest.raises(PluginError, match="repr"):
            sandbox.run(lambda: "y" * 500)

    def test_threaded_peak_memory(self):
        sandbox = Sandbox(
            limits=SandboxLimits(threaded=True, time_limit=5.0, max_result_repr=1_000_000)
        )
        result = sandbox.run(lambda: bytearray(100_000))
        assert result.peak_memory > 0

    def test_worker_thread_is_daemon(self):
        # A run that finishes must not leave a non-daemon thread behind.
        # (Threads abandoned by timeout tests may still be alive, so assert
        # on daemon-ness rather than on absence of the name.)
        sandbox = Sandbox(limits=SandboxLimits(threaded=True, time_limit=5.0))
        sandbox.run(lambda: 1)
        sandbox_threads = [t for t in threading.enumerate() if t.name == "llmdiff-sandbox"]
        assert all(t.daemon for t in sandbox_threads)


class TestRunSandboxed:
    def test_convenience_wrapper(self):
        result = run_sandboxed(lambda: "ok")
        assert result.value == "ok"

    def test_custom_limits(self):
        result = run_sandboxed(lambda: 1, limits=SandboxLimits(time_limit=2.0, threaded=False))
        assert result.value == 1


class TestApplyRlimits:
    def test_apply_rlimits_no_crash(self):
        # Only safe to call with a huge limit so we don't constrain ourselves.
        sandbox = Sandbox(limits=SandboxLimits(memory_limit=1 << 40))
        sandbox.apply_rlimits()


class TestScoreLikeUsage:
    """Exercise the sandbox the way the registry does: wrap a score call."""

    def test_score_call(self, make_context):
        sandbox = Sandbox(limits=SandboxLimits(time_limit=1.0))

        def score(output: str, context) -> float:
            return 1.0 if output else 0.0

        ctx = make_context(baseline="a", candidate="b")
        result = sandbox.run(score, "hello", ctx)
        assert result.value == 1.0

    def test_slow_score_times_out(self, make_context):
        sandbox = Sandbox(limits=SandboxLimits(time_limit=0.05))

        def score(output: str, context) -> float:
            time.sleep(1.0)
            return 1.0

        ctx = make_context()
        with pytest.raises(SandboxTimeoutError):
            sandbox.run(score, "x", ctx)
