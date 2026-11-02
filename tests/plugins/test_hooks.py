"""Tests for llmdiff.plugins.hooks."""

from __future__ import annotations

import pytest

from llmdiff.plugins.hooks import (
    HOOK_POINTS,
    Hook,
    HookError,
    HookManager,
    collect_hooks,
    hook,
    make_hook_manager,
    subscribe_module,
)


class TestHookPoints:
    def test_all_expected_points_present(self):
        expected = {
            "before_eval",
            "after_eval",
            "before_case",
            "after_case",
            "before_replay",
            "after_replay",
            "before_report",
            "after_report",
            "on_error",
        }
        assert set(HOOK_POINTS) == expected

    def test_points_are_unique(self):
        assert len(HOOK_POINTS) == len(set(HOOK_POINTS))


class TestSubscribe:
    def test_subscribe_and_emit(self):
        manager = HookManager()
        seen: list[str] = []
        manager.subscribe("before_eval", lambda **kw: seen.append("called"))
        report = manager.emit("before_eval", corpus=None)
        assert seen == ["called"]
        assert report.ok
        assert report.results() == [None]

    def test_unknown_point_rejected(self):
        manager = HookManager()
        with pytest.raises(HookError, match="unknown hook point"):
            manager.subscribe("nope", lambda: None)

    def test_non_callable_rejected(self):
        manager = HookManager()
        with pytest.raises(HookError, match="not callable"):
            manager.subscribe("before_eval", "not-a-callable")  # type: ignore[arg-type]

    def test_hook_dataclass_validates_point(self):
        with pytest.raises(HookError):
            Hook(point="bogus", callback=lambda: None)

    def test_priority_ordering(self):
        manager = HookManager()
        order: list[int] = []
        manager.subscribe("after_eval", lambda **kw: order.append(1), priority=200)
        manager.subscribe("after_eval", lambda **kw: order.append(2), priority=50)
        manager.subscribe("after_eval", lambda **kw: order.append(3), priority=100)
        manager.emit("after_eval")
        assert order == [2, 3, 1]

    def test_subscribers_listing(self):
        manager = HookManager()
        manager.subscribe("before_eval", lambda **kw: None, plugin="a")
        manager.subscribe("before_eval", lambda **kw: None, plugin="b")
        subs = manager.subscribers("before_eval")
        assert len(subs) == 2
        assert {s.plugin for s in subs} == {"a", "b"}

    def test_points_only_nonempty(self):
        manager = HookManager()
        assert manager.points() == ()
        manager.subscribe("before_eval", lambda **kw: None)
        assert manager.points() == ("before_eval",)

    def test_len_counts_all_hooks(self):
        manager = HookManager()
        assert len(manager) == 0
        manager.subscribe("before_eval", lambda **kw: None)
        manager.subscribe("after_eval", lambda **kw: None)
        assert len(manager) == 2


class TestUnsubscribeClear:
    def test_unsubscribe_removes(self):
        manager = HookManager()
        cb = lambda **kw: None  # noqa: E731
        manager.subscribe("before_eval", cb)
        assert manager.unsubscribe("before_eval", cb) is True
        assert len(manager) == 0
        assert manager.unsubscribe("before_eval", cb) is False

    def test_unsubscribe_unknown_point(self):
        manager = HookManager()
        with pytest.raises(HookError):
            manager.unsubscribe("bogus", lambda: None)

    def test_clear_single_point(self):
        manager = HookManager()
        manager.subscribe("before_eval", lambda **kw: None)
        manager.subscribe("after_eval", lambda **kw: None)
        manager.clear("before_eval")
        assert len(manager) == 1
        assert manager.points() == ("after_eval",)

    def test_clear_all(self):
        manager = HookManager()
        manager.subscribe("before_eval", lambda **kw: None)
        manager.subscribe("after_eval", lambda **kw: None)
        manager.clear()
        assert len(manager) == 0

    def test_clear_unknown_point(self):
        manager = HookManager()
        with pytest.raises(HookError):
            manager.clear("bogus")


class TestEmit:
    def test_unknown_point_emit(self):
        manager = HookManager()
        with pytest.raises(HookError, match="unknown hook point"):
            manager.emit("bogus")

    def test_kwargs_filtered_to_signature(self):
        manager = HookManager()
        received: dict[str, object] = {}

        def cb(*, corpus):  # only accepts corpus
            received["corpus"] = corpus

        manager.subscribe("before_eval", cb)
        manager.emit("before_eval", corpus="C", config="CFG", extra=1)
        assert received == {"corpus": "C"}

    def test_var_keyword_receives_everything(self):
        manager = HookManager()
        received: dict[str, object] = {}

        def cb(**kwargs):
            received.update(kwargs)

        manager.subscribe("before_eval", cb)
        manager.emit("before_eval", corpus="C", config="CFG")
        assert received == {"corpus": "C", "config": "CFG"}

    def test_hook_error_collected_not_raised(self):
        manager = HookManager()

        def bad(**kw):
            raise ValueError("boom")

        manager.subscribe("after_eval", bad)
        report = manager.emit("after_eval", result=None)
        assert not report.ok
        assert len(report.errors) == 1
        assert isinstance(report.errors[0].error, ValueError)
        assert "ValueError" in report.errors[0].tb

    def test_fail_fast_reraises(self):
        manager = HookManager(fail_fast=True)

        def bad(**kw):
            raise RuntimeError("fatal")

        manager.subscribe("after_eval", bad)
        with pytest.raises(HookError, match="fatal"):
            manager.emit("after_eval", result=None)

    def test_continue_after_error(self):
        manager = HookManager()
        calls: list[str] = []

        def bad(**kw):
            raise ValueError("x")

        manager.subscribe("after_eval", bad, priority=1)
        manager.subscribe("after_eval", lambda **kw: calls.append("good"), priority=2)
        report = manager.emit("after_eval")
        assert calls == ["good"]
        assert len(report.errors) == 1

    def test_report_to_dict(self):
        manager = HookManager()

        def bad(**kw):
            raise ValueError("nope")

        manager.subscribe("on_error", bad, plugin="p1")
        report = manager.emit("on_error", case=None, error=ValueError("e"))
        data = report.to_dict()
        assert data["point"] == "on_error"
        assert data["failed"] == 1
        assert data["ok"] is False
        assert data["errors"][0]["plugin"] == "p1"

    def test_emit_on_error_swallow(self):
        manager = HookManager()
        manager.subscribe("on_error", lambda **kw: True)
        assert manager.emit_on_error(case="c", error=ValueError("e")) is True

    def test_emit_on_error_no_swallow(self):
        manager = HookManager()
        manager.subscribe("on_error", lambda **kw: False)
        assert manager.emit_on_error(case="c", error=ValueError("e")) is False

    def test_multiple_results_in_order(self):
        manager = HookManager()
        manager.subscribe("before_report", lambda **kw: "a", priority=2)
        manager.subscribe("before_report", lambda **kw: "b", priority=1)
        report = manager.emit("before_report", result=None, format="md")
        assert report.results() == ["b", "a"]


class TestHookDecorator:
    def test_decorator_attaches_metadata(self):
        @hook("after_eval", plugin="mine", priority=5)
        def handler(*, result):
            return result

        meta = handler.__llmdiff_hook__
        assert meta["point"] == "after_eval"
        assert meta["plugin"] == "mine"
        assert meta["priority"] == 5

    def test_decorator_unknown_point(self):
        with pytest.raises(HookError):
            hook("bogus")(lambda: None)

    def test_decorator_rejects_positional_only(self):
        with pytest.raises(HookError, match="positional-only"):

            @hook("before_eval")
            def bad(a, /, b):
                return a

    def test_collect_hooks_finds_decorated(self):
        import types

        module = types.ModuleType("fake")

        @hook("before_eval", plugin="p")
        def one(*, corpus):
            return 1

        @hook("after_eval")
        def two(*, result):
            return 2

        module.one = one
        module.two = two
        module.unrelated = 42
        found = collect_hooks(module)
        points = {f[0] for f in found}
        assert points == {"before_eval", "after_eval"}

    def test_subscribe_module(self):
        import types

        module = types.ModuleType("fake2")

        @hook("before_eval", plugin="auto")
        def handler(*, corpus):
            return "ok"

        module.handler = handler
        manager = HookManager()
        hooks = subscribe_module(manager, module, plugin="fallback")
        assert len(hooks) == 1
        assert hooks[0].plugin == "auto"
        report = manager.emit("before_eval", corpus=None)
        assert report.results() == ["ok"]

    def test_make_hook_manager(self):
        manager = make_hook_manager(
            [("before_eval", lambda **kw: 1), ("after_eval", lambda **kw: 2)],
            plugin="p",
        )
        assert len(manager) == 2
        assert manager.emit("before_eval").results() == [1]
        assert manager.emit("after_eval").results() == [2]


class TestHookLifecycleScenarios:
    """Simulate a full eval lifecycle with hooks at every point."""

    def test_full_lifecycle(self):
        manager = HookManager()
        log: list[str] = []
        manager.subscribe("before_eval", lambda **kw: log.append("before_eval"))
        manager.subscribe("before_case", lambda **kw: log.append("before_case"))
        manager.subscribe("after_case", lambda **kw: log.append("after_case"))
        manager.subscribe("after_eval", lambda **kw: log.append("after_eval"))
        manager.subscribe("before_report", lambda **kw: log.append("before_report"))
        manager.subscribe("after_report", lambda **kw: log.append("after_report"))

        manager.emit("before_eval", corpus="corpus", config="cfg")
        manager.emit("before_case", case="case1")
        manager.emit("after_case", case="case1", case_diff="diff")
        manager.emit("after_eval", result="result", config="cfg")
        manager.emit("before_report", result="result", format="json")
        manager.emit("after_report", result="result", format="json", rendered="...")

        assert log == [
            "before_eval",
            "before_case",
            "after_case",
            "after_eval",
            "before_report",
            "after_report",
        ]

    def test_replay_hooks(self):
        manager = HookManager()
        seen: list[str] = []
        manager.subscribe("before_replay", lambda **kw: seen.append("b"))
        manager.subscribe("after_replay", lambda **kw: seen.append("a"))
        manager.emit("before_replay", case="c", provider="p")
        manager.emit("after_replay", case="c", provider="p", output="o")
        assert seen == ["b", "a"]
