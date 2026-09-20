"""管理员覆写热重载（KNOWN_ISSUES §1.1）。

在此之前：``register_reload_hook`` 在整个生产代码里零订阅者，``signal_reload()``
是打进一个空列表的，四个消费者一个都不换，而写入接口报 ``reloaded=True``。

这份测试盯三件事：

1. **手动执行路径真的换了** —— ``SafetyGuard.reload()``，包括「覆写被删掉之后
   要退回出厂值」这个方向（它比「换成新值」更容易写错：在已合并的 ``_limits``
   上再合并一次，旧的放宽值会永远粘住）。
2. **agent 路径被安排重建** —— 私聊图清缓存 + 群聊图后台重建。agent 侧的包络是
   在 build 时写进 pydantic ``Field(ge=, le=)`` 的，重新合并 gate 没用。
3. **报告不说谎** —— 群聊重建因为任务在跑而没做时，``restart_required`` 必须
   是 True，哪怕手动路径已经刷新了。这一条正是 §1.1 里写明「只做一半更危险」的
   那个场景。
"""

from __future__ import annotations

import json

import pytest

from mast.admin import reload_wiring
from mast.admin.override_store import SAFETY_LIMITS, ConfigOverrideRegistry
from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard


@pytest.fixture
def ovr_dir(tmp_path):
    """A private overrides dir wired into the SINGLETON (SafetyGuard reads it).

    Reset on both sides: a singleton left pointing at a tmp dir would silently
    follow other tests into a deleted directory.
    """
    d = tmp_path / "config" / "overrides"
    d.mkdir(parents=True)
    ConfigOverrideRegistry.reset()
    reload_wiring.reset_for_tests()
    ConfigOverrideRegistry.get(d)
    yield d
    ConfigOverrideRegistry.reset()
    reload_wiring.reset_for_tests()


def _write(d, payload: dict) -> None:
    (d / SAFETY_LIMITS).write_text(json.dumps(payload), encoding="utf-8")


# ── 1. 手动执行路径 ───────────────────────────────────────────────────────


class TestSafetyGuardReload:
    def test_reload_picks_up_a_new_override(self, ovr_dir):
        guard = SafetyGuard(SafetyLimits())
        assert guard._limits.bias_max_v == 10.0

        _write(ovr_dir, {"bias_max_v": 3.0})
        ConfigOverrideRegistry.get().reload()
        guard.reload()

        assert guard._limits.bias_max_v == 3.0
        # …and the pre-resolved check tuples were rebuilt, not just _limits.
        bias = [c for c in guard._resolved_checks if c[0] == "bias_v"]
        assert bias and bias[0][3] == 3.0

    def test_reload_reverts_when_the_override_is_removed(self, ovr_dir):
        """The direction that is easy to get wrong, and wrong in the unsafe way.

        Re-merging onto the ALREADY merged limits would leave a widened value
        stuck forever — "reset to code defaults" would silently do nothing.
        """
        _write(ovr_dir, {"bias_max_v": 42.0})
        ConfigOverrideRegistry.get().reload()
        guard = SafetyGuard(SafetyLimits())
        assert guard._limits.bias_max_v == 42.0

        (ovr_dir / SAFETY_LIMITS).unlink()
        ConfigOverrideRegistry.get().reload()
        guard.reload()

        assert guard._limits.bias_max_v == 10.0

    def test_resolved_checks_are_published_whole(self, ovr_dir):
        """Readers must never see a half-built check list.

        Not a race test — it pins the STRUCTURE that makes the race impossible:
        _apply_overrides binds locals and assigns at the end, so the list object
        a reader holds is never the one being appended to.
        """
        guard = SafetyGuard(SafetyLimits())
        before = guard._resolved_checks
        guard.reload()
        assert guard._resolved_checks is not before, (
            "reload must publish a NEW list, not mutate the live one"
        )


# ── 2. 接线：signal_reload 真的推动到组件上 ───────────────────────────────


class _FakeEngine:
    def __init__(self):
        self.invalidated = 0

    def invalidate(self, agent_id=None):
        self.invalidated += 1


class _FakeRuntime:
    """Just the three attributes the wiring reaches for."""

    def __init__(self, *, guard=None, engine=None, orch_suffix="（agent 工具表后台重建中，数秒后生效）"):
        self._safety = guard
        self._conv_engine = engine
        self._orch_suffix = orch_suffix
        self.rebuild_requests = 0

    def _request_composite_rebuild(self) -> str:
        self.rebuild_requests += 1
        return self._orch_suffix


class TestWiring:
    def test_signal_reload_now_has_a_subscriber(self, ovr_dir):
        """``reloaded`` in the write response is derived from this count.

        It used to be a hardcoded True over an empty hook list.
        """
        reg = ConfigOverrideRegistry.get()
        assert reg.signal_reload() == 0  # nothing wired yet

        rt = _FakeRuntime(guard=SafetyGuard(SafetyLimits()), engine=_FakeEngine())
        assert reload_wiring.wire_override_reload(rt, reg) is True
        assert reg.signal_reload() == 1

    def test_save_and_reload_refreshes_guard_and_schedules_agent_rebuild(self, ovr_dir):
        guard = SafetyGuard(SafetyLimits())
        engine = _FakeEngine()
        rt = _FakeRuntime(guard=guard, engine=engine)
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)

        fired = reg.save_and_reload(SAFETY_LIMITS, {"xy_max_m": 2.52e-6})

        assert fired == 1
        assert guard._limits.xy_max_m == 2.52e-6      # manual path: immediate
        assert engine.invalidated == 1                 # private chat: next turn
        assert rt.rebuild_requests == 1                # group chat: background
        assert reload_wiring.agent_refresh_pending() is False

    def test_wiring_is_idempotent(self, ovr_dir):
        rt = _FakeRuntime(guard=SafetyGuard(SafetyLimits()), engine=_FakeEngine())
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)
        reload_wiring.wire_override_reload(rt, reg)
        assert reg.signal_reload() == 1, "re-wiring must not stack duplicate hooks"

    def test_one_failing_path_does_not_block_the_others(self, ovr_dir):
        class _BoomEngine:
            def invalidate(self, agent_id=None):
                raise RuntimeError("engine exploded")

        guard = SafetyGuard(SafetyLimits())
        rt = _FakeRuntime(guard=guard, engine=_BoomEngine())
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)

        reg.save_and_reload(SAFETY_LIMITS, {"bias_max_v": 4.0})

        assert guard._limits.bias_max_v == 4.0     # manual path still refreshed
        assert rt.rebuild_requests == 1            # orchestrator still requested
        assert reload_wiring.agent_refresh_pending() is True  # …and reported


# ── 3. 报告不说谎 ─────────────────────────────────────────────────────────


class TestHonestReporting:
    def test_task_active_means_the_agent_path_is_still_pending(self, ovr_dir):
        """§1.1 的核心场景：手动路径刷新了，agent 路径没有 —— 必须报 True。

        ``_request_composite_rebuild`` 在任务运行中时**刻意不重建**（重建会换掉
        正在跑的图）。它返回的那句中文就是唯一的信号。
        """
        rt = _FakeRuntime(
            guard=SafetyGuard(SafetyLimits()), engine=_FakeEngine(),
            orch_suffix="（任务运行中，agent 工具表暂不重建——任务结束后重试或重启生效）",
        )
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)

        reg.save_and_reload(SAFETY_LIMITS, {"xy_max_m": 2.52e-6})

        outcome = reload_wiring.last_refresh()
        assert outcome is not None
        assert outcome.guard_reloaded is True
        assert outcome.orchestrator == reload_wiring.ORCH_SKIPPED_TASK_ACTIVE
        assert reload_wiring.agent_refresh_pending() is True

    def test_unwired_process_reports_unknown_not_false(self, ovr_dir):
        """从没跑过 ≠ 已生效。这条区别正是整条链最初出错的方式。"""
        assert reload_wiring.agent_refresh_pending() is None

    def _busy_runtime(self):
        """一个「任务正在跑」的 CoreRuntime —— 不 setup，不连硬件，不起线程。

        只装 request_agent_rebuild 的 busy 分支真正会读的三样东西。
        """
        import threading

        from mast.core.runtime import CoreRuntime

        rt = CoreRuntime.__new__(CoreRuntime)
        rt._agents_api_state = {"task": {"active": True}}
        rt._pending_agent_rebuild = None
        rt._pending_rebuild_lock = threading.Lock()
        rt._orchestrator = object()
        return rt

    def test_task_active_queues_the_rebuild_instead_of_dropping_it(self):
        """任务运行中不重建是对的；**不重建就到此为止**是 修复项 修掉的那个洞。

        以前 ``_request_composite_rebuild`` 返回一句中文就结束了，没有任何东西
        记得回来补做 —— 用户改完包络、看到「暂不重建」，然后 agent 一直用着旧
        工具表，直到有人想起来重启。现在它进队列，任务结束时 drain。
        """
        rt = self._busy_runtime()
        code, suffix = rt.request_agent_rebuild("probe")

        assert code == reload_wiring.ORCH_PENDING_QUEUED
        assert rt.pending_agent_rebuild() is not None, "排队了却没记下来 = 又丢了"
        assert rt.pending_agent_rebuild()["reason"] == "probe"
        assert "任务运行中" in suffix, (
            "这句文案是遗留 reload_wiring 分支唯一的信号 —— 改掉它，"
            "老版本的 _rebuild_orchestrator 会把「没重建」误判成「重建中」"
        )

    def test_a_queued_rebuild_is_still_pending_not_done(self):
        """排队 ≠ 已生效。

        本模块开头那条纪律（不知道就报 None，别报 False）在这里的形态是：一件
        已经安排好、但还没发生的事，报「已跟上」就是在替未来打包票。
        """
        outcome = reload_wiring.RefreshOutcome(
            guard_reloaded=True, chat_invalidated=True,
            orchestrator=reload_wiring.ORCH_PENDING_QUEUED, at=0.0,
        )
        assert outcome.agent_path_pending is True

    def test_draining_takes_the_slot_before_rebuilding(self):
        """compare-and-clear：先取走再做。

        否则一个在重建期间到达的新请求会被这次 drain 的「清空」吞掉 —— 那正是
        「排了队但没人做」的第二种长法。
        """
        rt = self._busy_runtime()
        rt.request_agent_rebuild("first")
        assert rt.pending_agent_rebuild() is not None

        rt._agents_api_state = {"task": {}}      # 任务结束
        rt._orchestrator = None                  # 没有图可建 → drain 立即返回
        assert rt.drain_pending_agent_rebuild() == "first"
        assert rt.pending_agent_rebuild() is None, "drain 之后槽必须是空的"
        assert rt.drain_pending_agent_rebuild() is None, "空队列 drain 必须是 no-op"

    def test_legacy_substring_branch_survives_for_old_runtimes(self):
        """没有 request_agent_rebuild 的 runtime（测试替身、旧嵌入方）仍要判对。

        新的结构化通道不能顺手把老路拆了 —— 那会让一批只有
        ``_request_composite_rebuild`` 的调用方静默退化成「以为重建了」。
        """
        class _OldRuntime:
            def _request_composite_rebuild(self):
                return "（任务运行中，agent 工具表暂不重建——任务结束后重试或重启生效）"

        assert (reload_wiring._rebuild_orchestrator(_OldRuntime())
                == reload_wiring.ORCH_SKIPPED_TASK_ACTIVE)

    def test_restart_required_follows_the_agent_side(self, ovr_dir, monkeypatch):
        """把 §1.1 的判据钉在路由上，而不是只钉在 wiring 里。

        构造：手动 guard 已与磁盘一致（``pending_restart`` → False），但 agent
        侧因任务运行中没重建。旧实现直接返回 ``pending_restart`` 的 False —— 那
        正是「只对了一半的『已生效』」。
        """
        from mast.api.routes import admin as admin_routes

        monkeypatch.setattr(
            "mast.api.safety_view.pending_restart", lambda ctx: False
        )
        rt = _FakeRuntime(
            guard=SafetyGuard(SafetyLimits()), engine=_FakeEngine(),
            orch_suffix="（任务运行中，agent 工具表暂不重建——任务结束后重试或重启生效）",
        )
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)
        reg.signal_reload()

        assert admin_routes._restart_required(None, "safety_limits", True) is True

    def test_restart_required_false_only_when_both_sides_are_confirmed(
        self, ovr_dir, monkeypatch
    ):
        from mast.api.routes import admin as admin_routes

        monkeypatch.setattr(
            "mast.api.safety_view.pending_restart", lambda ctx: False
        )
        rt = _FakeRuntime(guard=SafetyGuard(SafetyLimits()), engine=_FakeEngine())
        reg = ConfigOverrideRegistry.get()
        reload_wiring.wire_override_reload(rt, reg)
        reg.signal_reload()

        assert admin_routes._restart_required(None, "safety_limits", True) is False

    def test_unwired_process_keeps_the_pre_fix_answer(self, ovr_dir, monkeypatch):
        """没有接线的进程（离线 API / 测试夹具）行为不变：以手动比较为准。"""
        from mast.api.routes import admin as admin_routes

        monkeypatch.setattr(
            "mast.api.safety_view.pending_restart", lambda ctx: False
        )
        assert admin_routes._restart_required(None, "safety_limits", True) is False
        monkeypatch.setattr(
            "mast.api.safety_view.pending_restart", lambda ctx: True
        )
        assert admin_routes._restart_required(None, "safety_limits", True) is True
