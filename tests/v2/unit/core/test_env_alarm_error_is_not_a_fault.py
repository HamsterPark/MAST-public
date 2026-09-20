"""环境读数不可用与物理量越限必须分开处理。

status=error 只说明读取失败，应记录并通知；status=alarm 表示明确越限，
继续执行既定的停止、退针和急停流程。不能因一次未知读数触发硬件故障处置。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.execution_context import (  # noqa: E402
    ABORT_REASON_ATTR,
    ExecutionContext,
    mark_abort,
)
from mast.core.runtime import CoreRuntime  # noqa: E402


class _Pool:
    """记下每一次 safe_call —— 退针有没有发生,看这里。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def safe_call(self, method, *args, **kw):
        self.calls.append((method, args, kw))
        return SimpleNamespace(error="", return_value=None)


def _app() -> CoreRuntime:
    """一个只装了本测试需要的那几个字段的 CoreRuntime,不跑 __init__ 的重活。"""
    app = CoreRuntime.__new__(CoreRuntime)
    app._orch_abort = threading.Event()
    app._orch_abort_emergency = False
    app._orch_abort_why = ""
    app._orch_run_aborts = {}
    app._orch_run_aborts_lock = threading.Lock()
    app._executor = None
    app._storage = None
    app._pool = _Pool()
    app._env_alarm_log = []
    return app


def _reading(status: str, value: float = 0.0):
    return SimpleNamespace(status=status, value=value, unit="A")


# ── error:读不到 ────────────────────────────────────────────────────────

def test_error_does_not_retract_the_tip():
    """环境读失败只表示观测不可用，不应触发硬件动作。"""
    app = _app()
    app._on_env_alarm("tunnel_current", _reading("error"), "ok")
    assert app._pool.calls == [], (
        f"status=error 时碰了硬件:{app._pool.calls}。"
        "读失败不携带任何关于世界的信息,不该成为退针的理由。")


def test_error_does_not_latch_the_emergency():
    """更不许挂闩 —— 闩一挂,群聊私聊全线拒绝一切仪器动作。"""
    app = _app()
    app._on_env_alarm("tunnel_current", _reading("error"), "ok")
    assert not app._orch_abort_emergency, "status=error 挂上了急停闩"
    assert not app._orch_abort.is_set(), "status=error 置了全局 abort"


def test_error_is_still_recorded():
    """读失败应写入可见告警，保持只读错误的可观测性。"""
    app = _app()
    app._on_env_alarm("tunnel_current", _reading("error"), "ok")
    assert len(app._env_alarm_log) == 1
    assert app._env_alarm_log[0]["sensor"] == "tunnel_current"
    assert app._env_alarm_log[0]["status"] == "error"


# ── alarm:世界真的坏了 ─────────────────────────────────────────────────

def test_a_real_alarm_still_stops_and_retracts():
    """真越限那条路一个字都不许改。

    没有这一条,上面三条会诱使人把整个 stop-loss 拆掉 —— 而真空失效 / 热失控
    时停机退针是**对的**,那是这段代码存在的理由。
    """
    app = _app()
    app._on_env_alarm("pressure", _reading("alarm", 1e-3), "ok")
    assert app._orch_abort.is_set(), "真告警没有停掉运行"
    assert app._orch_abort_emergency, "真告警没有挂闩"
    assert any(c[0] == "ZCtrl_Withdraw" for c in app._pool.calls), (
        f"真告警没有退针:{app._pool.calls}")


def test_a_real_alarm_records_why_it_stopped():
    """并且要留下**为什么** —— 否则下游那句拒绝语只能替它编一个。"""
    app = _app()
    app._on_env_alarm("pressure", _reading("alarm", 1e-3), "ok")
    assert app._orch_abort_why, "挂了闩却没留原因"
    assert "pressure" in app._orch_abort_why
    assert getattr(app._orch_abort, ABORT_REASON_ATTR, "") == app._orch_abort_why


@pytest.mark.parametrize("status", ["ok", "warning", "unavailable"])
def test_soft_statuses_change_nothing(status):
    """warning / unavailable / ok 一如既往:不停、不退针。"""
    app = _app()
    app._on_env_alarm("tunnel_current", _reading(status), "ok")
    assert app._pool.calls == []
    assert not app._orch_abort.is_set()


# ── 闩要解得开 ──────────────────────────────────────────────────────────

def test_the_latch_can_actually_be_released():
    """2026-08-13 之前:挂得上,解不开,直到进程重启。"""
    app = _app()
    app._on_env_alarm("pressure", _reading("alarm", 1e-3), "ok")
    assert app._orch_abort_emergency

    was = app.clear_emergency_latch("test")
    assert was is True, "clear_emergency_latch 没报告它解掉了一个闩"
    assert not app._orch_abort_emergency
    assert not app._orch_abort.is_set(), "闩解了,但全局 abort 还 set 着 —— 症状一样"


def test_clearing_also_releases_the_per_run_events():
    """per-run 事件当时也被一起 set 了,只清全局的等于没清。"""
    app = _app()
    ev = threading.Event()
    app._orch_run_aborts["run-1"] = ev
    app._on_env_alarm("pressure", _reading("alarm", 1e-3), "ok")
    assert ev.is_set(), "真告警应当停掉每一个在跑的 run"

    app.clear_emergency_latch("test")
    assert not ev.is_set(), (
        "解闩之后 per-run 事件还 set 着 —— 那个 run 的上下文照样被拦,"
        "用户看到的症状和没解一模一样。")


def test_latch_state_reports_why():
    """状态要连原因一起给:只说 True 的接口逼读的人自己编原因。"""
    app = _app()
    assert app.emergency_latch_state()["latched"] is False
    app._on_env_alarm("pressure", _reading("alarm", 1e-3), "ok")
    st = app.emergency_latch_state()
    assert st["latched"] is True
    assert st["abort_set"] is True
    assert "pressure" in st["why"]


# ── 拒绝语不许造原因 ────────────────────────────────────────────────────

def test_context_reports_the_reason_that_was_recorded():
    ev = threading.Event()
    mark_abort(ev, "环境告警:pressure 越限")
    ctx = ExecutionContext(pool=None, state=None, registry=None, abort_event=[ev])
    assert ctx.check_abort() is True
    assert ctx.abort_reason() == "环境告警:pressure 越限"


def test_an_unexplained_abort_says_nothing_rather_than_guessing():
    """未记录中止原因时返回空串，不应推断为用户主动中止。"""
    ev = threading.Event()
    ev.set()  # 裸 set,没有原因
    ctx = ExecutionContext(pool=None, state=None, registry=None, abort_event=[ev])
    assert ctx.check_abort() is True
    assert ctx.abort_reason() == ""


def test_reason_comes_from_the_event_that_is_actually_set():
    """多个停止源时,原因要取**真正 set 了的**那个。"""
    quiet = threading.Event()
    setattr(quiet, ABORT_REASON_ATTR, "这个没触发")
    fired = threading.Event()
    mark_abort(fired, "这个触发了")
    ctx = ExecutionContext(pool=None, state=None, registry=None,
                           abort_event=[quiet, fired])
    assert ctx.abort_reason() == "这个触发了"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
