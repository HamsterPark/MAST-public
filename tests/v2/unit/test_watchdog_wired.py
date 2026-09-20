"""SafetyWatchdog 的触发行为及运行时接线回归。

测试覆盖阈值触发、解析失败、角色选择和未越限状态，并检查 CoreRuntime 启动路径。
生产阈值应从配置派生，显式测试参数不能替代对默认接线的核验。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.core.watchdog import SafetyWatchdog

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of


class _Rec:
    def __init__(self, current):
        self.error = ""
        self.return_value = (0, 0, [current])   # Current_Get shape: parsed[2][0]


class _Pool:
    def __init__(self, current):
        self._c = current
        self.calls = []

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, role))
        return _Rec(self._c)


def test_fires_on_sustained_high_current():
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9, interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(2.0) is True   # all-window > threshold → anomaly
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_no_fire_on_normal_current():
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9, interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(0.25) is False  # below threshold → never fires
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_polls_monitor_role():
    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        current_threshold_a=100e-9, interval_s=0.005, window_size=2)
    pool = wd._pool
    wd.daemon = True
    wd.start()
    import time
    time.sleep(0.05)
    wd.stop()
    wd.join(timeout=1)
    assert any(c == ("Current_Get", "monitor") for c in pool.calls)


class _BadRec:
    """A Current_Get reply that cannot be parsed into a current value."""

    def __init__(self, return_value):
        self.error = ""
        self.return_value = return_value


class _AltPool:
    """Pool that returns a scripted sequence of records (cycling)."""

    def __init__(self, records):
        self._records = list(records)
        self._i = 0
        self.calls = []
        self._lock = threading.Lock()

    def safe_call(self, method, *args, role="main"):
        with self._lock:
            self.calls.append((method, role))
            rec = self._records[min(self._i, len(self._records) - 1)]
            self._i += 1
        return rec


def test_parse_failure_does_not_reset_window():
    """A malformed Current_Get reply must NOT inject 0.0 into the window.

    Bug: parse failure injected 0.0, which is always below threshold, so it
    diluted/reset the all-readings-high check and delayed/masked a real
    overcurrent (tip crash). Here, an unparseable tick is interleaved with
    sustained high readings; the anomaly must still fire promptly.
    """
    high = 200e-9
    fired = threading.Event()
    # Interleave: high, garbage, high, garbage, high, ... — if garbage injected
    # 0.0 the all(>threshold) over a window of 3 could never be satisfied.
    records = [
        _Rec(high),
        _BadRec(None),
        _Rec(high),
        _BadRec("not a tuple"),
        _Rec(high),
        _BadRec((0, 0, [])),
        _Rec(high),
    ]
    wd = SafetyWatchdog(_AltPool(records), on_anomaly=fired.set,
                        current_threshold_a=100e-9, interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(2.0) is True
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_extract_current_returns_none_on_bad_shapes():
    extract = SafetyWatchdog._extract_current
    assert extract(None) is None
    assert extract("nope") is None
    assert extract((0, 0)) is None          # len <= 2
    assert extract((0, 0, [])) is None       # empty payload
    assert extract((0, 0, "x")) is None      # payload[0] not floatable
    assert extract((0, 0, [float("nan")])) is None
    assert extract((0, 0, [float("inf")])) is None
    assert extract((0, 0, [200e-9])) == 200e-9


def test_app_has_watchdog_wiring():
    """执行器暴露 start/stop,而 **CoreRuntime** 有 `_ensure_watchdog`。

    原来这里断言的是 ``mast.gui.app.MASTApp`` —— 那个模块随 TS 重写删掉了,
    而这一行是整个文件被删的唯一原因。看门狗的接线本身搬到了 ``CoreRuntime``,
    判据不变:**有人负责启动它**。
    """
    from mast.core.executor import SkillExecutor
    from mast.core.runtime import CoreRuntime
    assert hasattr(SkillExecutor, "start_watchdog")
    assert hasattr(SkillExecutor, "stop_watchdog")
    assert hasattr(CoreRuntime, "_ensure_watchdog")


# 看门狗阈值从 cm_sat_current_a 派生，避免多个互不一致的满量程来源。

def test_the_production_path_does_not_pin_a_number_of_its_own():
    """start_watchdog 的默认阈值应为 None，并从 cm_sat_current_a 获取当前配置，避免另设默认量程。"""
    import inspect

    from mast.core.executor import SkillExecutor

    sig = inspect.signature(SkillExecutor.start_watchdog)
    assert sig.parameters["current_threshold_a"].default is None, (
        "看门狗又自带了一个阈值 —— 会偏离当前配置的量程")


def test_threshold_follows_cm_sat_current_a_live(monkeypatch):
    """改 ``cm_sat_current_a``,看门狗下一次解析就跟着变(live-read)。

    钉的是**派生关系本身**,不是某个具体数字 —— 数字是用户的,关系是我们的。
    """
    from mast.monitoring.thresholds import (
        get_monitor_thresholds, set_monitor_thresholds,
    )

    original = get_monitor_thresholds()
    try:
        def getter():
            return float(get_monitor_thresholds().cm_sat_current_a)

        wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                            threshold_getter=getter,
                            interval_s=0.005, window_size=2)
        set_monitor_thresholds({"cm_sat_current_a": 10e-9})
        assert wd.effective_threshold_a == 10e-9
        # 运行时更新前放量程配置 —— 不重启也要跟上。
        set_monitor_thresholds({"cm_sat_current_a": 45e-9})
        assert wd.effective_threshold_a == 45e-9
    finally:
        set_monitor_thresholds(original)


def test_a_synthetic_rail_fires_when_the_threshold_is_derived():
    """合成平台电流超过动态阈值时应触发；对照更高的固定阈值验证阈值来源确实影响行为。"""
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(21e-9), on_anomaly=fired.set,
                        threshold_getter=lambda: 20e-9,
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(2.0) is True, "20 nA 贴轨没有触发退针"
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_a_synthetic_rail_does_NOT_fire_at_the_old_hardcoded_100na():
    """对照:同一股电流,在旧的 100 nA 下确实不会响 —— 证明上一条测的是派生,
    不是「反正都会响」。"""
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(21e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9,
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(0.3) is False
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_unreadable_threshold_is_loud_and_still_armed(caplog):
    """读不到真源时:**不静默回退**,而且**仍然武装**。

    `lookup(name) || DEFAULT` 那个形状 —— name 写错就得到一个能跑的错版本 ——
    正是这次事故的本体。降级必须在 WARNING 以上说话。
    """
    import logging

    def boom():
        raise RuntimeError("thresholds gone")

    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        threshold_getter=boom, interval_s=0.005, window_size=2)
    with caplog.at_level(logging.WARNING, logger="mast.core.watchdog"):
        value = wd.effective_threshold_a

    assert value is not None and value > 0, "读不到真源就不武装了 —— 比阈值偏保守更糟"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), \
        "降级了但一个字都没说 —— 这就是静默回退"
    # 兜底值必须仍然来自同一个字段的出厂默认,不是一个新写的字面量。
    from mast.monitoring.thresholds import MonitorThresholds
    assert value == MonitorThresholds().cm_sat_current_a


def test_a_transient_getter_failure_reuses_the_last_good_value(caplog):
    """临时读不到时用**上次的好值**,而不是掉回出厂默认。

    刚刚还正确的值比一个没在这台机器上标定过的出厂默认更接近真实前放量程。
    """
    import logging

    state = {"ok": True}

    def flaky():
        if not state["ok"]:
            raise RuntimeError("transient")
        return 12e-9

    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        threshold_getter=flaky, interval_s=0.005, window_size=2)
    assert wd.effective_threshold_a == 12e-9
    state["ok"] = False
    with caplog.at_level(logging.WARNING, logger="mast.core.watchdog"):
        assert wd.effective_threshold_a == 12e-9, "没有沿用上次的好值"
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════
# 抑制:蓄意扎针期间不判 —— 用**活谓词**,不是配对的 disable/enable
# ══════════════════════════════════════════════════════════════════════

def test_deliberate_tip_work_suppresses_the_net():
    """扎针期间电流本来就该贴轨(poke_dwell_s 上界 10 s,而本网 4 s 就开火)。"""
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9,
                        suppress_getter=lambda: "PokeConditionTip",
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(0.3) is False, "蓄意扎针被当成撞针退针了"
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_the_net_comes_back_by_itself_when_the_token_is_released():
    """令牌一还就恢复 —— **不需要任何人调 enable()**。

    这是选活谓词而不是配对 disable 的全部理由:配对调用漏一次,安全网就静默永久关闭。
    """
    holder = {"skill": "PokeConditionTip"}
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9,
                        suppress_getter=lambda: holder["skill"],
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(0.2) is False
        holder["skill"] = ""          # 修针结束,令牌还回去 —— 没有人调 enable()
        assert fired.wait(2.0) is True, "令牌还了,安全网没有自己恢复"
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_an_unreadable_suppressor_does_not_suppress():
    """读不到修针意图时**不抑制** —— 失败方向必须是「保护照常生效」。"""
    def boom():
        raise RuntimeError("no lock")

    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9, suppress_getter=boom,
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert fired.wait(2.0) is True, "读不到意图就把保护关了"
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_the_suppressor_is_tip_intent_not_a_second_predicate():
    """接的必须是 ``tip_intent`` 那一份判据,不是新造的第二份。

    这个仓栽过「差点建成第二真源」的跟头:同一个语义两处判定,迟早分叉,
    而分叉那天没人知道。这里直接对着**产出方**核 —— 不抄它的字面量。
    """
    import inspect

    from mast.core.executor import SkillExecutor

    src = source_of(SkillExecutor.start_watchdog)
    assert "active_tip_work" in src, "看门狗没有接 tip_intent 的判据"
    assert "suppress_getter" in src, "判据没有被传进看门狗"
    # 「现在动仪器的是我们还是人」同样必须接上,而且必须接**令牌**——
    # 不是「最近发过写命令」(一次长扫描是持续操作、零写命令)。
    assert "idle_getter" in src, "没接「MAST 在不在驱动」的判据 ⇒ 抑制从未生效"
    assert "instrument_lock" in src, (
        "「MAST 在不在驱动」接的不是仪器令牌 —— 写命令代理无法覆盖纯读取的长操作")


# ══════════════════════════════════════════════════════════════════════
# disable():必须自己会过期
# ══════════════════════════════════════════════════════════════════════

def test_disable_expires_by_itself_without_anyone_calling_enable():
    """**硬要求。** 一个靠配对 enable() 才能恢复的 disable,就是我们在修的那个病:
    某条路径抛异常 / 提前 return / 忘了 finally ⇒ 安全网静默永久关闭,
    而从外面看和武装着一模一样。"""
    fired = threading.Event()
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                        current_threshold_a=100e-9,
                        interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    try:
        assert wd.disable(for_s=0.15) == 0.15
        assert fired.wait(0.1) is False          # 关着的时候确实不响
        # 从这里往后**没有任何 enable() 调用**。
        assert fired.wait(3.0) is True, "disable 到期后安全网没有自己活过来"
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_disable_is_capped_so_a_forgotten_call_cannot_park_it_forever():
    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        current_threshold_a=100e-9)
    from mast.core.watchdog import MAX_DISABLE_S
    assert wd.disable(for_s=10 * MAX_DISABLE_S) == MAX_DISABLE_S


def test_disable_with_a_nonpositive_duration_disables_nothing():
    """``for_s<=0`` **不是**「用默认值」。

    `_pos()` 那个「v<=0 → 出厂默认」的写法 2026-08-09 刚砍断过一次 ForgeAuTip
    (想填 0 关掉上限的人拿到了 30)。同一个形状放在**安全网的关闭时长**上更糟:
    「关 0 秒」被读成「关 15 秒」= 把保护关掉了一段没人要求的时间。
    """
    from mast.core.watchdog import DEFAULT_DISABLE_S

    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        current_threshold_a=100e-9)
    for bad in (0, -1, 0.0, float("nan"), "nope", None):
        assert wd.disable(for_s=bad) == 0.0, f"for_s={bad!r} 关掉了保护"
        assert wd.disable(for_s=bad) != DEFAULT_DISABLE_S


# ══════════════════════════════════════════════════════════════════════
# 人在手动操作时不要拔他的针 —— 但这个抑制**自己要会过期**
#
# 手动用 Nanonis GUI 修针尖期间,电流突变是操作本身引入的,不是故障;而
# cm_sat_current_a 已标定到 20 nA ⇒ 看门狗一定够得着 ⇒ 会在手动操作时把针拔走。
# 一个会那样做的系统,操作者只会直接关掉 —— 那样看门狗直接失效。
# ══════════════════════════════════════════════════════════════════════

def _fast(**kw):
    """一台开火很快的看门狗:窗口 3×5 ms,便于在测试里量「持续多久」。"""
    base = dict(current_threshold_a=100e-9, interval_s=0.005, window_size=3)
    base.update(kw)
    return base


class TestHumanAtTheControls:

    def test_mast_driving_still_retracts(self):
        """**主路径不许被新抑制关掉。** 令牌被持有 ⇒ 照常退针。"""
        fired = threading.Event()
        wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                            idle_getter=lambda: None,      # None = 正被持有
                            **_fast())
        wd.daemon = True
        wd.start()
        try:
            assert fired.wait(2.0) is True, "MAST 在驱动却不退针了 —— 主路径被关掉了"
        finally:
            wd.stop(); wd.join(timeout=1)

    def test_a_long_scan_with_zero_writes_still_retracts(self):
        """仪器操作令牌在长扫描期间持续有效，即使只有状态查询而没有新的写命令，仍应视为软件驱动。"""
        fired = threading.Event()
        # 令牌一直被持有(长扫描),期间零写命令 —— idle 恒为 None。
        wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                            idle_getter=lambda: None, **_fast())
        wd.daemon = True
        wd.start()
        try:
            assert fired.wait(2.0) is True, "长扫描期间贴轨没退针"
        finally:
            wd.stop(); wd.join(timeout=1)

    def test_just_released_still_retracts(self):
        """桥:令牌刚放手不到一个确认窗口 ⇒ 仍算 MAST 在驱动。"""
        fired = threading.Event()
        wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                            idle_getter=lambda: 0.001,     # 刚放手
                            **_fast())
        wd.daemon = True
        wd.start()
        try:
            assert fired.wait(2.0) is True, "技能之间的空档里贴轨没退针"
        finally:
            wd.stop(); wd.join(timeout=1)

    def test_human_at_the_controls_is_not_interrupted(self):
        """令牌空了很久 ⇒ 判定为人工 ⇒ **不退针**(但仍然记录)。"""
        fired = threading.Event()
        wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                            idle_getter=lambda: 3600.0,    # 一小时没碰过
                            human_rail_override_s=1e9,     # 过期时限设得极远
                            **_fast())
        wd.daemon = True
        wd.start()
        try:
            assert fired.wait(0.4) is False, "在用户手里把针拔走了"
        finally:
            wd.stop(); wd.join(timeout=1)

    def test_the_human_suppression_expires_by_itself(self):
        """依据空闲推断的人为操作抑制必须有明确时限。
        连续贴轨超过配置的覆盖时限时，应恢复急停能力。"""
        fired = threading.Event()
        wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                            idle_getter=lambda: 3600.0,    # 判定为人工
                            human_rail_override_s=0.2,     # T,测试里调小
                            **_fast())
        wd.daemon = True
        wd.start()
        try:
            assert fired.wait(0.1) is False, "还没到 T 就动手了"
            assert fired.wait(3.0) is True, (
                "贴轨持续超过 T 之后仍然没退针 —— 抑制变成永久的了")
        finally:
            wd.stop(); wd.join(timeout=1)

    def test_an_absent_or_broken_idle_getter_stays_armed(self):
        """判据缺席/读不到 ⇒ **照常武装**。

        抑制是新加的,而它关掉的是这棵树上唯一一条会自己动手保护针尖的路。
        **读不清就武装**,不是「读不清就闭嘴」。
        """
        for getter in (None, (lambda: (_ for _ in ()).throw(RuntimeError("x")))):
            fired = threading.Event()
            wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=fired.set,
                                idle_getter=getter, **_fast())
            wd.daemon = True
            wd.start()
            try:
                assert fired.wait(2.0) is True, f"判据={getter!r} 时安全网被关掉了"
            finally:
                wd.stop(); wd.join(timeout=1)

    def test_the_rail_clock_is_reset_by_the_run_loop_when_the_rail_clears(self):
        """贴轨断了就重新计时 —— 否则零散的短贴轨会累加成一次「持续」,
        而「持续」正是那条过期判据的全部依据。

        ⚠️ 测的是**运行循环**里那一行,不是 ``_rail_held_for`` 自己。
        第一版直接调 helper 自测,于是「把循环里的重置删掉」的变异**没能把它弄红**
        —— 又是「测了原语没测调用点」。这个错误在这一批里我已经犯过两次。
        """
        wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,  # 一直低于阈值
                            idle_getter=lambda: None, **_fast())
        wd._rail_since = 12345.0          # 假装刚才在贴轨
        wd.daemon = True
        wd.start()
        try:
            import time as _t
            deadline = _t.monotonic() + 2.0
            while _t.monotonic() < deadline and wd._rail_since is not None:
                _t.sleep(0.01)
        finally:
            wd.stop(); wd.join(timeout=1)
        assert wd._rail_since is None, "贴轨已经断了,运行循环却没把计时清掉"


class TestTheIdleSignalComesFromTheToken:

    def test_idle_s_is_none_while_held_and_counts_after_release(self):
        """``instrument_lock.idle_s()`` 的三态:持有中 / 放手后 / 从未持有。"""
        from mast.core.instrument_lock import InstrumentLock

        lock = InstrumentLock()
        assert lock.idle_s() == float("inf"), "从未持有过应当是 inf,不是 0"
        with lock.hold(owner="t", skill="SetBias"):
            assert lock.idle_s() is None, "持有期间必须是 None"
        idle = lock.idle_s()
        assert idle is not None and idle < 1.0

    def test_a_nested_release_still_reads_as_driving(self):
        """composite 的子步 release 之后,外层还持着 ⇒ 仍然是「MAST 在驱动」。

        这是这条链路真正承重的性质:一次 ``ForgeAuTip`` 里有几十个子步,
        每个都 release 一次;如果其中任何一次让 ``idle_s()`` 变成非 None,
        看门狗就会在修针中途把「MAST 在驱动」读成「人在操作」。

        ⚠️ 诚实的边界:``release()`` 里那句「只在 depth 归 0 时写
        ``_last_released_at``」**目前不可观测** —— ``idle_s()`` 先看 depth,
        所以就算嵌套时也写了,外层放手时又会被正确覆盖一次。那句写法是
        **防御性的、语义上正确的**,但它现在没有一条测试能证伪。
        (变异 H6 因此没被杀死 —— 记在这里,而不是假装它被覆盖了。)
        """
        from mast.core.instrument_lock import InstrumentLock

        lock = InstrumentLock()
        with lock.hold(owner="t", skill="ForgeAuTip"):          # 外层
            for _ in range(3):                                  # 几个子步
                with lock.hold(owner="t", skill="TipShape"):
                    pass
                assert lock.idle_s() is None, "子步 release 之后被读成「人在操作」了"
            assert lock.idle_s() is None
        assert lock.idle_s() is not None, "外层放手之后应当开始计时"


def test_enable_is_early_restore_not_the_only_route_back():
    """``enable()`` 仍然能用 —— 它只是提前恢复,不是恢复的唯一途径。"""
    wd = SafetyWatchdog(_Pool(1e-9), on_anomaly=lambda: None,
                        current_threshold_a=100e-9)
    wd.disable(for_s=30.0)
    assert wd._enabled.is_set() is False
    wd.enable()
    assert wd._enabled.is_set() is True
    assert wd._disabled_until is None, "提前恢复之后计时器还挂着"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
