"""扫描等待预算与进展宽限回归。

预算不足时，仅在行数仍推进的情况下给予有界延长；停滞、不可读和换帧
分别报告。失败说明必须同时给出预算、实际等待、采集进度和帧时估计。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.skills.builtins.scan_utils import WaitScanComplete  # noqa: E402
from mast.skills.composite.scan_at import ScanAt  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# 传递链 —— 先把那两个假诊断钉死,免得下一个人再走一遍
# ══════════════════════════════════════════════════════════════════════

def _budget_ms(**params) -> int:
    p = {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 100e-9}
    p.update(params)
    wait = [s for s in ScanAt().plan(p) if s.step_id == "wait_scan"][0]
    return int(wait.params["timeout_ms"])


def test_the_headroom_really_is_applied_on_the_scan_at_path():
    """「这条路径没吃到 +30% 余量」—— 不成立,而且是从失败信息里推出来的。"""
    est = ScanAt()._resolve(
        {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 100e-9}
    ).estimated_scan_s
    assert _budget_ms() == int((max(300.0, est * 1.3 + 30.0)) * 1000)
    assert _budget_ms() > int(est * 1000), "预算必须严格大于估计帧时"


def test_an_explicit_wait_timeout_is_a_floor_not_a_cap():
    """显式等待值是下限；取显式值与几何估计的较大者。
    分别验证慢扫描不能被短下限截断，以及长下限仍能覆盖较快的扫描。"""
    # 慢扫档(est≈2048 s):显式 300 不再能把一帧 34 分钟的扫描按死在 300 s 上
    est = ScanAt()._resolve(
        {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 100e-9}
    ).estimated_scan_s
    assert _budget_ms(wait_timeout_s=300.0) == int(max(300.0, est * 1.3 + 30.0) * 1000)
    # 快扫档(est≈77 s ⇒ 几何值 < 300):显式值仍然生效,这才叫「下限」
    assert _budget_ms(pixels=256, line_time_s=0.15,
                      wait_timeout_s=300.0) == 300_000
    # 显式值大于几何值时它胜出 —— ForgeAuTip 的 1300 s 不许被缩短
    assert _budget_ms(pixels=256, line_time_s=0.15,
                      wait_timeout_s=1300.0) == 1_300_000


# ══════════════════════════════════════════════════════════════════════
# 行为:还在推进就别judge死它
# ══════════════════════════════════════════════════════════════════════

class _Rig:
    """扫描按 ``line_at(elapsed)`` 推进;状态永远是「在扫」直到行数满。"""

    def __init__(self, total=512, per_line_s=0.002, stall_at=None):
        self.total = total
        self.per_line_s = per_line_s
        self.stall_at = stall_at
        self.t0 = None
        self.stopped = False

    def _elapsed(self):
        import time
        if self.t0 is None:
            self.t0 = time.monotonic()
        return time.monotonic() - self.t0

    def lines_now(self) -> int:
        n = int(self._elapsed() / self.per_line_s)
        if self.stall_at is not None:
            n = min(n, self.stall_at)
        return min(n, self.total)

    def safe_call(self, method, *args, **kw):
        class _R:
            error = ""
            return_value = ("", b"", [0])
            method = ""
            args = ()
        if method == "Scan_Action":
            self.stopped = True
        if method == "Scan_StatusGet":
            running = 0 if self.lines_now() >= self.total else 1
            _R.return_value = ("", b"", [running])
        return _R()

    def check_abort(self):
        return False

    def emit_progress(self, p):
        pass

    def get_progress(self, name):
        return None

    def checkpoint_flush(self):
        pass


def _run(rig, timeout_s, *, poll=0.002):
    """驱动一次真等待。

    ``poll_interval_s`` 走**参数**传进去 —— 必须在 ``max_polls`` 从它推导出来
    **之前**生效。第一版是在 plan_dynamic 之后打补丁的,于是轮询预算仍按 0.5 s
    算出来只有 2 次,延长根本没机会发生:测试红了,而红的是测试台不是产品。
    """
    skill = WaitScanComplete()
    skill._measure_lines = lambda ctx: {          # type: ignore[assignment]
        "lines_done": rig.lines_now(), "lines_total": rig.total,
        "lines_verified": True,
        "stopped_early": rig.lines_now() < rig.total,
    }
    return skill.execute(rig, {"timeout_ms": int(timeout_s * 1000),
                               "poll_interval_s": poll})


def test_a_frame_that_finishes_just_past_the_budget_is_not_killed():
    """合成扫描略晚于原预算完成，进展证据应允许有界延长。"""
    rig = _Rig(total=100, per_line_s=0.004)      # 需要 ~0.40 s
    res = _run(rig, timeout_s=0.30)              # 预算比它短
    assert res.success
    assert res.data["timed_out"] is False, res.data
    assert res.data["outcome"] == "completed"
    assert res.data["extensions"] >= 1, "没有延长,就是又赛跑了一次"


def test_a_stalled_scan_still_dies_on_schedule():
    """延长必须是**挣来的**:行数不涨就不给。

    否则这就不是「有界宽限」,是把超时保护关掉 —— 而超时保护要抓的正是这个场景。"""
    rig = _Rig(total=100, per_line_s=0.004, stall_at=20)
    res = _run(rig, timeout_s=0.20)
    assert res.data["timed_out"] is True
    assert res.data["extensions"] == 0, "卡住的扫描拿到了延长"
    assert rig.stopped, "超时后必须停扫描"


def test_extensions_are_bounded():
    """一帧永远采不完的图,延长次数必须封顶。"""
    rig = _Rig(total=100_000, per_line_s=0.0005)   # 永远采不完
    res = _run(rig, timeout_s=0.20)
    assert res.data["timed_out"] is True
    assert res.data["extensions"] <= WaitScanComplete._MAX_EXTENSIONS


def test_unreadable_line_count_does_not_earn_an_extension():
    """读不到行数 ⇒ 没有正面证据 ⇒ 退回旧行为(判超时)。

    读不到就延长的话,一台缓冲区读不出来的机器会在每次超时上都无限等下去。"""
    rig = _Rig(total=100, per_line_s=0.004)
    skill = WaitScanComplete()
    skill._measure_lines = lambda ctx: {}          # type: ignore[assignment]
    res = skill.execute(rig, {"timeout_ms": 200, "poll_interval_s": 0.002})
    assert res.data["timed_out"] is True
    assert res.data["extensions"] == 0


def test_the_budget_is_reported_on_every_outcome():
    """预算/等待时长不能只在超时那条路上算 —— 只在需要它时才缺席的数字最坏。"""
    rig = _Rig(total=20, per_line_s=0.002)
    res = _run(rig, timeout_s=5.0)
    assert res.data["outcome"] == "completed"
    assert res.data["budget_s"] == pytest.approx(5.0)
    assert res.data["elapsed_s"] > 0
    assert res.data["extensions"] == 0


# ══════════════════════════════════════════════════════════════════════
# 那句话必须印什么
# ══════════════════════════════════════════════════════════════════════

def test_the_timeout_message_prints_the_budget_not_only_the_estimate():
    """两个假诊断都是这句话喂出来的:它唯一的数字是**估计值**。"""
    from mast.skills.composite.graph_executor import CompositeProgress

    progress = CompositeProgress(composite_name="ScanAt", total_steps=1)
    progress.partial_data.update({
        "wait_timed_out": True, "budget_s": 1980.0, "elapsed_s": 1985.0,
        "extensions": 2, "lines_done": 255, "lines_total": 256,
    })
    skill = ScanAt()
    skill._resolved_snapshot = {"estimated_scan_s": 1500.0}
    ok, msg = skill._decide_outcome(True, progress, {})
    assert ok is False
    assert "1980" in msg, "没印等待预算 —— 这正是当初无法判断的那个量"
    assert "1985" in msg, "没印实际等了多久"
    assert "255/256" in msg, "没印放弃时采到哪一行"
    assert "1500" in msg, "估计值仍然要在(它说明估计偏小了多少)"
    # 并且要告诉读者「几乎采满」意味着什么 —— 否则下一个人还是会去查错方向。
    assert "估计帧时偏小" in msg and "wait_timeout_s" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ══════════════════════════════════════════════════════════════════════
# 连续扫描:等不到的东西,别报成「卡住了」（合成连续扫描）
# ══════════════════════════════════════════════════════════════════════

class _ContinuousRig(_Rig):
    """连续扫描的合成仪器：行号在换帧时回绕，运行状态始终保持扫描中。"""

    def lines_now(self) -> int:
        return int(self._elapsed() / self.per_line_s) % self.total

    def safe_call(self, method, *args, **kw):
        rec = super().safe_call(method, *args, **kw)
        if method == "Scan_StatusGet":
            rec.return_value = ("", b"", [1])   # 永不为 0
        return rec


def test_a_line_count_that_goes_DOWN_is_a_new_frame_not_a_stuck_one():
    """合成行号回退表示当前帧已更换，必须区别于行号不变的停滞状态。"""
    rig = _ContinuousRig(total=256, per_line_s=0.001)
    skill = WaitScanComplete()
    seq = iter([180, 120])                      # 播种一次,到点一次
    skill._measure_lines = lambda ctx: {         # type: ignore[assignment]
        "lines_done": next(seq, 120), "lines_total": 256,
        "lines_verified": True, "stopped_early": True,
    }
    res = skill.execute(rig, {"timeout_ms": 200, "poll_interval_s": 0.002})
    assert res.data["timed_out"] is True
    assert res.data["frame_restarted"] is True, res.data
    assert res.data["outcome"] == "restarted", res.data
    assert res.data["extensions"] == 0, "换帧不该换来延长 —— 再等也等不到"
    assert rig.stopped, "等不到的东西要停掉,不是接着等"


def test_an_unchanged_line_count_is_still_stuck_not_restarted():
    """「不涨」和「变少」必须各说各的。

    没有这条,上面那条修完就会走到另一侧:一个真卡住的扫描被报成「仪器在连续扫」,
    把人送去关一个根本没开的开关。两种错话的代价一样 —— 都是让人查错地方。
    """
    rig = _Rig(total=100, per_line_s=0.004, stall_at=20)
    res = _run(rig, timeout_s=0.20)
    assert res.data["timed_out"] is True
    assert res.data["frame_restarted"] is False, res.data
    assert res.data["outcome"] == "timed_out"


def test_the_inherited_instrument_state_no_longer_kills_the_next_task():
    """显式等待值是下限，继承的慢扫描参数仍应生成足够长的几何预算。
    用独立构造的像素数和线时长验证这一点。"""
    est_s = 384 * 3.0 * 2  # 双向
    budget_ms = _budget_ms(pixels=384, line_time_s=3.0, wait_timeout_s=300.0)
    assert budget_ms > 300_000, (
        "还是被那个 300 s 的常数按死了：%d ms" % budget_ms)
    assert budget_ms >= int(est_s * 1.3 * 1000), (
        "预算 %d ms 不够扫完这一帧（估计 %.0f s）—— 无人值守里必然超时"
        % (budget_ms, est_s))
