"""进针等待相的进展见证：模块运行时长与电流判定窗口分别报告。

有状态的合成仪器维护模块开关、压电往复运动和噪声底电流。测试核验
粗动推进、模块停下和电流达标是不同状态，不能用某一个等待时长代替全部过程。
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.core.types import NanonisCallRecord
from mast.skills.builtins.approach import AutoApproach


# ── 一台贯穿状态的假机器 ──────────────────────────────────────────────────────

#: 独立合成压电范围。
_Z_TOP = 200e-9
#: 合成回转点留在下限内侧。
_Z_BOTTOM = -180e-9
#: 合成往复循环的 Z 序列（顶 → 底 → 顶）。
_WOODPECKER = (_Z_TOP, 60e-9, -60e-9, _Z_BOTTOM, 60e-9)


@dataclass
class FakeRig:
    """Nanonis 的一台小替身:模块开关 + 啄木鸟 Z + 噪声底电流。

    时间由**轮询次数**推进,不由挂钟推进 —— 一个靠 sleep 才对得上的测试,在慢机器上
    会用另一种方式绿。
    """

    #: 模块运行指定轮询次数后自行停止。
    stop_after_polls: int = 40
    #: 在第 N 次轮询上谎报一次 running=0(瞬态),之后照常。None = 不谎报。
    flap_at_poll: "int | None" = None
    #: 压电动不动。False = 模块报在跑但 Z 纹丝不动(另一种故障)。
    piezo_moves: bool = True
    #: 电流(A)。默认噪声底 —— 4 K 下针尖还远,永远够不到判据。
    current_a: float = 4e-14
    setpoint_a: float = 5e-11

    running: bool = False
    polls: int = 0
    z_index: int = 0
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    #: 模块被 OnOffSet(0) 停掉时的轮询序号(None = 从没被停过)。
    stopped_at_poll: "int | None" = None

    # -- Nanonis 面 ---------------------------------------------------------
    def safe_call(self, method: str, *args: Any, **kw: Any) -> NanonisCallRecord:
        self.calls.append((method, args))
        if method == "AutoApproach_Open":
            return self._ok([])
        if method == "AutoApproach_OnOffSet":
            on = bool(args[0]) if args else False
            if not on and self.running:
                self.stopped_at_poll = self.polls
            self.running = on
            return self._ok([])
        if method == "AutoApproach_OnOffGet":
            self.polls += 1
            if self.running:
                if self.piezo_moves:
                    self.z_index += 1
                if self.polls >= self.stop_after_polls:
                    self.running = False          # 模块自己停(到达停止判据)
                elif self.polls == self.flap_at_poll:
                    return self._ok([0])          # 瞬态谎报,模块其实还在跑
            return self._ok([1 if self.running else 0])
        if method == "ZCtrl_ZPosGet":
            z = _WOODPECKER[self.z_index % len(_WOODPECKER)] if self.piezo_moves \
                else _Z_BOTTOM
            return self._ok([z])
        if method == "Current_Get":
            return self._ok([self.current_a])
        if method == "ZCtrl_SetpntGet":
            return self._ok([self.setpoint_a])
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")

    @staticmethod
    def _ok(vals: list) -> NanonisCallRecord:
        return NanonisCallRecord(method="", args=(), return_value=("", b"", vals))

    # -- 技能面 -------------------------------------------------------------
    def run(self, skill_name: str, params: dict):
        """⑫ 的参数组切换会走这条路。这台替身没有参数组 ⇒ 一律失败 ⇒ 不切、不放回。"""
        raise RuntimeError(f"no skill {skill_name} on this fake rig")


@pytest.fixture
def skill(monkeypatch) -> AutoApproach:
    """一个把所有等待窗口都缩短、但**每一层机制都还在**的 AutoApproach。"""
    monkeypatch.setattr(AutoApproach, "_poll_interval_s", 0.0, raising=False)
    monkeypatch.setattr(AutoApproach, "_grace_s", 0.0, raising=False)
    monkeypatch.setattr(AutoApproach, "_z_sample_every_s", 1e-9, raising=False)
    monkeypatch.setattr(AutoApproach, "_engage_interval_s", 0.0, raising=False)
    monkeypatch.setattr(AutoApproach, "_engage_budget_s", 0.02, raising=False)
    monkeypatch.setattr(AutoApproach, "_crosstalk_every_s", 0.0, raising=False)
    return AutoApproach()


def _run(skill: AutoApproach, rig: FakeRig, **params):
    res = skill.run_composite(rig, dict(params))
    return res


# ── 模块跑了很久,报文却让人读成「秒停」 ────────────────────────────────────

def test_failure_names_how_long_the_module_ran(skill):
    """报文必须说出**模块自己**跑了多久。

    去掉 ``_engage_failure_text`` 的 ``progress=`` 参数 → 这条红。
    """
    rig = FakeRig(stop_after_polls=40)
    res = _run(skill, rig)

    assert res.success is False
    assert "模块实际运行" in res.error, res.error
    assert "读到 running=1" in res.error, res.error


def test_the_only_second_in_the_message_is_not_the_settle_window(skill):
    """电流判定窗口必须标明自身含义，避免被误读成模块总运行时长。"""
    rig = FakeRig(stop_after_polls=40)
    res = _run(skill, rig)

    assert "电流判定窗口" in res.error, res.error
    # 旧措辞不能回来:它读起来像「这趟一共等了这么久」。
    assert "等了 " not in res.error, res.error


def test_the_stage_was_advancing_and_the_message_says_so(skill):
    """Z 满摆 = 粗动在推进,而这件事必须出现在报文里。

    去掉等待循环里的 ``progress.note_z(...)`` → 这条红(报文改说「没读到 Z」)。
    """
    rig = FakeRig(stop_after_polls=40)
    res = _run(skill, rig)

    assert "粗动确实在推进" in res.error, res.error
    wp = (res.data or {}).get("wait_progress") or {}
    assert wp["z_travel_m"] > 0
    assert wp["z_cycles_approx"] is not None and wp["z_cycles_approx"] >= 1


def test_a_frozen_piezo_is_a_different_sentence(skill):
    """「在动只是慢」和「根本没动」必须是两句话。

    这条是那个区分本身的名字。把 ``motion_text`` 里 ``z_travel_m <= 0`` 那一支删掉
    (两种情形共用一句话)→ 这条红。
    """
    rig = FakeRig(stop_after_polls=40, piezo_moves=False)
    res = _run(skill, rig)

    assert "纹丝不动" in res.error, res.error
    assert "粗动确实在推进" not in res.error, res.error
    assert (res.data or {}).get("wait_progress", {})["z_travel_m"] == 0


def test_never_started_is_not_the_same_as_ran_then_stopped(skill):
    """启动被拒(从没读到 running=1)要说得出来,而不是和「跑完停了」共用一句话。"""
    rig = FakeRig(stop_after_polls=0)      # OnOffSet(1) 之后立刻就是停的
    res = _run(skill, rig)

    assert res.success is False
    assert "0 次读到 running=1" in res.error, res.error
    assert "启动可能被拒绝" in res.error, res.error


# ── 单次读数不该停掉一趟正在走的进针 ─────────────────────────────────────────

def test_one_transient_zero_does_not_kill_a_running_approach(skill):
    """状态位抖一下 ≠ 模块停了 —— 复读确认之前不许动手停机器。

    去掉 ``_confirm_stopped`` 的调用 → 等待相在第 12 次轮询就收工,``polls`` 远小于
    ``stop_after_polls``,而且 ``status_flap_n`` 归零 → 这条红。
    """
    rig = FakeRig(stop_after_polls=40, flap_at_poll=12)
    res = _run(skill, rig)

    wp = (res.data or {}).get("wait_progress") or {}
    assert wp["status_flap_n"] == 1, wp
    # 抖动之后仍然把模块跑到了它自己的终点(而不是在第 12 次轮询上被掐掉)。
    assert rig.polls >= 40, rig.polls
    assert wp["running_polls_n"] > 12, wp


def test_a_real_stop_still_ends_the_wait(skill):
    """复读确认不能把「真的停了」拖成一次挂起 —— 反向的变异守卫。"""
    rig = FakeRig(stop_after_polls=8)
    res = _run(skill, rig)

    assert res.success is False
    wp = (res.data or {}).get("wait_progress") or {}
    assert wp["status_flap_n"] == 0
    # 停下之后不再空转:轮询数只比模块的终点多一点点(确认读 + 收尾)。
    assert rig.polls <= 12, rig.polls


# ── 预算砍断一趟仍在推进的进针,要说清它不是「卡住」 ──────────────────────────

def test_timeout_message_distinguishes_progressing_from_stuck(skill):
    """跑满预算 ≠ 卡住。报文要说出它当时还在推进,以及再调一次会接着走。

    把 timeout 分支改回只有一句 "did not reach the setpoint within …" → 这条红。
    """
    rig = FakeRig(stop_after_polls=10**9)      # 模块自己永远不停
    res = _run(skill, rig, wait_timeout_s=0.05)

    assert res.success is False
    assert "粗动确实在推进" in res.error, res.error
    assert "从当前粗动位置继续" in res.error, res.error
    assert rig.stopped_at_poll is not None      # 上限到了仍然要把模块停掉


# ── 结果里带得走这些数字(不只是印在一句话里) ────────────────────────────────

def test_wait_progress_rides_in_the_result_data(skill):
    """进展观测要进 ``data``,否则只有人读得到、下游读不到。"""
    rig = FakeRig(stop_after_polls=40)
    res = _run(skill, rig)

    wp = (res.data or {}).get("wait_progress")
    assert isinstance(wp, dict)
    assert wp["module_ran_s"] >= 0.0
    assert wp["running_polls_n"] >= 30
    assert wp["polls_n"] >= wp["running_polls_n"]
    assert wp["z_reads_n"] > 0
