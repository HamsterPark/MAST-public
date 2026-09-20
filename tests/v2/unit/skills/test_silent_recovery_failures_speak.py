"""恢复/急停动作失败时，调用方必须知道 —— 四条静默路径的承重钉。

判据不是「有没有打日志」。日志只有事后翻日志的人看得见，而**调用方拿到的东西
和成功时一模一样**的话，它就还是静默。所以每条测试问的都是同一个问题：

    这次失败之后，SkillResult / 返回值里有没有一个字与成功时不同？

2026-08-10 修的四条：

  1. ``RelocateCoarseXY._panic``        急停 + 退针发不出去 → 原来 ``except: pass``
  2. ``RelocateCoarseXY._prove_clear``  qPlus 取证抛异常 → 原来键直接不出现，
                                        与「本机没 qPlus」长得一模一样
  3. ``AutoApproach._stop_module``      进针模块停机命令没下发 → 原来 ``except: pass``
  4. ``preflight_modules``              探针抛异常 → 返回 ``None``，与「全都验过了」同值

外加一条**反方向**的钉子：``AutoApproach._confirm_stopped`` 读不到时返回 True
是**刻意的**（见其 docstring），不许被这一轮「静默失败都要说话」的势头顺手改掉。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_silent_recovery_failures_speak.py -x -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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
from mast.skills.composite._preflight import preflight_modules
from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY


@dataclass
class ExplodingCtx:
    """``safe_call`` 对指定动词抛异常，其余正常返回。"""

    boom: tuple[str, ...] = ()
    calls: list[str] = field(default_factory=list)

    def safe_call(self, method: str, *args: Any, **kw: Any) -> NanonisCallRecord:
        self.calls.append(method)
        if method in self.boom:
            raise RuntimeError(f"TCP 断了: {method}")
        return NanonisCallRecord(method=method, args=args, return_value=None)


@dataclass
class RefusingCtx:
    """``safe_call`` 不抛，但回包带 error —— 仪器拒了这条命令。"""

    refuse: tuple[str, ...] = ()

    def safe_call(self, method: str, *args: Any, **kw: Any) -> NanonisCallRecord:
        err = "module not running" if method in self.refuse else ""
        return NanonisCallRecord(method=method, args=args, error=err)


# ── ① RelocateCoarseXY._panic ────────────────────────────────────────────────
def test_panic_that_could_not_be_sent_leaves_a_trace():
    skill = RelocateCoarseXY()
    skill._call_log = []
    skill._panic_failures = []
    ctx = ExplodingCtx(boom=("Motor_StopMove", "ZCtrl_Withdraw"))

    skill._panic(ctx)

    assert len(skill._panic_failures) == 2, (
        "两个急停动作都抛了，_panic_failures 却没记满 —— 调用方看不出针尖没退开")
    errored = [r for r in skill._call_log if getattr(r, "error", "")]
    assert len(errored) == 2, (
        "失败的急停动作必须在 nanonis_calls 里留下带 error 的记录，"
        "否则调用日志读起来像两条命令都发出去了")
    note = skill._panic_note()
    assert "紧急停止未能完整下发" in note and "针尖已退开" in note, (
        f"急停失败的那句话必须点名「去确认针尖退开了没有」，实得: {note!r}")


def test_panic_that_worked_says_nothing():
    """反方向：都发出去了就不许制造噪音（否则用户会学会忽略这句话）。"""
    skill = RelocateCoarseXY()
    skill._call_log = []
    skill._panic_failures = []
    skill._panic(ExplodingCtx(boom=()))
    assert skill._panic_failures == []
    assert skill._panic_note() == ""
    assert not [r for r in skill._call_log if getattr(r, "error", "")]


# ── ② RelocateCoarseXY._prove_clear ──────────────────────────────────────────
def test_a_failed_qplus_probe_is_not_the_same_as_no_probe(monkeypatch):
    """「取证没做成」和「这台机器没 qPlus」必须是两个不同的返回值。"""
    skill = RelocateCoarseXY()
    skill._call_log = []

    def _boom(_ctx):
        raise RuntimeError("读振幅失败")

    monkeypatch.setattr("mast.skills.builtins._tip_evidence.qplus_recovered",
                        _boom, raising=False)
    out = skill._prove_clear(ExplodingCtx())

    assert out.get("qplus_probe_failed") is True, (
        "qPlus 取证抛异常之后，返回的 dict 必须带一个说得出「试过但没成」的字段")
    assert out.get("qplus_recovered", "MISSING") is None, (
        "必须是显式的 None（试过没成），不能是键缺席（与「没 qPlus」同形）")
    assert "未完成" in str(out.get("qplus_note", "")), (
        f"note 要说清没成，实得: {out.get('qplus_note')!r}")


# ── ③ AutoApproach._stop_module ──────────────────────────────────────────────
def test_a_stop_command_that_never_left_says_so():
    skill = AutoApproach()
    skill._call_log = []
    skill._stop_failures = []

    skill._stop_module(ExplodingCtx(boom=("AutoApproach_OnOffSet",)))

    assert skill._stop_failures, (
        "停机命令抛了却没记 —— 调用方会以为进针模块已经停了")
    errored = [r for r in skill._call_log if getattr(r, "error", "")]
    assert errored, "失败的停机命令要在 nanonis_calls 里留下带 error 的记录"
    note = skill.stop_note()
    assert "停机命令没有成功下发" in note and "仍在推进" in note, (
        f"这句话必须说出「模块可能还在推针尖」，实得: {note!r}")


def test_a_stop_command_the_instrument_refused_also_says_so():
    """没抛异常、但回包带 error —— 同样是「没停下来」。

    只判异常的话，一台**回了错误码**的机器会被算成停成功。
    """
    skill = AutoApproach()
    skill._call_log = []
    skill._stop_failures = []

    skill._stop_module(RefusingCtx(refuse=("AutoApproach_OnOffSet",)))

    assert skill._stop_failures, "仪器拒了停机命令，也必须记成失败"
    assert "停机命令没有成功下发" in skill.stop_note()


def test_a_stop_command_that_worked_says_nothing():
    skill = AutoApproach()
    skill._call_log = []
    skill._stop_failures = []
    skill._stop_module(RefusingCtx(refuse=()))
    assert skill._stop_failures == []
    assert skill.stop_note() == ""


# ── ④ preflight_modules ──────────────────────────────────────────────────────
def test_a_preflight_that_could_not_check_is_not_a_preflight_that_passed():
    """两次调用都返回 None，但 ``unchecked`` 必须把它们分开。"""
    def _good(_ctx):
        return NanonisCallRecord(method="Good_Get", args=(), error="")

    def _boom(_ctx):
        raise RuntimeError("探针坏了")

    passed: list[str] = []
    assert preflight_modules(None, [(_good, "OK模块")], unchecked=passed) is None
    assert passed == [], "所有探针都答了，不该有 unchecked"

    skipped: list[str] = []
    assert preflight_modules(None, [(_boom, "坏探针")], unchecked=skipped) is None
    assert len(skipped) == 1 and "坏探针" in skipped[0], (
        "探针抛异常时必须记进 unchecked —— 否则「没验成」和「验过了」同为 None")
    assert "探针抛异常" in skipped[0]


def test_preflight_without_the_out_param_behaves_exactly_as_before():
    """``unchecked`` 是可选出参：不传时行为逐字不变（九个调用点没改）。"""
    def _boom(_ctx):
        raise RuntimeError("探针坏了")

    assert preflight_modules(None, [(_boom, "坏探针")]) is None


# ── 反方向：被否掉的方案钉成测试 ──────────────────────────────────────────────
def test_confirm_stopped_deliberately_returns_true_when_it_cannot_read():
    """``_confirm_stopped`` 读不到时返回 True 是**刻意的**，不是漏网的静默。

    理由写在它自己的 docstring 里：这个闩存在是为了不让**单次** 0 读数掐掉一趟
    正在走的进针；读不到时返回 True = 不推翻已经读到的那个 0。若改成 False，
    一条坏链路会让等待循环永远结束不了 —— **把一个已知的失败换成一个静默的挂起**，
    比它要防的问题更糟。

    这条钉子是给下一个人的：他会（正因为看起来显然对）重新想到「读不到就别说
    已停」。要推翻它，需要先回答：等待循环在链路坏掉时靠什么退出？
    """
    skill = AutoApproach()
    skill._call_log = []

    class _Progress:
        status_flap_n = 0

    assert skill._confirm_stopped(ExplodingCtx(boom=("AutoApproach_OnOffGet",)),
                                  _Progress()) is True
    assert skill._confirm_stopped(
        RefusingCtx(refuse=("AutoApproach_OnOffGet",)), _Progress()) is True


def test_selfcheck_the_helpers_are_the_real_ones():
    """一条匹配不到真东西的测试文件会一直绿。这里确认钉的是真源上的方法。"""
    for owner, name in ((RelocateCoarseXY, "_panic"),
                        (RelocateCoarseXY, "_panic_note"),
                        (RelocateCoarseXY, "_prove_clear"),
                        (AutoApproach, "_stop_module"),
                        (AutoApproach, "stop_note"),
                        (AutoApproach, "_confirm_stopped")):
        assert callable(getattr(owner, name, None)), (
            f"{owner.__name__}.{name} 不存在了 —— 本文件钉的是一个已经不在的东西")
    import inspect
    assert "unchecked" in inspect.signature(preflight_modules).parameters
