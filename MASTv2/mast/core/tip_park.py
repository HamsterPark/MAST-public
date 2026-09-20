"""针尖退到位了没有 —— 只读谓词,全仓唯一的一句「已退到静态安全态」。

为什么有这个模块
================
``SafeRetract`` 从 v1 抄过来的时候是这样的::

    record = context.safe_call("ZCtrl_Withdraw", 0, 1)
    return SkillResult(..., data={"retracted": True})   # ← 命令发出去了,不是针退到了

``(0, 1)`` = 不等待、1 ms 超时。命令一发出就报 ``retracted: True``,而压电这时
**还在往上爬**。``WithdrawTip`` 2026-07-03 修过同一个缺陷(``builtins/approach.py``
的 ``ZCtrl_Withdraw(1, -1)``:等它真的走完再报)。这里补上另一半:**回读状态**。

判据:Withdraw 是**静态态**,不是过程
=====================================
用户的定义(STM 通用,不是某台机器的):**Z 反馈环断开** 且 **Z 压电停在收回端**。

* ``ZCtrl_StatusGet`` 的码 6 叫 ``Withdrawing`` —— 那是**正在退**,是过程,不是这个
  状态。判「到位没有」不能读它。
* 「反馈环断开」问的是**实时控制器**(``ZCtrl_OnOffGet``),不是 Z-Controller 模块
  (``ZCtrl_StatusGet``)。Nanonis 手册自己说模块可以显示 Off 而实时控制器还没跟上,
  两者不可互换 —— 这条判断全仓只有一份实现,在 :func:`mast.skills.verify
  .verify_z_controller`,本模块调它,不另写。
* 「收回端是哪一端」由 ``z_extend_sign`` 决定,**本机没声明过就必须说不知道**
  (:func:`mast.core.instrument_profile.z_extend_sign_or_none`)。猜错的代价不对称:
  猜错方向会把「针停在伸长端」(顶到样品那一侧)判成 parked。这正是
  既有教训 那一族 —— 保护动作和它依赖的符号是同一个机制。
* 「两端各在哪」用 :func:`mast.core.envelope_reconcile.resolve_z_travel`,
  全仓唯一的一份 Z 行程算法(它同时管住了「``ZCtrl_LimitsGet`` 那对数字只有
  ``ZCtrl_LimitsEnabledGet == 1`` 时才算数」这条)。

三态,而且「读不到」还要再分两种
================================
``state`` ∈ ``parked`` / ``not_parked`` / ``unreadable``。**读不到永远不折叠成
具体值** —— 既不是「退到了」,也不是「没退到」。

``unreadable`` 里两种成因分开放在两个字段,因为它们的**处置完全不同**:

* :attr:`ParkVerdict.unreadable` —— 硬件读失败。可能是瞬时的,**再等一下有意义**。
* :attr:`ParkVerdict.undeclared` —— 本机从来没声明过这一项(如 ``z_extend_sign``)。
  等多久都不会变,**再轮询是白等**,要人去仪器档案里填。

与 ``core/state.py`` 的关系(**已知重复,待收敛**)
================================================
``state.py`` 的 ``HardwareState.withdrawn`` 算的是同一件事,但用的是
``ZCtrl_StatusGet``(模块,不是实时控制器)+ ``ZCtrl_LimitsGet[0]`` 当收回端
(**写死了「高端=收回端」,等价于假定 ``z_extend_sign == -1``**)+ ``< 1e-12`` 的
浮点相等,而且它是 1 s 缓存刷新的产物 —— 那份缓存有**读失败时沿用旧值**的语义,
拿它做退针确认会用一份几秒前的读数说「已经退到了」。所以本模块**每次都自己读**,
不走缓存。两份实现要合并成一份(``state.py`` 当前在别人的未提交批次里,不在本次
改动范围内)。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

# ``_floats``:回包解包,与 ``envelope_reconcile`` 同一份(同包,不再抄第三份)。
# 两个被 import 的 core 模块顶层都只依赖标准库,无环。
from mast.core.envelope_reconcile import _floats, resolve_z_travel
from mast.core.instrument_profile import z_extend_sign_or_none

logger = logging.getLogger(__name__)

# ZCtrl_StatusGet 的码表。与 ``skills/verify.py:_MODULE_STATUS`` /
# ``core/state.py`` 同一张表 —— 这里只用来把码翻成给人看的词,不参与判定。
_MODULE_STATUS = {1: "Off", 2: "On", 3: "Hold", 4: "SwitchingOff",
                  5: "SafeTip", 6: "Withdrawing"}

PARKED = "parked"
NOT_PARKED = "not_parked"
UNREADABLE = "unreadable"

#: 「Z 停在收回端」的容差,取 Z 行程的这个比例。
#
# 为什么是派生的而不是一个绝对值:Z 行程本身随温度腰斩(本机 RT 全程 720 nm、
# LHe 339 nm),写死一个纳米数在其中一档上就是错的。
#
# 为什么是 1%:两侧的代价不对称。太紧 → 把一次真的退针判成 not_parked(烦,安全,
# 顶多多等一轮);太松 → 把没退到的针判成 parked(危险)。比例乘的是**全程**
# (``ZTravel.span_m``),本机 ⇒ 7.2 nm(RT,全程 720 nm)/ 3.4 nm(LHe,全程 339 nm):
# 比 float32 回包噪声(相对 ~4e-9)高好几个数量级,又比「针还在隧穿区」离收回端的
# 距离(约一整个行程)小两个数量级 —— 两边都不挨着。
_RAIL_TOL_FRAC = 0.01


@dataclass(frozen=True)
class ParkVerdict:
    """针尖退到位了没有,以及**凭什么这么说**。

    ``state`` 是结论,其余字段是证据。证据必须跟着结论走:一个「没退到」如果不带
    上「Z 现在在哪、收回端在哪」,下游只能重新去读一遍,或者干脆不信。
    """

    state: str                      # parked / not_parked / unreadable
    reason: str                     # 人话,给用户和 agent 看的那一句
    feedback_on: bool | None        # 实时控制器(ZCtrl_OnOffGet);None = 没读到
    module_status: str | None       # Z-Controller 模块状态词(GUI 显示的那个),仅作证据
    z_m: float | None               # ZCtrl_ZPosGet
    rail_m: float | None            # 收回端的 Z 坐标
    rail_side: str | None           # "high" / "low" —— 收回端是行程的哪一端
    gap_m: float | None             # |z - rail|
    tolerance_m: float | None       # 判「到端了」的容差
    travel_source: str | None       # 行程是哪条 Nanonis 读答出来的
    unreadable: tuple[str, ...]     # 读失败的项 —— 可能是瞬时的,再读有意义
    undeclared: tuple[str, ...]     # 本机没声明过的项 —— 等多久都不会变
    read_at: float                  # time.time(),读数拍于何时

    @property
    def is_parked(self) -> bool:
        """**只有确认到位才是 True。** ``unreadable`` 和 ``not_parked`` 都是 False。

        想区分「没退到」和「不知道」的调用方必须看 :attr:`state` —— 这个布尔故意
        不承担三态,免得有人拿 ``not verdict.is_parked`` 当「确认没退到」用。
        """
        return self.state == PARKED

    @property
    def retry_useful(self) -> bool:
        """再轮询一次有没有可能改变结论。

        ``undeclared`` 非空 ⇒ 白等(要人去填仪器档案),调用方应当**立刻**停止轮询
        并如实说是配置缺口,而不是耗光预算再报一个看起来像超时的东西。
        """
        return not self.undeclared

    def evidence(self) -> str:
        """一行证据。数字带单位,不带任何没测过的猜测。"""
        parts: list[str] = []
        if self.feedback_on is None:
            parts.append("Z 反馈: 读不到")
        else:
            parts.append(f"Z 反馈: {'闭合' if self.feedback_on else '断开'}")
        if self.module_status:
            parts.append(f"模块: {self.module_status}")
        if self.z_m is not None:
            parts.append(f"Z = {self.z_m * 1e9:.1f} nm")
        if self.rail_m is not None:
            parts.append(f"收回端 = {self.rail_m * 1e9:.1f} nm")
        if self.gap_m is not None and self.tolerance_m is not None:
            parts.append(f"差 {self.gap_m * 1e9:.1f} nm(容差 {self.tolerance_m * 1e9:.1f} nm)")
        if self.travel_source:
            parts.append(f"行程来源: {self.travel_source}")
        if self.unreadable:
            parts.append(f"读不到: {', '.join(self.unreadable)}")
        if self.undeclared:
            parts.append(f"本机未声明: {', '.join(self.undeclared)}")
        return "；".join(parts)


def park_verdict_from_readings(
    *,
    feedback_on: bool | None,
    module_status: str | None,
    z_m: float | None,
    travel: Any,
    z_extend_sign: int | None,
    unreadable: "tuple[str, ...] | list[str]" = (),
    read_at: float | None = None,
) -> ParkVerdict:
    """把一组读数判成一个结论。**纯函数,不碰硬件** —— 判据在这里,读在上面。

    分开是为了测试能直接喂读数(不用假 context 也能钉住判据),也是为了
    ``core/state.py`` 那份重复实现日后能收敛到这一个,而不用把它的读法一起搬过来。

    参数:
        feedback_on: 实时控制器说反馈环闭没闭合;``None`` = 没读到。
        module_status: Z-Controller 模块状态词,**只作证据不作判据**(见模块说明)。
        z_m: 当前 Z 压电位置(米)。
        travel: :class:`mast.core.envelope_reconcile.ZTravel` 或 ``None``。
        z_extend_sign: ``+1`` = 伸长(朝样品)表现为 Z 增大 ⇒ 收回端在**低**端;
            ``-1`` = 收回端在**高**端;``None`` = 本机没声明过,不许猜。
        unreadable: 已知读失败的项名。
    """
    at = time.time() if read_at is None else read_at
    missing = list(unreadable)
    undeclared: list[str] = []

    def _verdict(state: str, reason: str, *, rail_m=None, rail_side=None,
                 gap_m=None, tol_m=None) -> ParkVerdict:
        return ParkVerdict(
            state=state, reason=reason,
            feedback_on=feedback_on, module_status=module_status, z_m=z_m,
            rail_m=rail_m, rail_side=rail_side, gap_m=gap_m, tolerance_m=tol_m,
            travel_source=getattr(travel, "source", None),
            unreadable=tuple(missing), undeclared=tuple(undeclared),
            read_at=at,
        )

    # ── 先看两个「一条读数就能定死」的否定 ────────────────────────────────
    # 它们不需要行程和方向符号,所以在那两项读不到 / 没声明的机器上照样给得出
    # 一个确定的「还没到」。确定的否定比一个 unreadable 有用得多。
    if feedback_on is True:
        return _verdict(NOT_PARKED,
                        "Z 反馈环仍然闭合(实时控制器回报 ON)—— 针没有退到静态安全态。")
    if module_status == "Withdrawing":
        # 码 6:模块自己说退针**正在进行**。这是过程,不是那个状态。
        return _verdict(NOT_PARKED, "退针正在进行中(模块状态 Withdrawing),还没到位。")

    # ── 再看能不能判 ──────────────────────────────────────────────────────
    if feedback_on is None:
        if "feedback_on" not in missing:
            missing.append("feedback_on")
    if z_m is None and "z_m" not in missing:
        missing.append("z_m")
    if travel is None and "z_travel" not in missing:
        missing.append("z_travel")
    if z_extend_sign is None:
        undeclared.append("z_extend_sign")

    if undeclared:
        return _verdict(
            UNREADABLE,
            "判不了退针到位没有:本机从没声明过 ``z_extend_sign``(Z 压电伸长朝哪个"
            "方向),所以不知道行程的哪一端是收回端。**不猜** —— 猜反了会把停在伸长端"
            "(朝样品那一侧)的针判成已退针。请在仪器档案里声明后再判。")
    if missing:
        return _verdict(
            UNREADABLE,
            f"判不了退针到位没有:{', '.join(missing)} 读不到。"
            "读不到不等于没退到,也不等于退到了。")

    assert travel is not None and z_m is not None  # 上面已把 None 全挡掉
    # z_extend_sign = +1:伸长(朝样品)= Z 增大 ⇒ 收回端在低端;-1 反之。
    if z_extend_sign > 0:
        rail_m, rail_side, far_m = travel.lo_m, "low", travel.hi_m
    else:
        rail_m, rail_side, far_m = travel.hi_m, "high", travel.lo_m

    tol_m = abs(travel.span_m) * _RAIL_TOL_FRAC
    gap_m = abs(float(z_m) - float(rail_m))

    if gap_m <= tol_m:
        note = "" if module_status in (None, "Off") else f"(模块状态 {module_status})"
        return _verdict(
            PARKED,
            f"已退到静态安全态:Z 反馈环断开,Z 停在收回端{note}。",
            rail_m=rail_m, rail_side=rail_side, gap_m=gap_m, tol_m=tol_m)

    # 停在**伸长端**要单独说一句:那是朝样品的那一侧,和「还差一点」不是一回事。
    at_far_end = abs(float(z_m) - float(far_m)) <= tol_m
    where = "而是顶在**伸长端**(朝样品那一侧)" if at_far_end else f"还差 {gap_m * 1e9:.1f} nm"
    return _verdict(
        NOT_PARKED,
        f"Z 反馈环已断开,但 Z 没停在收回端 —— {where}。",
        rail_m=rail_m, rail_side=rail_side, gap_m=gap_m, tol_m=tol_m)


def tip_parked(context) -> ParkVerdict:
    """针尖退到位了没有 —— 单次读数,只读,不取仪器令牌,不长等。

    读五条(全部是 GET):``ZCtrl_OnOffGet``(经 ``verify_z_controller``,连带
    ``ZCtrl_StatusGet``)、``ZCtrl_ZPosGet``、``Piezo_RangeGet``、
    ``ZCtrl_LimitsGet``、``ZCtrl_LimitsEnabledGet``;再加一条本地配置
    ``z_extend_sign``。**不缓存** —— 见模块说明里为什么不能走 ``state.py`` 那份缓存。

    **永不抛异常。** 一个自己会炸的确认步骤不是安全网。任何读失败都变成
    ``unreadable`` 里的一项,结论是 :data:`UNREADABLE`,而不是一个看起来正常的判断。

    看门狗可以调:它只发读命令,``ExecutionContext.safe_call`` 对读一律放行(连
    abort 之后也放行),也不排队等仪器令牌。走的是 ``context`` 自己那条 role ——
    调用方若怀疑主 socket 卡住,要自己换一个绑在别的 role 上的 context。
    """
    # 延迟导入:``mast.core`` 在导入期不依赖 ``mast.skills``(``core/runtime.py``
    # 对 skills 的引用也是这个形状)。``verify`` 本身只依赖标准库,无环。
    from mast.skills.verify import verify_z_controller

    read_at = time.time()
    unreadable: list[str] = []

    feedback_on: bool | None = None
    module_status: str | None = None
    try:
        v = verify_z_controller(context, expect=None, settle=False)
        feedback_on = v.get("on")
        if not v.get("verified"):
            unreadable.append("feedback_on")
        code = v.get("module_status")
        if code is not None:
            module_status = _MODULE_STATUS.get(int(code), f"Unknown({code})")
    except Exception as exc:  # noqa: BLE001 — 谓词不许炸
        logger.warning("tip_parked: verify_z_controller raised: %s", exc)
        unreadable.append("feedback_on")

    def _read(key: str, thunk, n: int) -> "list[float] | None":
        try:
            rec = thunk()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tip_parked: %s raised: %s", key, exc)
            unreadable.append(key)
            return None
        if rec is None or getattr(rec, "error", None):
            unreadable.append(key)
            return None
        vals = _floats(getattr(rec, "return_value", None), n)
        if vals is None:
            unreadable.append(key)
            return None
        return vals

    # 每个 verb 都写成 safe_call("字面量") ——  全仓的 abort 策略检查、安全审计、
    # API 覆盖率普查都靠 grep 这个形状找 Nanonis 调用(见
    # ``tests/v2/unit/core/test_safe_call_verbs_are_literal.py``)。
    z_vals = _read("z_m", lambda: context.safe_call("ZCtrl_ZPosGet"), 1)
    piezo = _read("piezo_range", lambda: context.safe_call("Piezo_RangeGet"), 3)
    zlim = _read("z_limits", lambda: context.safe_call("ZCtrl_LimitsGet"), 2)
    zen = _read("z_limits_enabled", lambda: context.safe_call("ZCtrl_LimitsEnabledGet"), 1)

    travel = resolve_z_travel(
        piezo_z_full_m=(piezo[2] if piezo else None),
        z_limits_m=(zlim if zlim else None),
        # ``None``(没问到)按未启用处理 —— 不知道软限启没启用时拿它当边界就是猜。
        z_limits_enabled=(None if zen is None else bool(zen[0])),
    )
    if travel is not None:
        # 行程算出来了,那两条各自的读失败就不是缺口了(一条读到就够)。
        for key in ("piezo_range", "z_limits", "z_limits_enabled"):
            while key in unreadable:
                unreadable.remove(key)

    return park_verdict_from_readings(
        feedback_on=feedback_on,
        module_status=module_status,
        z_m=(z_vals[0] if z_vals else None),
        travel=travel,
        z_extend_sign=z_extend_sign_or_none(),
        unreadable=tuple(unreadable),
        read_at=read_at,
    )


__all__ = [
    "ParkVerdict",
    "park_verdict_from_readings",
    "tip_parked",
    "PARKED",
    "NOT_PARKED",
    "UNREADABLE",
]
