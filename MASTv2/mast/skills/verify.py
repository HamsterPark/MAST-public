"""Confirm, don't assume — read the hardware back after you write to it.

WHY THIS MODULE EXISTS
======================
A recurring, expensive failure in this system: a skill issues a write, the TCP call
returns without an error, and the skill then REPORTS the value it asked for as
though it were the value the instrument has.

    rec = context.safe_call("ZCtrl_OnOffSet", 0)
    return SkillResult(..., data={"z_controller_on": False})   # ← a CLAIM, not a READING

That is the same defect the 2026-07-10 kept surfacing under
different names — "AutoApproach 宣布进针成功 at 0.17 pA", "硬件『停止』≠『达标』".
The agent downstream believes `z_controller_on: False` and runs a COARSE APPROACH.
If the loop is in fact still closed, that is a tip crash.

THE ONE THAT MATTERS MOST: TWO DIFFERENT "IS THE Z LOOP OFF?"
=============================================================
Nanonis exposes two, and they are not interchangeable. From the V5e TCP manual:

  ``ZCtrl_StatusGet``  — the Z-Controller **module's** status (Off/On/Hold/
                         SwitchingOff/SafeTip/Withdrawing). This is what the GUI
                         shows.
  ``ZCtrl_OnOffGet``   — "Returns the status of the Z-Controller. This function
                         returns the status **from the real-time controller** (i.e.
                         **not from the Z-Controller module**). This function is
                         useful to make sure that the Z-controller is **really off**
                         before starting an experiment. Due to the communication
                         delay …"

Nanonis is telling you, in its own manual, that the module can say "off" while the
real-time controller has not caught up — and that if you are about to start
something that depends on the loop being off, you must ask the RT controller.

MAST wrapped ``StatusGet`` and not ``OnOffGet``. So every "I turned the Z loop off"
in this codebase was the module's opinion at best, and usually not even that — just
the skill repeating its own request back to itself.

USAGE
=====
Any skill that switches the Z loop and then acts on that fact must call
:func:`verify_z_controller`. It costs one extra TCP round-trip. A crashed tip costs
a day.
"""

from __future__ import annotations

import logging
import math
import time as _time
from typing import Any

logger = logging.getLogger(__name__)


def _scalar(value: Any) -> Any:
    """The scalar out of a Nanonis return value.

    The shape is ``(header, error, body)`` — the payload is at index **2**. Walking
    the whole structure for "the first number" finds the HEADER's zero instead, which
    reads as OFF no matter what the hardware said. (It did exactly that until a test
    caught it: ``_ok(1)`` — the loop is ON — came back as ``off``. A verification
    helper that always answers "off" is worse than no helper, because everything
    downstream is built on believing it.)

    Mirrors the convention every other skill uses (zcontrol._scalar, datalog._values).
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) > 2:
            body = value[2]
            if isinstance(body, (list, tuple)):
                return body[0] if body else None
            return body
        return None
    except (TypeError, IndexError):  # pragma: no cover — defensive
        return None


# How long the real-time controller may LEGITIMATELY lag a write.
#
# Nanonis Z-Controllers have a per-controller SWITCH-OFF DELAY. From the Mimea
# manual (Z-Controller Configuration §6): when the controller receives a request
# to switch off, "it starts to average the position over the specified amount of
# time on the RT engine **while the controller is still running**… This procedure
# doesn't switch off the controller immediately". Recommended values are 10–100 ms.
#
# Our write→read gap is ~0.2 ms on loopback and 1–3 ms over LAN — one to two
# ORDERS OF MAGNITUDE inside that window. So reading back immediately does not
# race occasionally, it reads the PRE-WRITE state essentially every time
# ("要求 ON，实时控制器回报 OFF" and its mirror image).
#
# The delay is a per-controller property, which is why this began showing up
# after the rig switched its control signal to log Current — switching the active
# controller switches its switch-off delay along with it.
#
# v1 diagnosed this exact race in 2026-03 and fixed it with a 100 ms retry, but
# that fix lives in core/executor.py and only covers the manual/GUI path; agents
# go through wrap_skill → skill.execute and bypass it entirely.
_SETTLE_POLL_S = 0.05
_SETTLE_FALLBACK_S = 0.5    # budget when the rig won't tell us its delay
_SETTLE_CEILING_S = 10.0    # a mis-configured delay must not hang the skill

# ZCtrl_StatusGet codes — 4 (SwitchingOff) is direct evidence of "still closing".
_MODULE_STATUS = {1: "Off", 2: "On", 3: "Hold", 4: "SwitchingOff",
                  5: "SafeTip", 6: "Withdrawing"}


def _switch_off_delay_s(context) -> float | None:
    """The ACTIVE controller's switch-off delay in seconds, or None if unreadable.

    Per-controller property — see the module note above. Read only to size the
    settle budget; never to decide anything on its own.
    """
    try:
        rec = context.safe_call("ZCtrl_SwitchOffDelayGet")
    except Exception:  # noqa: BLE001 — diagnostics must never break the caller
        return None
    if getattr(rec, "error", None):
        return None
    try:
        return float(_scalar(getattr(rec, "return_value", None)))
    except (TypeError, ValueError):
        return None


def _module_status(context) -> int | None:
    """Z-Controller MODULE status code, or None. Diagnostic only.

    Note this disagrees with the RT controller by design during the switch-off
    window: the module reports 4 (SwitchingOff) while the loop is still closed.
    """
    try:
        rec = context.safe_call("ZCtrl_StatusGet")
    except Exception:  # noqa: BLE001
        return None
    if getattr(rec, "error", None):
        return None
    try:
        return int(_scalar(getattr(rec, "return_value", None)))
    except (TypeError, ValueError):
        return None


def verify_z_controller(context, expect: bool | None = None, *,
                        settle: bool = True,
                        timeout_s: float | None = None) -> dict:
    """Ask the REAL-TIME CONTROLLER whether the Z feedback loop is closed.

    Returns a dict — never raises, because a verification step that can itself
    explode is not a safety net:

        {"on": bool | None,      # None = we could NOT determine it
         "verified": bool,       # did the read actually succeed?
         "matches": bool | None, # None when expect is None or we couldn't read
         "error": str | None,
         "record": <NanonisRecord>,
         "waited_s": float,      # how long we polled
         "switch_off_delay_s": float | None,
         "module_status": int | None}

    **``on=None`` is not ``on=False``.** A caller that is about to do something only
    safe with the loop open must treat "could not determine" as "assume it is
    CLOSED" — the unsafe state — and refuse. That is the whole point: the failure
    of the measurement chain must never be the trigger for the dangerous action.
    (The same fail-closed rule already saved TryEngageController from recommending
    a blind coarse approach when every current read failed.)

    ``settle=True`` (the default): if the first read disagrees with *expect*, do
    NOT conclude failure — poll within the rig's own switch-off-delay budget
    until it agrees or the budget runs out. See the module note above for why an
    immediate read is wrong. Defaulting to on is deliberate: a call site that
    forgets the argument should get the safe behaviour, not fall back into the
    2026-07 race.

    **Timing out still fails closed.** Settling only removes "we read too early"
    as a cause of failure; it never upgrades "could not read" into "read".
    """
    import time

    def _read():
        rec = context.safe_call("ZCtrl_OnOffGet")
        if rec.error:
            logger.warning("verify_z_controller: ZCtrl_OnOffGet failed: %s", rec.error)
            return None, False, str(rec.error), rec
        raw = _scalar(getattr(rec, "return_value", None))
        if raw is None:
            return None, False, "ZCtrl_OnOffGet returned nothing parseable", rec
        # ZCtrl_OnOffGet: 0 = off, 1 = on.
        return bool(int(raw)), True, None, rec

    t0 = time.monotonic()
    on, verified, err, rec = _read()

    delay = None
    if settle and expect is not None and (not verified or on != bool(expect)):
        delay = _switch_off_delay_s(context)
        if timeout_s is not None:
            budget = timeout_s
        elif delay is not None:
            budget = min(float(delay) + 0.2, _SETTLE_CEILING_S)
        else:
            budget = _SETTLE_FALLBACK_S
        abort = getattr(context, "check_abort", None)
        while (time.monotonic() - t0) < budget:
            # Abort leaves the LOOP, not the conclusion — being cancelled must
            # never be reported as "confirmed off".
            if callable(abort) and abort():
                break
            time.sleep(_SETTLE_POLL_S)
            on, verified, err, rec = _read()
            if verified and on == bool(expect):
                break

    matches = None if (expect is None or not verified) else (on == bool(expect))
    return {"on": on, "verified": verified, "matches": matches, "error": err,
            "record": rec, "waited_s": time.monotonic() - t0,
            "switch_off_delay_s": delay,
            "module_status": _module_status(context) if not matches else None}


def z_off_or_reason(context) -> tuple[bool, str | None, Any]:
    """'Is the Z loop REALLY open?' — the fail-closed form.

    Returns ``(is_off, reason_if_not, record)``. ``is_off`` is True ONLY when the
    real-time controller was successfully read AND reports the loop open. Anything
    else — a failed read, an unparseable answer, the loop still closed — returns
    False with a reason you can put in front of a human.

    Use this to guard anything that is only safe with the feedback open: a coarse
    approach, a gain change, a preamp switch, a bias sweep that must not be chased.
    """
    v = verify_z_controller(context, expect=False)
    if not v["verified"]:
        return False, (
            f"无法确认 Z 反馈是否已断开（ZCtrl_OnOffGet 读取失败：{v['error']}）。"
            "读不到就当它还闭着——反馈环还在的时候做粗进针/换增益就是撞针。"
        ), v["record"]
    if v["on"]:
        return False, (
            "Z 反馈环仍然闭合（实时控制器回报 ON）。"
            "注意：Z-Controller 模块可能已显示 Off——两者不是一回事，"
            "Nanonis 手册要求以实时控制器为准。"
        ), v["record"]
    return True, None, v["record"]


#: Relative tolerance for "the hardware took the number I sent it".
#
# Nanonis packs these over TCP as float32, so a float64 3e-12 comes back as
# 2.9999999880125916e-12 — a relative error of ~4e-9. 1e-3 sits six decades above
# that round-trip noise and still catches any real corruption: the failure mode we
# are guarding against is off by 1e12, not by 0.1%.
READBACK_REL_TOL = 1e-3


def values_match(
    requested: object,
    actual: object,
    *,
    rel_tol: float = READBACK_REL_TOL,
) -> "tuple[bool, str]":
    """Did the instrument take the number we sent it? Decided HERE, in Python.

    THIS COMPARISON MUST NOT BE DELEGATED TO THE MODEL
    ==================================================
    On 2026-08-03 an agent wrote a Z gain, read it back, and was shown both
    numbers::

        requested  3e-12
        read back  3.0

    It reported: "p_gain = 3.0（= 3e-12）… 偏差都在 float32 表示精度范围内,
    数值通道工作正常." It had walked them item by item and still called them equal.

    That is not a lapse of attention, it is structural: the same weakness that
    drops an exponent on the way out reads the dropped exponent back as intact on
    the way in. Two errors, one cause, pointing the same direction — so they
    corroborate instead of cancelling, and the result is a self-consistent report
    that the channel is healthy while the hardware holds a gain of three metres.

    So the verdict is a boolean computed by this function, and a mismatch is a
    skill FAILURE. Never put the two numbers in a tool result and ask whether they
    agree — the one being asked is the one that got it wrong.

    Returns:
        ``(ok, detail)``. ``detail`` is empty when ok; otherwise it states both
        values and the ratio between them, so the size of the discrepancy is
        explicit rather than something to be re-derived.
    """
    try:
        want = float(requested)  # type: ignore[arg-type]
        got = float(actual)      # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False, f"无法比较: 请求 {requested!r}, 读回 {actual!r}"
    if math.isnan(want) or math.isnan(got):
        return False, f"请求 {requested!r}, 读回 {actual!r}(NaN 不可比较)"
    if math.isclose(want, got, rel_tol=rel_tol, abs_tol=0.0):
        return True, ""
    if want == 0.0:
        return False, f"请求 0, 读回 {got!r}"
    ratio = got / want
    return False, f"请求 {want!r}, 读回 {got!r}(相差 {ratio:.3g} 倍)"


#: 收尾回读的四个量 → ``(Nanonis 方法, 结果字段)``。
#:
#: **不走 ``InstrumentState.refresh()``**,尽管那里已经读了同样这几个寄存器。理由是
#: 它的失败语义与这里要的正好相反:refresh 在任何一个读失败时会把**上一次缓存的值
#: 结转过来**(``_PATCHABLE_FIELDS`` 那个循环),而 ``stale`` 只在**一个核心读都没
#: 成功**时才置位 —— 于是「只有 Bias_Get 失败」这种情况会把几分钟前的偏压当成刚读
#: 到的值交出来,``stale`` 还是 False。那正是本模块存在的理由:结转值与实测值长得
#: 一模一样,而这里唯一的产品就是「这个数是不是我刚量到的」。
#:
#: 所以这里逐个直读,读不到就是 ``None`` + 一句为什么,永不结转。
#: 形状是 ``(字段, 动词, thunk)`` 而不是 ``(字段, 动词)`` + ``safe_call(动词)``:
#: 动词必须**字面量地**待在 ``safe_call(...)`` 里,否则本仓所有靠 grep 找 Nanonis
#: 调用的工具(中止策略检查、安全审计、API 覆盖率普查)全都看不见这几个读 ——
#: 那正是 ``test_safe_call_verbs_are_literal`` 钉的东西,它已经抓到过两次表驱动循环。
#: 动词在这里出现两次(一次给人看的错误话,一次给 grep),两者一致由
#: ``test_forge_report_is_a_reading_not_a_plan`` 里那条测试盯着。
_JUNCTION_READS: tuple[tuple[str, str, Any], ...] = (
    ("bias_v", "Bias_Get", lambda c: c.safe_call("Bias_Get")),
    ("setpoint_a", "ZCtrl_SetpntGet", lambda c: c.safe_call("ZCtrl_SetpntGet")),
    ("current_a", "Current_Get", lambda c: c.safe_call("Current_Get")),
    ("z_m", "ZCtrl_ZPosGet", lambda c: c.safe_call("ZCtrl_ZPosGet")),
)


def _read_one(context, method: str, thunk) -> "tuple[float | None, str | None]":
    """``(值, 读不到的原因)``。两者恰有一个是 None。"""
    try:
        rec = thunk(context)
    except Exception as exc:  # noqa: BLE001 — 回读永远不许把调用方炸掉
        return None, f"{method} 抛异常:{type(exc).__name__}: {exc}"
    err = getattr(rec, "error", None)
    if err:
        return None, f"{method} 报错:{err}"
    raw = _scalar(getattr(rec, "return_value", None))
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, (f"{method} 回包读不懂(repr 前 80 字:"
                      f"{str(getattr(rec, 'return_value', None))[:80]})")
    if math.isnan(val) or math.isinf(val):
        return None, f"{method} 读回 {val!r}"
    return val, None


def read_junction_state(context) -> dict:
    """**仪器现在到底是什么样** —— 一次实测回读,不是任何人的计划。

    收尾状态必须来自回读，不能使用流程计划值；入口即被拒绝时也同样如此。

    返回(所有量纲为 SI,读不到一律 ``None``)::

        {"bias_v", "setpoint_a", "current_a", "z_m": float | None,
         "feedback_on": bool | None,          # 实时控制器,不是模块
         "z_controller_status": str | None,   # 模块状态原词(GUI 显示的那个)
         "unreadable": {字段: 为什么读不到},
         "any_read": bool,                    # 有没有任何一个量真的读到了
         "read_at": float, "read_at_iso": str,  # 快照拍于何时
         "span_s": float}                     # 这一组读之间的跨度(不是同时读的)

    **读数必须连时刻一起交出去。** 一个真的、只是过期了的数,比读不到更难发现 ——
    调用方渲染时要说「截至 X 的读数」,不要说「现在是」。

    **``None`` 不是 0,也不是 False。** 调用方渲染时必须把 ``unreadable`` 里的话说
    出来,不许拿计划值顶替 —— 拿计划值顶替正是这个函数要终结的那件事。

    ``feedback_on`` 走 :func:`verify_z_controller`(实时控制器 ``ZCtrl_OnOffGet``),
    与全仓同一个判据,``settle=False``:这是收尾陈述,不是在等某个状态到达。

    永不抛异常。读全部失败时返回的是一份**全 None + 四条原因**的表,而不是一个
    看起来正常的零值表。中止路径上照样可用:``ExecutionContext.safe_call`` 放行
    一切读取,只拦写入。
    """
    t0 = _time.time()
    out: dict = {"bias_v": None, "setpoint_a": None, "current_a": None,
                 "z_m": None, "feedback_on": None, "z_controller_status": None,
                 "unreadable": {}}
    for field, method, thunk in _JUNCTION_READS:
        val, why = _read_one(context, method, thunk)
        out[field] = val
        if why:
            out["unreadable"][field] = why

    # 模块状态搭 ``verify_z_controller`` 的顺风车:``expect=None`` 时 ``matches``
    # 是 None,而它正是在 ``not matches`` 的分支里读 ``ZCtrl_StatusGet`` 的 ——
    # 所以这里再单独读一次就是同一个寄存器读两遍。
    code = None
    try:
        v = verify_z_controller(context, expect=None, settle=False)
        out["feedback_on"] = v["on"]
        code = v.get("module_status")
        if not v["verified"]:
            out["unreadable"]["feedback_on"] = (
                v["error"] or "ZCtrl_OnOffGet 没有可解析的回包")
    except Exception as exc:  # noqa: BLE001
        out["unreadable"]["feedback_on"] = (
            f"verify_z_controller 抛异常:{type(exc).__name__}: {exc}")

    if code is None:
        out["unreadable"]["z_controller_status"] = "ZCtrl_StatusGet 读不到"
    else:
        out["z_controller_status"] = _MODULE_STATUS.get(int(code),
                                                        f"Unknown({code})")

    out["any_read"] = any(out[k] is not None for k in
                          ("bias_v", "setpoint_a", "current_a", "z_m",
                           "feedback_on", "z_controller_status"))

    # 状态快照必须附带读取时刻，不能把旧读数称作当前状态。
    # span_s 表示这组顺序读取的跨度；发生重试或延迟时，结果可能不属于同一时刻，
    # 必须向调用方说明，不能伪装成同时采集的完整状态。
    out["read_at"] = t0
    out["span_s"] = max(0.0, _time.time() - t0)
    try:
        out["read_at_iso"] = _time.strftime("%H:%M:%S", _time.localtime(t0))
    except Exception:  # noqa: BLE001 — 时间格式化不该毁掉一次成功的回读
        out["read_at_iso"] = ""
    return out


__all__ = [
    "verify_z_controller",
    "z_off_or_reason",
    "read_junction_state",
    "values_match",
    "READBACK_REL_TOL",
]
