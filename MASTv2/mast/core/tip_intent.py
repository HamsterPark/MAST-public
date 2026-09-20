"""「现在是不是在**故意**改针尖」—— 一个判据，若干消费者。

## 这个模块原来要解决的事故（历史，2026-08-05 ⑬）

视觉扫描监控在扫描中途判到针尖突变，会发 CRITICAL ``tip_quality_drop``；
``runtime.make_tip_halt_hook`` 把它变成 run 级 halt；``GraphExecutor`` 在**每个步
边界**消费它并中止整条 composite（``graph_executor.py`` 的 ``_check_halt``）。

这套链路对「针尖自己坏了」是对的，对「我正在把针尖修好」是灾难性的：修针流程本
来就是**故意**改变针尖形貌，而且每扎一次就要扫一张小图确认簇的形状 —— 那张图上
的针尖当然与上一张不同。于是流程在自己造出的证据上被中止，理由还是「视觉判定针
尖状态恶化…修针尖后可重跑」——建议你去修一个你正在修的针尖。

被打断的位置最坏:``poke_phase`` 是「断反馈下压 → 扫图看簇 → 再扎」的循环,在扫
图之后那个步边界停下,意味着一次修针只做了一半。

## ⑰-C1（2026-08-09）：上面那个事故现在**不可能发生了**

定案把**视觉来源的 halt 整条割掉** —— 不只是修针场景，**任何模式、任何场景**
下视觉针尖判定都不再中止 composite（判据 :func:`runtime.tip_halt_source`）。所以
``TIP_WORK_PATTERNS`` 不再是「视觉 halt 的豁免表」。

**但这张表没有作废，它换了消费者**，现在是这三个：

1. ``runtime._tip_work_suppresses_tip_halt``（经 :func:`exempt_during_tip_work`）——
   ⑰-C2：修针期间**电流监控的瞬变类**不中止 composite；
2. ``buffer_hitl.classify_event`` —— 给通知措辞：修针期间的瞬变印成「本职动作的
   签名」，而不是「巨幅瞬变」；
3. ``skills/builtins/tip_forge_selfcheck`` —— 上机前自检锻造技能名有没有被覆盖。

## 判据为什么必须窄

电流监控那半边有一张同类的表(``monitoring.service.SUPPRESS_SKILL_PATTERNS``),
但它回答的是**另一个**问题:「这个技能期间电流看起来暴力是正常的吗」——所以那张
表里有 approach / spectroscopy / sweep / setbias。那些技能**不改针尖形貌**,当年
拿它们来豁免视觉针尖判定就是把保护关掉。两张表都是子串匹配、都对着
``instrument_lock().snapshot()["skill"]``,但语义不同,不合并。

## 豁免的边界(只关一件事)

命中时只吸收「在步边界中止 composite」这一个动作。以下一律不受影响:

* ``current_monitor`` 的**持续类**(饱和 / 冻结)—— 物理安全:针尖被压进表面、
  前放到轨、测量链死了。「正在修针」从来不等于「关掉物理保护」。
* 用户中止与 E_STOP(abort 闩锁,另一条路)。
* 撞针检测与 ``crash_guard``(``tip_crash_tracker``,另一条路)。
* 事件本身的**发射、记录、面板可见、诊断台账**:视觉照样看、照样存、照样进
  ``event_refs`` 与 ``ReadHardwareEvents``。
  （⑰ 之前这里还写着「配方跑完照样在下一次 model call 前弹给用户」——
  那个确认框已经整条割掉，见 ``agents/_shared/buffer_hitl.py``。）

也就是说:证据一条不少,只是不让它在流程半途把流程自己杀掉。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 蓄意改变针尖顶端形貌的技能名子串(小写匹配)。
#:
#: 判据是「这个技能的**目的**就是改针尖」,不是「这个技能可能碰到针尖」。扫描、
#: 进针、扫谱都可能把针尖弄坏,但那是事故不是意图,视觉判定对它们必须保持有效。
TIP_WORK_PATTERNS: frozenset[str] = frozenset({
    # 硬件整形 / 电脉冲(叶子技能)
    "tipshape", "tippulse", "biaspulse",
    # 修针复合技能
    "conditiontip", "shapetip", "preparenobletip",
    # 特殊针尖锻造(本仓 2026-08-02 新增,见 docs/v2/design/special_tip_forging.md)
    "spectroscopytip", "resolutiontip", "biaswiggle",
    # Au(111) 全流程修针外环(2026-08-05,见 docs/v2/design/au111_tip_forge_
    # uninterrupted.md)。**这一条必须在表里**:令牌记的是最外层技能名,而外环
    # 里每个子步骤(脉冲、扎针、快扫评估、换区、重新进针)都在它名下跑 —— 表里
    # 没有它,内层那些评估图会让视觉判定把整条外环从中间掐断,而那些图正是外环
    # 自己为了看修针效果扫出来的。
    "forgeau",
})


def is_tip_work(skill_name: str) -> bool:
    """*skill_name* 是不是一个蓄意改针尖的技能。"""
    if not skill_name:
        return False
    low = str(skill_name).lower()
    return any(pat in low for pat in TIP_WORK_PATTERNS)


# ─────────────────────────────────────────────────────────────────────
# 电流监控的物理信号:**瞬变 vs 持续**(⑭ 引入,⑰-C2 搬到这里)
# ─────────────────────────────────────────────────────────────────────
#
# ## 这个区分要回答的问题
#
# 「修针期间电流上打出的这个台阶,是**本职动作的签名**,还是**事故**?」
#
# ⑭(2026-08-06 ForgeAuTip 首演)给出的答案:**修针的本职就是制造瞬变** —— 脉冲、
# 扎针、进针,每一下都在电流上打出远超结电流的台阶。首演里 giant_spike 几分钟报一次,
# **零真阳性**,把自动运行打成了值守运行。
#
# 边界是**瞬变 vs 持续**,不是「全部物理类」:
#
# * **瞬变**(giant_spike):一次动作制造的、本来就会自己过去的台阶。修针期间它是本职
#   动作的签名;
# * **持续**(贴轨/饱和、冻结):即使在修针期间也是真事故 —— **修针不该造成持续贴轨**,
#   那说明针已经压进表面出不来了,或者信号链死了。这两类**任何时候都不豁免**。
#
# ## 为什么定义在 core 而不在 buffer_hitl(⑰-C2,2026-08-09)
#
# ⑭ 当时把这两张表写在 ``agents/_shared/buffer_hitl.py`` 里,因为那时的消费者只有
# 确认框。⑰ 把确认框整条割掉之后,``buffer_hitl`` 只拿它**给通知措辞**,而真正会
# **停下仪器**的消费者是 ``runtime.make_tip_halt_hook`` —— 让「会停仪器的那一方」去
# 一个「只写措辞的模块」里 import 判据,方向是反的。所以定义搬到这里,
# ``buffer_hitl`` 改成 re-export。
#
# ## ⑭ 只改了一半,⑰-C2 补上另一半
#
# ⑭ 把瞬变类从**确认框**里豁免了,却没有从 **halt** 里豁免:
# ``runtime._tip_work_suppresses_tip_halt()`` 里有一条「``source=current_monitor``
# 直接返回空」的分支,把整个电流监控来源一刀切掉了。后果正是用户在首演里挨的那一下
# —— 弹框不弹了,外环照样被从中间掐断。
#
# 这不是新决定,是同一个决定的另一半。
#
# ## 要把 halt 豁免**反转回来**,需要观测到什么
#
# 具体一条:**举出一次实例 —— 修针期间的一个瞬变类事件,确实预示了事故,而且持续类
# (贴轨/冻结)、撞针状态机、针尖包络都没有接住它。** 截至 2026-08-09 这样的实例
# **一个都没有被观测到**(⑭ 记的是「N 次弹窗零真阳性」)——「至今没被观测到」,
# 不是「不可能存在」。真出现了,把它记在这里再讨论。

#: 物理类电流信号的兜底表。真源是告警引擎(见 :func:`physical_current_signals`),
#: 这里的字面量只在 import 失败时顶上。
_PHYSICAL_SIGNAL_FALLBACK: frozenset[str] = frozenset({
    "current_saturation",     # 前放到轨 —— 针被压进表面
    "current_freeze",         # 测量链死了
    "current_giant_spike",    # 远超结电流能做出的台阶
})

#: **瞬变类** —— 一次动作制造的、本来就会自己过去的台阶。修针期间豁免的**唯一**一组。
TRANSIENT_PHYSICAL_SIGNALS: frozenset[str] = frozenset({
    "current_giant_spike",
})

#: **持续类** —— 任何时候都不豁免。修针不该造成持续贴轨。
SUSTAINED_PHYSICAL_SIGNALS: frozenset[str] = frozenset({
    "current_saturation",
    "current_freeze",
})


def physical_current_signals() -> frozenset[str]:
    """「越过了物理界限」而不是「针尖看起来变差了」的那些信号。

    并集,永远不是替换:派生集权威且**只增不减**。上游改名最坏退化成「老三样仍被
    认成物理类」,而不是「什么都不再被认成物理类」。
    """
    try:
        from mast.monitoring.alerts import critical_signals

        return frozenset(critical_signals()) | _PHYSICAL_SIGNAL_FALLBACK
    except Exception:  # noqa: BLE001 — 分类器在热路径上,绝不抛
        return _PHYSICAL_SIGNAL_FALLBACK


def is_transient_physical(signal: str) -> bool:
    """这个信号是不是**显式归进瞬变类**的那一组。

    白名单不是黑名单:明天新加一条判据如果没被归进瞬变类,它默认**不豁免** ——
    不该仅仅因为没人记得排除它,就获得了在修针期间沉默的权利。
    """
    return str(signal or "") in TRANSIENT_PHYSICAL_SIGNALS


def exempt_during_tip_work(signal: str) -> bool:
    """这个物理信号此刻是否属于「蓄意修针的本职动作签名」。

    两个条件都要:① 信号显式归进瞬变类;② 此刻确实有修针技能持着仪器令牌。

    ``active_tip_work()`` 读不到令牌时返回 ``""`` ⇒ 不豁免 —— 豁免的失败模式必须是
    「保护照常生效」,而不是「因为看不清所以放行」。永不抛。
    """
    if not is_transient_physical(signal):
        return False
    try:
        return bool(active_tip_work())
    except Exception:  # noqa: BLE001 — 调用方多半在事件发布线程上,绝不能抛
        logger.debug("读不到修针意图(不豁免)", exc_info=True)
        return False


def active_tip_work() -> str:
    """正在跑的蓄意修针技能名;没有就是 ``""``。

    读的是仪器令牌的持有者 —— composite 的子步骤不会改写它(令牌是可重入的,
    ``_skill`` 只在最外层那次 acquire 时写),所以 ``PokeConditionTip`` 内部那张
    ``ScanAt`` 期间,这里返回的仍然是 ``PokeConditionTip``。这正是要的:那张图是
    修针流程的一部分。

    **读不到一律返回 ``""``**(= 不豁免)。豁免的失败模式必须是「保护照常生效」,
    而不是「因为看不清所以放行」。
    """
    try:
        from mast.core.instrument_lock import instrument_lock

        holder = instrument_lock().snapshot() or {}
        skill = str(holder.get("skill") or "")
    except Exception:  # noqa: BLE001 — 调用方多半在事件发布线程上，绝不能抛
        logger.debug("读不到仪器令牌持有者(不豁免)", exc_info=True)
        return ""
    return skill if is_tip_work(skill) else ""


__all__ = [
    "SUSTAINED_PHYSICAL_SIGNALS",
    "TIP_WORK_PATTERNS",
    "TRANSIENT_PHYSICAL_SIGNALS",
    "active_tip_work",
    "exempt_during_tip_work",
    "is_transient_physical",
    "is_tip_work",
    "physical_current_signals",
]
