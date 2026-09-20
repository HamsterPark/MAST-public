"""「现在真正在生效的安全限值是多少」—— 一个真源，两个路由共用。

写入接口能返回 `{"ok":true,"reloaded":true}`，而 `GET /api/safety/limits`
依旧报出厂默认的 ±1.5 µm —— 两处各自独立的失真：

* ``routes/safety.py`` 里 ``get_safety_limits()`` 直接 ``return SafetyLimits()``
  —— 返回的是**代码常量**，从来没看过覆写层。它的模块 docstring 甚至写着
  「Phase 2 returns the defaults … 等 Phase 3 管理员覆写落地后再折进来」。
  Phase 3 早就落地了，这一句没人回来改，于是一句 TODO 变成了一句假话。
* ``routes/admin.py`` 的写入响应里 ``reloaded=True`` 是**字面量**，不是回报。
  而 ``ConfigOverrideRegistry.register_reload_hook`` 在整个生产代码里**零订阅
  者** —— ``signal_reload()`` 是打进一个空列表的。override_store 自己的
  docstring 已经写明了这件事该怎么处理：「consumers that have NOT registered a
  hook … callers whose change targets such a consumer should tell the user
  『重启后生效』」。调用方做的恰好相反。

两者叠加的后果不是「少显示了一个数」，而是**用户拿不到任何一条能反驳「已生效」
的证据**：改完看一眼限值接口，看到旧值，无从判断是自己写错了、还是接口在骗人。

真正在生效的那份限值住在运行中的 ``SafetyGuard`` 实例里（``SafetyGuard.__init__``
在构造时就把覆写合并进 ``_limits`` 定住了）。所以「在生效的是什么」这个问题只能
向那个活对象要答案，不能靠重算一遍覆写合并来**推断** —— 推断出来的值恰好就是
重启之后才会成立的那个值，那是另一句形式不同的假话。

用法：

    limits, source = in_force_safety_limits(ctx)
    # source ∈ {"live_guard", "merged", "defaults"}

``pending_restart(ctx)`` 回答的是另一个问题：磁盘上已持久化的覆写，与进程里当前
真正在用的，是不是已经不一样了。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# in_force_safety_limits() 的 source 取值。写成常量是为了让路由与测试引用同一份
# 字面量 —— 这三个字符串会出现在 API 响应里，拼错一个字母就是一个静默的契约漂移。
SOURCE_LIVE = "live_guard"   # 取自运行中的 SafetyGuard —— 这才是真正在拦参数的那份
SOURCE_MERGED = "merged"     # 没有活 guard；代码默认值 + 磁盘覆写现算的
SOURCE_DEFAULTS = "defaults"  # 覆写层也读不到，退回代码常量


def live_safety_guard(ctx: Any) -> Any | None:
    """运行中的 SafetyGuard —— 真正在拦参数的那一个，取不到返回 None。

    ``runtime.setup()`` 建的 ``rt._safety`` 是手动执行路径（SkillExecutor）用的
    那一份；``bootstrap`` 已把 ``ctx.runtime`` 接好。任何一步取不到都返回 None，
    调用方退回下一档 —— 这个函数只用来**报告**，绝不能让它把请求打挂。
    """
    for holder in (getattr(ctx, "runtime", None), getattr(ctx, "live_app", None),
                   getattr(ctx, "app", None)):
        guard = getattr(holder, "_safety", None) if holder is not None else None
        if guard is not None and getattr(guard, "_limits", None) is not None:
            return guard
    return None


def merged_safety_limits() -> Any:
    """代码默认值 + 磁盘上的管理员覆写，现算一份。

    与 ``core.safety._get_effective_limits`` 同一套合并语义（同一个
    ``get_safety_limits()`` 覆写读取口），失败一律退回代码默认值并记 WARNING ——
    一个坏掉的覆写文件绝不能让「限值是多少」这个问题变成 500。
    """
    from mast.config import SafetyLimits

    # 直接复用手动执行路径那一份 —— 合并语义与收紧顺序**只有一处实现**。
    # 这里曾经是第二份抄写：`pending_restart` 拿这份值去和活 guard 比，两边算法
    # 一旦漂开就永远不相等，「要重启」这个提示恒亮，接着没人再看它
    # （KNOWN_ISSUES §1.1 的教训）。抄写就是漂移的种子，所以不留第二份。
    try:
        from mast.core.safety import _get_effective_limits

        return _get_effective_limits(SafetyLimits())
    except Exception as exc:  # noqa: BLE001 — 报告用途，永不因此失败
        logger.warning("merged_safety_limits: 合并失败，退回代码默认值: %s", exc)
        return SafetyLimits()


def in_force_safety_limits(ctx: Any) -> tuple[Any, str]:
    """(此刻真正在生效的 SafetyLimits, 来源标签)。

    三档，从「最接近事实」往下退：活 guard → 现算的合并值 → 代码默认值。
    """
    guard = live_safety_guard(ctx)
    if guard is not None:
        return guard._limits, SOURCE_LIVE
    merged = merged_safety_limits()
    try:
        from mast.config import SafetyLimits

        return merged, (SOURCE_MERGED if merged != SafetyLimits() else SOURCE_DEFAULTS)
    except Exception:  # noqa: BLE001
        return merged, SOURCE_MERGED


def pending_restart(ctx: Any) -> bool | None:
    """已持久化的覆写是否还没进到进程里（True＝要重启才生效）。

    None ＝ 判断不了（拿不到活 guard），此时不能报 False —— 「不知道」和「已生效」
    是两回事，把前者说成后者正是这一整条链最初出错的方式。
    """
    guard = live_safety_guard(ctx)
    if guard is None:
        return None
    return guard._limits != merged_safety_limits()
