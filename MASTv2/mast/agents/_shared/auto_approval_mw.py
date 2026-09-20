"""审批 → 提醒:本来会弹框等人的工具调用,现在**执行 + 留痕 + 通知**(⑰,2026-08-08)。

这个中间件占了两个前任的位置:

* ``HumanInTheLoopMiddleware(interrupt_on=_derive_hitl_map(registry))`` ——
  每个 ``safety_level=DANGEROUS`` 的技能一次 ``interrupt()``,等用户在审批面板
  按批准/编辑/拒绝;
* ``ModeGatedPulseHITLMiddleware`` —— SEMI 模式下每一发电脉冲一次 ``interrupt()``。

两者现在都不再产生 ``interrupt()``。判据本身没删也没放宽 —— 它整个搬进了
``mast.core.auto_approval.would_have_asked``,由这里、``skill_adapter``(写
``approvals`` 审计行)和 ``core/executor`` + ``core/execution_context``(原来在那里
拒绝并让人去审批面板)共用同一份答案。

## 为什么是 ``wrap_tool_call`` 而不是 ``after_model``

两个前任都挂在 ``after_model``:那是**能改写工具调用**的位置(批准/编辑/拒绝要能
把 tool_call 换掉或删掉)。现在没有任何东西要改写,只要在调用真正发生的那一刻说
一句话 —— ``wrap_tool_call`` 拿得到 ``request.tool``(带 ``skill_metadata``)和这次
调用的实参,而 ``after_model`` 只拿得到名字和参数、拿不到 metadata,还得自己维护一
张 ``caps_by_name`` 快照(前任就是这么做的,而那张快照在 graph 构建时冻结)。

## 顺序:必须挂在 SafetyGateMiddleware **里面**

它说的是「不再等批准,直接执行」。如果 SafetyGate 随后把这次调用拒了,那句话就成了
假话。中间件链里先列的在外层,所以这条排在 ``SafetyGateMiddleware`` 之后。

**注意这不是完备保证**:更内层(技能自己的 ``validate_params``、针尖包络、qPlus 软
门)仍可能拒绝,那时通知说的「直接执行」比事实早了一步。可接受,因为每一次拒绝
都会在同一本台账里留下自己那一行(``safety_block`` / ``precondition_block`` …),
而真正带审计意义的 ``approvals`` 行是在**技能确实跑完之后**由记录管线写的。

## 要把弹框加回来,需要观测到什么

见 ``mast/core/auto_approval.py`` 顶部:举出一次「批准框拦下了真实损害、而拒绝型
防护接不住」的实例。截至 2026-08-08 零例。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class AutoApprovalNoticeMiddleware(AgentMiddleware):
    """本来要等人批准的工具调用:照跑,并且在台账里留一行。

    Parameters
    ----------
    get_mode:
        实时操作模式读取器。只影响 SEMI 电脉冲那一支判据;``None`` ⇒ 判据自己
        去读全局实时值(``core.operating_mode``),这样这里和 ``skill_adapter``
        得到的是同一个答案。
    extra_names:
        **额外**要通知的技能名(在 ``would_have_asked`` 之外)。生产上由
        ``instrument_control/graph`` 传 ``_derive_hitl_map`` 的键 —— 与判据同源
        (都是 ``safety_level == DANGEROUS``),所以生产上两者必然一致,这里只是
        让 ``build(..., interrupt_on={...})`` 这个参数继续**真的有作用**。

        它只能**加**不能减:漂移的失败模式因此是「多一条通知」,不是「少一条」。
        留着这个口子的理由是 ⑰ 之前 ``interrupt_on`` 就是测试用来在不改生产
        metadata 的前提下演练这条路径的口子;把它变成一个只影响日志的装饰,
        等于在代码里留一个读起来像可调项、调了没反应的死配置。
    """

    def __init__(self, get_mode: "Callable[[], Any] | None" = None,
                 extra_names: "set[str] | frozenset[str] | None" = None):
        super().__init__()
        self._get_mode = get_mode
        self._extra_names = frozenset(extra_names or ())
        # 生命周期累计数,给测试和遥测。「一次没通知过」和「这条中间件没装上」
        # 从外面看必须不一样。
        self.notified = 0

    @property
    def name(self) -> str:
        return "AutoApprovalNoticeMiddleware"

    # ── internals ────────────────────────────────────────────────────

    def _mode(self) -> Any:
        if self._get_mode is None:
            return None
        try:
            return self._get_mode()
        except Exception:  # noqa: BLE001 — 读模式失败不该影响一次工具调用
            logger.debug("AutoApprovalNotice: get_mode 失败", exc_info=True)
            return None

    def _announce(self, request: Any) -> None:
        """如果这次调用本来会等人批准,发一条通知。永不抛。"""
        try:
            tool = getattr(request, "tool", None)
            meta = (getattr(tool, "metadata", None) or {}).get("skill_metadata")
            if meta is None:
                return  # 不是技能工具 —— handoff / buffer / 分析,从来不过审批
            tool_call = getattr(request, "tool_call", None)
            args = (
                tool_call.get("args", {}) if isinstance(tool_call, dict)
                else getattr(tool_call, "args", {})
            ) or {}
            name = getattr(tool, "name", "") or str(getattr(meta, "name", "") or "")

            from mast.core.auto_approval import notify, would_have_asked

            reason = would_have_asked(meta, tool_name=name, args=args,
                                      mode=self._mode())
            if reason is None and name in self._extra_names:
                reason = f"显式门控名单里的技能({name})"
            if reason is None:
                return
            self.notified += 1
            notify(
                name,
                f"{reason} —— 已直接执行并通知,不再等待人工批准",
                args=args,
                safety_level=getattr(
                    getattr(meta, "safety_level", None), "value", None),
                agent_path="tool_call",
            )
        except Exception:  # noqa: BLE001 — 通知坏了绝不能让工具调用失败
            logger.debug("AutoApprovalNotice: 通知失败(已忽略)", exc_info=True)

    # ── hooks ────────────────────────────────────────────────────────

    def wrap_tool_call(self, request, handler):
        self._announce(request)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        self._announce(request)
        return await handler(request)


__all__ = ["AutoApprovalNoticeMiddleware"]
