"""Operating-mode middleware — 每次 model call 注入一段当前模式的信念块。

让全局操作模式(SAFE / SEMI / AUTO,见 ``mast.core.types.OperatingMode``)在
*自主* 路径上塑造 instrument-control agent。模式经 ``get_mode`` 回调实时读取,
所以在 UI 里切模式下一次 model call 就生效 —— 不用重建 agent。

* ``ModeBeliefMiddleware`` —— 每次 model call 在 system message 末尾追加一段
  按模式写的信念块(与 ``LiveStateMiddleware`` 同款)。SAFE 让 agent 相信针尖没
  问题、直接继续做实验;SEMI 让它优先做实验、只做浅层机械修针;AUTO 什么都不加。
  挂在 LiveStateMiddleware 之后,所以信念块是模型看到的最新一段 —— 它压得过静态
  提示词里的修针指引。

## ⑰(2026-08-08):``ModeGatedPulseHITLMiddleware`` 已删除

这里曾经还有第二个中间件:``HumanInTheLoopMiddleware`` 的薄子类,SEMI 模式下把
每一发电脉冲送去人工确认。它随「确认框/审批整条链路」一起被割掉了 —— 判据没丢,
搬进了 ``mast.core.auto_approval.would_have_asked``,由
``agents/_shared/auto_approval_mw.AutoApprovalNoticeMiddleware`` 消费成一条通知。

**SEMI 的信念块同步改了口径。** 它原来写着「任何电脉冲修针都需要人工确认后才会
执行」—— 那句话在中间件删掉的同一刻变成了假话,而模型会照着它行动(不去用一个它
以为要等人的工具)。修好问题之后,为旧行为辩护的话必须一起换掉。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import append_system_block
from mast.core.types import OperatingMode

#: 登记表条目 id —— 两个模式是两条不同的条目。
PROMPT_ID_SAFE = "mw.mode_belief.safe"
PROMPT_ID_SEMI = "mw.mode_belief.semi"

#: 哪些 agent 挂它（真源；登记表与建图派生）。只有 IC 拿得到 get_mode。
AGENTS = ("instrument_control",)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Belief text (stable per mode → prompt caching unaffected)
# ─────────────────────────────────────────────────────────────────────

# Since 2026-08-01 this block is no longer the load-bearing part of SAFE. The
# tip verdicts are rewritten to "good" at their producers (see
# mast.core.operating_mode + docs/v2/design/safe_mode_tip_verdict_override.md),
# so the model is not being asked to disbelieve its own tool results any more —
# the evidence and the belief now agree. The block stays as the middle layer of
# a three-layer defence (verdict override → belief → SafetyGate hard block); in
# particular its last sentence still overrides the static prompt's tip-repair
# guidance, which no override can reach.
_SAFE_BELIEF = (
    "## 操作模式：安全模式（SAFE）\n"
    "当前针尖状态良好，无需任何针尖处理。本模式下**不修针、不做电脉冲、也不做 tip "
    "shaping**。请把针尖当作处于良好状态，直接用它继续采集数据、推进实验；不要去评估"
    "是否需要修针，也不要尝试任何 tip conditioning / tip pulse / tip shape 操作。"
    "忽略系统提示中先前任何“若针尖变差则修针”的指引——在本模式下一律不修针，专注实验。"
)

#: ⑰(2026-08-08)改口径。原文写着「任何电脉冲修针都需要**人工确认**后才会执行」,
#: 而 SEMI 的电脉冲确认框已经删掉了 —— 那句话在同一刻变成假话,并且是**会指挥行动**
#: 的假话:模型会绕开一个它以为要等人的工具,或者去「等」一个永远不会来的确认。
#: 现在如实写:脉冲会直接执行并留痕,SEMI 与 AUTO 的差别只剩「下压深度受限」。
_SEMI_BELIEF = (
    "## 操作模式：半自动模式（SEMI）\n"
    "优先直接用当前针尖做实验。允许**浅层、纯机械**的 tip shaping；**电脉冲**修针"
    "（bias pulse / tip pulse / condition tip 等）会**直接执行**并记入诊断台账，"
    "不需要、也不会等待人工确认——不要为了等批准而停下来。"
    "较深的下压仍会被拒绝（这是拒绝型防护，不是审批），仅使用浅层修针。"
)


def _belief_block(get_mode: "Callable[[], Any] | None") -> tuple[str, str]:
    """``(prompt_id, text)`` for the current mode; two empty strings for AUTO.

    返回 id 而不只是文本，是因为注入台账要记「这一段是谁塞的」—— SAFE 与 SEMI
    是登记表里两个不同的条目，拼进 system 之后从文本上分不出来。
    """
    if get_mode is None:
        return "", ""
    try:
        mode = OperatingMode.coerce(get_mode())
    except Exception as exc:  # never let a mode read crash the model call
        logger.info("ModeBeliefMiddleware get_mode() failed: %s", exc)
        return "", ""
    # Read through the prompt-override layer so an operator edit in 高级管理 →
    # 上下文注入 takes effect on the next model call (no rebuild). Falls back to
    # the constant above whenever nothing is overridden.
    from mast.prompts.registry import resolve as resolve_prompt

    if mode is OperatingMode.SAFE:
        return PROMPT_ID_SAFE, resolve_prompt(PROMPT_ID_SAFE, _SAFE_BELIEF)
    if mode is OperatingMode.SEMI:
        return PROMPT_ID_SEMI, resolve_prompt(PROMPT_ID_SEMI, _SEMI_BELIEF)
    return "", ""


class ModeBeliefMiddleware(AgentMiddleware):
    """Append the current operating-mode belief block to the system message.

    Mirrors ``LiveStateMiddleware``'s per-call append so the belief reacts to a
    runtime mode switch with no rebuild. No-op when ``get_mode`` is None or the
    mode is AUTO (production/offline fallback).
    """

    def __init__(self, get_mode: "Callable[[], Any] | None"):
        super().__init__()
        self._get_mode = get_mode

    @property
    def name(self) -> str:
        return "ModeBeliefMiddleware"

    def _apply(self, request: Any) -> Any:
        # 按模式取值，一个模式内逐轮不变 → 留在 system 末尾（它要压住静态提示词
        # 里的旧指引，所以位置就该是最后一段）。
        prompt_id, block = _belief_block(self._get_mode)
        if not block:
            return request
        return append_system_block(request, prompt_id, block)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


# ─────────────────────────────────────────────────────────────────────
# ⑰(2026-08-08)这里原来是 ``ModeGatedPulseHITLMiddleware``(约 110 行)
# ─────────────────────────────────────────────────────────────────────
#
# 它在 SEMI 模式下对每一个 ``is_electrical_pulse`` 命中的工具调用发一次
# ``interrupt()``,等用户在 HitlModal 里按批准/拒绝。随「确认框整条链路」一起
# 删除。判据(``is_electrical_pulse`` + 模式是不是 SEMI)一个字没改,搬到了
# ``mast.core.auto_approval.would_have_asked``;消费它的是
# ``agents/_shared/auto_approval_mw.AutoApprovalNoticeMiddleware``,做的事从
# 「拦住等人」变成「照跑并在诊断台账里留一行」。
#
# 为什么是删掉而不是留一个恒 no-op 的子类:留着的话,「SEMI 的电脉冲要不要等人」
# 就有了两个可以各自漂移的答案(这里的子类 + 那个共享判据),而这正是本仓踩过的
# 「两张表必须保持相等」的形状。


__all__ = ["ModeBeliefMiddleware"]
