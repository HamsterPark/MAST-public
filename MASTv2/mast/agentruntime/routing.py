"""路由 —— 把 ``llm_route`` 接成编排器要的 :class:`RouteDecision`。

为什么**不**在这里重写一遍
--------------------------
``agents/_shared/llm_route.py``（138 行）是一次 provider 可移植性修复的
沉淀。它里面每一条约束都对应一家 provider 的一种挂法，全仓测过：

* ``with_structured_output`` **必须** 用 ``method="function_calling"`` 且喂扁平化的
  纯文本消息 —— ``response_format`` / prefill / 重放的 tool id 各自会打断一家；
* 文本降级的提示里**必须**出现字面量 "json"（qwen 要它），且**必须**以 user 轮结尾
  （sonnet 不支持 assistant prefill）；
* 裸名字兜底只在**恰好一次**词边界命中时生效 —— 含糊的散文必须返回 None，
  **绝不猜一个 agent**。

这类问题的形状值得记住：全量测试在默认的 Kimi 上路由完全正常，换任何一家其它
provider，整个多智能体系统就**静默地路由到 ``__end__``**（或 400），一个 agent 都
没派出去。所以这个模块只做适配，不碰那 138 行——重写等于把六家五样的坑重踩一遍。

这里做的三件事
--------------
1. 组装路由消息（父转录 + 可用 agent + 已有产物）；
2. 调 ``route_decision``，把 ``{next_agent, reason}`` 变成 :class:`RouteDecision`；
3. **失败时结束，而不是猜**。路由不出来就把控制权还给用户——一个猜出来的目标会
   驱动真实仪器。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Sequence

from mast.agentruntime.context import RunContext
from mast.agentruntime.orchestrator import RouteDecision
from mast.agentruntime.state import TaskState

logger = logging.getLogger(__name__)

#: 结束的伪目标。路由器说它 = 这一轮到此为止。
END = "__end__"

#: 文本降级那一轮的提示。**必须含字面量 "json"**（qwen 会因为没有它而拒绝），
#: 并且整条消息以 user 轮结尾（sonnet 不支持 assistant prefill）。改这句话之前
#: 先读 ``llm_route`` 的模块 docstring。
_JSON_HINT = (
    '只回一个 json 对象，不要任何别的文字：'
    '{"next_agent": "<目标名>", "reason": "<一句话理由>"}。'
    'next_agent 必须是给定名单里的一个，或者 "__end__" 表示任务已经完成。')


def build_routing_messages(state: TaskState, targets: Sequence[str], *,
                           system_prompt: str = "") -> list[dict]:
    """路由器看到的东西：系统提示 + 已有产物 + 父转录。

    **产物要给路由器看。** 旧世界里 supervisor 是全系统唯一看不见产物的角色——它是
    裸函数，不走中间件栈，而产物注入是中间件做的。于是「每个阶段只走一遍」这类
    路由提示只能是行为性猜测，因为它没法查。这里它查得到。
    """
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if state.artifacts:
        have = "、".join(sorted(state.artifacts))
        msgs.append({"role": "system",
                     "content": f"目前已经产出：{have}。已经在那里的东西不要重复派人做"
                                "——这是读数，不是推测。"})
    if state.visit_count:
        been = "、".join(f"{k}×{v}" for k, v in sorted(state.visit_count.items())
                        if k != "supervisor")
        if been:
            msgs.append({"role": "system", "content": f"各 agent 已被进入：{been}。"})
    msgs.extend(state.messages)
    msgs.append({"role": "user",
                 "content": f"可派发的目标：{'、'.join(targets)}，或 {END} 结束。"})
    return msgs


def make_llm_router(model: Any, targets: Sequence[str], *,
                    system_prompt: str = "",
                    schema: Any = None,
                    ) -> Callable[[TaskState, RunContext], RouteDecision]:
    """造一个供 :class:`~mast.agentruntime.orchestrator.OrchestratorLoop` 用的路由函数。

    ``model`` 是 langchain 的 chat model（一期），因为 ``route_decision`` 要用它的
    ``with_structured_output``——那是**这一层**的实现细节，循环与中间件都不知道它。
    二期换自研 provider client 时，改的也只有这个文件。
    """
    valid = tuple(targets) + (END,)
    route_schema = schema or _default_schema(valid)

    def _route(state: TaskState, ctx: RunContext) -> RouteDecision:
        messages = build_routing_messages(state, targets,
                                          system_prompt=system_prompt)
        try:
            from mast.agents._shared.llm_route import route_decision

            decision, path = route_decision(
                model, messages, schema=route_schema, valid_targets=valid,
                json_hint=_JSON_HINT, field="next_agent")
        except Exception as exc:  # noqa: BLE001
            # ★ 路由不出来就**结束**，不猜。一个猜出来的目标会驱动真实仪器。
            logger.warning("routing failed (%s); ending the run rather than "
                           "guessing a target", exc)
            return RouteDecision(targets=[],
                                 reason=f"路由无法解析（{type(exc).__name__}），"
                                        "已停下等用户。")

        target = str(decision.get("next_agent") or "").strip()
        reason = str(decision.get("reason") or "").strip()
        logger.info("route → %s (%s) via %s", target or END, reason, path)
        if not target or target == END:
            return RouteDecision(targets=[], reason=reason or "任务已完成。")
        return RouteDecision(targets=[target], reason=reason)

    return _route


def _default_schema(valid: Iterable[str]):
    """路由的结构化输出 schema。

    用 ``total=False`` 基类而不是 ``NotRequired``：在 ``from __future__ import
    annotations`` 下 PEP 563 把注解变成字符串，``TypedDict`` 算
    ``__required_keys__`` 时看不透 ``"NotRequired[str]"``，字段会**变成必填**
    （全仓 25 处 ``NotRequired`` 实测 optional 数 = 0）。这条坑在 orchestrator 的
    路由 schema 上已经踩过一次并改成了 total=False。
    """
    from typing import TypedDict

    class Route(TypedDict, total=False):
        next_agent: str
        reason: str

    Route.__doc__ = ("下一步派给谁。next_agent 必须是可派发名单里的一个，"
                     f"或 {END} 表示任务完成。")
    return Route


__all__ = ["make_llm_router", "build_routing_messages", "END"]

