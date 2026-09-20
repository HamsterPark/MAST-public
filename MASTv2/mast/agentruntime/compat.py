"""桥：让**既有的** ``langchain.agents.middleware.AgentMiddleware`` 在新栈里跑。

为什么是桥，不是重写
--------------------
要移植的那 14 个中间件是四个月现场反馈的沉淀（SafetyGate 794 行、StallGuard 的阶梯与
台账、AlertDelivery、记忆召回、心愿单回读……）。它们**不是框架适配层**，它们的逻辑就是
业务本身。把它们逐个重抄一遍，等于把四个月的教训重新誊写一次，每一次誊写都是一次可能
抄错的机会——而抄错一个安全件的症状是「守卫看起来在，实际不拦」。

更现实的一条：那 13 个 ``_shared/*_mw.py`` 改动面很大，机械重写整批文件容易出错。
桥接**一行都不碰它们**，把改写风险降到零。

所以顺序是：先桥接（逻辑零风险地跑起来），等工作树落定、新运行时在生产上烤熟之后，
再逐个把桥拆掉换成原生实现——那时每拆一个都有对照测试兜着。

它翻译什么
----------
============================  ==================================================
我们的 ``ModelRequest``       langchain 的 ``ModelRequest``（真类，不是替身）
``system_prompt: str``        ``system_message: SystemMessage | None``
``settings: dict``            ``model_settings``
``TurnView``                  ``before_model(state, runtime)`` 的 ``state``/返回的 update
``ToolCallView``              ``ToolCallRequest``
============================  ==================================================

**用真的 ``ModelRequest`` 而不是自造替身**：``inject.append_system_block`` 这类共用助手
优先走 ``request.override(...)``，只有在替身没有它时才退回直接赋值。喂一个没有 override
的替身，走的就是那条**降级**路径——测得再绿，跑的也不是生产那条路。

⚠️ 桥不吞 ``wrap_*`` 的异常
---------------------------
与 :class:`~mast.agentruntime.middleware.MiddlewareStack` 同一条纪律：``before_model`` /
``after_model`` 失败降级（少注入一段提示，回合还能跑），但**包住模型/工具调用的中间件
抛出来就是整条链停**——SafetyGate 在那条链上，「守卫抛异常所以放行」是这个仓库最不该有
的形状。
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from mast.agentruntime.middleware import Middleware, ToolCallView, TurnView
from mast.agentruntime.model import ModelRequest, ModelResponse
from mast.agentruntime.tools import ToolResult, coerce_result

logger = logging.getLogger(__name__)


def _overrides(mw: Any, hook: str) -> bool:
    """这个中间件**覆写**了 ``hook`` 吗？

    ⚠️ 不能用 ``getattr(mw, hook, None) is None`` 判断 —— ``AgentMiddleware`` 基类
    **定义了全部四个挂点**，属性永远在。第一版就是这么写的：于是只实现了
    ``before_model`` 的中间件也被当成实现了 ``wrap_model_call``，桥去调它，基类抛
    「Synchronous implementation is not available」，整轮以 error 收场。

    这是本仓记过的「看着在防护其实没有」的形状：检查语句读起来像在检查存在性，
    而被检查的东西恒存在。

    比的是**函数对象**：子类没覆写时 ``type(mw).hook`` 就是基类那一个。
    """
    own = getattr(type(mw), hook, None)
    if own is None:
        return False
    for base in type(mw).__mro__[1:]:
        if base.__name__ == "AgentMiddleware":
            return own is not getattr(base, hook, None)
    # 不是 AgentMiddleware 的子类（鸭子类型的中间件）：属性在就算覆写。
    return True


def _async_only(mw: Any, hook: str) -> bool:
    """只实现了异步版、没实现同步版 —— 桥**不能**默默跳过它。

    静默跳过一个 SafetyGate，症状是「守卫挂着、一次都没拦过」。生产里的 14 个
    中间件目前都是同步/异步双份（CLI 改同步之后异步那半成了死代码），所以这条在
    今天不该触发；留着是因为将来有人只写异步版时，要**当场知道**。
    """
    return not _overrides(mw, hook) and _overrides(mw, f"a{hook}")


class _RuntimeStub:
    """``runtime`` 参数的最小替身。

    实测那 14 个中间件里只有两处真的读它（``.config`` / ``.configurable``），其余
    「runtime.」命中都是注释里指 ``core/runtime.py``。所以这里只承诺这两个属性，
    多的**不假装有** —— 一个什么都答得上来的替身，会让「中间件依赖了我们没给的
    东西」这件事测不出来。
    """

    __slots__ = ("config", "configurable", "context", "store")

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.configurable = self.config.get("configurable", {})
        self.context = None
        self.store = None


class LangChainMiddlewareBridge(Middleware):
    """把一个 ``AgentMiddleware`` 实例包成新栈能用的 :class:`Middleware`。"""

    def __init__(self, mw: Any, *, agent_id: str = "", config: dict | None = None):
        self._mw = mw
        self._agent_id = agent_id
        self._runtime = _RuntimeStub(config)

    @property
    def name(self) -> str:
        # 带前缀，好让日志里一眼看出「这条还挂在桥上」——拆桥的进度因此是可读的。
        inner = getattr(self._mw, "name", None) or type(self._mw).__name__
        return f"lc:{inner}"

    @property
    def wrapped(self) -> Any:
        """被包着的那个中间件。拆桥时用来核对「换掉的是同一个」。"""
        return self._mw

    def _runs(self, hook: str) -> bool:
        """这个挂点要不要真的调过去。

        两件事合成一个判断：

        * **没覆写** → 跳过。不能用「属性在不在」判断，基类定义了全部四个挂点
          （见 :func:`_overrides`）。
        * **只覆写了异步版** → **抛**，不是跳过。静默跳过一个只写了 ``awrap_tool_call``
          的 SafetyGate，症状是「守卫挂着、一次都没拦过」—— 那正是这次迁移要根除的
          东西，不该由迁移工具自己制造一个新的。
        """
        if _async_only(self._mw, hook):
            raise RuntimeError(
                f"{self.name} 只实现了 a{hook}，没有同步版。新运行时是全同步的，"
                f"桥接它等于让这个中间件的逻辑一次都不跑 —— 请补上同步实现。")
        return _overrides(self._mw, hook)

    # ── before / after ────────────────────────────────────────────────
    def before_model(self, turn: TurnView) -> None:
        if not self._runs("before_model"):
            return
        fn = self._mw.before_model
        state = self._state(turn.messages)
        update = fn(state, self._runtime)
        self._apply_update(turn, update)

    def after_model(self, turn: TurnView, response: ModelResponse) -> None:
        if not self._runs("after_model"):
            return
        fn = self._mw.after_model
        msgs = list(turn.messages)
        if response.message is not None:
            msgs = msgs + [response.message]
        fn(self._state(msgs), self._runtime)

    # ── wrap_model_call ───────────────────────────────────────────────
    def wrap_model_call(
        self, request: ModelRequest,
        call_next: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not self._runs("wrap_model_call"):
            return call_next(request)
        fn = self._mw.wrap_model_call

        lc_request = self._to_lc_request(request)
        captured: dict = {}

        def _handler(lc_req):
            # 中间件可能换了一个新的 request（``override`` 返回新对象），所以要从
            # **handler 收到的那一个**读回改动，而不是从我们传出去的那一个。
            captured["response"] = call_next(self._from_lc_request(lc_req, request))
            return self._to_lc_response(captured["response"])

        lc_out = fn(lc_request, _handler)

        if "response" not in captured:
            # 中间件短路了（没调 handler）—— 它自己造了一个回答。这是合法的
            # （限流中间件就这么干），但结果要能被翻译回来。
            return self._from_lc_response(lc_out)
        return self._merge_response(captured["response"], lc_out)

    # ── wrap_tool_call ────────────────────────────────────────────────
    def wrap_tool_call(self, call: ToolCallView, ctx,
                       call_next: Callable[[ToolCallView], Any]) -> Any:
        if not self._runs("wrap_tool_call"):
            return call_next(call)
        fn = self._mw.wrap_tool_call

        lc_req = self._to_lc_tool_request(call)
        captured: dict = {}

        def _handler(req):
            # 中间件可能改了参数（SI 归一、参数精修）——从它交回来的那份读。
            tc = getattr(req, "tool_call", None) or {}
            call.args = dict(tc.get("args", call.args))
            captured["result"] = coerce_result(call_next(call))
            return self._to_lc_tool_message(call, captured["result"])

        lc_out = fn(lc_req, _handler)

        if "result" not in captured:
            # 守卫拦下了这次调用（SafetyGate 拒绝、审批否决）。它给的那条
            # ToolMessage 就是模型该看到的东西。
            return self._from_lc_tool_message(lc_out)
        return self._merge_tool_result(captured["result"], lc_out)

    # ── 翻译 ───────────────────────────────────────────────────────────
    def _state(self, messages: list) -> dict:
        """``before_model`` / ``after_model`` 看到的 state。

        实测中间件只读 ``messages``（两处）与 ``event_refs``（一处）。给一个普通
        dict 而不是 ``MASTState``：新运行时没有 typed channel，假装有一个只会让
        「谁在读什么」更难看清。
        """
        return {"messages": list(messages), "event_refs": [],
                "agent_id": self._agent_id}

    def _model_state(self, request: ModelRequest) -> dict:
        """``wrap_model_call`` 看到的 state：消息 + 本次 run 累计的 state 键。

        ``ToolVisibilityMiddleware`` 读 ``state["loaded_tool_packs"]`` 决定放宽哪些
        工具；那个键由 ``load_tool_pack`` 的 ``Command(update=…)`` 写出、被循环合进
        ``state_delta``。桥接层不把它带过来，工具包就永远「已加载」而不可见
        （2026-08-28 真 provider 首跑：模型把 StartScan 调成了 StopScan）。
        """
        base = self._state(request.messages)
        extra = getattr(request, "state", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in ("messages",):
                    base[k] = v
        return base

    def _apply_update(self, turn: TurnView, update: Any) -> None:
        if not isinstance(update, dict):
            return
        msgs = update.get("messages")
        if msgs is None:
            return
        # 压缩类中间件返回 ``[RemoveMessage(ALL), summary, *保留]``。新运行时里
        # 消息列表归我们所有，所以 RemoveMessage 只是一个「从这里截断」的标记。
        cleaned, removed_all = _strip_remove_markers(msgs)
        if removed_all:
            turn.replace_messages(cleaned, removed=len(turn.messages),
                                  kept=len(cleaned))
        else:
            turn.messages = list(turn.messages) + list(cleaned)

    def _to_lc_request(self, request: ModelRequest):
        from langchain.agents.middleware import ModelRequest as LCReq
        from langchain_core.messages import SystemMessage

        sm = SystemMessage(content=request.system_prompt) if request.system_prompt \
            else None
        return LCReq(model=None, messages=list(request.messages), system_message=sm,
                     tool_choice=None, tools=list(request.tools),
                     response_format=None, state=self._model_state(request),
                     runtime=self._runtime, model_settings=dict(request.settings))

    def _from_lc_request(self, lc_req: Any, original: ModelRequest) -> ModelRequest:
        out = original.copy()
        sm = getattr(lc_req, "system_message", None)
        if sm is not None:
            content = getattr(sm, "content", sm)
            out.system_prompt = _flatten_text(content)
        elif getattr(lc_req, "system_message", "missing") is None:
            out.system_prompt = ""
        msgs = getattr(lc_req, "messages", None)
        if msgs is not None:
            out.messages = list(msgs)
        tools = getattr(lc_req, "tools", None)
        if tools is not None:
            out.tools = list(tools)
        settings = getattr(lc_req, "model_settings", None)
        if isinstance(settings, dict):
            out.settings = dict(settings)
        return out

    def _to_lc_response(self, response: ModelResponse):
        from langchain.agents.middleware import ModelResponse as LCResp

        result = [response.message] if response.message is not None else []
        return LCResp(result=result, structured_response=None)

    def _from_lc_response(self, lc_out: Any) -> ModelResponse:
        """中间件短路时给的东西 → 我们的 ModelResponse。

        它可能给 ``ModelResponse``、也可能直接给一条 ``AIMessage``（上游签名允许
        三种返回）。两种都要接住 —— 认不出的形状**抛**而不是静默变成空回答：
        一个空回答会被循环当成「模型没话说」，于是回合以 ``final`` 结束，
        而真相是中间件说了什么我们没听懂。
        """
        msg = None
        result = getattr(lc_out, "result", None)
        if isinstance(result, list) and result:
            msg = result[-1]
        elif hasattr(lc_out, "content"):
            msg = lc_out
        if msg is None:
            raise TypeError(
                f"{self.name} 短路了模型调用，但返回的东西认不出来："
                f"{type(lc_out).__name__}。不敢当成空回答放行。")
        return ModelResponse(message=msg,
                             tool_calls=list(getattr(msg, "tool_calls", None) or []),
                             text=_flatten_text(getattr(msg, "content", "")),
                             usage=dict(getattr(msg, "usage_metadata", None) or {}))

    def _merge_response(self, ours: ModelResponse, lc_out: Any) -> ModelResponse:
        """中间件调了 handler，但可能又改了返回值（很少见，但合法）。"""
        try:
            replaced = self._from_lc_response(lc_out)
        except TypeError:
            return ours
        return replaced if replaced.message is not ours.message else ours

    def _to_lc_tool_request(self, call: ToolCallView):
        from langchain.agents.middleware import ToolCallRequest as LCTCR

        tool_call = {"name": call.name, "args": dict(call.args),
                     "id": call.tool_call_id, "type": "tool_call"}
        return LCTCR(tool_call=tool_call, tool=None,
                     state=self._state([]), runtime=self._runtime)

    def _to_lc_tool_message(self, call: ToolCallView, result: ToolResult):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=result.text or "",
                           tool_call_id=call.tool_call_id, name=call.name)

    def _from_lc_tool_message(self, lc_out: Any) -> ToolResult:
        """守卫拦下调用时给的 ToolMessage / Command → 我们的 ToolResult。"""
        content = getattr(lc_out, "content", None)
        if content is None and hasattr(lc_out, "update"):
            # 一个 ``Command`` —— 取它 update 里的最后一条消息。
            msgs = (getattr(lc_out, "update", None) or {}).get("messages") or []
            content = getattr(msgs[-1], "content", "") if msgs else ""
        return ToolResult(text=_flatten_text(content or ""), ok=False)

    def _merge_tool_result(self, ours: ToolResult, lc_out: Any) -> ToolResult:
        """中间件调了 handler 之后又改了结果（留痕类中间件通常原样返回）。"""
        text = _flatten_text(getattr(lc_out, "content", "") or "")
        if text and text != (ours.text or ""):
            return ToolResult(text=text, handoff=ours.handoff,
                              state_delta=ours.state_delta, ok=ours.ok,
                              preview=ours.preview)
        return ours


def _strip_remove_markers(messages: list) -> tuple[list, bool]:
    """去掉 ``RemoveMessage`` 标记，并说明是不是「全删」。

    新运行时的消息列表归编排器所有，裁剪是切片不是标记；但被桥接的压缩中间件仍然
    按 LangGraph 的协议返回标记，所以在边界上翻译一次。
    """
    out, removed_all = [], False
    for m in messages:
        if type(m).__name__ == "RemoveMessage":
            removed_all = True
            continue
        out.append(m)
    return out, removed_all


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(content or "")


def bridge_all(middlewares, *, agent_id: str = "",
               config: dict | None = None) -> list[Middleware]:
    """把一串既有中间件整体桥接过来，保持顺序。

    顺序就是挂载顺序，与今天 ``create_agent`` 的语义一致——移植时可以照抄各
    ``graph.py`` 里那张表，不用重新推导谁该在外层。
    """
    return [LangChainMiddlewareBridge(m, agent_id=agent_id, config=config)
            for m in (middlewares or [])]


__all__ = ["LangChainMiddlewareBridge", "bridge_all"]
