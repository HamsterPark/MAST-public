"""Workflow ``agent`` node — 把一个域 agent 当作受预算约束的节点（P3-B）。

「agent 节点」是能动性谱系的最高一档（decide < llm < agent）：把一段任务
完整委托给某个域 agent（文献检索、数据分析、写作、评审、实验设计），它带
自己的工具循环。约束：

  * **instrument_control 永不可委托** —— 仪器动作必须以 step 节点逐技能
    出现在图上（经 ExecutionContext.run 安全收口），不存在「让 IC 自由发挥」
    的节点。这是最小能动性原则的硬边界，写进 spec 校验。
  * 预算硬上限：max_model_calls ≤ 12、max_tool_calls ≤ 40、墙钟超时
    （超时/失败走 on_error 槽，绝不让工作流崩在节点里）。
  * 结果（最终文本）绑定到节点 id 供下游 $expr / llm 节点消费；
  * 委托记录进决策日志（mechanism="agent"），resume 重放复用缓存结果。

standalone 调用复用各 agent 现有 ``graph.build(buf, ...)``（与 orchestrator
同一构建器，中间件齐全）。agent 在独立图里调 handoff_* 工具会因无父图而
异常——prompt 注入「不要 handoff」+ 异常时从 checkpointer 状态抢救最后的
实质性回答（容错而非假装没发生，note 字段如实标注）。
"""

from __future__ import annotations

import logging
import threading
import time

from mast.agents._shared.call_limits import derive_recursion_limit

logger = logging.getLogger(__name__)

#: 可委托 agent 白名单 —— instrument_control 被刻意排除（最小能动性）。
#: research_director 2026-08-21 加入：判据与排除 IC 的那条一样（工作流节点里
#: 由模型决定下一步，所以不给会驱动仪器的那一个），而不是「它是新来的所以先别给」。
DELEGATABLE_AGENTS = ("research_director",
                      "literature", "data_processing", "experiment_design",
                      "paper_writing", "paper_review")

MAX_MODEL_CALLS_CAP = 12
MAX_TOOL_CALLS_CAP = 40
DEFAULT_TIMEOUT_S = 600

_NO_HANDOFF = ("\n\n[工作流委托约束] 你在一个独立的工作流节点中运行，没有"
               "orchestrator：不要调用任何 handoff_* 工具，完成任务后直接"
               "给出最终文字回答。")


def _build_agent_graph(agent_id: str, *, max_model_calls: int,
                       max_tool_calls: int):
    """Patchable seam（tests 注入 fake graph）。Lazy heavy import."""
    import importlib

    from langgraph.checkpoint.memory import InMemorySaver
    mod = importlib.import_module(f"mast.agents.{agent_id}.graph")
    return mod.build(None, checkpointer=InMemorySaver(),
                     max_model_calls=max_model_calls,
                     max_tool_calls=max_tool_calls)


def _v2_engine_enabled() -> bool:
    """``engine_v2_workflow_agent`` —— 退出 LangGraph 的第一个切换面。

    **DEFAULT OFF，live-read，fail-safe OFF**（照抄 ``orchestrator_auto_background``
    的形状）：设置缺失、store 读不出来、任何异常，一律走旧路径。开关每次委托时读一次，
    所以翻回去在下一次委托即刻生效，不用重启。

    为什么第一站选这里：面最小。工作流的 ``agent`` 节点是 invoke→最终文本，没有流式、
    没有 HITL、没有历史（thread 一次性），而且 ``DELEGATABLE_AGENTS`` 本来就排除
    instrument_control —— 零硬件风险。失败也早就被设计成走 ``on_error`` 槽而不是抛。
    """
    try:
        from mast.webui.settings_store import settings_store_for_runtime

        return bool(settings_store_for_runtime().get("engine_v2_workflow_agent"))
    except Exception:  # noqa: BLE001 — 读不到开关 = 用旧路径，不是用新路径
        return False


def _run_via_loop(agent_id: str, task_text: str, *, max_model_calls: int,
                  max_tool_calls: int) -> dict:
    """v2 路径：用 ``agentruntime.AgentLoop`` 跑同一次委托。

    与旧路径的两处**可见**差别（都是改善，都要在回归对照里被看见）：

    * 结局是显式的。旧路径靠「有没有文本」判断 ok，拿不到文本时只能说「agent 无文本
      输出」——那句话对四种完全不同的结局（限流、停机、模型报错、模型确实没话说）
      是同一句。这里 ``outcome`` 与 ``stop_reason`` 直接进 note。
    * 不需要「从 checkpointer 抢救」。旧路径要在异常之后去 ``get_state`` 里捞最后一条
      实质回答，因为交棒工具在独立图里会抛。这里交棒工具**根本没给**（见
      ``assembly.build_agent_loop``）——移除诱因，而不是在提示词里说服模型。
    """
    from mast.agentruntime.assembly import build_agent_loop
    from mast.agentruntime.context import RunContext

    loop = build_agent_loop(agent_id, buf=None,
                            max_model_calls=max_model_calls,
                            max_tool_calls=max_tool_calls,
                            system_suffix=_NO_HANDOFF)
    ctx = RunContext(agent_id=agent_id, run_id=f"wf-agent-{agent_id}")
    gen = loop.run([{"role": "user", "content": task_text}], ctx)
    while True:                      # 事件在这条路径上没有消费者，跑干即可
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break

    text = result.final_text or ""
    ok = bool(text) and result.outcome in ("final", "handoff")
    if ok:
        note = ""
    elif result.stop_reason:
        note = f"{result.outcome}: {result.stop_reason}"
    else:
        note = f"{result.outcome}: agent 无文本输出"
    return {"ok": ok, "text": text, "note": note}


def _extract_final_text(messages) -> str:
    """最后一条有实质文本的 assistant 消息。"""
    for m in reversed(messages or []):
        cls = type(m).__name__.lower()
        role = m.get("role") if isinstance(m, dict) else (
            "assistant" if "ai" in cls else "")
        if role != "assistant" and "ai" not in cls:
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
        text = str(content or "").strip()
        if text:
            return text
    return ""


def run_agent_task(node: dict, task_text: str) -> dict:
    """执行一次受限委托。绝不 raise —— 失败/超时返回 ok=False 由调用方走
    on_error 槽。返回 dict 全 JSON 可序列化（缓存进 partial_data）。"""
    agent_id = str(node.get("agent") or "")
    t0 = time.perf_counter()

    def _out(ok, text, note):
        return {"ok": ok, "text": text, "note": note, "agent": agent_id,
                "duration_ms": int((time.perf_counter() - t0) * 1000)}

    if agent_id not in DELEGATABLE_AGENTS:
        return _out(False, "", f"agent {agent_id!r} 不可委托")
    mmc = min(int(node.get("max_model_calls") or 8), MAX_MODEL_CALLS_CAP)
    mtc = min(int(node.get("max_tool_calls") or 24), MAX_TOOL_CALLS_CAP)
    timeout_s = min(float(node.get("timeout_s") or DEFAULT_TIMEOUT_S), 3600.0)

    result: dict = {}

    fell_back: list[str] = []

    def _work():
        # v2 引擎（退出 LangGraph 的第一个切换面）。任何失败都退回旧路径 ——
        # 双轨期的第一条纪律：新引擎不许把一次本来能完成的委托变成失败。
        if _v2_engine_enabled():
            try:
                result.update(_run_via_loop(agent_id, task_text,
                                            max_model_calls=mmc,
                                            max_tool_calls=mtc))
                return
            except Exception as exc:  # noqa: BLE001
                # **回退必须留痕。** 一个静默的兜底会让「新引擎其实一直在挂」看起来
                # 和「新引擎工作正常」完全一样 —— 那正是这次迁移要根除的形状，
                # 不该在迁移工具自己身上重演一遍。痕迹进 note（用户看得见）
                # 也进日志（排障看得见）。
                fell_back.append(f"{type(exc).__name__}: {exc}")
                logger.warning("v2 agent loop failed for %s (%s); "
                               "falling back to the graph path", agent_id, exc)
        try:
            graph = _build_agent_graph(agent_id, max_model_calls=mmc,
                                       max_tool_calls=mtc)
            # ``2 * (mmc + mtc) + 10`` prices one model→tool round at TWO
            # super-steps. That is not what a round costs: every middleware with
            # a ``before_model`` / ``after_model`` hook is its own node, so the
            # real price is ``len(nodes) - 2`` — five for these five agents today,
            # ELEVEN on the instrument_control chat graph. The old formula is not
            # currently starving anyone (74 budget vs 39 needed at mmc=8, measured
            # 2026-08-04), but it is the same premise that silently cut the
            # private chat down to three tool calls per turn when middleware grew
            # under a literal 50. So: derive from the graph, and keep the old
            # formula as a FLOOR — this can only ever grow the budget.
            _floor = 2 * (mmc + mtc) + 10
            cfg = {"configurable": {"thread_id": f"wf-agent-{id(node)}-{t0}"},
                   "recursion_limit": max(_floor, derive_recursion_limit(
                       graph, model_calls_per_run=mmc, fallback=_floor))}
            try:
                out = graph.invoke(
                    {"messages": [{"role": "user",
                                   "content": task_text + _NO_HANDOFF}]}, cfg)
                result["text"] = _extract_final_text(out.get("messages"))
                result["ok"] = bool(result["text"])
                result["note"] = "" if result["ok"] else "agent 无文本输出"
            except Exception as exc:  # noqa: BLE001 — handoff/预算等中途异常
                # 从 checkpointer 状态抢救最后的实质回答（容错且如实标注）
                text = ""
                try:
                    st = graph.get_state(cfg)
                    text = _extract_final_text(
                        (getattr(st, "values", None) or {}).get("messages"))
                except Exception:  # pragma: no cover
                    pass
                if text:
                    result.update(ok=True, text=text,
                                  note=f"中途异常后抢救输出：{type(exc).__name__}")
                else:
                    result.update(ok=False, text="",
                                  note=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — 构建失败
            result.update(ok=False, text="", note=f"构建 agent 失败：{exc}")

    th = threading.Thread(target=_work, name=f"wf-agent-{agent_id}",
                          daemon=True)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        # 线程无法强杀（daemon 化放弃）；预算上限保证它终将自止。
        return _out(False, "", f"超时 {timeout_s:.0f}s（已放弃等待）")
    note = result.get("note", "")
    if fell_back:
        marker = f"（v2 引擎失败已回退：{fell_back[0]}）"
        note = f"{note} {marker}".strip() if note else marker
    return _out(bool(result.get("ok")), result.get("text", ""), note)


__all__ = ["DELEGATABLE_AGENTS", "run_agent_task",
           "MAX_MODEL_CALLS_CAP", "MAX_TOOL_CALLS_CAP"]
