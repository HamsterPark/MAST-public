"""Brainstorm subgraph — facilitated multi-viewpoint discussion of an experiment.

Design: docs/v2/design/agentic-cognition.md §4. Mirrors a coding agent's multi-agent
discussion, adapted to STM: a facilitator sets the agenda, ~6 viewpoint agents
take turns (reading the grounding + transcript + the user's injected viewpoints),
the facilitator summarises each round and decides whether to run another (capped
at ``max_rounds``), and a final node synthesises a summary that can be written to
memory.

Hard rules honoured here:
  * **No hardware tools.** Viewpoint agents only read grounding *text*; this graph
    never imports a skill/instrument tool and never touches Nanonis. The only LLM
    interface used is ``model.invoke(messages) -> message`` (or our rule-based
    fallback) — there is no tool-calling agent loop.
  * **Non-blocking steps.** No ``time.sleep`` / unbounded ``.get()`` / ``.join()``
    anywhere — each step is a pure state->state transform.
  * **JSON-only state.** Everything written into BrainstormState is str / int /
    list / dict of scalars (no tensor/handle/socket).
  * **Honesty.** Every viewpoint line and the summary carry the 非实测 banner.
  * **No cross-agent imports.** This module imports only stdlib,
    mast.logging.storage, mast.memory.store, and its sibling state/prompts.

2026-08-27 — 不再是一张图
-------------------------
这里曾经是一张四节点的 ``StateGraph``（全仓唯一用到 ``add_conditional_edges``
的地方）。四个步骤本来就是纯变换，中间没有并发、没有暂停、没有 checkpoint 需求 ——
那张图除了「一个 while 循环」之外没有表达任何东西，却带来一个用**超步**计数的预算
（``recursion_limit = 4 * max_rounds + 12``）和一份对 LangGraph 的依赖。

展开成 :func:`run_discussion` 之后：预算的单位变回**轮数**、失败路径（主持人不收敛）
有一个读得懂的上限并且会说话，而 ``run_brainstorm`` 的对外签名一个字没变。
这是退出 LangGraph 的 Step 5 的一部分。

Offline degradation: ``llm=None`` runs a dependency-free rule-based discussion
that reports the *real* skill counts / statuses / memory entries from the record
(no fabricated numbers) and synthesises a rule-based summary.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .prompts import (
    BRAINSTORM_TAG,
    FACILITATOR_AGENDA_INSTRUCTION,
    FACILITATOR_ROUND_SUMMARY_INSTRUCTION,
    FACILITATOR_SYSTEM,
    SUMMARY_INSTRUCTION,
    SUMMARY_SYSTEM,
    VIEWPOINT_ORDER,
    viewpoint_name,
    viewpoint_system,
)
from .state import BrainstormState

logger = logging.getLogger(__name__)

# Memory entries land under this namespace/path family when written.
_MEMORY_NS_GLOBAL = "global"


# ─────────────────────────────────────────────────────────────────────
# Grounding — read the real experiment record + memory (no fabrication).
# Mirrors mast.memory.dreaming.gather_context: get_actions returns
# ActionRecord OBJECTS, so fields are read via getattr (here _field()).
# ─────────────────────────────────────────────────────────────────────

def _field(obj, key, default=None):
    """Read ``key`` from a dict OR a dataclass/record object (getattr fallback)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def gather_grounding(
    db_path,
    experiment_id: str | None = None,
    *,
    memory_store=None,
    n_memory: int = 12,
    topic: str = "",
) -> dict:
    """Assemble the read-only context the discussion is grounded in.

    Returns an all-JSON dict::

        {
          "experiment": {id, name, status, goal} | None,
          "skill_counts": {skill: n},   "statuses": {status: n},
          "actions_total": int,         "samples": [{name, status}],
          "memory": [{path, title, kind, excerpt}],
        }

    Best-effort: a missing DB / table never raises (the brainstorm must still
    run, just with thinner grounding).
    """
    out: dict = {
        "experiment": None,
        "skill_counts": {},
        "statuses": {},
        "actions_total": 0,
        "samples": [],
        "memory": [],
    }

    # ── experiment record (experiments / samples / actions) ────────────
    try:
        from mast.logging.storage import ExperimentStorage
        st = ExperimentStorage(str(db_path))
    except Exception as exc:  # pragma: no cover — storage import/open failure
        logger.debug("brainstorm: storage open failed: %s", exc)
        st = None

    if st is not None:
        try:
            if experiment_id:
                exp = st.get_experiment(experiment_id)
                if exp:
                    out["experiment"] = {
                        "id": _field(exp, "id"),
                        "name": _field(exp, "name", ""),
                        "status": _field(exp, "status", "?") or "?",
                        "goal": _field(exp, "goal_text", "") or _field(exp, "goal", ""),
                    }
                try:
                    for s in st.get_samples(experiment_id):
                        out["samples"].append({
                            "name": _field(s, "name", ""),
                            "status": _field(s, "status", "?") or "?",
                        })
                except Exception:
                    pass
                actions = _safe_actions(st, experiment_id)
            else:
                # No specific experiment → summarise the most recent ones.
                actions = []
                try:
                    exps = st.list_experiments(limit=8)
                except Exception:
                    exps = []
                for e in exps:
                    status = _field(e, "status", "?") or "?"
                    out["statuses"][status] = out["statuses"].get(status, 0) + 1
                    actions.extend(_safe_actions(st, _field(e, "id")))
            for a in actions:
                sk = _field(a, "skill_name", "?") or "?"
                out["skill_counts"][sk] = out["skill_counts"].get(sk, 0) + 1
            out["actions_total"] = len(actions)
            if experiment_id and out["experiment"]:
                status = out["experiment"]["status"]
                out["statuses"][status] = out["statuses"].get(status, 0) + 1
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("brainstorm: grounding read failed: %s", exc)

    # ── memory (list + optional topic search) ──────────────────────────
    if memory_store is not None:
        seen: set = set()
        rows: list = []
        try:
            if topic.strip():
                rows.extend(memory_store.search(topic.strip(), limit=n_memory) or [])
        except Exception:
            pass
        try:
            rows.extend(memory_store.list(limit=n_memory) or [])
        except Exception:
            pass
        for r in rows:
            path = _field(r, "path", "") or ""
            if path in seen:
                continue
            seen.add(path)
            content = (_field(r, "content", "") or "").replace("\n", " ").strip()
            out["memory"].append({
                "path": path,
                "title": _field(r, "title", "") or "",
                "kind": _field(r, "kind", "") or "",
                "excerpt": content[:200],
            })
            if len(out["memory"]) >= n_memory:
                break

    return out


def _safe_actions(storage, experiment_id) -> list:
    """get_actions but never raises (returns [] on any failure / no id)."""
    if not experiment_id:
        return []
    try:
        return storage.get_actions(experiment_id) or []
    except Exception:
        return []


def grounding_to_text(grounding: dict) -> str:
    """Flatten grounding into a compact text block for the prompts (real data
    only — no fabricated numbers)."""
    g = grounding or {}
    lines: list[str] = []
    exp = g.get("experiment")
    if exp:
        lines.append(
            f"实验: {exp.get('name') or '(未命名)'} "
            f"[状态={exp.get('status', '?')}]"
        )
        if exp.get("goal"):
            lines.append(f"目标: {exp['goal']}")
    else:
        lines.append("实验: (未指定具体实验 / 跨近期实验汇总)")
    lines.append(f"已记录动作总数: {g.get('actions_total', 0)}")
    statuses = g.get("statuses") or {}
    if statuses:
        lines.append("实验状态分布: " + ", ".join(f"{k}×{v}" for k, v in statuses.items()))
    skills = sorted((g.get("skill_counts") or {}).items(),
                    key=lambda kv: kv[1], reverse=True)
    if skills:
        lines.append("高频技能: " + ", ".join(f"{s}×{n}" for s, n in skills[:10]))
    else:
        lines.append("高频技能: (尚无动作记录)")
    samples = g.get("samples") or []
    if samples:
        lines.append("样品: " + ", ".join(
            f"{s.get('name', '?')}[{s.get('status', '?')}]" for s in samples[:6]))
    mem = g.get("memory") or []
    if mem:
        lines.append("相关记忆条目:")
        for m in mem[:8]:
            tag = f"({m.get('kind', '')})" if m.get("kind") else ""
            title = m.get("title") or m.get("path") or "记忆"
            lines.append(f"  - {title} {tag}: {m.get('excerpt', '')}")
    else:
        lines.append("相关记忆条目: (无)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# LLM adapter — accept either a langchain chat model (.invoke([...]) ->
# message with .content) or a plain callable(prompt:str)->str. None => rule.
# ─────────────────────────────────────────────────────────────────────

def _llm_text(llm: Any, system: str, user: str) -> str:
    """Call the model and return plain text. Tolerant of both langchain chat
    models and simple callables; never raises (falls back to '')."""
    if llm is None:
        return ""
    try:
        # langchain BaseChatModel style: invoke a list of messages.
        if hasattr(llm, "invoke"):
            from langchain_core.messages import HumanMessage, SystemMessage
            msg = llm.invoke([SystemMessage(content=system),
                              HumanMessage(content=user)])
            content = getattr(msg, "content", msg)
            if isinstance(content, list):  # some providers return content parts
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content)
            return str(content).strip()
        # plain callable(prompt) -> str
        if callable(llm):
            return str(llm(f"{system}\n\n{user}")).strip()
    except Exception as exc:
        logger.debug("brainstorm: llm call failed (%s); using rule-based line", exc)
    return ""


# ─────────────────────────────────────────────────────────────────────
# Rule-based fallback lines (real grounding, no fabrication).
# ─────────────────────────────────────────────────────────────────────

def _rule_agenda(topic: str, grounding: dict) -> list[str]:
    g = grounding or {}
    n_actions = g.get("actions_total", 0)
    skills = sorted((g.get("skill_counts") or {}).items(),
                    key=lambda kv: kv[1], reverse=True)
    items = [f"围绕主题『{topic or '当前实验与进度'}』,各视角给出设想与分歧"]
    if n_actions:
        top = skills[0][0] if skills else "已记录操作"
        items.append(f"已记录 {n_actions} 个动作(高频:{top}),讨论下一步该做什么")
    else:
        items.append("当前尚无动作记录,讨论应从哪一步开始")
    items.append("识别风险/不确定项,以及哪些结论需要补测才能确认")
    return items


def _rule_viewpoint_line(key: str, topic: str, grounding: dict,
                         user_viewpoints: list[str]) -> str:
    """A grounded, non-fabricated rule-based line for one viewpoint."""
    g = grounding or {}
    n_actions = g.get("actions_total", 0)
    skills = sorted((g.get("skill_counts") or {}).items(),
                    key=lambda kv: kv[1], reverse=True)
    top_skill = skills[0][0] if skills else None
    statuses = g.get("statuses") or {}
    mem = g.get("memory") or []
    name = viewpoint_name(key)

    base = {
        "design": (
            f"从实验设计看,记录显示已执行 {n_actions} 个动作"
            + (f"(最常用 `{top_skill}`)" if top_skill else "(尚无操作)")
            + "。设想:下一步应明确变量与对照,确认扫描范围/偏压是否匹配目标。"
        ),
        "safety": (
            "从风险/安全看,需检查是否有不可逆或危险步骤(大脉冲、tip conditioning)。"
            + (f"失败状态实验 {statuses.get('failed', 0)} 个,值得回顾共因。"
               if statuses.get("failed") else "建议危险操作走 HITL 确认。")
        ),
        "analysis": (
            f"从数据分析看,现有 {n_actions} 个动作的产出需要明确判据后再下结论;"
            "区分『可由现有记录推断』与『需补测』,注意漂移与噪声。"
        ),
        "literature": (
            "从文献依据看,可参考同体系的一般性现象与典型参数区间(仅作参考,非权威定值)。"
            + (f"记忆库有 {len(mem)} 条相关条目可借鉴。" if mem else "暂无可借鉴的记忆条目。")
        ),
        "feasibility": (
            "从操作可行性看,需评估在 Nanonis V5e 上的落地耗时与稳定性(漂移/针尖寿命),"
            "并确认所需前置状态是否就绪。"
        ),
        "critic": (
            "从批判质疑看,前面的设想可能过度乐观:最弱的假设是默认当前针尖/样品状态良好;"
            "被忽略的失败模式与替代方案值得在动手前再核一遍。"
        ),
    }
    line = base.get(key, f"从{name}视角,基于现有记录提出设想,并指出未解决的问题。")
    if user_viewpoints:
        line += f" 另外回应用户观点之一:『{str(user_viewpoints[0])[:80]}』——值得纳入考量。"
    return f"{BRAINSTORM_TAG} {line}"


def _rule_round_summary(round_idx: int, transcript: list[dict]) -> str:
    this_round = [t for t in transcript
                  if t.get("round") == round_idx and t.get("role") not in ("facilitator",)]
    speakers = ", ".join(dict.fromkeys(t.get("speaker", "?") for t in this_round))
    return (
        f"{BRAINSTORM_TAG} 第 {round_idx} 轮汇总:{len(this_round)} 位视角发言"
        f"({speakers})。共识在于均同意先明确目标与判据;主要分歧/风险集中在"
        "操作安全与结论是否需要补测。仍待解决:下一步具体参数与失败预案。"
    )


def _rule_summary(state: BrainstormState) -> str:
    transcript = state.get("transcript", []) or []
    user_vps = state.get("user_viewpoints", []) or []
    topic = state.get("topic", "") or "当前实验"
    grounding = state.get("grounding", {}) or {}
    g = grounding
    n_actions = g.get("actions_total", 0)
    n_views = len({t.get("role") for t in transcript
                   if t.get("role") not in ("facilitator", "user", "")})
    skills = sorted((g.get("skill_counts") or {}).items(),
                    key=lambda kv: kv[1], reverse=True)
    lines = [
        BRAINSTORM_TAG,
        "",
        f"## 头脑风暴综合摘要 — {topic}",
        "",
        f"参与视角 {n_views} 个,共 {len([t for t in transcript])} 条发言;"
        f"基于 {n_actions} 个已记录动作"
        + (f"(高频技能 {skills[0][0]})" if skills else "")
        + "。",
        "",
        "**核心共识(设想)**: 先明确实验目标、变量/对照与判据,再决定下一步操作。",
        "**主要分歧/风险**: 操作安全(危险步骤需 HITL)与『现有数据能否支撑结论』。",
        "**建议下一步(设想,非指令)**: 补齐对照与关键测量,危险操作走人工确认。",
        "**仍需补测的开放问题**: 具体扫描参数、失败预案、针尖/样品当前真实状态。",
    ]
    if user_vps:
        lines += ["", "**已纳入的用户观点**:"]
        lines += [f"- {str(v)[:120]}" for v in user_vps[:5]]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# LangGraph nodes (pure state->partial-state; non-blocking).
# ─────────────────────────────────────────────────────────────────────

def _make_nodes(llm: Any) -> dict:
    """讨论的四个步骤，都是纯 ``state -> 部分 state``（``llm=None`` 走规则版）。

    它们曾经是一张 ``StateGraph`` 的四个节点。保持「纯变换」这个形状是刻意的：
    正因为它们没有副作用、不并发、不需要暂停，那张图才可以被一个 while 循环替掉
    （见 :func:`run_discussion`）。
    """

    def facilitator_open(state: BrainstormState) -> dict:
        topic = state.get("topic", "") or ""
        grounding = state.get("grounding", {}) or {}
        gtext = grounding_to_text(grounding)
        agenda: list[str] = []
        txt = _llm_text(llm, FACILITATOR_SYSTEM,
                        f"topic: {topic}\n\n## grounding\n{gtext}\n\n"
                        f"{FACILITATOR_AGENDA_INSTRUCTION}")
        if txt:
            for ln in txt.splitlines():
                ln = ln.strip().lstrip("-•*0123456789. )、").strip()
                if ln:
                    agenda.append(ln)
        if not agenda:
            agenda = _rule_agenda(topic, grounding)
        agenda = agenda[:4]
        opening = (
            f"{BRAINSTORM_TAG} 本场头脑风暴主题:{topic or '当前实验与进度'}。议程:\n"
            + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(agenda))
        )
        turn = {"speaker": "主持人", "role": "facilitator",
                "content": opening, "round": 0}
        # seed transcript with any user viewpoints up-front so viewpoints react.
        transcript = list(state.get("transcript", []) or [])
        for v in (state.get("user_viewpoints", []) or []):
            transcript.append({"speaker": "用户", "role": "user",
                               "content": str(v), "round": 0})
        transcript.append(turn)
        return {"agenda": agenda, "transcript": transcript, "round": 1, "done": False}

    def viewpoints_round(state: BrainstormState) -> dict:
        round_idx = state.get("round", 1)
        topic = state.get("topic", "") or ""
        grounding = state.get("grounding", {}) or {}
        gtext = grounding_to_text(grounding)
        user_vps = [str(v) for v in (state.get("user_viewpoints", []) or [])]
        transcript = list(state.get("transcript", []) or [])
        # transcript-so-far text the viewpoints may read (read-only).
        prior = "\n".join(f"[{t.get('speaker')}] {t.get('content')}"
                          for t in transcript)[-4000:]
        for key in VIEWPOINT_ORDER:
            name = viewpoint_name(key)
            user_block = ("\n## 用户观点\n" + "\n".join(f"- {v}" for v in user_vps)
                          if user_vps else "")
            prompt = (
                f"主题: {topic}\n第 {round_idx} 轮。\n\n## grounding(只读)\n{gtext}"
                f"{user_block}\n\n## 目前的讨论\n{prior}\n\n"
                "请以你的视角发言(3-6 句,鼓励提出分歧)。"
            )
            txt = _llm_text(llm, viewpoint_system(key), prompt)
            if txt:
                content = txt if BRAINSTORM_TAG in txt else f"{BRAINSTORM_TAG} {txt}"
            else:
                content = _rule_viewpoint_line(key, topic, grounding, user_vps)
            turn = {"speaker": name, "role": key,
                    "content": content, "round": round_idx}
            transcript.append(turn)
            prior = (prior + f"\n[{name}] {content}")[-4000:]
        return {"transcript": transcript}

    def facilitator_summarize(state: BrainstormState) -> dict:
        round_idx = state.get("round", 1)
        max_rounds = max(1, int(state.get("max_rounds", 2)))
        transcript = list(state.get("transcript", []) or [])
        this_round = "\n".join(
            f"[{t.get('speaker')}] {t.get('content')}"
            for t in transcript if t.get("round") == round_idx)
        txt = _llm_text(llm, FACILITATOR_SYSTEM,
                        f"## 第 {round_idx} 轮发言\n{this_round}\n\n"
                        f"{FACILITATOR_ROUND_SUMMARY_INSTRUCTION}")
        summary_line = txt if txt else _rule_round_summary(round_idx, transcript)
        if BRAINSTORM_TAG not in summary_line:
            summary_line = f"{BRAINSTORM_TAG} {summary_line}"
        transcript.append({"speaker": "主持人", "role": "facilitator",
                           "content": summary_line, "round": round_idx})
        done = round_idx >= max_rounds
        return {"transcript": transcript,
                "round": round_idx + 1,
                "done": done}

    def summarize(state: BrainstormState) -> dict:
        transcript = state.get("transcript", []) or []
        convo = "\n".join(f"[{t.get('speaker')}] {t.get('content')}"
                          for t in transcript)
        user_vps = [str(v) for v in (state.get("user_viewpoints", []) or [])]
        user_block = ("\n\n## 用户观点\n" + "\n".join(f"- {v}" for v in user_vps)
                      if user_vps else "")
        txt = _llm_text(llm, SUMMARY_SYSTEM,
                        f"## 讨论纪要\n{convo}{user_block}\n\n{SUMMARY_INSTRUCTION}")
        if txt:
            summary = txt if BRAINSTORM_TAG in txt else f"{BRAINSTORM_TAG}\n\n{txt}"
        else:
            summary = _rule_summary(state)
        return {"summary": summary}

    return {"facilitator_open": facilitator_open,
            "viewpoints_round": viewpoints_round,
            "facilitator_summarize": facilitator_summarize,
            "summarize": summarize}


#: 循环步数的兜底上限。正常情况下 ``done`` 会在 ``max_rounds`` 轮之后置位；这个数
#: 只在「主持人一直不收敛」时兜底。它替代的是原来的
#: ``recursion_limit = 4 * max_rounds + 12`` —— 同一个作用，但**单位是轮数**而不是
#: 图的超步数，所以它不会因为将来给讨论加一个环节而悄悄变小。
_MAX_ROUNDS_HARD_CAP = 24


def run_discussion(state: BrainstormState, llm: Any | None = None) -> BrainstormState:
    """跑完一场讨论：主持人开场 → N 轮观点 → 收敛 → 综述。

    这里原本是一张四节点的 ``StateGraph``（唯一用到 ``add_conditional_edges``
    的地方）。四个节点本来就是纯 ``state -> 部分 state`` 的变换，中间没有并发、
    没有暂停、没有 checkpoint 需求 —— 也就是说，那张图除了「一个 while 循环」之外
    没有表达任何东西，却带来了一个用超步计数的预算（``recursion_limit``）和一份
    对 LangGraph 的依赖。

    展开成循环之后：预算的单位变回**轮数**，失败路径（主持人不收敛）有一个读得懂
    的上限，而 ``run_brainstorm`` 的对外签名一个字没变。
    """
    nodes = _make_nodes(llm)
    state.update(nodes["facilitator_open"](state))

    steps = 0
    while True:
        state.update(nodes["viewpoints_round"](state))
        state.update(nodes["facilitator_summarize"](state))
        steps += 1
        if state.get("done"):
            break
        if steps >= _MAX_ROUNDS_HARD_CAP:
            # 说出来而不是静默停 —— 一个悄悄截断的讨论，读的人无从知道它被截断了。
            logger.warning("brainstorm hit the %d-round hard cap without the "
                           "facilitator converging; summarising anyway",
                           _MAX_ROUNDS_HARD_CAP)
            break

    state.update(nodes["summarize"](state))
    return state


# ─────────────────────────────────────────────────────────────────────
# Clean entry point.
# ─────────────────────────────────────────────────────────────────────

def run_brainstorm(
    db_path,
    experiment_id: str | None = None,
    *,
    topic: str = "",
    user_viewpoints: list[str] | None = None,
    max_rounds: int = 2,
    llm: Any | None = None,
    memory_store=None,
) -> dict:
    """Run one facilitated brainstorm and return ``{"transcript", "summary"}``.

    Args:
        db_path:         experiment SQLite DB (read for grounding).
        experiment_id:   the experiment to discuss; None = recent-experiments view.
        topic:           discussion topic / question.
        user_viewpoints: opinions the user injects into the discussion.
        max_rounds:      convergence cap (>=1).
        llm:             a langchain chat model OR callable(prompt)->str. None =>
                         rule-based offline discussion (no network / API key).
        memory_store:    a mast.memory.store.MemoryStore. When given, (a) its
                         entries are read into the grounding and (b) the final
                         summary is written back as a kind="brainstorm" memory
                         entry carrying the 非实测 banner.

    Returns:
        {"transcript": list[Turn], "summary": str}. Never raises on a missing /
        empty experiment (the discussion still runs on thin grounding).
    """
    user_viewpoints = [str(v) for v in (user_viewpoints or []) if str(v).strip()]
    max_rounds = max(1, int(max_rounds))

    grounding = gather_grounding(
        db_path, experiment_id, memory_store=memory_store, topic=topic)

    init: BrainstormState = {
        "topic": topic or "",
        "experiment_id": experiment_id or "",
        "grounding": grounding,
        "user_viewpoints": user_viewpoints,
        "max_rounds": max_rounds,
        "round": 0,
        "agenda": [],
        "transcript": [],
        "summary": "",
        "done": False,
    }

    try:
        final = run_discussion(init, llm=llm)
    except Exception as exc:  # pragma: no cover — 讨论本身不该抛
        logger.warning("brainstorm discussion failed (%s); returning rule-based "
                       "fallback", exc)
        init["transcript"] = init.get("transcript") or []
        init["summary"] = _rule_summary(init)
        final = init

    transcript = final.get("transcript", []) or []
    summary = final.get("summary", "") or _rule_summary(final)

    # Persist the summary as a brainstorm memory entry (honesty banner enforced).
    if memory_store is not None and summary.strip():
        content = summary if BRAINSTORM_TAG in summary else f"{BRAINSTORM_TAG}\n\n{summary}"
        ns = f"experiment:{experiment_id}" if experiment_id else _MEMORY_NS_GLOBAL
        slug = (topic or "session").strip().replace("/", "_")[:40] or "session"
        path = f"brainstorms/{slug}.md"
        try:
            memory_store.write(
                ns, path, content,
                title=f"头脑风暴: {topic or '当前实验'}",
                kind="brainstorm",
                experiment_id=experiment_id,
                author="brainstorm",
                tags=["brainstorm", "非实测"],
            )
        except Exception as exc:  # pragma: no cover — memory write best-effort
            logger.debug("brainstorm summary → memory failed: %s", exc)

    return {"transcript": transcript, "summary": summary}


__all__ = [
    "run_brainstorm",
    "build",
    "gather_grounding",
    "grounding_to_text",
    "BrainstormState",
    "BRAINSTORM_TAG",
]
