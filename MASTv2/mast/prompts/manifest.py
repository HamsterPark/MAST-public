"""注入矩阵：哪个 agent 每次调用收到哪些块，按**实际挂载顺序**。

在这个模块出现之前，这个问题只能靠人读七份 ``graph.py`` 加一张共享表推出来 ——
没有文档、没有测试固化它，而登记表里有四条的归属就是这么标错的（``mw.mode_belief``
/ ``mw.experiment_prefs`` / ``mw.instrument_profile`` 实际只挂 IC(+XD)，UI 上却
显示成「全局」，改覆写的人会以为影响全部七家）。

## 顺序从哪来

优先取 :mod:`mast.prompts.builds` 里那次真实建图的中间件列表（``order_source
= "build"``）；进程还没建过图时退回登记表的声明顺序（``"declared"``）。**这个
区别必须说出来**：一份「声明顺序」被当成「实际顺序」读，会让人以为自己在看
运行时事实。

## 这不是调度器

它只描述，不决定。谁挂不挂由中间件自己的 ``AGENTS`` 常量决定（
:func:`mast.agents._shared.shared_stack.applies_to` 读它，登记表的
``agents_from`` 也读它）。测试 ``tests/v2/unit/prompts/test_manifest_matrix.py``
双向对账：栈里有的必须在册，在册的必须在栈里。
"""

from __future__ import annotations

import logging
from typing import Any

from mast.prompts import builds as _builds
from mast.prompts import registry as _reg

logger = logging.getLogger(__name__)

#: 七个流水线 agent + 两个侧通道。**对账真源**是
#: ``mast/agents/orchestrator/graph.py:_AGENT_NAMES``；这里的顺序是展示顺序
#: （决策链从上游到下游），由 ``test_manifest_agents_match_the_orchestrator_list``
#: 钉住不许漂。
AGENTS: tuple[str, ...] = (
    "research_director",
    "literature",
    "experiment_design",
    "instrument_control",
    "data_processing",
    "paper_writing",
    "paper_review",
)

#: 不是流水线 agent，但确实各自发模型调用、各自有系统提示。
SIDE_CHANNELS: tuple[str, ...] = ("orchestrator", "buffer_summarizer")

LABELS: dict[str, str] = {
    "research_director": "科研策划 RD",
    "literature": "文献 LIT",
    "experiment_design": "实验设计 XD",
    "instrument_control": "仪器控制 IC",
    "data_processing": "数据处理 DP",
    "paper_writing": "论文写作 PW",
    "paper_review": "论文审稿 PR",
    "orchestrator": "编排与协调 SUP",
    "buffer_summarizer": "缓冲区摘要 BUF",
}


def all_agents() -> tuple[str, ...]:
    return AGENTS + SIDE_CHANNELS


def label_for(agent: str) -> str:
    return LABELS.get(agent, agent)


def entries_by_middleware() -> dict[str, _reg.PromptEntry]:
    """中间件类名 → 登记条目。反向闸门用它问「这个注入器在册吗」。"""
    out: dict[str, _reg.PromptEntry] = {}
    for e in _reg.entries():
        if e.middleware:
            out.setdefault(e.middleware, e)
    return out


def _order_key(entry: _reg.PromptEntry, mw_order: list[str],
               base_order: list[str]) -> tuple:
    """排序键：先按落点大类，再按建图里的真实先后。

    ``base_order`` 是建图时静态系统提示的**拼接顺序**（IC = 系统提示，然后
    Nanonis 速查）。不用它就只能按 id 字母序，而那会把 ``manual_index`` 排到
    ``system`` 前面 —— 页面于是显示成「先接手册索引，再接系统提示」，与真实
    拼接顺序相反。这一页存在的意义就是别让人读到这种反的东西。
    """
    if entry.when == _reg.WHEN_BUILD_TIME:
        rank = base_order.index(entry.id) if entry.id in base_order else len(base_order)
        return (0, rank, entry.id)
    if entry.middleware and entry.middleware in mw_order:
        return (1, mw_order.index(entry.middleware), entry.id)
    return (2, 0, entry.id)


def manifest_for(agent: str, *, path: str = "group") -> dict[str, Any]:
    """*agent* 每次调用会收到的块，按挂载顺序。

    Returns:
        ``{"agent", "label", "order_source", "blocks": [PromptEntry, …],
        "tool_surface", "tool_surface_note"}``
    """
    rec = _builds.last_build(agent)
    mw_order = list(rec.middleware) if rec else []
    base_order = [i for i, _ in (rec.system_blocks if rec else ())]
    order_source = "build" if mw_order else "declared"

    blocks = [e for e in _reg.entries()
              if _reg.applies_to(e, agent) and path in (e.paths or ("group",))]
    blocks.sort(key=lambda e: _order_key(e, mw_order, base_order))

    surface = rec.tool_surface if rec else None
    if surface is None:
        from mast.prompts import tool_surface as _ts
        surface = _ts.get(agent)
    note = ""
    if surface is None:
        note = ("这个进程还没有建过 %s 的图，所以量不到它的工具面。"
                "建一次（跑一次对话或一次任务）之后这里就有数字了 —— "
                "这里不做静态估算，估出来的数字看起来和实测一模一样。"
                % label_for(agent))
    elif surface.note:
        note = surface.note

    return {
        "agent": agent,
        "label": label_for(agent),
        "order_source": order_source,
        "blocks": blocks,
        "tool_surface": surface,
        "tool_surface_note": note,
        "build": rec,
    }


def matrix(*, path: str = "group") -> dict[str, Any]:
    """行 = 块，列 = agent。回答「不同角色有没有针对性的注入」。"""
    agents = list(all_agents())
    rows: list[dict[str, Any]] = []
    for e in _reg.entries():
        got = _reg.agents_of(e)
        rows.append({
            "entry": e,
            "agents": got,
            "cells": {a: (_reg.ALL_AGENTS in got or a in got) for a in agents},
            "shared": _reg.ALL_AGENTS in got,
            "exclusive": (_reg.ALL_AGENTS not in got and len(got) == 1),
        })
    return {"agents": agents, "rows": rows, "path": path}


__all__ = [
    "AGENTS", "LABELS", "SIDE_CHANNELS",
    "all_agents", "entries_by_middleware", "label_for", "manifest_for", "matrix",
]
