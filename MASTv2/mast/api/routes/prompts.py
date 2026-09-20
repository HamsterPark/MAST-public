"""「上下文注入」的**只读**读侧：谁每次调用收到什么、各占多大。

与 ``routes/admin.py`` 里那一组 ``/admin/prompts*`` 的分工：

* ``/admin/prompts*``  —— 管理员改话术（读 + 写 + 覆写历史），在 PIN 门后面；
* 这里             —— 面向用户的**只读**视图，回答两个 admin 侧答不了的问题：
                     「不同角色有针对性的注入吗」与「这一次调用由什么构成」。

只读是这个模块的不变式，不是巧合：它不在 PIN 门后面，所以它一旦能写，PIN 就形同
虚设。``tests/v2/unit/api/test_prompts_manifest_api.py``
的 ``test_the_read_side_has_no_write_routes`` 钉着这一条。

HOUSE STYLE（同 routes/admin.py）：模块级 ``router``；每个 handler 收
``request: Request``；每个都有 ``response_model``；**优雅降级** —— 单独起的 API
进程也要能启动，读不到就 ``degraded=true``，绝不 500；重后端在 handler 里
lazy-import。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request, Response

from mast.api.schemas_prompts import (
    AgentInfo,
    AgentInjectionResponse,
    InjectionBlock,
    InjectionMatrixResponse,
    LatestCaptureResponse,
    ToolPackInfo,
    ToolSurfaceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prompts"])


# ── 组装 ─────────────────────────────────────────────────────────────────

def _preview(text: str, limit: int = 220) -> str:
    one = " ".join((text or "").split())
    return one if len(one) <= limit else one[:limit] + "…"


def _block(entry: Any) -> InjectionBlock:
    from mast.api.routes.admin import _prompt_bodies
    from mast.prompts import registry as reg

    default_text, override_text, effective, reason = _prompt_bodies(entry)
    agents = list(reg.agents_of(entry))
    return InjectionBlock(
        id=entry.id,
        label=entry.label,
        category=entry.category,
        agent=entry.agent,
        availability=entry.availability,
        source=entry.source,
        note=entry.note,
        overridable=entry.overridable,
        overridden=bool((override_text or "").strip()),
        default_chars=len(default_text),
        effective_chars=len(effective),
        preview=_preview(effective),
        unavailable_reason=reason,
        agents=agents,
        when=entry.when,
        position=entry.position,
        middleware=entry.middleware,
        paths=list(entry.paths or ()),
        requires=entry.requires,
        exclusive=(reg.ALL_AGENTS not in agents and len(agents) == 1),
    )


def _tool_surface(rec: Any, surface: Any) -> ToolSurfaceInfo | None:
    if surface is None:
        return None
    from mast.agents._shared import tool_packs as tp

    catalog = getattr(rec, "tool_catalog", None) if rec is not None else None
    packs: list[ToolPackInfo] = []
    core_tools = core_chars = None
    if catalog is not None:
        core_tools = len(catalog.core)
        core_chars = catalog.core_chars
        packs = [ToolPackInfo(name=p, label=tp.pack_label(p),
                              tools=len(catalog.tools_by_pack.get(p, ())),
                              chars=catalog.chars_by_pack.get(p, 0))
                 for p in catalog.known_packs()]
    return ToolSurfaceInfo(
        count=surface.count, schema_chars=surface.schema_chars,
        top=[{"name": n, "chars": c} for n, c in surface.top],
        fmt=surface.fmt, core_tools=core_tools, core_chars=core_chars,
        packs=packs)


# ── 路由 ─────────────────────────────────────────────────────────────────

@router.get("/prompts/manifest", response_model=InjectionMatrixResponse)
def injection_matrix(request: Request) -> InjectionMatrixResponse:
    """行 = 注入块，列 = agent。回答「不同角色有针对性的注入吗」。"""
    try:
        from mast.prompts import builds, manifest
    except Exception as exc:  # noqa: BLE001
        logger.info("prompts manifest unavailable: %s", exc)
        return InjectionMatrixResponse(degraded=True,
                                       note="注入清单模块读不到（不是「没有注入」）。")
    agents = list(manifest.all_agents())
    infos: list[AgentInfo] = []
    for a in agents:
        rec = builds.last_build(a)
        surf = rec.tool_surface if rec else None
        infos.append(AgentInfo(
            id=a, label=manifest.label_for(a),
            tool_count=(surf.count if surf else None),
            tool_chars=(surf.schema_chars if surf else None),
            system_chars=(rec.system_chars if rec else None)))

    blocks: list[InjectionBlock] = []
    cells: dict[str, dict[str, bool]] = {}
    for entry in _entries():
        b = _block(entry)
        blocks.append(b)
        cells[entry.id] = {a: (("*" in b.agents) or (a in b.agents)) for a in agents}

    note = ""
    if all(i.tool_count is None for i in infos):
        note = ("这个进程还没有建过任何 agent 的图，所以工具面栏是空的 —— "
                "跑一次对话或一次任务之后就有数字了。这里不做静态估算。")
    return InjectionMatrixResponse(agents=infos, blocks=blocks, cells=cells,
                                   note=note)


def _entries():
    from mast.prompts import registry as reg

    return reg.entries()


@router.get("/prompts/manifest/{agent}", response_model=AgentInjectionResponse)
def agent_injection(agent: str, request: Request,
                    response: Response) -> AgentInjectionResponse:
    """*agent* 每次调用收到的块，按实际挂载顺序，外加它的工具面。"""
    try:
        from mast.prompts import manifest
    except Exception as exc:  # noqa: BLE001
        logger.info("prompts manifest unavailable: %s", exc)
        return AgentInjectionResponse(agent=agent, degraded=True)

    if agent not in manifest.all_agents():
        response.status_code = 404
        return AgentInjectionResponse(agent=agent, label=agent,
                                      tool_surface_note=f"没有名为 {agent} 的 agent。")

    m = manifest.manifest_for(agent)
    rec = m.get("build")
    return AgentInjectionResponse(
        agent=agent,
        label=m["label"],
        blocks=[_block(e) for e in m["blocks"]],
        order_source=m["order_source"],
        tool_surface=_tool_surface(rec, m.get("tool_surface")),
        tool_surface_note=m.get("tool_surface_note", ""),
        middleware=list(rec.middleware) if rec else [],
    )


#: 没有快照的四种原因。**分开说** —— 一个统一的「暂无数据」把它们揉成一种，
#: 而它们没有一种能靠再点一次解决。
_REASONS = {
    "disabled": "快照记录已关闭，所以这里是空的 —— 不是没有发生过模型调用。",
    "no_calls": "本进程还没有发生过任何模型调用。这里只记录真实请求，不做离线模拟。",
    "no_calls_for_agent": "本进程跑过模型调用，但没有一次是这个 agent 发的。",
    "unknown_agent": "没有这个 agent。",
}


@router.get("/prompts/capture/latest/{agent}", response_model=LatestCaptureResponse)
def latest_capture(agent: str, request: Request,
                   response: Response) -> LatestCaptureResponse:
    """这个 agent **最近一次主模型调用**收到的完整消息列表 + 分段归属。

    「最近一次」指主模型节点那一次：中间件内部的压缩 / 精炼子调用用的是同一个
    source，不筛的话这里会返回一条摘要请求，而它看起来和真的一模一样。
    """
    try:
        from mast.prompts import capture, manifest
    except Exception as exc:  # noqa: BLE001
        logger.info("prompt capture unavailable: %s", exc)
        return LatestCaptureResponse(found=False, reason_code="degraded",
                                     reason="注入记录模块读不到。", degraded=True)

    if agent not in manifest.all_agents():
        response.status_code = 404
        return LatestCaptureResponse(found=False, reason_code="unknown_agent",
                                     reason=_REASONS["unknown_agent"])

    ring = capture.get_ring()
    snap = ring.latest(agent)
    if snap is None:
        response.status_code = 404
        if not ring.enabled:
            code = "disabled"
        elif ring.total_seen == 0:
            code = "no_calls"
        else:
            code = "no_calls_for_agent"
        return LatestCaptureResponse(found=False, reason_code=code,
                                     reason=_REASONS[code])

    return LatestCaptureResponse(
        found=True,
        index=0,
        seq=snap.seq,
        ts=snap.ts,
        age_s=max(0.0, time.time() - snap.ts),
        source=snap.source,
        model_id=snap.model_id,
        provider=snap.provider,
        messages=[{"role": m.role, "content": m.content, "chars": m.chars,
                   "truncated": m.truncated, "blocks": m.blocks}
                  for m in snap.messages],
        total_chars=snap.total_chars,
        dropped_messages=snap.dropped_messages,
        input_tokens=snap.input_tokens,
        output_tokens=snap.output_tokens,
        cache_read_tokens=snap.cache_read_tokens,
        cache_creation_tokens=snap.cache_creation_tokens,
        tokens_source=snap.tokens_source,
        tool_count=snap.tool_count,
        tools_chars=snap.tools_chars,
        tools_top=[{"name": n, "chars": c} for n, c in (snap.tools_top or [])],
        tools_source=snap.tools_source,
        blocks_source=snap.blocks_source,
        node=snap.node,
        thread_id=snap.thread_id,
    )


__all__ = ["router"]
