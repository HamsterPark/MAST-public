"""笔记 / 向操作员发问 / 交接报告 —— 外部 agent 与 MAST 其余部分的双向通道。

* **笔记**写进 MAST 自己的记忆库（``CognitionContext.remember``，与实验记录同一个库）。
  内部 agent 每个用户回合会经 ``MemoryRecallMiddleware`` 自动召回相关记忆 —— 所以外部
  agent 在这里留下的结论，下一个接手的内部 agent 不用问就能看到；反过来，内部 agent
  写的记忆也能在这里搜到。命名空间与内部一致：有当前实验记 ``experiment:<id>``，
  否则 ``global``；路径 ``ext/<名字>/<内容哈希>`` ⇒ 同内容重发是幂等的。
* **发问**进心愿单的「agent → 操作员」请求（与内部 agent 的 ``request_user_action``
  同一张表），操作员在界面上答；答复里 ``path`` 是一等字段（最常见的问题是「文件在哪」）。
* **交接报告**把这个调用方的作业、动作与笔记汇成一份 markdown，存进文档库
  （``kind=experiment_report``，``created_by=ext:<名字>``），操作员在「报告」页能看到。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request

from mast.api import direct_exec
from mast.api.ext.common import (
    ExtError,
    caller_of,
    cognition_of,
    memory_store_of,
    scope_ids,
    storage_of,
)
from mast.api.ext.schemas import HandoverBody, NoteBody, RequestBody

logger = logging.getLogger(__name__)

router = APIRouter(tags=["collab"])

_NOTE_KINDS = ("note", "insight", "summary", "hypothesis", "protocol")


def _entry(r: dict, *, content_cap: int = 2000) -> dict:
    c = str(r.get("content") or "")
    return {"namespace": r.get("namespace"), "path": r.get("path"), "title": r.get("title"),
            "kind": r.get("kind"), "author": r.get("author"), "tags": r.get("tags") or [],
            "experiment_id": r.get("experiment_id"), "updated_at": r.get("updated_at"),
            "pinned": bool(r.get("pinned")),
            "content": c if len(c) <= content_cap else c[:content_cap] + "…(截断)"}


# ─────────────────────────────────────────────────────────────────────
# 笔记
# ─────────────────────────────────────────────────────────────────────

@router.post("/notes")
def write_note(body: NoteBody, request: Request):
    """写一条笔记进 MAST 记忆库（内部 agent 会自动召回）。同标题同内容重发不产生重复。"""
    from mast.agents._shared.cognition import _namespace_for

    caller = caller_of(request)
    eid, _ = scope_ids()
    warnings: list[str] = []
    if body.scope == "experiment" and not eid:
        warnings.append("没有当前实验 —— 这条笔记存进了 global 命名空间")
    ns = _namespace_for(eid if body.scope == "experiment" else None)
    kind = body.kind if body.kind in _NOTE_KINDS else "note"
    digest = hashlib.sha1(f"{body.title}\n{body.content}".encode("utf-8")).hexdigest()[:12]
    path = f"ext/{caller.actor}/{digest}"
    tags = [str(t)[:40] for t in (body.tags or [])][:12]
    cog = cognition_of(request)
    try:
        if cog is not None and hasattr(cog, "remember"):
            res = cog.remember(ns, path, body.content, title=body.title, kind=kind, tags=tags,
                               experiment_id=eid if body.scope == "experiment" else None,
                               author=caller.agent_id)
        else:
            store = memory_store_of(request)
            if store is None:
                raise ExtError(503, "not_wired", "记忆库未接线", missing=["cognition"])
            res = store.write(ns, path, body.content, title=body.title, kind=kind, tags=tags,
                              experiment_id=eid if body.scope == "experiment" else None,
                              author=caller.agent_id)
            warnings.append("语义索引未接线：这条笔记只能被子串检索到")
    except ExtError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtError(503, "write_failed", f"笔记写不进去：{exc}") from None
    return {"ok": True, "namespace": res.get("namespace", ns), "path": res.get("path", path),
            "id": res.get("id"), "author": caller.agent_id, "warnings": warnings}


@router.get("/notes")
def search_notes(request: Request, q: str = Query("", max_length=500),
                 scope: str = Query("both", description="both | experiment | global"),
                 limit: int = Query(10, ge=1, le=50)):
    """检索笔记：``q`` 非空时语义召回（当前实验 ∪ global；语义索引不可用时退回子串），
    ``q`` 为空时列最近的。内部 agent 写的记忆也在这里。"""
    from mast.agents._shared.cognition import _namespace_for

    eid, _ = scope_ids()
    store = memory_store_of(request)
    cog = cognition_of(request)
    if store is None:
        raise ExtError(503, "not_wired", "记忆库未接线", missing=["cognition"])
    scope = scope if scope in ("both", "experiment", "global") else "both"
    namespaces = {"both": [_namespace_for(eid), "global"] if eid else ["global"],
                  "experiment": [_namespace_for(eid)] if eid else [],
                  "global": ["global"]}[scope]
    rows: list[dict] = []
    try:
        if q and scope == "both" and cog is not None and hasattr(cog, "recall"):
            rows = list(cog.recall(q, experiment_id=eid, k=limit) or [])
        elif q:
            for ns in namespaces:
                rows += list(store.search(q, namespace=ns, limit=limit) or [])
        else:
            for ns in namespaces:
                rows += list(store.list(ns, limit=limit) or [])
    except Exception as exc:  # noqa: BLE001
        raise ExtError(503, "read_failed", f"记忆库读不了：{exc}") from None
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r.get("namespace"), r.get("path"))
        if key in seen:
            continue
        seen.add(key)
        out.append(_entry(r))
    return {"query": q, "scope": scope, "namespaces": namespaces, "count": len(out[:limit]),
            "notes": out[:limit]}


# ─────────────────────────────────────────────────────────────────────
# 向操作员发问
# ─────────────────────────────────────────────────────────────────────

@router.post("/requests")
def ask_operator(body: RequestBody, request: Request):
    """向操作员发一个请求 / 问题（界面上会亮起来）。同一个未答的问题重发不产生重复。
    **不要等在这里** —— 答复是异步的，之后用 GET /requests/{id} 或简报的
    ``operator_requests`` 段看。"""
    from mast.wishlist import post_agent_request

    caller = caller_of(request)
    eid, _ = scope_ids()
    rec = post_agent_request(caller.agent_id, body.message, kind=(body.kind or "question"),
                             experiment_id=eid)
    if not isinstance(rec, dict) or rec.get("error"):
        raise ExtError(422, "request_rejected", str((rec or {}).get("error") or "请求没被接受"))
    return {"ok": True, "request": direct_exec.jsonable(rec)}


#: 请求状态的对外名字 → 心愿单里的存储名。未答的在心愿单里叫 ``pending``；``open`` 是同义词。
_REQUEST_STATUS_ALIASES = {"pending": "pending", "open": "pending", "done": "done",
                           "dismissed": "dismissed"}


@router.get("/requests")
def my_requests(request: Request,
                status: str = Query("", description="pending (= open) | done | dismissed；空 = 全部")):
    """本调用方发过的请求（未答的在前）与答复。"""
    from mast.wishlist import list_agent_requests

    caller = caller_of(request)
    wanted = None
    if status.strip():
        wanted = _REQUEST_STATUS_ALIASES.get(status.strip().lower())
        if wanted is None:           # 未知状态回 422，不回一个永远为空的列表
            raise ExtError(422, "unknown_status", f"没有状态 {status!r}",
                           available=sorted(_REQUEST_STATUS_ALIASES))
    rows = [r for r in list_agent_requests(wanted)
            if r.get("agent_id") == caller.agent_id]
    return {"count": len(rows), "requests": direct_exec.jsonable(rows)}


@router.get("/requests/{request_id}")
def get_request(request_id: str, request: Request):
    """单条请求与答复（``note`` 是文字答复，``path`` 是操作员给的文件/目录路径）。"""
    from mast.wishlist import get_request as _get

    rec = _get(request_id)
    if rec is None:
        raise ExtError(404, "unknown_request", f"没有请求 {request_id!r}")
    return {"request": direct_exec.jsonable(rec)}


# ─────────────────────────────────────────────────────────────────────
# 交接报告
# ─────────────────────────────────────────────────────────────────────

def _since_ok(ts: str | None, since: str | None) -> bool:
    if not since:
        return True
    return str(ts or "") >= str(since)


def _signed_by(context: Any, agent_id: str) -> bool:
    """v1 ``actions.context`` 是 ``ext:<actor>`` 或 ``ext:<actor>/<session>``。
    不能用裸前缀：``ext:tester`` 是 ``ext:tester2`` 的前缀。"""
    c = str(context or "")
    return c == agent_id or c.startswith(agent_id + "/")


@router.post("/handover")
def handover(body: HandoverBody, request: Request):
    """生成交接报告存进文档库。汇总：你写的总结与下一步、你这次提交的作业及结局、
    记在当前实验里署你名字的动作、你写的笔记。操作员在「报告」页能看到。"""
    caller = caller_of(request)
    eid, sid = scope_ids()
    st = storage_of(request)
    jm = getattr(request.app.state, "jobs", None)

    jobs = [j for j in (jm.list(actor=caller.actor, limit=200) if jm else [])
            if _since_ok(j.created_at, body.since)]
    actions: list[dict] = []
    if st is not None and eid and hasattr(st, "recent_actions"):
        try:
            actions = [r for r in st.recent_actions(eid, 500)
                       if _signed_by(r.get("context"), caller.agent_id)
                       and _since_ok(r.get("timestamp"), body.since)]
        except Exception as exc:  # noqa: BLE001
            logger.debug("handover: 读动作失败(%s)", exc)
    notes: list[dict] = []
    store = memory_store_of(request)
    if store is not None:
        try:
            from mast.agents._shared.cognition import _namespace_for

            for ns in ([_namespace_for(eid)] if eid else []) + ["global"]:
                notes += [r for r in (store.list(ns, limit=200) or [])
                          if r.get("author") == caller.agent_id
                          and _since_ok(r.get("updated_at"), body.since)]
        except Exception as exc:  # noqa: BLE001
            logger.debug("handover: 读笔记失败(%s)", exc)

    steps = body.next_steps if isinstance(body.next_steps, list) else (
        [body.next_steps] if str(body.next_steps or "").strip() else [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = body.title.strip() or f"外部 agent 交接 · {caller.actor} · {now}"
    lines = [f"# {title}", "",
             f"- 署名：`{caller.agent_id}`（会话 `{caller.thread_id}`）",
             f"- 实验 / 样品：`{eid or '无'}` / `{sid or '无'}`",
             f"- 生成时间：{now}", "", "## 总结", "", body.summary.strip(), ""]
    if steps:
        lines += ["## 下一步", ""] + [f"- {s}" for s in steps if str(s).strip()] + [""]
    lines += ["## 作业", ""]
    if jobs:
        lines += ["| 时间 | 技能 | 结局 | 说明 |", "|---|---|---|---|"]
        for j in reversed(jobs):
            res = j.result or {}
            why = (res.get("error") or res.get("summary") or "").replace("|", "/").replace("\n", " ")
            lines.append(f"| {j.created_at[:19]} | {j.skill} | {j.state}"
                         f"{'（' + j.refused_by + '）' if j.refused_by else ''} | {why[:160]} |")
    else:
        lines.append("（本进程里没有这个调用方的作业记录）")
    lines += ["", "## 实验记录里署这个名字的动作", ""]
    if actions:
        for r in reversed(actions):
            mark = "✓" if r.get("success") else "✗"
            lines.append(f"- {str(r.get('timestamp') or '')[:19]} {r.get('skill_name')} {mark}"
                         + (f" — {str(r.get('error'))[:160]}" if r.get("error") else ""))
    else:
        lines.append("（没有 —— 没有当前实验时动作不进记录）" if not eid else "（没有）")
    lines += ["", "## 笔记", ""]
    if notes:
        for r in notes:
            lines.append(f"- **{r.get('title') or r.get('path')}**（{r.get('kind')}）："
                         f"{' '.join(str(r.get('content') or '').split())[:300]}")
    else:
        lines.append("（没有）")
    md = "\n".join(lines) + "\n"
    try:
        from mast.documents.store import store as doc_store

        res = doc_store().save(text=md, kind="experiment_report", title=title,
                               experiment_id=eid, sample_id=sid,
                               created_by=caller.agent_id, note="外部 agent 交接报告",
                               use_turn_context=False)
    except Exception as exc:  # noqa: BLE001
        raise ExtError(503, "save_failed", f"交接报告存不进文档库：{exc}", markdown=md) from None
    if not getattr(res, "ok", False):
        raise ExtError(503, "save_failed", str(getattr(res, "error", "") or "文档库拒绝了保存"),
                       markdown=md)
    return {"ok": True, "doc_id": res.doc_id, "version": res.version, "path": res.path,
            "title": title, "jobs": len(jobs), "actions": len(actions), "notes": len(notes)}


__all__ = ["router"]
