"""健康 / 铁律 / 状态 / 简报 / 作用域 / 急停。

## 简报为什么是这个形状

内部 agent 每一轮都由中间件白拿一批上下文块（针尖、仪器档案、live 读数、操作员偏好、
续工、记忆召回），外部 agent 只拿到裸的技能返回 —— 它不知道针尖是钨还是 PtIr、不知道
这台仪器 Z 朝哪边是退针、不知道十分钟前另一个 agent 在同一块样品上做了什么。简报把
那些块原样渲染出来，再加上「在我之前谁做了什么」。

三条纪律：

* **逐段独立降级**：任何一段读不到只让那一段 ``ok:false`` 并进 ``degraded``，其余照给。
  读不到 ≠ 一个值 —— 一段缺席的针尖块不能被读成「没有针尖」。
* **零硬件 I/O**：只读缓存快照与内存状态，绝不 ``refresh()``、绝不发 TCP（不走
  ``records_export._scan_search_dirs``，它会去问 Nanonis 会话目录）。
* **有界**：每段文字截断；续工块会全量加载实验动作，缓存 10 秒。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from fastapi import APIRouter, Query, Request

from mast.api import direct_exec
from mast.api.ext import guide as _guide
from mast.api.ext.common import (
    API_VERSION,
    Caller,
    ExtError,
    active_log,
    caller_of,
    cognition_of,
    ctx_of,
    memory_store_of,
    operating_mode,
    registry_of,
    runtime_of,
    scope_ids,
    storage_of,
)
from mast.api.ext.schemas import EstopBody, ScopeBody

logger = logging.getLogger(__name__)

router = APIRouter(tags=["overview"])

#: 单段文字上限 / 整份简报文字上限（字符）。
SECTION_TEXT_CAP = 2400
BRIEFING_TEXT_CAP = 12000

_LIVE_FIELDS = ("bias_v", "current_a", "setpoint_a", "z_pos_m", "x_pos_m", "y_pos_m",
                "z_controller_on", "z_controller_status", "withdrawn", "scan_running", "stale")


def _cap(text: str, n: int = SECTION_TEXT_CAP) -> str:
    t = str(text or "").strip()
    return t if len(t) <= n else t[:n] + "\n…(截断)"


# ─────────────────────────────────────────────────────────────────────
# 健康 / 铁律
# ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health(request: Request):
    """外部面自己的健康：各子系统接没接线。``missing`` 非空时相关端点会 503。"""
    ctx, rt = ctx_of(request), runtime_of(request)
    wired = {
        "runtime": rt is not None,
        "registry": registry_of(request) is not None,
        "pool": getattr(ctx, "connection_pool", None) is not None,
        "state": (getattr(ctx, "state", None) or getattr(rt, "_state", None)) is not None,
        "storage": storage_of(request) is not None,
        "cognition": cognition_of(request) is not None,
        "jobs": getattr(request.app.state, "jobs", None) is not None,
    }
    try:
        from mast.api.version import get_version

        ver = get_version()
    except Exception:  # noqa: BLE001
        ver = "unknown"
    return {"ok": True, "api_version": API_VERSION, "mast_version": ver,
            "wired": wired, "missing": [k for k, v in wired.items() if not v]}


@router.get("/guide")
def guide(lang: str = Query("en", description="zh | en")):
    """面向 agent 的铁律清单。完整指南在仓库的 docs/external/。"""
    lang = "zh" if str(lang).lower().startswith("zh") else "en"
    return {"lang": lang, "rules": _guide.rules(lang), "docs_hint": _guide.DOCS_HINT[lang]}


# ─────────────────────────────────────────────────────────────────────
# 状态
# ─────────────────────────────────────────────────────────────────────

def _live(request: Request) -> dict | None:
    ctx, rt = ctx_of(request), runtime_of(request)
    state = getattr(ctx, "state", None) or getattr(rt, "_state", None)
    snap = getattr(state, "snapshot", None)
    if not callable(snap):
        return None
    hw = snap()
    return {k: direct_exec.jsonable(getattr(hw, k, None)) for k in _LIVE_FIELDS}


def _connection(request: Request) -> dict:
    pool = getattr(ctx_of(request), "connection_pool", None)
    roles: dict[str, bool] = {}
    if pool is not None:
        for role in ("main", "monitor", "data", "emergency"):
            try:
                pool.get(role)          # 没连上就抛；不重连、不发命令
                roles[role] = True
            except Exception:  # noqa: BLE001
                roles[role] = False
    return {"wired": pool is not None, "roles": roles}


def _abort(request: Request) -> dict | None:
    fn = getattr(runtime_of(request), "emergency_latch_state", None)
    if not callable(fn):
        return None
    st = fn() or {}
    return {"set": bool(st.get("abort_set")), "emergency": bool(st.get("latched")),
            "why": str(st.get("why") or "")}


def _lock() -> dict:
    from mast.core.instrument_lock import instrument_lock

    snap = instrument_lock().snapshot()
    if not snap:
        return {"held": False}
    return {"held": True, "owner": snap.get("owner"), "skill": snap.get("skill"),
            "held_s": round(float(snap.get("held_s") or 0.0), 1)}


def _scope(request: Request) -> dict:
    eid, sid = scope_ids()
    st = storage_of(request)
    exp = st.get_experiment(eid) if (st is not None and eid) else None
    smp = st.get_sample(sid) if (st is not None and sid) else None
    rt = runtime_of(request)
    return {
        "experiment": ({"id": eid, "name": (exp or {}).get("name") or "",
                        "goal": (exp or {}).get("goal_text") or ""} if eid else None),
        "sample": ({"id": sid, "name": (smp or {}).get("name") or "",
                    "sample_type": (smp or {}).get("sample_type") or ""} if sid else None),
        "recording": {
            "v1": bool(st is not None and eid),
            "v2": bool(getattr(rt, "_v2_repos", None) is not None
                       and getattr(rt, "_v2_eid", None)),
            "note": ("" if eid else "没有当前实验：你的动作不会进实验记录（v1 外键拒绝空实验）。"
                     "先用 POST /scope 选择或新建实验。"),
        },
    }


def _status(request: Request) -> tuple[dict, list[str]]:
    degraded: list[str] = []
    out: dict[str, Any] = {"mode": operating_mode()}
    for key, fn in (("abort", lambda: _abort(request)), ("lock", _lock),
                    ("connection", lambda: _connection(request)),
                    ("live", lambda: _live(request)), ("scope", lambda: _scope(request))):
        try:
            out[key] = fn()
            if out[key] is None:
                degraded.append(key)
        except Exception as exc:  # noqa: BLE001
            out[key] = None
            degraded.append(key)
            logger.debug("ext status %s failed: %s", key, exc)
    jm = getattr(request.app.state, "jobs", None)
    out["jobs"] = ({"running": jm.running_count(), "max": jm.max_concurrent}
                   if jm is not None else None)
    out["degraded"] = degraded
    return out, degraded


@router.get("/status")
def status(request: Request):
    """快速状态：运行模式、中止/急停闩、仪器锁持有者、连接、live 读数、作用域、作业数。
    只读内存与缓存快照，零硬件 I/O。"""
    out, _ = _status(request)
    return out


def _status_text(s: dict) -> str:
    lines = [f"- 运行模式 mode: {s.get('mode')}"]
    ab = s.get("abort")
    if ab is None:
        lines.append("- 中止/急停: 读不到")
    elif ab.get("set") or ab.get("emergency"):
        lines.append(f"- ⚠ 中止事件已置位（急停闩={ab.get('emergency')}）：{ab.get('why') or '未留原因'}"
                     " —— 此时一切写动作都会被拒；急停闩由操作员解。")
    else:
        lines.append("- 中止/急停: 未置位")
    lk = s.get("lock") or {}
    if lk.get("held"):
        lines.append(f"- 仪器锁: 被 {lk.get('owner')} 持有（{lk.get('skill')}，{lk.get('held_s')} s）")
    else:
        lines.append("- 仪器锁: 空闲")
    conn = s.get("connection") or {}
    roles = conn.get("roles") or {}
    if not conn.get("wired"):
        lines.append("- Nanonis 连接: 未接线")
    else:
        marks = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in roles.items())
        lines.append(f"- Nanonis 连接: {marks}")
    lv = s.get("live") or {}
    if lv:
        lines.append(f"- live: bias={lv.get('bias_v')} V, I={lv.get('current_a')} A, setpoint={lv.get('setpoint_a')} A, "
                     f"Z={lv.get('z_pos_m')} m, Z控制={lv.get('z_controller_status') or lv.get('z_controller_on')}, "
                     f"扫描中={lv.get('scan_running')}, 退针={lv.get('withdrawn')}, stale={lv.get('stale')}")
    jb = s.get("jobs") or {}
    if jb:
        lines.append(f"- 外部作业: 在跑 {jb.get('running')} / 上限 {jb.get('max')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 简报
# ─────────────────────────────────────────────────────────────────────

_RESUME_CACHE: dict[str, Any] = {"key": None, "at": 0.0, "text": None}


def _sec_status(request, caller):
    s, _ = _status(request)
    return {"text": _status_text(s), "data": s}


def _sec_scope(request, caller):
    sc = _scope(request)
    e, smp = sc.get("experiment"), sc.get("sample")
    lines = [f"- 实验: {e['name']} (id {e['id']})" if e else "- 实验: 无 —— 先 POST /scope",
             f"- 样品: {smp['name']} (id {smp['id']}, {smp.get('sample_type') or '类型未填'})"
             if smp else "- 样品: 无 —— 产数据的技能（扫描/谱学）会在门口被拒"]
    if e and e.get("goal"):
        lines.append(f"- 目标: {_cap(e['goal'], 300)}")
    if sc["recording"]["note"]:
        lines.append(f"- ⚠ {sc['recording']['note']}")
    return {"text": "\n".join(lines), "data": sc}


def _sec_resume(request, caller):
    from mast.agents._shared.resume_context import build_experiment_resume_block

    eid, sid = scope_ids()
    key = f"{eid}/{sid}"
    now = time.monotonic()
    if _RESUME_CACHE["key"] == key and now - _RESUME_CACHE["at"] < 10.0:
        text = _RESUME_CACHE["text"]
    else:
        text = build_experiment_resume_block(
            active_log(), storage_of(request),
            plan_store=getattr(runtime_of(request), "_plan_store", None))
        _RESUME_CACHE.update(key=key, at=now, text=text)
    return {"text": _cap(text or "（没有当前实验，无可续工的上下文）")}


def _sec_tip(request, caller):
    from mast.core.instrument_profile import get_profile
    from mast.core.tip_state import current_tip_facts, format_tip_block, get_current_tip

    text = format_tip_block(get_current_tip(), get_profile())
    return {"text": _cap(text or "（针尖未登记 —— 修针类技能会退回通用保守参数）"),
            "data": direct_exec.jsonable(current_tip_facts())}


def _sec_instrument(request, caller):
    from mast.core.instrument_profile import format_profile_block, get_profile

    return {"text": _cap(format_profile_block(get_profile()) or "（仪器档案未设置）")}


def _sec_live(request, caller):
    from mast.agents._shared.live_state_mw import format_live_state_block

    ctx, rt = ctx_of(request), runtime_of(request)
    state = getattr(ctx, "state", None) or getattr(rt, "_state", None)
    if state is None:
        raise RuntimeError("仪器状态未接线")
    return {"text": _cap(format_live_state_block(state.snapshot()) or "（live 读数为空）")}


def _sec_prefs(request, caller):
    from mast.agents._shared.experiment_prefs import format_prefs_block, get_prefs

    return {"text": _cap(format_prefs_block(get_prefs()) or "（操作员没有设置默认参数偏好）")}


def _sec_safety(request, caller):
    from mast.api.routes.safety import get_latches
    from mast.api.safety_view import in_force_safety_limits

    limits, source = in_force_safety_limits(ctx_of(request))
    lim = limits.model_dump() if hasattr(limits, "model_dump") else dict(limits)
    keep = {k: v for k, v in lim.items()
            if any(s in k for s in ("bias", "setpoint", "current", "xy", "scan_size", "z_"))}
    lat = get_latches(request)
    latched = [f"{s.name}（{s.why or '未留原因'}；解除：{s.release or '无'}）"
               for s in (lat.latches or []) if s.latched]
    unreadable = list(lat.unreadable or [])
    lines = [f"- 生效包络（来源 {source}）: " + ", ".join(f"{k}={v}" for k, v in sorted(keep.items()))]
    lines.append("- 挂着的闩: " + ("；".join(latched) if latched else "无"))
    if unreadable:
        lines.append(f"- 读不到的闩: {unreadable}")
    return {"text": _cap("\n".join(lines)),
            "data": {"limits_source": source, "limits": direct_exec.jsonable(keep),
                     "latched": latched, "unreadable": unreadable}}


def _sec_recent_actions(request, caller):
    eid, _ = scope_ids()
    st = storage_of(request)
    if not eid:
        return {"text": "（没有当前实验）", "data": []}
    if st is None or not hasattr(st, "recent_actions"):
        raise RuntimeError("实验记录未接线")
    rows = st.recent_actions(eid, 12)
    lines = []
    for r in rows:
        who = r.get("context") or "MAST agent"
        mark = "✓" if r.get("success") else ("✗" if r.get("success") is False else "?")
        err = f" — {r.get('error')[:120]}" if r.get("error") else ""
        lines.append(f"- {str(r.get('timestamp') or '')[:19]} [{who}] {r.get('skill_name')} {mark}{err}")
    return {"text": "\n".join(lines) or "（本实验还没有动作）", "data": rows}


def _static_roots(request) -> list[str]:
    """数据根目录 —— **不**去问 Nanonis 会话目录（那会发 TCP）。"""
    rt = runtime_of(request)
    dirs: list[str] = []
    for d in (getattr(rt, "_scan_search_dirs", None) or ()):
        if d:
            dirs.append(str(d))
    cfg = getattr(ctx_of(request), "config", None)
    if getattr(cfg, "experiments_dir", None):
        dirs.append(str(cfg.experiments_dir))
    try:
        from mast._runtime_paths import project_root

        dirs += [str(project_root() / "working-sessions"), str(project_root() / "experiments")]
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast.core.experiment_paths import experiment_root

        dirs.append(str(experiment_root()))
    except Exception:  # noqa: BLE001
        pass
    seen: set[str] = set()
    return [d for d in dirs if not (d in seen or seen.add(d))]


def _sec_recent_files(request, caller):
    from mast.webui.scan_preview import collect_scan_stats

    stats = collect_scan_stats(*_static_roots(request))[:8]
    rows = [{"path": str(s.path), "name": s.path.name, "mtime_ns": s.mtime_ns,
             "size_bytes": s.size_bytes} for s in stats]
    text = "\n".join(f"- {r['name']}  ({r['size_bytes']} B)  {r['path']}" for r in rows)
    return {"text": text or "（数据目录里还没有扫描/谱文件）", "data": rows}


def _sec_alarms(request, caller):
    rt = runtime_of(request)
    log = getattr(rt, "_env_alarm_log", None)
    if not isinstance(log, list):
        raise RuntimeError("环境监控未接线")
    rows = [dict(r) for r in log[-5:] if isinstance(r, dict)]
    mon = getattr(ctx_of(request), "environment_monitor", None)
    overall = "unknown"
    if mon is not None:
        try:
            overall = str(mon.overall_status() or "ok")
        except Exception:  # noqa: BLE001
            overall = "unknown"
    lines = [f"- 环境总体: {overall}"] + [
        f"- {r.get('t')} {r.get('sensor')}: {r.get('status')} ({r.get('value')} {r.get('unit') or ''})"
        for r in rows]
    return {"text": "\n".join(lines), "data": {"overall": overall, "recent": direct_exec.jsonable(rows)}}


def _sec_notes(request, caller):
    from mast.agents._shared.cognition import _namespace_for

    store = memory_store_of(request)
    if store is None:
        raise RuntimeError("记忆库未接线")
    eid, _ = scope_ids()
    rows = store.list(_namespace_for(eid), limit=6) if eid else []
    lines = [f"- [{r.get('kind')}] {r.get('title') or r.get('path')} —— {r.get('author') or '?'}"
             f"：{_cap(' '.join(str(r.get('content') or '').split()), 160)}" for r in rows]
    return {"text": "\n".join(lines) or "（本实验还没有笔记）",
            "data": [{k: r.get(k) for k in ("namespace", "path", "title", "kind", "author",
                                            "updated_at", "pinned")} for r in rows]}


def _sec_recording(request, caller):
    sc = _scope(request)
    rec = sc["recording"]
    text = (f"- 这次会话的动作署名: {caller.thread_id}\n"
            f"- v1 实验记录: {'会写入' if rec['v1'] else '不会写入'}；v2: {'会写入' if rec['v2'] else '不会写入'}")
    if rec["note"]:
        text += f"\n- ⚠ {rec['note']}"
    return {"text": text, "data": {**rec, "agent_id": caller.agent_id, "thread_id": caller.thread_id}}


def _sec_jobs(request, caller):
    jm = getattr(request.app.state, "jobs", None)
    if jm is None:
        raise RuntimeError("作业管理器未接线")
    rows = jm.list(actor=caller.actor, limit=6)
    lines = [f"- {j.job_id} {j.skill} → {j.state}" + (f"（{j.refused_by}）" if j.refused_by else "")
             for j in rows]
    return {"text": "\n".join(lines) or "（你还没有提交过作业）",
            "data": [{"job_id": j.job_id, "skill": j.skill, "state": j.state,
                      "created_at": j.created_at} for j in rows]}


def _sec_operator_requests(request, caller):
    from mast.wishlist import list_agent_requests

    rows = [r for r in list_agent_requests() if r.get("agent_id") == caller.agent_id][:6]
    lines = []
    for r in rows:
        # 说明与路径两个都给：操作员常常只在路径里作答（「在这个目录」）。
        ans = "；".join(str(x) for x in (r.get("note"), r.get("path")) if x)
        lines.append(f"- {r.get('id')} [{r.get('status')}] {_cap(r.get('message') or '', 120)}"
                     + (f" → 答复：{_cap(ans, 200)}" if ans else ""))
    return {"text": "\n".join(lines) or "（你没有向操作员发过请求）",
            "data": direct_exec.jsonable(rows)}


SECTIONS: dict[str, Callable[[Request, Caller], dict]] = {
    "status": _sec_status,
    "scope": _sec_scope,
    "resume": _sec_resume,
    "tip": _sec_tip,
    "instrument": _sec_instrument,
    "live": _sec_live,
    "prefs": _sec_prefs,
    "safety": _sec_safety,
    "recent_actions": _sec_recent_actions,
    "recent_files": _sec_recent_files,
    "alarms": _sec_alarms,
    "notes": _sec_notes,
    "recording": _sec_recording,
    "jobs": _sec_jobs,
    "operator_requests": _sec_operator_requests,
}


@router.get("/briefing")
def briefing(request: Request,
             sections: str = Query("", description="逗号分隔的段名；空 = 全部")):
    """分段简报。每段 ``{ok, text?, data?, error?}``；读不到的段名进 ``degraded``。
    顶层 ``text`` 是各段文字拼成的一份 markdown（给模型直接读）。零硬件 I/O。"""
    caller = caller_of(request)
    wanted = [s.strip() for s in (sections or "").split(",") if s.strip()] or list(SECTIONS)
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise ExtError(422, "unknown_section", f"没有这些段：{unknown}", available=list(SECTIONS))
    out: dict[str, dict] = {}
    degraded: list[dict] = []
    for name in wanted:
        try:
            sec = SECTIONS[name](request, caller) or {}
            out[name] = {"ok": True, **sec}
        except Exception as exc:  # noqa: BLE001 — 一段坏了不许拖垮其余
            reason = f"{type(exc).__name__}: {exc}"[:300]
            out[name] = {"ok": False, "error": reason}
            degraded.append({"section": name, "reason": reason})
    parts = []
    for name in wanted:
        sec = out[name]
        body = sec.get("text") if sec.get("ok") else f"（读不到：{sec.get('error')}）"
        parts.append(f"## {name}\n{body}")
    text = "\n\n".join(parts)
    if len(text) > BRIEFING_TEXT_CAP:
        text = text[:BRIEFING_TEXT_CAP] + "\n…(简报截断；用 ?sections= 单独取某几段)"
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "actor": caller.agent_id,
            "sections": out, "degraded": degraded, "text": text}


# ─────────────────────────────────────────────────────────────────────
# 作用域
# ─────────────────────────────────────────────────────────────────────

@router.get("/scope")
def get_scope(request: Request):
    """当前实验与样品，以及「这次的动作会不会进记录」。"""
    return _scope(request)


@router.post("/scope")
def set_scope(body: ScopeBody, request: Request):
    """切换或新建实验 / 样品。给 ``id`` 就切换到已有的；给 ``name`` 就按名字复用或新建。

    仪器正被占用时默认拒绝（在跑的动作会记到新作用域下）；``force: true`` 才切。
    """
    caller = caller_of(request)
    log, st = active_log(), storage_of(request)
    if log is None or st is None:
        raise ExtError(503, "not_wired", "记录系统未接线",
                       missing=[n for n, v in (("experiment_log", log), ("storage", st)) if v is None])
    if body.experiment is None and body.sample is None:
        raise ExtError(422, "empty_request", "experiment 与 sample 至少给一个")
    holder = _lock()
    warnings: list[str] = []
    if holder.get("held"):
        if not body.force:
            raise ExtError(409, "instrument_busy",
                           f"仪器正被 {holder.get('owner')} 占用（{holder.get('skill')}）——"
                           "现在切作用域，它接下来的动作会记到新的实验/样品下。等它结束，或传 force:true。",
                           holder=holder)
        warnings.append(f"仪器正被 {holder.get('owner')} 占用，已按 force 切换")
    reason = (body.reason or f"外部 agent {caller.agent_id}")[:200]
    done: list[str] = []
    e = body.experiment
    if e is not None:
        if e.id:
            sc = log.switch_experiment(e.id, source="ext", reason=reason)
            if not sc.ok:
                raise ExtError(404 if sc.block_code == "unknown_experiment" else 409,
                               sc.block_code or "scope_refused", sc.error or "切换被拒")
            done.append(f"experiment→{e.id}")
        elif e.name:
            eid = log.start_experiment(e.name.strip(), e.goal or "", reuse_open=True)
            reused = bool(getattr(log, "_last_start_reused", False))
            done.append(f"experiment {'reused' if reused else 'created'}→{eid}")
        else:
            raise ExtError(422, "bad_experiment", "experiment 要给 id 或 name")
    s = body.sample
    if s is not None:
        if s.id:
            sc = log.switch_sample(s.id, source="ext", reason=reason)
            if not sc.ok:
                raise ExtError(404 if sc.block_code == "unknown_sample" else 409,
                               sc.block_code or "scope_refused", sc.error or "切换被拒")
            done.append(f"sample→{s.id}")
        elif s.name:
            if not log.current_experiment_id:
                raise ExtError(409, "no_experiment", "没有当前实验 —— 先定实验再建样品")
            sid = log.start_sample(s.name.strip(), s.description or "",
                                   sample_type=s.sample_type or "", reuse_active=True)
            reused = bool(getattr(log, "_last_sample_reused", False))
            done.append(f"sample {'reused' if reused else 'created'}→{sid}")
        else:
            raise ExtError(422, "bad_sample", "sample 要给 id 或 name")
    logger.info("ext-gateway: %s 改了作用域：%s", caller.agent_id, done)
    return {**_scope(request), "changed": done, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────
# 急停
# ─────────────────────────────────────────────────────────────────────

@router.post("/estop")
def estop(body: EstopBody, request: Request):
    """硬件急停：与 ``/api/safety/emergency-stop`` 同一个动作（中止、停运动、退针、
    E_STOP），闩上的原因如实写成这个外部 agent；再取消所有外部作业。
    **解闩是操作员的事**（``/api/safety/clear-emergency``）。"""
    caller = caller_of(request)
    rt = runtime_of(request)
    fn = getattr(rt, "emergency_stop", None)
    if not callable(fn):
        raise ExtError(503, "not_wired", "没有活的 runtime / 急停钩子")
    reason = body.reason or "未说明"
    why = f"外部 agent {caller.agent_id} 触发急停：{reason}"[:200]
    logger.critical("ext-gateway E-STOP: %s", why)
    try:
        res = fn(why=why)
    except TypeError:                       # 旧签名（没有 why 参数）
        res = fn()
    jm = getattr(request.app.state, "jobs", None)
    cancelled = jm.cancel_all(by=caller.agent_id, reason=f"estop: {reason}") if jm else []
    return {"ok": True, **direct_exec.jsonable(res or {}), "why": why,
            "cancelled_jobs": cancelled,
            "note": "急停闩已挂；此后一切写动作被拒，直到操作员解闩。"}


__all__ = ["SECTIONS", "router"]
