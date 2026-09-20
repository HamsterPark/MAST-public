"""作用域 API —— 「当前实验 / 当前样品」的唯一真源，以及切换。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §13

为什么需要一个 ``/api/experiments/current``
-------------------------------------------

在此之前前端**没有**这个端点，于是每个组件自己从实验列表里猜当前实验，而且
猜法不一致：右栏取 ``experiments[0]``，顶栏还在用
``find(status === "active")``。两处能显示不同的实验。

现在服务端有一个显式指针（``active_scope`` 单行表），这个端点把它读出来。所有
前端组件读同一个端点、同一个 queryKey —— **结构上不可能再分歧**。

排序也在这里修：``/api/experiments`` 按 ``start_time DESC`` 排，对"回到上个月
那个实验"是错的排序（它是上个月**建**的，但可能是昨天才动过）。
``/api/experiments/recent`` 按 ``last_active_at DESC`` 排。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scope"])


# ── schemas ───────────────────────────────────────────────────────────

class ScopeExperiment(BaseModel):
    id: str
    name: str = ""
    goal: str = ""
    start_time: str | None = None
    last_active_at: str | None = None
    dir_name: str | None = None


class ScopeSample(BaseModel):
    id: str
    name: str = ""
    sample_type: str = ""
    sample_subtype: str = ""
    description: str = ""
    start_time: str | None = None
    last_active_at: str | None = None
    dir_name: str | None = None
    index: int | None = None


class CurrentScopeResponse(BaseModel):
    experiment: ScopeExperiment | None = None
    sample: ScopeSample | None = None
    has_experiment: bool = False
    has_sample: bool = False
    #: 无样品时给用户的一句话。产数据的操作会被拦，但对话不受影响。
    hint: str = ""
    folder_path: str | None = None
    degraded: bool = False


class ScopeChangeResult(BaseModel):
    ok: bool = False
    changed: bool = False
    experiment: ScopeExperiment | None = None
    sample: ScopeSample | None = None
    blocked: bool = False
    block_code: str = ""
    block_reason: str = ""
    can_force: bool = False
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False


class ActivateBody(BaseModel):
    sample_id: str | None = None
    force: bool = False
    reason: str = ""


class ClearSampleBody(BaseModel):
    reason: str = ""


class RecentExperiment(ScopeExperiment):
    sample_count: int = 0
    action_count: int = 0
    is_current: bool = False


class RecentExperimentsResponse(BaseModel):
    experiments: list[RecentExperiment] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class SampleItem(ScopeSample):
    is_current: bool = False
    action_count: int = 0


class SamplesResponse(BaseModel):
    samples: list[SampleItem] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class PreflightResponse(BaseModel):
    """新建实验弹窗的重名检测 —— "严谨"的实质内容。

    create-always 会重复创建同名实验，重名检测应让用户明确选择继续或新建。
    把已存在的同名实验连同它的规模一起摆给用户看，让他选"继续该实验"还是
    "仍然新建一个"，是从源头堵住重复的地方。
    """
    existing: RecentExperiment | None = None
    current: ScopeExperiment | None = None
    degraded: bool = False


class AdvisoryResponse(BaseModel):
    busy: bool = False
    orchestrator_running: bool = False
    background_active: int = 0
    instrument_holder: dict | None = None
    blocking: bool = False
    can_force: bool = True
    reasons: list[str] = Field(default_factory=list)
    degraded: bool = False


# ── helpers ───────────────────────────────────────────────────────────

def _exp_model(row: dict | None) -> ScopeExperiment | None:
    if not row:
        return None
    return ScopeExperiment(
        id=str(row.get("id") or ""),
        name=row.get("name") or "",
        goal=row.get("goal_text") or row.get("goal") or "",
        start_time=_s(row.get("start_time")),
        last_active_at=_s(row.get("last_active_at")),
        dir_name=row.get("dir_name") or None,
    )


def _sample_model(row: dict | None) -> ScopeSample | None:
    if not row:
        return None
    return ScopeSample(
        id=str(row.get("id") or ""),
        name=row.get("name") or "",
        sample_type=row.get("sample_type") or "",
        sample_subtype=row.get("sample_subtype") or "",
        description=row.get("description") or "",
        start_time=_s(row.get("start_time")),
        last_active_at=_s(row.get("last_active_at")),
        dir_name=row.get("dir_name") or None,
        index=row.get("sample_index"),
    )


def _s(v) -> str | None:
    return str(v) if v is not None else None


def _log():
    """The live ExperimentLog singleton, or None (standalone dev / tests)."""
    try:
        from mast.logging.experiment_log import get_active_log
        return get_active_log()
    except Exception:  # noqa: BLE001
        return None


def _result(sc, storage) -> ScopeChangeResult:
    """ScopeChange → 传输模型。"""
    exp = storage.get_experiment(sc.experiment_id) if (storage and sc.experiment_id) else None
    smp = storage.get_sample(sc.sample_id) if (storage and sc.sample_id) else None
    return ScopeChangeResult(
        ok=sc.ok, changed=sc.changed,
        experiment=_exp_model(exp), sample=_sample_model(smp),
        blocked=not sc.ok, block_code=sc.block_code, block_reason=sc.error,
        can_force=sc.can_force, warnings=list(sc.warnings),
    )


_NO_SAMPLE_HINT = (
    "未选定样品 —— 扫描/谱学等产生数据的操作已暂停（对话、提问、查看状态不受影响）。"
    "请选择或新建一个样品。"
)
_NO_EXPERIMENT_HINT = "还没有进行中的实验。新建或选择一个实验后即可开始。"


# ── endpoints ─────────────────────────────────────────────────────────

@router.get("/experiments/current", response_model=CurrentScopeResponse)
def current_scope(request: Request) -> CurrentScopeResponse:
    """当前实验与样品。**前端的唯一真源。**"""
    storage = request.app.state.ctx.experiment_storage
    log = _log()
    if storage is None:
        return CurrentScopeResponse(degraded=True, hint=_NO_EXPERIMENT_HINT)

    eid = sid = None
    if log is not None:
        eid, sid = log.current_experiment_id, log.current_sample_id
    else:
        try:
            row = storage.get_active_scope() or {}
            eid, sid = row.get("experiment_id"), row.get("sample_id")
        except Exception as exc:  # noqa: BLE001
            logger.warning("active_scope read failed: %s", exc)
            return CurrentScopeResponse(degraded=True)

    try:
        exp = storage.get_experiment(eid) if eid else None
        smp = storage.get_sample(sid) if sid else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("scope rows unreadable: %s", exc)
        return CurrentScopeResponse(degraded=True)

    folder = None
    if exp and exp.get("dir_name"):
        try:
            from mast.core.experiment_paths import experiment_dir
            folder = str(experiment_dir(exp["dir_name"]))
        except Exception:  # noqa: BLE001
            folder = None

    hint = ""
    if not exp:
        hint = _NO_EXPERIMENT_HINT
    elif not smp:
        hint = _NO_SAMPLE_HINT

    return CurrentScopeResponse(
        experiment=_exp_model(exp), sample=_sample_model(smp),
        has_experiment=bool(exp), has_sample=bool(smp),
        hint=hint, folder_path=folder,
    )


@router.post("/experiments/{experiment_id}/activate", response_model=ScopeChangeResult)
def activate_experiment(experiment_id: str, body: ActivateBody,
                        request: Request) -> ScopeChangeResult:
    """切换当前实验。**不改变任何实验行** —— 只移动指针。"""
    storage = request.app.state.ctx.experiment_storage
    log = _log()
    if log is None or storage is None:
        return ScopeChangeResult(degraded=True, blocked=True,
                                 block_reason="记录系统未接线")
    adv = _advisory(request)
    if adv.blocking and not body.force:
        return ScopeChangeResult(
            ok=False, blocked=True, block_code="instrument_busy",
            block_reason=_busy_reason(adv), can_force=True,
            warnings=adv.reasons)
    sc = log.switch_experiment(experiment_id, sample_id=body.sample_id,
                               source="gui", reason=body.reason)
    res = _result(sc, storage)
    if sc.ok and adv.busy:
        res.warnings = list(res.warnings) + adv.reasons
    return res


@router.post("/experiments/{experiment_id}/samples/{sample_id}/activate",
             response_model=ScopeChangeResult)
def activate_sample(experiment_id: str, sample_id: str, body: ActivateBody,
                    request: Request) -> ScopeChangeResult:
    storage = request.app.state.ctx.experiment_storage
    log = _log()
    if log is None or storage is None:
        return ScopeChangeResult(degraded=True, blocked=True,
                                 block_reason="记录系统未接线")
    adv = _advisory(request)
    if adv.blocking and not body.force:
        return ScopeChangeResult(
            ok=False, blocked=True, block_code="instrument_busy",
            block_reason=_busy_reason(adv), can_force=True,
            warnings=adv.reasons)
    sc = log.switch_sample(sample_id, source="gui", reason=body.reason)
    res = _result(sc, storage)
    if sc.ok and adv.busy:
        res.warnings = list(res.warnings) + adv.reasons
    return res


@router.post("/scope/clear-sample", response_model=ScopeChangeResult)
def clear_sample(body: ClearSampleBody, request: Request) -> ScopeChangeResult:
    """取消选中样品（物理出样时用）。**不写样品行的任何字段。**"""
    storage = request.app.state.ctx.experiment_storage
    log = _log()
    if log is None or storage is None:
        return ScopeChangeResult(degraded=True, blocked=True,
                                 block_reason="记录系统未接线")
    return _result(log.clear_sample(source="gui", reason=body.reason), storage)


@router.get("/experiments/recent", response_model=RecentExperimentsResponse)
def recent_experiments(request: Request, limit: int = 30,
                       q: str = "") -> RecentExperimentsResponse:
    """按**上次活动时间**倒序 —— 切换器的数据源。"""
    storage = request.app.state.ctx.experiment_storage
    if storage is None:
        return RecentExperimentsResponse(degraded=True)
    log = _log()
    cur = log.current_experiment_id if log else None
    try:
        rows = storage.list_experiments_recent(limit=limit, query=q)
    except Exception as exc:  # noqa: BLE001
        logger.warning("recent experiments failed: %s", exc)
        return RecentExperimentsResponse(degraded=True)
    out = []
    for r in rows:
        base = _exp_model(r)
        if base is None:
            continue
        out.append(RecentExperiment(
            **base.model_dump(),
            sample_count=int(r.get("sample_count") or 0),
            action_count=int(r.get("action_count") or 0),
            is_current=(str(r.get("id")) == cur),
        ))
    return RecentExperimentsResponse(experiments=out, count=len(out))


@router.get("/experiments/preflight", response_model=PreflightResponse)
def preflight(request: Request, name: str = "") -> PreflightResponse:
    """新建实验前的重名检测。"""
    storage = request.app.state.ctx.experiment_storage
    if storage is None:
        return PreflightResponse(degraded=True)
    log = _log()
    cur_row = None
    try:
        if log and log.current_experiment_id:
            cur_row = storage.get_experiment(log.current_experiment_id)
    except Exception:  # noqa: BLE001
        cur_row = None

    existing = None
    nm = (name or "").strip()
    if nm:
        try:
            row = storage.find_experiment_by_name(nm)
            if row:
                base = _exp_model(row)
                counts = _counts_for(storage, row["id"])
                existing = RecentExperiment(**base.model_dump(), **counts,
                                            is_current=bool(cur_row and
                                                            cur_row["id"] == row["id"]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("preflight lookup failed: %s", exc)
    return PreflightResponse(existing=existing, current=_exp_model(cur_row))


def _counts_for(storage, eid: str) -> dict:
    try:
        samples = storage.get_samples(eid) or []
    except Exception:  # noqa: BLE001
        samples = []
    try:
        actions = storage.get_actions(eid) or []
    except Exception:  # noqa: BLE001
        actions = []
    return {"sample_count": len(samples), "action_count": len(actions)}


@router.get("/experiments/{experiment_id}/samples", response_model=SamplesResponse)
def list_samples(experiment_id: str, request: Request) -> SamplesResponse:
    """轻量样品列表（选择器的数据源）。

    刻意不用 ``GET /api/experiments/{id}`` —— 那个会把 actions + markers +
    feedback 一起拖出来，对一个下拉框来说太重了。
    """
    storage = request.app.state.ctx.experiment_storage
    if storage is None:
        return SamplesResponse(degraded=True)
    log = _log()
    cur = log.current_sample_id if log else None
    try:
        rows = storage.get_samples(experiment_id) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("samples list failed: %s", exc)
        return SamplesResponse(degraded=True)
    out = []
    for r in rows:
        base = _sample_model(r)
        if base is None:
            continue
        out.append(SampleItem(**base.model_dump(),
                              is_current=(str(r.get("id")) == cur)))
    # 上次用过的排最前 —— 与切换器的心智模型一致。
    out.sort(key=lambda s: (s.last_active_at or "", s.start_time or ""), reverse=True)
    return SamplesResponse(samples=out, count=len(out))


class NanonisDirPlan(BaseModel):
    ok: bool = False
    needs_change: bool = False
    inplace_active: bool = False
    target: str | None = None
    current_session_path: str | None = None
    previous_session_path: str | None = None
    stale_sample_dir: str | None = None
    stale_warning: str = ""
    warnings: list[str] = Field(default_factory=list)
    reason: str = ""
    degraded: bool = False


class NanonisDirBody(BaseModel):
    #: True = 指向当前样品；False = 恢复到 previous_session_path
    restore: bool = False


@router.get("/scope/nanonis-dir", response_model=NanonisDirPlan)
def nanonis_dir_plan(request: Request) -> NanonisDirPlan:
    """「把 Nanonis 保存目录指向当前样品」的**计划**（不执行）。

    纯读：算出目标路径、当前是否已经指对、有没有换样品后没跟着改的情况。
    """
    rt = _runtime(request)
    log = _log()
    storage = request.app.state.ctx.experiment_storage
    if rt is None or log is None or storage is None:
        return NanonisDirPlan(degraded=True, reason="记录系统或运行时未接线")

    try:
        from mast.core import nanonis_session as ns
        scope = rt._ensure_scope_dirs() if hasattr(rt, "_ensure_scope_dirs") else None
        exp_dir, sample_dir = scope if scope else (None, None)
        cur = _current_session_path(rt)
        exp = storage.get_experiment(log.current_experiment_id) if log.current_experiment_id else None
        smp = storage.get_sample(log.current_sample_id) if log.current_sample_id else None
        plan = ns.plan_repoint(cur, exp_dir, sample_dir,
                               experiment_title=(exp or {}).get("name", "") or "",
                               sample_name=(smp or {}).get("name", "") or "")
        stale_dir = ns.session_path_sample(cur, exp_dir)
        stale = ""
        if stale_dir and sample_dir and stale_dir != sample_dir:
            stale = ns.stale_warning(stale_dir, sample_dir,
                                     current_sample_name=(smp or {}).get("name", "") or "")
        prev = None
        try:
            from mast.logging.v2.manifest import read_json
            prev = (read_json(Path(exp_dir) / "experiment.json").get("nanonis") or {}
                    ).get("previous_session_path") if exp_dir else None
        except Exception:  # noqa: BLE001
            prev = None
        return NanonisDirPlan(
            ok=plan["ok"], needs_change=plan["needs_change"],
            inplace_active=ns.is_inplace_active(cur, exp_dir, sample_dir),
            target=plan["target"], current_session_path=cur,
            previous_session_path=prev or plan.get("previous_session_path"),
            stale_sample_dir=stale_dir if stale else None,
            stale_warning=stale, warnings=list(plan["warnings"]),
            reason=plan["reason"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("nanonis dir plan failed: %s", exc)
        return NanonisDirPlan(degraded=True, reason=str(exc))


@router.post("/scope/nanonis-dir", response_model=NanonisDirPlan)
def nanonis_dir_apply(body: NanonisDirBody, request: Request) -> NanonisDirPlan:
    """执行：把 Nanonis 保存目录指向当前样品，或恢复原目录。

    走 ``SetSessionPath`` skill（``SafetyLevel.CONFIRM``）。改完**立刻回读校验**
    并失效 runtime 的 ~15 s session 缓存 —— 不然后续的文件发现会用着旧路径。
    """
    rt = _runtime(request)
    log = _log()
    if rt is None or log is None:
        return NanonisDirPlan(degraded=True, reason="运行时未接线")

    plan = nanonis_dir_plan(request)
    if plan.degraded or not plan.ok:
        return plan

    if body.restore:
        target = plan.previous_session_path
        if not target:
            plan.reason = "没有记录到改动前的保存目录，无法一键恢复。"
            plan.ok = False
            return plan
    else:
        if not plan.needs_change:
            return plan
        target = plan.target
        try:
            Path(target).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            plan.ok = False
            plan.reason = f"无法创建目标目录：{exc}"
            return plan

    ec = _execution_context(request)
    if ec is None:
        plan.ok = False
        plan.reason = "Nanonis 未连接——无法修改保存目录。"
        return plan
    try:
        res = ec.run("SetSessionPath", {"session_path": str(target),
                                        "save_settings_to_previous": True})
        if not getattr(res, "success", False):
            plan.ok = False
            plan.reason = f"SetSessionPath 失败：{getattr(res, 'error', '') or res}"
            return plan
    except Exception as exc:  # noqa: BLE001
        plan.ok = False
        plan.reason = f"SetSessionPath 未送达：{exc}"
        return plan

    # 主动失效 ~15 s 的 session 缓存，否则后续的文件发现还在用旧路径。
    # 然后立刻回读校验 —— 不校验就等于不知道到底改没改。
    try:
        rt._session_dir_cache = None
        rt._session_path = None
        rt._session_dir = None
    except Exception:  # noqa: BLE001
        pass

    after = nanonis_dir_plan(request)
    # 记录可逆性锚点：previous_session_path 让 UI 能提供「恢复原保存目录」。
    try:
        from mast.logging.v2.manifest import set_nanonis_state
        scope = rt._ensure_scope_dirs()
        if scope:
            set_nanonis_state(
                scope[0],
                session_path_at_last_check=after.current_session_path,
                previous_session_path=(None if body.restore
                                       else plan.current_session_path),
                inplace_mode=(not body.restore))
    except Exception:  # noqa: BLE001
        pass
    return after


def _execution_context(request: Request) -> Any:
    """一次性 ExecutionContext（与 routes/signals.py 同一范式）。

    任一单例缺席就返回 None → 降级，不在半接线的进程里碰硬件。
    """
    ctx = request.app.state.ctx
    pool = getattr(ctx, "connection_pool", None)
    state = getattr(ctx, "state", None) or getattr(ctx, "instrument_state", None)
    registry = getattr(ctx, "skill_registry", None) or getattr(ctx, "registry", None)
    if pool is None or state is None or registry is None:
        return None
    try:
        from mast.core.execution_context import ExecutionContext
        app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
        abort = getattr(app, "_orch_abort", None)
        return ExecutionContext(pool=pool, state=state, registry=registry,
                                abort_event=abort, approval_source="human",
                                owner="设置页 · Nanonis 保存目录")
    except Exception:  # noqa: BLE001
        return None


def _runtime(request: Request):
    rt = getattr(request.app.state, "runtime", None)
    if rt is None:
        rt = getattr(request.app.state.ctx, "runtime", None)
    return rt


def _current_session_path(rt) -> str | None:
    try:
        fn = getattr(rt, "_resolve_session_dir", None)
        if callable(fn):
            fn()
        p = getattr(rt, "_session_path", None) or getattr(rt, "_session_dir", None)
        return str(p) if p else None
    except Exception:  # noqa: BLE001
        return None


class FolderHealth(BaseModel):
    """实验文件夹的健康度 —— 归档到底有没有在工作。

    这个端点存在的理由：一个静默失效的归档管线看起来和"还没扫过图"一模一样。
    ``failed``/``dropped`` 非零、或者 ``last_error`` 有值，用户就该知道。
    """
    folder_path: str | None = None
    exists: bool = False
    raw_files: int = 0
    raw_bytes: int = 0
    samples: int = 0
    ingest: dict = Field(default_factory=dict)
    env_csv: dict = Field(default_factory=dict)
    quarantined: int = 0
    degraded: bool = False


@router.get("/scope/folder-health", response_model=FolderHealth)
def folder_health(request: Request) -> FolderHealth:
    rt = _runtime(request)
    log = _log()
    if log is None or not log.current_experiment_id:
        return FolderHealth(degraded=True)
    out = FolderHealth()
    try:
        scope = rt._ensure_scope_dirs() if (rt and hasattr(rt, "_ensure_scope_dirs")) else None
        if scope:
            exp_dir = scope[0]
            out.folder_path = str(exp_dir)
            out.exists = Path(exp_dir).is_dir()
            samples_dir = Path(exp_dir) / "samples"
            if samples_dir.is_dir():
                out.samples = sum(1 for p in samples_dir.iterdir() if p.is_dir())
            n = b = 0
            for p in Path(exp_dir).glob("samples/*/raw/*/*"):
                if p.is_file():
                    n += 1
                    try:
                        b += p.stat().st_size
                    except OSError:
                        pass
            out.raw_files, out.raw_bytes = n, b
    except Exception as exc:  # noqa: BLE001
        logger.debug("folder health scan failed: %r", exc)
    try:
        sink = getattr(rt, "_ingest", None)
        out.ingest = sink.stats() if sink is not None else {"enabled": False}
    except Exception:  # noqa: BLE001
        out.ingest = {"enabled": False}
    try:
        csv_sink = getattr(rt, "_env_csv", None)
        out.env_csv = csv_sink.stats() if csv_sink is not None else {"enabled": False}
    except Exception:  # noqa: BLE001
        out.env_csv = {"enabled": False}
    try:
        from mast.core.experiment_paths import quarantine_dir
        qi = quarantine_dir() / "index.jsonl"
        if qi.is_file():
            out.quarantined = sum(1 for line in qi.read_text(encoding="utf-8").splitlines()
                                  if line.strip())
    except Exception:  # noqa: BLE001
        pass
    return out


class MaintenanceBody(BaseModel):
    #: verify | reindex | export_db | export_rocrate
    action: str = "verify"
    #: verify 时是否重算 sha256（慢，但这是唯一真的读字节的校验）
    deep: bool = False
    dry_run: bool = False


class MaintenanceResult(BaseModel):
    ok: bool = False
    action: str = ""
    result: dict = Field(default_factory=dict)
    artifact: str | None = None
    message: str = ""
    degraded: bool = False


@router.post("/scope/maintenance", response_model=MaintenanceResult)
def folder_maintenance(body: MaintenanceBody, request: Request) -> MaintenanceResult:
    """实验文件夹的维护动作（全部**按需**，没有一个是收尾步骤）。

    * ``verify`` —— 按 manifest 校验完整性。``deep=True`` 重算 sha256，这是
      ``scan_files.fixity_ok`` 这一列自建库以来第一次真的被验证。
    * ``reindex`` —— 从文件夹重建 DB 索引。**只补不改**，随时可跑；它是
      「DB 是索引、文件夹是记录」这句判断的可执行证明。
    * ``export_db`` / ``export_rocrate`` —— 抽一份可独立解读的归档到 ``exports/``。
    """
    storage = request.app.state.ctx.experiment_storage
    rt = _runtime(request)
    action = (body.action or "").strip()
    try:
        from mast.logging.v2 import reindex as rx

        if action == "verify":
            return MaintenanceResult(ok=True, action=action,
                                     result=rx.verify_all(deep=body.deep))

        if action == "reindex":
            if storage is None:
                return MaintenanceResult(degraded=True, action=action,
                                         message="记录系统未接线")
            repos = getattr(rt, "_v2_repos", None)
            res = rx.rebuild_from_folders(storage=storage, repos=repos,
                                          dry_run=body.dry_run)
            return MaintenanceResult(ok=True, action=action, result=res)

        scope = rt._ensure_scope_dirs() if (rt and hasattr(rt, "_ensure_scope_dirs")) else None
        if not scope:
            return MaintenanceResult(degraded=True, action=action,
                                     message="当前没有实验文件夹")
        exp_dir = scope[0]

        if action == "export_db":
            p = rx.export_record_db(exp_dir, storage=storage)
            return MaintenanceResult(ok=bool(p), action=action,
                                     artifact=str(p) if p else None,
                                     message="" if p else "导出失败")
        if action == "export_rocrate":
            repos = getattr(rt, "_v2_repos", None)
            v2_eid = getattr(rt, "_v2_eid", None)
            if repos is None or not v2_eid:
                return MaintenanceResult(degraded=True, action=action,
                                         message="v2 记录未接线")
            p = rx.export_rocrate(exp_dir, repos=repos, v2_experiment_id=v2_eid)
            return MaintenanceResult(ok=bool(p), action=action,
                                     artifact=str(p) if p else None,
                                     message="" if p else "导出失败")
        return MaintenanceResult(action=action, message=f"未知动作：{action}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("folder maintenance %s failed: %s", action, exc)
        return MaintenanceResult(action=action, message=str(exc))


@router.get("/scope/switch-advisory", response_model=AdvisoryResponse)
def switch_advisory(request: Request) -> AdvisoryResponse:
    """切换前的忙碌状态。全是本地内存读，**永不阻塞**。"""
    return _advisory(request)


def _advisory(request: Request) -> AdvisoryResponse:
    """当前有没有东西在跑，切换是否危险。

    **绝不自动中止任何任务** —— 把一次切换点击变成破坏性操作是不能接受的。
    这里只回报事实，由用户决定（弹窗提供「先中止任务」和「强制切换」）。
    """
    rt = getattr(request.app.state, "runtime", None)
    if rt is None:
        rt = getattr(request.app.state.ctx, "runtime", None)
    reasons: list[str] = []
    orch = False
    bg = 0
    holder = None
    try:
        orch = bool(getattr(rt, "_orch_running", False)) if rt is not None else False
        if orch:
            reasons.append("群聊任务正在运行；它接下来的仪器动作会记到新的实验/样品下")
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast.core.background_runs import background_runs
        bg = int(background_runs().active_count()) if hasattr(
            background_runs(), "active_count") else (
            1 if background_runs().has_active() else 0)
        if bg:
            reasons.append(f"有 {bg} 个后台任务正在运行")
    except Exception:  # noqa: BLE001
        bg = 0
    try:
        from mast.core.instrument_lock import instrument_lock
        holder = instrument_lock().snapshot()
    except Exception:  # noqa: BLE001
        holder = None

    blocking = bool(holder)
    if blocking:
        reasons.insert(0, _busy_reason_from(holder))
    return AdvisoryResponse(
        busy=bool(orch or bg or holder), orchestrator_running=orch,
        background_active=bg, instrument_holder=holder or None,
        blocking=blocking, can_force=True, reasons=reasons,
    )


def _busy_reason_from(holder: dict | None) -> str:
    h = holder or {}
    who = h.get("owner") or "另一个入口"
    what = h.get("skill") or "某个仪器动作"
    held = h.get("held_s")
    held_txt = f"，已 {held:.0f} 秒" if isinstance(held, (int, float)) else ""
    return (f"仪器正被占用：{who} 正在执行 {what}{held_txt}。"
            f"现在切换会让这次操作记到错误的样品下。")


def _busy_reason(adv: AdvisoryResponse) -> str:
    return _busy_reason_from(adv.instrument_holder)
