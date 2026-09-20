"""作业：提交 → 轮询 → 取消 → 幂等重发。外部 agent 不再被 HTTP 超时坑。

## 它解决的那件事

``POST /api/skills/{name}/execute`` 是同步的。一次粗动 50 s、一条谱七分钟，而客户端
的 HTTP 超时是 60 s 甚至更短：超时之后客户端以为「失败了」去做下一步，服务端那个
技能**还在跑、还占着仪器令牌**，于是之后每一次调用都撞锁 —— 看起来像随机失败，
实际是自己踩自己。这里把一次技能调用变成一个有身份的作业：

* 提交立刻返回 ``job_id``；执行在自己的线程里（仪器令牌按线程可重入，一个作业一条
  线程，互不串）；
* 轮询 ``GET /jobs/{id}?wait_s=`` 是 ``async`` 的，不占线程池；
* 取消 = 给作业专属的中止事件 ``mark_abort``（与进程级 ``_orch_abort`` 取并集）——
  **协作式**：技能在下一次查询处停，卡在单条阻塞的 Nanonis 命令里时要等它返回；
* 撞锁**不排队**：以 ``refused_busy`` 结束并附上持有者（「谁在开车」）；
* ``request_id`` 幂等：同一调用方同一 id，内容相同回原作业，不同就 409 ——
  传输层断了用同一个 id 重发，不会打两发；
* journal（JSONL）跨重启：进程重启时还没结束的作业标 ``lost_on_restart``，**绝不
  重放**；幂等表同样从 journal 重建，所以重启后用同一 id 重发拿到的是那条
  ``lost_on_restart``，而不是再打一发脉冲。

## 关停

``CoreRuntime.add_shutdown_hook`` 登记了 :meth:`JobManager.shutdown`：停止接新作业、
请在跑的作业停下、有界等待、如实记日志、**不强杀**（强杀线程会把 Nanonis 端口留在
事务中间 —— 那是这台仪器最贵的故障）。它排在解绑运行模式之前，SAFE 不会在关停窗口
里失效。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from mast.api import direct_exec
from mast.api.ext.common import (
    Caller,
    ExtError,
    caller_of,
    ctx_of,
    registry_of,
)
from mast.api.ext.schemas import CancelBody, JobSubmit

logger = logging.getLogger(__name__)

#: 终态。进了其中之一就不会再变。
TERMINAL: frozenset[str] = frozenset({
    "succeeded", "failed", "refused_busy", "cancelled", "crashed", "lost_on_restart",
})

#: 同时在跑（未终态）的作业上限。写技能反正经仪器令牌串行（拒绝不排队）；这个上限
#: 管的是只读 / 分析类作业的并发，以及一个失控客户端能起多少条线程。
MAX_CONCURRENT = 4

#: 内存里保留的作业数（终态的旧作业先丢；未终态的永远不丢）。
KEEP = 300

#: 作业结果里 ``data`` 的 JSON 体积上限。再大就只给键名 —— 大数组请走 ``/data/*``。
DATA_CAP_BYTES = 64_000

#: journal 超过这个体积就在加载时压实成「每个作业最后一版」。
JOURNAL_COMPACT_BYTES = 2_000_000

#: ``GET /jobs/{id}?wait_s=`` 的上限（秒）。更长的等待由客户端分几次轮询。
MAX_WAIT_S = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_journal_dir() -> Path:
    """``MAST_EXT_JOURNAL_DIR`` > 实验库所在目录下的 ``ext_gateway/``。"""
    env = os.environ.get("MAST_EXT_JOURNAL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    from mast.agents._shared.data_paths import experiment_db_path

    return experiment_db_path().parent / "ext_gateway"


def fingerprint(skill: str, params: dict) -> str:
    raw = json.dumps({"skill": skill, "params": params}, sort_keys=True,
                     ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _cap_data(data: Any) -> Any:
    j = direct_exec.jsonable(data)
    try:
        size = len(json.dumps(j, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        return {"_truncated": True, "note": "结果无法序列化"}
    if size <= DATA_CAP_BYTES:
        return j
    keys = list(j)[:60] if isinstance(j, dict) else []
    return {"_truncated": True, "bytes": size, "keys": keys,
            "note": "结果太大，只列键名；原始数据请用 /data/file 或 /data/frame 取。"}


@dataclass
class Job:
    job_id: str
    skill: str
    params: dict
    actor: str
    session: str = ""
    request_id: str | None = None
    fp: str = ""
    note: str = ""
    state: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None
    cancel_reason: str = ""
    result: dict | None = None
    refused_by: str | None = None
    busy_holder: dict | None = None
    abort: dict | None = None
    recorded: dict = field(default_factory=dict)
    params_used: dict | None = None
    # 运行期对象（不进 journal）
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def run_id(self) -> str:
        return f"ext-{self.job_id}"

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def view(self) -> dict:
        return {
            "job_id": self.job_id, "skill": self.skill, "params": direct_exec.jsonable(self.params),
            "params_used": direct_exec.jsonable(self.params_used) if self.params_used is not None else None,
            "actor": self.actor, "session": self.session, "request_id": self.request_id,
            "note": self.note, "run_id": self.run_id,
            "state": self.state, "terminal": self.terminal,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "elapsed_s": self.elapsed_s,
            "cancel_requested": self.cancel_event.is_set() or bool(self.cancel_reason),
            "cancel_reason": self.cancel_reason,
            "result": self.result, "refused_by": self.refused_by,
            "busy_holder": self.busy_holder, "abort": self.abort,
            "recorded": self.recorded,
        }

    @classmethod
    def from_view(cls, v: dict) -> "Job":
        j = cls(job_id=str(v["job_id"]), skill=str(v.get("skill") or ""),
                params=dict(v.get("params") or {}), actor=str(v.get("actor") or "anonymous"),
                session=str(v.get("session") or ""), request_id=v.get("request_id"),
                fp=str(v.get("fp") or ""), note=str(v.get("note") or ""),
                state=str(v.get("state") or "queued"),
                created_at=str(v.get("created_at") or _now()))
        j.started_at = v.get("started_at")
        j.finished_at = v.get("finished_at")
        j.elapsed_s = v.get("elapsed_s")
        j.cancel_reason = str(v.get("cancel_reason") or "")
        j.result = v.get("result")
        j.refused_by = v.get("refused_by")
        j.busy_holder = v.get("busy_holder")
        j.abort = v.get("abort")
        j.recorded = dict(v.get("recorded") or {})
        j.params_used = v.get("params_used")
        if j.terminal:
            j.done.set()
        return j


class JobManager:
    """进程内的作业表 + JSONL journal。线程安全。"""

    def __init__(self, journal_dir: Path | str | None = None, *,
                 max_concurrent: int = MAX_CONCURRENT) -> None:
        self._journal_dir = Path(journal_dir) if journal_dir else None
        self.max_concurrent = int(max_concurrent)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._by_request: dict[tuple[str, str], str] = {}
        self._loaded = False
        self._accepting = True
        self._compacted_bytes = 0

    # ── journal ──────────────────────────────────────────────────────

    @property
    def journal_path(self) -> Path:
        d = self._journal_dir or default_journal_dir()
        return Path(d) / "jobs.jsonl"

    def _append(self, job: Job) -> None:
        """把作业当前视图追加进 journal。写不进去只记日志 —— 执行不能因此失败。

        每次作业状态变化都追加一行 ⇒ 长跑的进程里文件只增不减；超过
        ``JOURNAL_COMPACT_BYTES`` 就在这里当场压实（每个作业只留最新一行），
        而不是只在下一次进程启动时。
        """
        rec = {"ev": "job", "job": {**job.view(), "fp": job.fp}}
        try:
            p = self.journal_path
            p.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                with open(p, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line + "\n")
                    size = fh.tell()
                # 「上次压实后的 2 倍」这一半：保留的作业本身就超过阈值时（结果数据最多 64 KB），
                # 不这样的话之后每追加一行都要整份重写一次。
                if size > max(JOURNAL_COMPACT_BYTES, 2 * self._compacted_bytes):
                    self._compact()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ext-gateway: journal 写入失败(%s)", exc)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            p = self.journal_path
            if not p.is_file():
                return
            latest: dict[str, dict] = {}
            try:
                with open(p, encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except ValueError:
                            continue          # 半截的最后一行（写到一半断电）
                        v = rec.get("job") if isinstance(rec, dict) else None
                        if isinstance(v, dict) and v.get("job_id"):
                            latest[str(v["job_id"])] = v
            except Exception as exc:  # noqa: BLE001
                logger.warning("ext-gateway: journal 读不了(%s) —— 从空表开始", exc)
                return
            lost: list[Job] = []
            for v in sorted(latest.values(), key=lambda x: str(x.get("created_at") or "")):
                job = Job.from_view(v)
                if not job.terminal:
                    # 进程重启时还没结束 ⇒ 不知道它在仪器上做到了哪一步。**绝不重放。**
                    job.state = "lost_on_restart"
                    job.finished_at = job.finished_at or _now()
                    job.done.set()
                    lost.append(job)
                self._jobs[job.job_id] = job
                if job.request_id:
                    self._by_request[(job.actor, job.request_id)] = job.job_id
            self._trim()
            try:
                if p.stat().st_size > JOURNAL_COMPACT_BYTES:
                    self._compact()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ext-gateway: journal 压实失败(%s)", exc)
        for job in lost:
            logger.warning("ext-gateway: 作业 %s (%s by %s) 在进程重启时未结束 —— 标为 "
                           "lost_on_restart，不重放", job.job_id, job.skill, job.actor)
            self._append(job)

    def _compact(self) -> None:
        p = self.journal_path
        tmp = p.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for job in self._jobs.values():
                fh.write(json.dumps({"ev": "job", "job": {**job.view(), "fp": job.fp}},
                                    ensure_ascii=False, default=str) + "\n")
            self._compacted_bytes = fh.tell()
        os.replace(tmp, p)

    def _trim(self) -> None:
        if len(self._jobs) <= KEEP:
            return
        for jid in [j.job_id for j in self._jobs.values() if j.terminal][: len(self._jobs) - KEEP]:
            job = self._jobs.pop(jid)
            if job.request_id and self._by_request.get((job.actor, job.request_id)) == jid:
                self._by_request.pop((job.actor, job.request_id), None)

    # ── 查询 ─────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        self._ensure_loaded()
        with self._lock:
            return self._jobs.get(str(job_id))

    def list(self, *, actor: str | None = None, state: str | None = None,
             limit: int = 50) -> list[Job]:
        self._ensure_loaded()
        with self._lock:
            rows = [j for j in self._jobs.values()
                    if (actor is None or j.actor == actor)
                    and (not state or j.state == state)]
        rows.sort(key=lambda j: j.created_at, reverse=True)
        return rows[: max(1, min(int(limit or 50), KEEP))]

    def running_count(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.terminal)

    # ── 提交 ─────────────────────────────────────────────────────────

    def submit(self, ctx: Any, caller: Caller, skill: str, params: dict, *,
               request_id: str | None = None, note: str = "") -> tuple[Job, bool]:
        """建作业并起线程。返回 ``(job, replay)``；``replay`` = 幂等命中了原作业。

        门口检查（抛 :class:`ExtError`）：注册表里有这个技能；技能与其声明式子步都不
        在本机关闭名单里（硬件模块 / 高级能力；订阅门不拦 —— 它不是安全机制）；
        幂等；并发上限。
        """
        self._ensure_loaded()
        if not self._accepting:
            raise ExtError(503, "shutting_down", "服务正在关停，不再接新作业。")
        registry = getattr(ctx, "skill_registry", None) or getattr(ctx, "registry", None)
        if registry is None:
            raise ExtError(503, "not_wired", "技能注册表未接线（独立开发模式？）",
                           missing=["skill_registry"])
        try:
            cls = registry.get(skill)
        except Exception:  # noqa: BLE001
            near = _did_you_mean(registry, skill)
            raise ExtError(404, "unknown_skill", f"注册表里没有 {skill!r}。先用 "
                           "/skills/search 按动作找技能。", did_you_mean=near)
        disabled = _disabled_names()
        blocked = sorted(({skill} | _spec_steps(cls)) & disabled)
        if blocked:
            raise ExtError(422, "skill_disabled",
                           f"本机关闭了：{'、'.join(blocked)}（硬件模块不存在或该能力未授予）。"
                           "关闭是操作员的决定，包一层组合技能不会打开它。",
                           disabled=blocked)
        fp = fingerprint(skill, params)
        with self._lock:
            if request_id:
                prior_id = self._by_request.get((caller.actor, request_id))
                prior = self._jobs.get(prior_id) if prior_id else None
                if prior is not None:
                    if prior.fp and prior.fp != fp:
                        raise ExtError(409, "request_id_conflict",
                                       f"request_id {request_id!r} 已经用于另一次内容不同的提交。",
                                       existing_job_id=prior.job_id)
                    return prior, True
            running = sum(1 for j in self._jobs.values() if not j.terminal)
            # 停止 / 退针类补救技能不受上限约束：上限挡的是堆积，而最需要停的时刻往往正是
            # 作业堆满的时刻（与仪器锁对它们放行同一个理由，同一份名单）。
            if running >= self.max_concurrent and not _is_remedy(skill):
                raise ExtError(429, "too_many_jobs",
                               f"同时在跑的作业已达上限 {self.max_concurrent}。等它们结束，"
                               "或取消不需要的。停止 / 退针类技能不受这个上限约束。",
                               running=running)
            job = Job(job_id=f"j_{secrets.token_hex(6)}", skill=skill, params=dict(params),
                      actor=caller.actor, session=caller.session, request_id=request_id,
                      fp=fp, note=note)
            self._jobs[job.job_id] = job
            if request_id:
                self._by_request[(caller.actor, request_id)] = job.job_id
            self._trim()
        self._append(job)
        t = threading.Thread(target=self._run, args=(job, ctx, caller),
                             name=f"ext-job-{job.job_id}", daemon=True)
        job.thread = t
        t.start()
        return job, False

    def _run(self, job: Job, ctx: Any, caller: Caller) -> None:
        t0 = time.monotonic()
        try:
            with self._lock:
                job.state = "running"
                job.started_at = _now()
            self._append(job)
            ec, missing = direct_exec.build_context(
                ctx, owner=caller.owner, extra_aborts=[job.cancel_event], run_id=job.run_id)
            if ec is None:
                job.result = {"success": False, "summary": "",
                              "error": f"执行上下文不可用，缺少：{missing}",
                              "data": {}, "nanonis_calls": 0, "elapsed_s": 0.0}
                job.refused_by = "not_wired"
                job.state = "failed"
                return
            run = direct_exec.run_and_record(
                ec, job.skill, job.params,
                runtime=direct_exec.live_runtime(ctx),
                agent_id=caller.agent_id, thread_id=caller.thread_id,
                tool_call_id=job.job_id, context=caller.thread_id,
                approval_source=direct_exec.APPROVAL_LLM)
            job.params_used = run.params
            job.result = {"success": run.success, "summary": run.summary, "error": run.error,
                          "data": _cap_data(run.data), "nanonis_calls": run.nanonis_calls,
                          "elapsed_s": round(run.elapsed_s, 3)}
            job.refused_by = run.refused_by
            job.busy_holder = direct_exec.jsonable(run.busy_holder)
            job.abort = run.abort
            job.recorded = direct_exec.jsonable(run.recorded)
            if run.refused_by == "busy":
                job.state = "refused_busy"
            elif job.cancel_event.is_set() and not run.success:
                job.state = "cancelled"
            elif run.success:
                job.state = "succeeded"
            else:
                job.state = "failed"
        except BaseException as exc:  # noqa: BLE001 — 线程里的任何意外都要落成一个终态
            logger.exception("ext-gateway: 作业 %s 执行线程异常", job.job_id)
            job.result = {"success": False, "summary": "",
                          "error": f"{type(exc).__name__}: {exc}",
                          "data": {}, "nanonis_calls": 0, "elapsed_s": 0.0}
            job.state = "crashed"
        finally:
            job.elapsed_s = round(time.monotonic() - t0, 3)
            job.finished_at = _now()
            self._append(job)
            job.done.set()

    # ── 取消 / 关停 ──────────────────────────────────────────────────

    def cancel(self, job_id: str, *, by: str, reason: str = "") -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.terminal:
            return job
        from mast.core.execution_context import mark_abort

        why = f"外部 agent 取消（{by}）" + (f"：{reason}" if reason else "")
        with self._lock:
            job.cancel_reason = reason or "(未说明)"
        mark_abort(job.cancel_event, why[:200])
        self._append(job)
        return job

    def cancel_all(self, *, by: str, reason: str = "") -> list[str]:
        ids = [j.job_id for j in self.list(limit=KEEP) if not j.terminal]
        for jid in ids:
            self.cancel(jid, by=by, reason=reason)
        return ids

    def shutdown(self, timeout_s: float = 10.0) -> None:
        """关停钩子：停止接新作业 → 请在跑的停下 → 有界等待 → 如实记日志。不强杀。"""
        self._accepting = False
        running = [j for j in self.list(limit=KEEP) if not j.terminal]
        if not running:
            return
        self.cancel_all(by="service", reason="服务正在关停")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        for job in running:
            t = job.thread
            if t is not None and t.is_alive():
                t.join(max(0.0, deadline - time.monotonic()))
        alive = [j.job_id for j in running if j.thread is not None and j.thread.is_alive()]
        if alive:
            logger.error("ext-gateway: 关停时仍有作业没停下（可能卡在一条阻塞的 Nanonis "
                         "命令里）：%s —— 不强杀，交给进程退出兜底", alive)
        else:
            logger.info("ext-gateway: %d 个在跑的作业已在关停前停下", len(running))

    async def wait(self, job: Job, wait_s: float) -> Job:
        deadline = time.monotonic() + max(0.0, min(float(wait_s or 0.0), MAX_WAIT_S))
        while not job.terminal and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        return job


# ─────────────────────────────────────────────────────────────────────
# 门口的小判据（永不抛）
# ─────────────────────────────────────────────────────────────────────

def _disabled_names() -> frozenset[str]:
    """本机关闭的技能：硬件模块 ∪ 高级能力。订阅门**不**算 —— 它不是安全机制。"""
    try:
        from mast.skills.tool_face import compute

        face = compute()
        return frozenset(face.hardware | face.advanced)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ext-gateway: 关闭名单读不到(%s) —— 本次不按它拒绝", exc)
        return frozenset()


def _is_remedy(skill: str) -> bool:
    """停止 / 退针这类补救技能（``instrument_lock.BYPASS_NAMES``：仪器锁对它们放行）。"""
    try:
        from mast.core.instrument_lock import BYPASS_NAMES

        return skill in BYPASS_NAMES
    except Exception:  # noqa: BLE001
        return False


def _spec_steps(cls: Any) -> set[str]:
    try:
        from mast.agents._shared.skill_forge_tools import _spec_step_skills_of

        return set(_spec_step_skills_of(cls))
    except Exception:  # noqa: BLE001
        return set()


def _did_you_mean(registry: Any, skill: str) -> list[str]:
    q = str(skill or "").lower()
    try:
        names = [m.name for m in registry.list_skills()]
    except Exception:  # noqa: BLE001
        return []
    hits = [n for n in names if q and (q in n.lower() or n.lower() in q)]
    return sorted(hits)[:8]


# ─────────────────────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["jobs"])


def _jm(request: Request) -> JobManager:
    jm = getattr(request.app.state, "jobs", None)
    if jm is None:
        raise ExtError(503, "not_wired", "作业管理器未接线")
    return jm


@router.post("/jobs", status_code=202)
def submit_job(body: JobSubmit, request: Request):
    """提交一个技能作业。新作业 202；``request_id`` 幂等命中原作业 200 +
    ``idempotent_replay: true``。技能自身失败不是 HTTP 错误 —— 作业以终态结束。"""
    caller = caller_of(request)
    job, replay = _jm(request).submit(ctx_of(request), caller, body.skill.strip(),
                                      dict(body.params or {}),
                                      request_id=(body.request_id or None), note=body.note)
    view = {**job.view(), "idempotent_replay": replay}
    return JSONResponse(status_code=200 if replay else 202, content=view)


@router.get("/jobs")
def list_jobs(request: Request, all: bool = Query(False, description="true = 所有调用方的作业"),  # noqa: A002
              state: str = Query("", description="按状态过滤"),
              limit: int = Query(50, ge=1, le=KEEP)):
    """本调用方最近的作业（新的在前）。丢了上下文之后用它找回自己提交过什么。"""
    caller = caller_of(request)
    rows = _jm(request).list(actor=None if all else caller.actor, state=state or None,
                             limit=limit)
    return {"count": len(rows), "jobs": [j.view() for j in rows]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request,
                  wait_s: float = Query(0.0, ge=0.0, le=MAX_WAIT_S,
                                        description="等到终态或超时（秒）")):
    """作业视图。``wait_s>0`` 时等到终态或超时再回（async，不占线程池）。"""
    jm = _jm(request)
    job = jm.get(job_id)
    if job is None:
        raise ExtError(404, "unknown_job", f"没有作业 {job_id!r}（进程重启前的作业在 "
                       "journal 里；超过保留数量的旧作业会被丢掉）")
    if wait_s > 0 and not job.terminal:
        await jm.wait(job, wait_s)
    return job.view()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, body: CancelBody, request: Request):
    """协作式取消：技能在下一次查询中止处停下。卡在单条阻塞的 Nanonis 命令里时
    要等那条命令返回；需要物理上立刻停，用不取令牌的 ``StopScan`` 这类技能或 ``/estop``。"""
    caller = caller_of(request)
    job = _jm(request).cancel(job_id, by=caller.agent_id, reason=body.reason)
    if job is None:
        raise ExtError(404, "unknown_job", f"没有作业 {job_id!r}")
    return job.view()


__all__ = ["Job", "JobManager", "MAX_CONCURRENT", "TERMINAL", "default_journal_dir",
           "fingerprint", "router"]
