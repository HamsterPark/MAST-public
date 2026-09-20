"""GET /api/safety/limits — the active hardware-safety bounds.

Returns the bounds **actually in force in this process** — the running
SafetyGuard's own merged limits when one is reachable, otherwise code defaults
folded with the on-disk admin overrides. The same Pydantic model is used client-
AND server-side (F4): client validation is for UX only; the server remains
authoritative (R6 — safety never lives in the API/UI).

This used to return a bare ``SafetyLimits()`` — the shipped constants,
never the override layer — under a docstring promising the merge would land "when
admin overrides land (Phase 3)". Phase 3 landed long ago, and the merge never did:
widening the XY envelope through the admin override endpoint got `{"ok":true}` from
the write and the *old* numbers from here, with nothing to tell you which one was
lying. Provenance and the restart question now live in ``mast.api.safety_view``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from mast.api.safety_view import (
    SOURCE_DEFAULTS,
    in_force_safety_limits,
    pending_restart,
)
from mast.api.schemas import SafetyLimits

logger = logging.getLogger(__name__)

router = APIRouter(tags=["safety"])


@router.get("/safety/limits", response_model=SafetyLimits)
def get_safety_limits(request: Request, response: Response) -> SafetyLimits:
    """The bounds in force right now — live SafetyGuard first, merged, defaults.

    Response headers carry the provenance without changing the body shape (the
    body model is re-exported straight from ``config.SafetyLimits``, shared with
    the core; API-only fields must not be bolted onto it):

      * ``x-safety-limits-source``   — live_guard | merged | defaults
      * ``x-safety-restart-pending`` — true | false | unknown

    Never raises: "what are the limits" degrades to the code defaults rather
    than 500-ing, because a caller who gets no answer here will assume the
    defaults anyway — better to hand them the defaults and say so.
    """
    ctx = getattr(request.app.state, "ctx", None)
    try:
        limits, source = in_force_safety_limits(ctx)
        pending = pending_restart(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("safety limits resolve failed, serving code defaults: %s", exc)
        limits, source, pending = SafetyLimits(), SOURCE_DEFAULTS, None
    response.headers["x-safety-limits-source"] = source
    response.headers["x-safety-restart-pending"] = (
        "unknown" if pending is None else ("true" if pending else "false")
    )
    return limits


class EmergencyStopResponse(BaseModel):
    """POST /api/safety/emergency-stop — result of a one-click hardware E-STOP."""
    ok: bool = False
    aborted: bool = False           # autonomous run abort signalled
    stopped_motion: bool = False    # auto-approach / motor / scan stopped
    retracted: bool = False         # tip withdrawn (emergency or main port)
    errors: list[str] = Field(default_factory=list)
    degraded: bool = False          # no live app wired


@router.post("/safety/emergency-stop", response_model=EmergencyStopResponse)
def emergency_stop(request: Request) -> EmergencyStopResponse:
    """Hardware EMERGENCY STOP: abort the run, stop powered motion, retract the
    tip, emit E_STOP. This is the one-click safety action the UI E-STOP button
    calls — before it, 'abort' only set a software flag and never touched the
    hardware. Degrades to a typed no-op with no live app."""
    ctx = request.app.state.ctx
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    estop = getattr(app, "emergency_stop", None)
    if not callable(estop):
        return EmergencyStopResponse(ok=False, degraded=True,
                                     errors=["no live app / emergency_stop hook"])
    try:
        r = estop()
        return EmergencyStopResponse(
            ok=True,
            aborted=bool(r.get("aborted")),
            stopped_motion=bool(r.get("stopped_motion")),
            retracted=bool(r.get("retracted")),
            errors=[str(e) for e in (r.get("errors") or [])],
            degraded=False,
        )
    except Exception as exc:  # pragma: no cover - never 500 on an E-STOP
        logger.critical("emergency_stop endpoint failed: %s", exc)
        return EmergencyStopResponse(ok=False, errors=[f"{type(exc).__name__}: {exc}"])


class EmergencyLatchState(BaseModel):
    """急停闩现在挂没挂,以及**为什么**。"""

    latched: bool = False       # 闩着 —— 新任务无法启动,仪器动作全被拒
    abort_set: bool = False     # 全局 abort 事件本身是否为 set
    why: str = ""               # 挂它的原因;空串 = 没留原因(不要猜)
    degraded: bool = False      # 没有活的 app


class ClearEmergencyRequest(BaseModel):
    reason: str = Field("", max_length=200,
                        description="解除理由,写进日志留痕(可空)")


class ClearEmergencyResponse(BaseModel):
    ok: bool = True
    was_latched: bool = False   # 解之前确实闩着
    cleared_why: str = ""       # 被解掉的那个闩当初是因为什么挂的
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)


@router.get("/safety/emergency-latch", response_model=EmergencyLatchState)
def get_emergency_latch(request: Request) -> EmergencyLatchState:
    """闩的状态 —— 挂没挂、为什么。永不 500。

    在这个端点存在之前,「闩着」这件事在系统里**没有任何地方看得见**:
    症状只有一个 —— 每次仪器调用都被拒,而拒绝语说是用户干的。
    """
    ctx = request.app.state.ctx
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    fn = getattr(app, "emergency_latch_state", None)
    if not callable(fn):
        return EmergencyLatchState(degraded=True)
    try:
        st = fn() or {}
        return EmergencyLatchState(
            latched=bool(st.get("latched")),
            abort_set=bool(st.get("abort_set")),
            why=str(st.get("why") or ""),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("emergency-latch read failed: %s", exc)
        return EmergencyLatchState(degraded=True)


@router.post("/safety/clear-emergency", response_model=ClearEmergencyResponse)
def clear_emergency(request: Request,
                    body: ClearEmergencyRequest) -> ClearEmergencyResponse:
    """解除急停闩,让仪器动作重新被允许。

    ## 为什么必须有这个端点

    ``POST /safety/emergency-stop`` **挂**闩;在 2026-08-13 之前,全系统**没有
    任何东西解**它 —— ``MASTApp.clear_emergency_latch()`` 存在但没有路由、没有
    按钮,是死代码。而闩会被三个来源自动挂上(急停按钮、任意 E_STOP 事件、
    环境告警),其中两个不需要人参与。

    这类情况真会发生:``main`` 端口抖了 21 秒,环境监控把「这一次读不到」判成
    硬故障 ⇒ 退针 + 挂闩;通知用的 E_STOP 又因为 reason 不在白名单里被静默
    丢掉,此后每一次仪器调用都只回一句
    「用户已中止本次运行」,别的什么都看不到。连接 21 秒后就自愈了,针和仪器全程完好,机器却
    锁死到进程重启为止。**能停不能解的开关不是安全措施,是死锁。**

    ## 它不做什么

    它**不**碰硬件:不进针、不开反馈、不恢复任何运行。它只把「拒绝一切动作」
    这个状态放开,之后做什么由人决定。解除前请先读 ``GET
    /safety/emergency-latch`` 的 ``why`` —— 如果闩是真的越限挂上的,那个原因
    还在那里,解开它不会让它消失。
    """
    ctx = request.app.state.ctx
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    state_fn = getattr(app, "emergency_latch_state", None)
    clear_fn = getattr(app, "clear_emergency_latch", None)
    if not callable(clear_fn):
        return ClearEmergencyResponse(ok=False, degraded=True,
                                      errors=["no live app / clear hook"])
    prev_why = ""
    if callable(state_fn):
        try:
            prev_why = str((state_fn() or {}).get("why") or "")
        except Exception:  # noqa: BLE001
            prev_why = ""
    try:
        was = bool(clear_fn(body.reason or "via API"))
        logger.warning("emergency latch cleared via API (was_latched=%s, "
                       "prev_why=%r, reason=%r)", was, prev_why, body.reason)
        return ClearEmergencyResponse(ok=True, was_latched=was,
                                      cleared_why=prev_why)
    except Exception as exc:  # pragma: no cover
        logger.critical("clear-emergency endpoint failed: %s", exc)
        return ClearEmergencyResponse(ok=False,
                                      errors=[f"{type(exc).__name__}: {exc}"])


class ReconnectRoleRequest(BaseModel):
    role: str = Field("main", description="main / monitor / data / emergency")
    reason: str = Field("", max_length=200, description="留痕用,可空")


class ReconnectRoleResponse(BaseModel):
    ok: bool = True
    reconnected: bool = False   # 真的换上了一条新连接
    role: str = ""
    degraded: bool = False      # 没有活的 app / 连接池
    errors: list[str] = Field(default_factory=list)


@router.post("/safety/reconnect-role", response_model=ReconnectRoleResponse)
def reconnect_role(request: Request,
                   body: ReconnectRoleRequest) -> ReconnectRoleResponse:
    """优雅地重建指定角色的 Nanonis TCP 连接。
    
    为自动重连未能识别或恢复的连接故障提供显式入口。它只更换连接，
    不进针、不开反馈、不改变扫描。沿连接池的 FIN、排空迟到字节、关闭和新建
    流程执行，避免强制关闭影响单客户端端口的恢复。"""
    ctx = request.app.state.ctx
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    pool = getattr(app, "_pool", None)
    fn = getattr(pool, "reconnect_role", None)
    if not callable(fn):
        return ReconnectRoleResponse(ok=False, degraded=True, role=body.role,
                                     errors=["no live pool / reconnect hook"])
    try:
        done = bool(fn(body.role, body.reason or "via API"))
        logger.warning("reconnect-role via API: role=%r ok=%s reason=%r",
                       body.role, done, body.reason)
        return ReconnectRoleResponse(ok=True, reconnected=done, role=body.role)
    except Exception as exc:  # pragma: no cover
        logger.critical("reconnect-role endpoint failed: %s", exc)
        return ReconnectRoleResponse(ok=False, role=body.role,
                                     errors=[f"{type(exc).__name__}: {exc}"])


# ══════════════════════════════════════════════════════════════════════════
# 闩族 —— 一次看全所有「挂上就拒绝/就停」的东西
#
# 上面那两个急停路由是**一个**闩的三件套(可读 / 可解 / 解了连带放开下游)。
# 2026-08-14 交接文档 §2 点名的是**五处**同形状,而其余四处各缺各的:有的能解
# 不能读(重连口在,连接健康度全仓零读者),有的读不到也解不掉(空转记账),
# 有的释放口只是别的动作的副作用(看门狗要靠「开始一个新任务」才重新布防)。
#
# 「发现一个看着在防护其实没有,就当场扫同形状」的反向用法:发现一个能挂不能解,
# 就把全家摆到同一页上,让缺的那一格自己显形。这个端点就是那一页。
#
# 三条纪律:
#   * **闭集枚举**。来源写死在 _LATCH_READERS 里,不做动态发现 —— 一个闩因为
#     没人注册而从列表里消失,和它不存在是一模一样的症状。
#   * **读不到就说读不到**。`latched` 是三态 `True/False/None`,None 必须配
#     `unreadable_reason`。把「读不到」折叠成 False 正是 2026-08-13 那一族事故
#     (一次读取失败被写成一个看起来完全合理的具体答案)。
#   * **`release` 为空 = 一份 bug 报告**,不是一个正常状态。
# ══════════════════════════════════════════════════════════════════════════

class LatchState(BaseModel):
    """一个停止源现在的状态,以及为什么、谁能解。"""

    id: str = ""                      # 闭集键:emergency / stall_guard / …
    name: str = ""                    # 中文名
    latched: "bool | None" = None     # True 挂着 / False 没挂 / **None = 读不到**
    unreadable_reason: str = ""       # 非空 ⟺ latched is None
    why: str = ""                     # 挂上的原因;空串 = 没留原因(不要猜)
    since: "float | None" = None      # epoch 秒;None = 不知道什么时候挂的
    effect: str = ""                  # 挂着的时候实际会发生什么
    release: str = ""                 # 释放口;空串 = **没有释放路径**
    release_actor: str = ""           # 谁能解:operator / 自动 / …
    detail: dict = Field(default_factory=dict)


class LatchFamilyResponse(BaseModel):
    latches: list[LatchState] = Field(default_factory=list)
    #: 三态。有任何一处读不到、而读得到的都没挂 ⇒ **None**,不是 False ——
    #: 「没查到」不能冒充「没事」。
    any_latched: "bool | None" = None
    unreadable: list[str] = Field(default_factory=list)   # 哪几处读不到
    without_release: list[str] = Field(default_factory=list)  # 哪几处没有释放口
    degraded: bool = False            # 没有活的 app,只能给出结构不能给出状态


def _app_of(request: Request):
    ctx = getattr(request.app.state, "ctx", None)
    return getattr(ctx, "live_app", None) or getattr(ctx, "app", None)


def _emergency_latch(app) -> LatchState:
    st = LatchState(
        id="emergency", name="急停闩",
        effect="新任务无法启动,仪器动作全被拒(拒绝语会说是用户中止的)",
        release="POST /api/safety/clear-emergency", release_actor="operator")
    fn = getattr(app, "emergency_latch_state", None)
    if not callable(fn):
        st.unreadable_reason = "没有活的 app / emergency_latch_state 钩子"
        return st
    try:
        d = fn() or {}
    except Exception as exc:  # noqa: BLE001
        st.unreadable_reason = f"读取抛异常:{type(exc).__name__}: {exc}"
        return st
    st.latched = bool(d.get("latched"))
    st.why = str(d.get("why") or "")
    st.detail = {"abort_set": bool(d.get("abort_set"))}
    return st


def _stall_latch(_app) -> LatchState:
    """空转记账 —— 五处里最后一处补上三件套的。

    「挂着」的定义:有签名已经**上膛**(escalation ≥ 上限)。上膛意味着下一次
    同签名的失败会**当场终结一个回合,且不再给智能体任何提示** —— 而这个计数
    按进程累计、不随运行重置,所以它可以是几天前某次早就过去的故障留下的。
    在这个端点之前,这个状态在系统里没有任何地方看得见,也没有任何东西能解。
    """
    st = LatchState(
        id="stall_guard", name="空转记账（stall-guard 升级阶梯）",
        effect="上膛的签名下一次再失败会当场终止该回合,且不再提示智能体;"
               "记账按进程累计,不随运行重置",
        release="POST /api/safety/clear-stall-guard", release_actor="operator")
    try:
        from mast.agents._shared.stall_guard_mw import latch_rows
        rows = latch_rows()
    except Exception as exc:  # noqa: BLE001
        st.unreadable_reason = f"读取抛异常:{type(exc).__name__}: {exc}"
        return st
    armed = [r for r in rows if r.get("armed")]
    st.latched = bool(armed)
    if armed:
        first = armed[0]
        st.why = (f"{first.get('agent')} 的 `{first.get('tool')}` 已上膛"
                  f"（提示 {first.get('escalation')} 次,"
                  f"最后一次同错失败 {first.get('last_count')} 连）"
                  + (f"，另有 {len(armed) - 1} 条" if len(armed) > 1 else ""))
        st.since = min((float(r.get("first_seen") or 0.0) for r in armed),
                       default=None) or None
    # 上膛的**一条都不省**(那是要回答的问题);还在爬阶梯的截断,并且**说出来**
    # 截断了 —— 一个悄悄少给几行的诊断口,读的人会拿它当全集。
    shown = armed + [r for r in rows if not r.get("armed")][:_STALL_ROWS_SHOWN]
    st.detail = {"armed": len(armed), "tracked": len(rows), "rows": shown,
                 "truncated": len(shown) < len(rows)}
    return st


def _comms_latch(app) -> LatchState:
    """TCP 通信熔断。连续 N 次 TCP 级失败后短路一切调用 —— 挂着的时候每个技能
    都失败,而失败原因说的是通信,不是它自己。

    `ConnectionPool.comms_snapshot()` 早就存在,**全仓零读者**(生产方接好了、
    消费方不存在的那一族)。这里是它的第一个读者。"""
    st = LatchState(
        id="comms_breaker", name="Nanonis TCP 通信熔断",
        effect="短路一切 safe_call:立即返回「通信中断」而不再碰 socket"
               "（刻意的 —— 端口很脆,重锤会把它锤到要重启 Nanonis）",
        release="冷却到期自动放一个探针;或 POST /api/safety/reconnect-role",
        release_actor="自动(冷却) + operator")
    pool = getattr(app, "_pool", None)
    fn = getattr(pool, "comms_snapshot", None)
    if not callable(fn):
        st.unreadable_reason = "没有活的连接池 / comms_snapshot 钩子"
        return st
    try:
        snap = dict(fn() or {})
    except Exception as exc:  # noqa: BLE001
        st.unreadable_reason = f"读取抛异常:{type(exc).__name__}: {exc}"
        return st
    state = str(snap.get("state") or "")
    if not state:
        st.unreadable_reason = "熔断器没有报出 state 字段"
        return st
    st.latched = state.upper() != "CLOSED"
    if st.latched:
        st.why = (f"连续 {snap.get('streak')} 次 TCP 级失败"
                  f"（阈值 {snap.get('fail_threshold')}）"
                  + (f"：{snap.get('last_reason')}" if snap.get("last_reason") else ""))
    st.detail = snap
    return st


def _watchdog_latch(app) -> LatchState:
    """看门狗的异常闩 —— **反向的闩**:挂着不代表拒绝什么,代表那张安全网
    **不再监视了**(一次性,触发退针后就停)。把它和「拒绝一切」的闩并成一个
    bool 会让人读反,所以 effect 里写清楚。"""
    st = LatchState(
        id="watchdog_anomaly", name="安全看门狗异常闩（反向：挂着=网已撤）",
        effect="一次性闩:触发过一次 tip-crash 退针之后看门狗停止监视,"
               "**不再拒绝任何动作,但也不再保护**,直到重新布防",
        release="POST /api/safety/rearm-watchdog（开始新任务时也会自动重新布防）",
        release_actor="operator + 自动(开始新任务)")
    ex = getattr(app, "_executor", None)
    wd = getattr(ex, "_watchdog", None) if ex is not None else None
    if wd is None:
        st.unreadable_reason = "没有活的 executor / watchdog"
        return st
    try:
        st.latched = bool(getattr(wd, "is_anomaly_triggered", False))
    except Exception as exc:  # noqa: BLE001
        st.unreadable_reason = f"读取抛异常:{type(exc).__name__}: {exc}"
        return st
    if st.latched:
        st.why = "看门狗触发过一次异常退针,尚未重新布防"
    return st


#: 空转账本里「还在爬阶梯」那部分最多回多少行(上膛的全回,不截断)。
_STALL_ROWS_SHOWN = 20

#: 闭集。顺序即展示顺序;加一个闩 = 在这里加一行 + 写它的读取器。
_LATCH_READERS = (
    ("emergency", _emergency_latch),
    ("stall_guard", _stall_latch),
    ("comms_breaker", _comms_latch),
    ("watchdog_anomaly", _watchdog_latch),
)


@router.get("/safety/latches", response_model=LatchFamilyResponse)
def get_latches(request: Request) -> LatchFamilyResponse:
    """全部停止源:挂没挂 / 为什么 / 谁能解。永不 500。

    ## 为什么是一个端点而不是四个

    2026-08-13 的伤害不是「机器停了」,是**人被送去查一个根本不存在的原因** ——
    每次仪器调用都被拒,拒绝语说是他自己中止的,而系统里没有任何一页能回答
    「现在到底是什么在拦我」。逐个闩各有各的端点仍然回答不了这个问题:
    要读的人先知道该问哪一个。这里一次给全。

    `latched` 是三态:`null` 表示**读不到**(配 `unreadable_reason`),
    绝不折叠成 `false`。同理 `any_latched` 在「有一处读不到且其余都没挂」时
    返回 `null` —— 那时诚实的答案是「不知道」。
    """
    app = _app_of(request)
    out: list[LatchState] = []
    for _id, reader in _LATCH_READERS:
        try:
            out.append(reader(app))
        except Exception as exc:  # noqa: BLE001 — one bad reader must not hide the rest
            logger.warning("latch reader %s failed: %s", _id, exc)
            out.append(LatchState(id=_id, name=_id,
                                  unreadable_reason=f"{type(exc).__name__}: {exc}"))
    unreadable = [s.id for s in out if s.latched is None]
    return LatchFamilyResponse(
        latches=out,
        any_latched=(True if any(s.latched for s in out)
                     else (None if unreadable else False)),
        unreadable=unreadable,
        without_release=[s.id for s in out if not s.release],
        degraded=app is None,
    )


class ClearStallRequest(BaseModel):
    agent: str = Field("", max_length=64,
                       description="只解某个 agent 的（空 = 全部）")
    signature: str = Field("", max_length=200,
                           description="签名子串,大小写不敏感（空 = 全部）")
    reason: str = Field("", max_length=200, description="留痕用,可空")


class ClearStallResponse(BaseModel):
    ok: bool = True
    released: int = 0                       # 真的解掉了几条
    signatures: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


@router.post("/safety/clear-stall-guard", response_model=ClearStallResponse)
def clear_stall_guard(body: ClearStallRequest) -> ClearStallResponse:
    """解除空转记账,让下一次同样的失败重新从「提示」开始。

    ## 为什么必须有

    stall-guard 的升级阶梯靠一本**每 agent 一个、进程生命期、只增不减**的账本
    (提示是注进单次模型调用的,从不写回 state,所以转录数不出来 —— 见
    `stall_guard_mw` 的模块说明)。后果:两次历史提示会把一个签名**永久上膛**,
    此后任何对话、任何实验、几天之后的一次偶发同错失败,都会当场终结一个回合,
    而且**一次提示都不给**。

    在这个端点之前,解开它的办法只有三个:那次击杀本身(代价就是被杀的回合)、
    账本满 64 条时的一次连坐大赦、重启进程。**「重启当释放」是被明确否掉的方案**
    (`test_latch_release_never_requires_restart`)。

    ## 它不做什么

    不碰硬件,不改安全限值,不让任何被安全门拒过的动作变成允许。它只把「这个
    失败签名已经被提示过两次」这条记账抹掉 —— 之后同样的失败会重新走完
    提示 → 再提示 → 终止本回合的完整阶梯。真正的故障因此仍然会被拦住。
    """
    try:
        from mast.agents._shared.stall_guard_mw import release

        rows = release(agent=body.agent, signature=body.signature,
                       why=body.reason or "via API")
        return ClearStallResponse(ok=True, released=len(rows),
                                  signatures=[str(r.get("signature", "")) for r in rows])
    except Exception as exc:  # pragma: no cover
        logger.critical("clear-stall-guard endpoint failed: %s", exc)
        return ClearStallResponse(ok=False, errors=[f"{type(exc).__name__}: {exc}"])


class RearmWatchdogResponse(BaseModel):
    ok: bool = True
    rearmed: bool = False       # 真的有一个活的看门狗被重新布防
    was_latched: "bool | None" = None   # 重新布防之前挂着没有;None = 读不到
    degraded: bool = False
    errors: list[str] = Field(default_factory=list)


@router.post("/safety/rearm-watchdog", response_model=RearmWatchdogResponse)
def rearm_watchdog(request: Request) -> RearmWatchdogResponse:
    """把安全看门狗重新布防。

    异常闩是一次性的:触发过一次 tip-crash 退针之后它就不再监视,而**唯一**
    让它重新布防的地方是「开始一个新的群聊任务」(`CoreRuntime.reset_watchdog`
    从那条路径上被调)。也就是说,想恢复保护就得先启动一个任务 ——
    释放口挂在别的动作的副作用上,和没有释放口只差一步。这里给它一个自己的。

    不碰硬件:只清异常标记与缓冲,让那张网重新开始看。"""
    app = _app_of(request)
    fn = getattr(app, "reset_watchdog", None)
    if not callable(fn):
        return RearmWatchdogResponse(ok=False, degraded=True,
                                     errors=["no live app / reset_watchdog hook"])
    was: "bool | None" = None
    try:
        ex = getattr(app, "_executor", None)
        wd = getattr(ex, "_watchdog", None) if ex is not None else None
        if wd is not None:
            was = bool(getattr(wd, "is_anomaly_triggered", False))
    except Exception:  # noqa: BLE001 — 读不到就是 None,不猜
        was = None
    try:
        done = bool(fn())
        logger.warning("watchdog re-armed via API (was_latched=%s, ok=%s)", was, done)
        return RearmWatchdogResponse(ok=True, rearmed=done, was_latched=was)
    except Exception as exc:  # pragma: no cover
        logger.critical("rearm-watchdog endpoint failed: %s", exc)
        return RearmWatchdogResponse(ok=False, was_latched=was,
                                     errors=[f"{type(exc).__name__}: {exc}"])
