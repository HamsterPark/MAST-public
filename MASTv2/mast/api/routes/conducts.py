"""``/api/conducts/*`` —— 多天 conduct 的用户入口。

设计:``docs/v2/design/campaign_director_design.md`` §7。

## 谁能按这些按钮:由**自主度策略**裁决,不由「有没有这个工具」裁决

2026-08-20 之前这里写的是「本路由**永远不注册为任何 agent 工具**,尤其是
approve —— 批准一份要连跑三天、要动针的流程是人的动作」。那个担心是对的,
**它选的实现方式不对**:把关建在「模型手里没有这个按钮」上,防的是模型想不到
去做,防不了它换一条路做;而它同时封死了另一件事 —— 夜里没有人在场时,什么也
批不了,而那正是全自动实验室要解决的问题。

现在把关在服务端:``mast/conduct/autonomy.py`` 的三档策略决定「谁点头算数」
(attended=仅人 / supervised=agent 可批但留撤销窗 / autonomous=即批即跑),
而**模板 lint、参数包络、approvable 判据在三档下完全一样**。agent 那一侧的
同名能力在 ``agents/_shared/conduct_tools.py``,走的是同一个裁决函数。

这个模块里仍然没有 ``@tool``(它是 HTTP 路由,不是工具模块)—— 但那已经不是
一道安全边界,只是一个分层事实。

## 两条并发纪律,写在这一层

1. **状态字段的单写者是 Director。** 本路由**不改** ``conducts`` 的状态列,
   只往 ``conduct_ops`` 写意图,由 Director 下一 tick 消费。
   例外只有两个,而且都是「人的动作,不是状态机的推进」:``create``(建草稿)与
   ``approve``(冻结参数)—— 那两个时刻 Director 还没有采纳这份 conduct。
   ``compile-from-plan`` **不是**第三个例外:它纯读,一个字都不写(既不写
   conduct 库也不写 plan 库),回的是一次干跑的结论。
2. **abort 不等 tick。** abort 落表的**同时**,API 线程立刻置当前步的 per-run
   abort Event。Director 卡在一次 ``executor.run`` 里时 tick 根本不会来,而
   abort 按钮必须还能用(设计 §4.7 写序纪律的唯一例外)。

## 引擎关着的时候

``conduct.cd_enabled`` 默认 0。关着时读端点回 ``degraded=true`` + 一句为什么,
写端点回 503 —— **不是**一个看起来正常的空响应。「引擎没开」和「没有 conduct」
必须长得不一样。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Query, Request, Response
from starlette import status as http

from mast.api.schemas_conduct import (
    AbortRequest,
    AckRequest,
    AckView,
    ActiveWaitView,
    AttendedRequest,
    BudgetView,
    ConditionView,
    ConductApproveRequest,
    ConductApproveResponse,
    ConductCompileError,
    ConductCompileResponse,
    ConductConfigResponse,
    ConductCreateRequest,
    ConductCreateResponse,
    ConductDetail,
    ConductKnob,
    ConductListResponse,
    ConductListRow,
    DetourView,
    FolderView,
    GateHistoryRow,
    HeartbeatView,
    IgnitionView,
    OpRequest,
    OpResponse,
    OverrideDecisionRequest,
    ParamEcho,
    ParamSpecRow,
    PendingDecisionView,
    StagePos,
    StageSummary,
    StepPos,
    TemplateListResponse,
    TemplateRow,
    TimelineRow,
    WaiveRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conduct"])

#: 引擎关着时的那句话。**一处定义** —— 三个端点回同一句,免得用户在不同页面
#: 读到三种说法然后以为是三件事。
ENGINE_OFF = ("conduct 指挥线程未启用(设置 → conduct.cd_enabled = 1 后重启生效)。"
              "已有的 conduct 状态仍然读得到。")


# ── 取件 ────────────────────────────────────────────────────────────

def _service():
    """当前的 ConductService,没有就 ``None``。永不抛。"""
    try:
        from mast.conduct.service import get_service

        return get_service()
    except Exception as exc:  # noqa: BLE001
        logger.debug("conduct service 取不到: %s", exc)
        return None


def _engine_enabled() -> bool:
    try:
        from mast.conduct.settings import get_conduct_knobs

        return bool(get_conduct_knobs().enabled)
    except Exception:  # noqa: BLE001
        return False


def _spec_for(row: dict):
    """这份 conduct 的模板。取不到回 ``None`` —— 调用方要如实报「取不到模板」,
    **不许**当成「没有阶段」(那会让面板显示一份空 conduct)。"""
    try:
        from mast.conduct.templates import get_template

        return get_template(str(row.get("spec_id") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("conduct 模板取不到: %s", exc)
        return None


def _live_app(request: Request):
    """进程里那个 ``CoreRuntime``(standalone dev 下没有,返回 ``None``)。"""
    ctx = getattr(request.app.state, "ctx", None)
    return getattr(ctx, "live_app", None) or getattr(ctx, "app", None)


def _plan_store(request: Request):
    """plan 库。**取不到就是取不到,这里不自己建一个。**

    ``PlanStore.__init__`` 会 ``mkdir`` 并 ``CREATE TABLE`` —— 一个「取不到就
    按默认路径建一份」的兜底,是在真实实验库上动手。测试污染真实数据在本仓已经
    五次,每一次的入口都是一个善意的默认路径(``ConductStore`` 因此干脆拒绝没有
    ``db_path`` 的构造)。取不到时端点回 503 + 一句为什么,由人去接线。
    """
    return getattr(_live_app(request), "_plan_store", None)


def _skill_registry(request: Request):
    ctx = getattr(request.app.state, "ctx", None)
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    return (getattr(ctx, "skill_registry", None) or getattr(ctx, "registry", None)
            or getattr(app, "_registry", None))


def _skills_and_analyses(request: Request):
    """approve 时校验器要的两样。``(skills, analyses, missing)``。

    ``skills=None`` 会让 ``validate_spec`` 把规则③记进 ``checks_skipped``,
    于是 ``approvable`` 是 False —— 「没检查」不会长得像「检查通过」。
    """
    skills = None
    missing: list[str] = []
    reg = _skill_registry(request)
    if reg is None:
        missing.append("skill_registry")
    else:
        try:
            from mast.conduct.validator import skill_index

            skills = skill_index(reg)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"skill_registry({exc})")
    try:
        from mast.conduct.analyses import known_names

        analyses = sorted(known_names())
    except Exception as exc:  # noqa: BLE001
        analyses = None
        missing.append(f"analyses({exc})")
    return skills, analyses, missing


# ── 旋钮目录 + 模板下拉(必须声明在 /conduct/{id} 之前)──────────────

@router.get("/conducts/config", response_model=ConductConfigResponse)
def get_conduct_config() -> ConductConfigResponse:
    """引擎开没开 + 旋钮目录(设置页动态渲染)。永不 5xx。

    ``enabled``(用户的意愿)与 ``director_running``(此刻线程活着没有)分开报:
    刚翻开开关而进程还没重启时两者不同,合成一个数会让「设了但没生效」看不出来。
    """
    try:
        from mast.conduct.settings import get_conduct_knobs, knob_catalog

        svc = _service()
        return ConductConfigResponse(
            enabled=bool(get_conduct_knobs().enabled),
            director_running=bool(getattr(svc, "is_running", False)),
            knobs=[ConductKnob(**k) for k in knob_catalog()])
    except Exception as exc:  # noqa: BLE001
        logger.warning("conduct config 读取失败: %s", exc)
        return ConductConfigResponse(degraded=True, reason=str(exc))


# ── 模板下拉(必须声明在 /conduct/{id} 之前)────────────────────────

def _param_rows(spec) -> list[ParamSpecRow]:
    """模板的 ``params_schema`` → 表单契约。**永不抛。**

    ``default`` 刻意不过河(见 :class:`ParamSpecRow` 的 docstring):表单渲染件
    拿不到那个数,就不会预填一个没人定过的工作点。这里回的是
    ``required = default is None`` —— 与 ``check_params`` 判必填用的是**同一个
    条件**,不是另写一遍。
    """
    out: list[ParamSpecRow] = []
    for p in getattr(spec, "params_schema", ()) or ():
        try:
            out.append(ParamSpecRow(
                name=str(p.name), type=str(p.type), unit=str(p.unit or ""),
                min_value=p.min_value, max_value=p.max_value,
                help=str(p.help or ""), choices=list(p.choices or ()),
                required=p.default is None))
        except Exception as exc:  # noqa: BLE001 —— 一个坏参数不该让整张表消失
            logger.warning("params_schema 一行转不了: %s", exc)
    return out


@router.get("/conducts/templates", response_model=TemplateListResponse)
def list_templates(request: Request) -> TemplateListResponse:
    """有哪些模板、**这台机器上批不批得下去**,以及**每个模板要填什么**。

    只列名字是不够的:一个引用了本机没有的技能的模板,在下拉框里和能跑的模板
    长得一样,直到 approve 那一刻才报错。这里当场把校验器跑一遍,把「缺什么」
    提前摆出来。

    ``params_schema`` 同理:没有它,新建表单渲染不出输入格,而「新建」这个动作
    就只剩 curl 一条路 —— 屏幕上不会有任何东西说明它去哪了。
    """
    try:
        from mast.conduct.templates import TEMPLATES
        from mast.conduct.validator import validate_spec
    except Exception as exc:  # noqa: BLE001
        return TemplateListResponse(degraded=True, reason=f"模板层不可用: {exc}")
    skills, analyses, _missing = _skills_and_analyses(request)
    rows: list[TemplateRow] = []
    for spec in TEMPLATES.values():
        try:
            rep = validate_spec(spec, skills=skills, analyses=analyses)
            rows.append(TemplateRow(
                spec_id=spec.spec_id, spec_version=int(spec.spec_version),
                title=spec.title, stages=[s.stage_id for s in spec.stages],
                approvable=bool(rep.approvable),
                findings=[str(f) for f in rep.findings],
                checks_skipped=list(rep.checks_skipped),
                params_schema=_param_rows(spec)))
        except Exception as exc:  # noqa: BLE001
            # 校验器炸了 ⇒ 批不下去,但**草稿仍然建得出来**(create 只跑
            # check_params)。所以这一行照样带上 params_schema —— 少了它,
            # 表单会变成一张空表,而空表看起来像「这个模板不需要参数」。
            rows.append(TemplateRow(spec_id=spec.spec_id, title=spec.title,
                                    approvable=False,
                                    findings=[f"校验器抛异常: {exc}"],
                                    params_schema=_param_rows(spec)))
    return TemplateListResponse(templates=rows)


# ── plan → conduct 干跑(必须声明在 /conducts/{conduct_id}/… 之前)──────
#
# 路由顺序在这一层是**语义**不是风格:``/conducts/compile-from-plan/{plan_id}``
# 与 ``/conducts/{conduct_id}/approve`` 段数相同,一个 plan_id 恰好叫
# ``approve`` 时,先声明的那条赢。静态段在前 ⇒ 赢的是这一条,而这一条才是
# 请求真正的意思。

def _compile_from_plan(request: Request, plan_id: str):
    """跑一次编译器。返回 ``(CompileResult | None, 取不到的理由)``。

    两个返回位是**两件事**:``result=None`` + 理由 = 编译**没能发生**
    (plan 库没接上);``result.ok=False`` = 编译发生了,结论是编不动。
    把前者伪装成后者,用户会去方案页找一个根本不存在的槽。
    """
    store = _plan_store(request)
    if store is None:
        return None, ("plan 库没接上(standalone dev 或 runtime 还没建 "
                      "PlanStore)—— 编译器不会自己按默认路径建一个库,"
                      "那是在真实实验库上动手")
    try:
        from mast.conduct.compiler import compile_plan_to_conduct_spec

        return compile_plan_to_conduct_spec(
            plan_id, plan_store=store, registry=_skill_registry(request)), ""
    except Exception as exc:  # noqa: BLE001 —— 读库失败 ≠ 「这份方案有问题」
        logger.warning("plan 编译失败(%s): %s", plan_id, exc, exc_info=True)
        return None, f"编译没能跑起来: {type(exc).__name__}: {exc}"


def _compile_errors(result) -> list[ConductCompileError]:
    return [ConductCompileError(code=e.code, stage_id=e.stage_id,
                                slot_name=e.slot_name, detail=e.detail)
            for e in result.errors]


@router.post("/conducts/compile-from-plan/{plan_id}",
             response_model=ConductCompileResponse)
def compile_from_plan(plan_id: str, request: Request,
                      response: Response) -> ConductCompileResponse:
    """把一份 **APPROVED** 的 plan 干跑一遍编译器。**什么都不建、什么都不写。**

    存在的理由:用户要在建 conduct **之前**看得见「这份方案缺哪几个数」。
    没有它,唯一的反馈路径是 ``POST /api/conducts`` 失败 —— 那时人已经按下了
    「新建」,而屏幕上说的是一串参数错误,看起来像是自己填错了表。

    ``ok=False`` 照样是 200:这是一个**答上来了的答案**(见
    :class:`ConductCompileResponse`)。plan 库没接上才是 503 —— 那是读不到。
    """
    result, why = _compile_from_plan(request, plan_id)
    if result is None:
        response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
        return ConductCompileResponse(plan_id=plan_id, degraded=True, reason=why)

    from mast.conduct.compiler import resolved_template

    spec = resolved_template(result)
    echo, stages, title = [], [], ""
    if spec is not None:
        title = spec.title
        stages = _stages_summary(spec)
        try:
            from mast.conduct.validator import check_params

            echo = _param_echo(spec, result.params,
                               check_params(spec, result.params))
        except Exception as exc:  # noqa: BLE001 —— 回显算不出来不该吞掉结论
            logger.warning("编译回显算不出来: %s", exc)
    return ConductCompileResponse(
        ok=bool(result.ok), plan_id=plan_id, spec_id=result.spec_id,
        spec_version=int(result.spec_version), title=title,
        params=dict(result.params), params_echo=echo, stages_summary=stages,
        errors=_compile_errors(result), warnings=list(result.warnings),
        checks_skipped=list(result.checks_skipped))


# ── 建 ──────────────────────────────────────────────────────────────

@router.post("/conducts", response_model=ConductCreateResponse,
             status_code=http.HTTP_201_CREATED)
def create_conduct(body: ConductCreateRequest, request: Request,
                    response: Response) -> ConductCreateResponse:
    """建一份 DRAFT。参数**超包络拒绝,不夹紧**(400,逐字段)。

    409 = 已经有一个未了结的 conduct(单活跃不变式由数据库执行)。

    ## 两条入口,一个出口

    * **表单**:``spec_id`` + ``params`` 由人填;
    * **编译**:给了 ``from_plan_id`` 而 ``params`` 留空 ⇒ 模板与参数都从那份
      APPROVED 的 plan 编译出来(``mast.conduct.compiler``),``created_by``
      记成 ``compiler:plan=<id>`` —— 事后要分得清这份 conduct 是谁装配的。

    两条路**汇到同一个 create**:参数校验、单活跃不变式、回显都只有一份实现。
    编译失败回 4xx 并**带上闭集码**(不是 500):那是「这份方案还编不动」,
    是一个答上来了的拒绝。
    """
    svc = _service()
    if svc is None:
        response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
        return ConductCreateResponse(degraded=True, errors=[ENGINE_OFF])
    try:
        from mast.conduct.templates import get_template
        from mast.conduct.validator import check_params
    except Exception as exc:  # noqa: BLE001
        response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
        return ConductCreateResponse(degraded=True, errors=[f"conduct 层不可用: {exc}"])

    params = dict(body.params)
    created_by = f"ui:from_plan={body.from_plan_id}" if body.from_plan_id else "ui"
    spec_id = body.spec_id
    if body.from_plan_id and not body.params:
        result, why = _compile_from_plan(request, body.from_plan_id)
        if result is None:
            response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
            return ConductCreateResponse(degraded=True, errors=[why])
        if not result.ok:
            # PLAN_NOT_FOUND 是 404(那个 plan 不在),其余是 400(方案编不动)。
            codes = {e.code for e in result.errors}
            response.status_code = (http.HTTP_404_NOT_FOUND
                                    if codes == {"PLAN_NOT_FOUND"}
                                    else http.HTTP_400_BAD_REQUEST)
            return ConductCreateResponse(
                errors=[str(e) for e in result.errors]
                       + [f"[未检查] {s}" for s in result.checks_skipped],
                compile_errors=_compile_errors(result))
        if spec_id and spec_id != result.spec_id:
            # 两个真源,而且**不一致** —— 明拒优于静默挑一个:挑错了的那次,
            # 建出来的 conduct 跑的是另一份实验,而请求里两个字段都各自「没错」。
            response.status_code = http.HTTP_400_BAD_REQUEST
            return ConductCreateResponse(errors=[
                f"body.spec_id={spec_id!r} 与 plan {body.from_plan_id!r} 编出来的 "
                f"{result.spec_id!r} 不一致 —— 走编译就把 spec_id 留空,"
                f"模板由方案说了算"])
        spec_id, params = result.spec_id, dict(result.params)
        created_by = f"compiler:plan={body.from_plan_id}"

    if not spec_id:
        response.status_code = http.HTTP_400_BAD_REQUEST
        return ConductCreateResponse(errors=[
            "spec_id 必填(或给 from_plan_id 并把 params 留空,让编译器定模板)"])
    try:
        spec = get_template(spec_id)
    except KeyError as exc:
        response.status_code = http.HTTP_400_BAD_REQUEST
        return ConductCreateResponse(errors=[str(exc)])

    findings = check_params(spec, params)
    echo = _param_echo(spec, params, findings)
    if findings:
        response.status_code = http.HTTP_400_BAD_REQUEST
        return ConductCreateResponse(params_echo=echo,
                                      stages_summary=_stages_summary(spec),
                                      errors=[str(f) for f in findings])
    attended = (spec.attended_default if body.attended is None
                else bool(body.attended))
    try:
        cid = svc.store.create(experiment_id=body.experiment_id,
                               spec_id=spec.spec_id,
                               spec_version=int(spec.spec_version),
                               params=params, attended=attended,
                               created_by=created_by)
    except Exception as exc:  # noqa: BLE001
        code = (http.HTTP_409_CONFLICT
                if type(exc).__name__ == "ActiveConductExists"
                else http.HTTP_400_BAD_REQUEST)
        response.status_code = code
        return ConductCreateResponse(params_echo=echo,
                                      stages_summary=_stages_summary(spec),
                                      errors=[str(exc)])
    return ConductCreateResponse(ok=True, conduct_id=cid, status="draft",
                                  params_echo=echo,
                                  stages_summary=_stages_summary(spec))


def _param_echo(spec, params: dict, findings) -> list[ParamEcho]:
    errs: dict[str, list[str]] = {}
    for f in findings:
        errs.setdefault(str(f.where), []).append(f.message)
    out = [ParamEcho(name=p.name,
                     value=params.get(p.name, p.default),
                     unit=p.unit,
                     ok=p.name not in errs,
                     error="；".join(errs.get(p.name, [])))
           for p in spec.params_schema]
    # 模板没声明的键也要回显 —— 一个悄悄被忽略的参数,填的人会一直以为它生效了。
    declared = {p.name for p in spec.params_schema}
    for name in sorted(set(params) - declared):
        out.append(ParamEcho(name=name, value=params[name], ok=False,
                             error="；".join(errs.get(name, ["spec 没有声明这个参数"]))))
    return out


def _stages_summary(spec) -> list[StageSummary]:
    return [StageSummary(stage_id=s.stage_id, title=s.title,
                         steps=len(s.all_steps), mandatory=bool(s.mandatory),
                         capabilities=sorted(s.capabilities))
            for s in spec.stages]


# ── 批(**只接受 UI 来源**)─────────────────────────────────────────

def _freeze_measured_params(spec, params: dict) -> "tuple[dict, dict, str]":
    """批准时冻结需要读取的参数，返回合并参数、审计信息与拒批原因。

coord_epoch 来源由 MEASURED_AT_APPROVE 定义，应读取当前活动实验的
坐标代次，不能信任手填值。参数批准后冻结，后续坐标核对据此执行。

读取失败必须拒批并说明原因，不能把 None 写成零；零明确表示该作用域
尚未粗动。此路径因此要求活动实验记录和可读的代次。"""
    from mast.conduct.compiler import MEASURED_AT_APPROVE

    by_name = {p.name: p for p in getattr(spec, "params_schema", ()) or ()}
    merged, audit = dict(params), {}
    for name in MEASURED_AT_APPROVE:
        p = by_name.get(name)
        if p is None:
            continue                      # 这份模板没有这个槽 —— 不关它的事
        from mast.core.coord_epoch import read_current_epoch

        epoch = read_current_epoch()
        if epoch is None:
            return params, {}, (
                f"批不下去:{name} 要在 approve 这一刻**实测**冻结,而当前坐标"
                f"代次**查不到**(没有活动实验记录 / 记录存储不可用)。"
                f"查不到不等于 0 —— 0 是「这个作用域还没粗动过」这个真实答案。"
                f"请先开一个实验记录再批。")
        under = p.min_value is not None and float(epoch) < float(p.min_value)
        over = p.max_value is not None and float(epoch) > float(p.max_value)
        if under or over:
            # 超包络**拒绝,不夹紧** —— 与 check_params 同一条纪律。夹紧会把一个
            # 越界的代次变成一次看起来正常的运行。
            return params, {}, (
                f"批不下去:实测到的 {name}={epoch} 超出模板声明的包络 "
                f"[{p.min_value}, {p.max_value}] —— 拒绝,不夹紧")
        merged[name] = int(epoch)
        audit[name] = int(epoch)
    return merged, audit, ""


@router.post("/conducts/{conduct_id}/approve",
             response_model=ConductApproveResponse)
def approve_conduct(conduct_id: str, body: ConductApproveRequest,
                     request: Request, response: Response) -> ConductApproveResponse:
    """批准一份 DRAFT:自主度分流 → 跑模板 lint → 冻结参数 → 渲染人读快照。

    **只看 ``approvable``**(= 没发现错误 **且** 该跑的检查都跑了)。只看
    ``ok`` 的话,一次「注册表读不到,规则③没跑」会长得和「检查通过」一模一样。

    ## 谁能按这个按钮

    以前这里的答案写死在 ``api/app.py`` 的注释里:「approve 端点**永远不注册
    为任何 agent 工具**:批准一份要跑三天的流程是人的动作」。那句话在有人值守
    的前提下是对的,而**它同时也是全自动实验室的天花板**——夜里三点没有人在场,
    于是什么也开不了工。

    现在这件事由 :mod:`mast.conduct.autonomy` 的三档策略裁决:``attended``
    保持旧行为(仅人可批);``supervised`` 允许 agent 批,但点火前留一个撤销窗;
    ``autonomous`` 即批即跑。**三档下这个函数后面的每一道检查完全一样**——
    模板 lint、参数包络、approvable 判据一个都不放宽。变的只是「谁点头算数」。
    """
    svc = _service()
    if svc is None:
        response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
        return ConductApproveResponse(conduct_id=conduct_id, degraded=True,
                                       errors=[ENGINE_OFF])
    row = svc.store.get(conduct_id)
    if row is None:
        response.status_code = http.HTTP_404_NOT_FOUND
        return ConductApproveResponse(conduct_id=conduct_id,
                                       errors=["没有这份 conduct"])
    if str(row.get("status")) != "draft":
        response.status_code = http.HTTP_409_CONFLICT
        return ConductApproveResponse(conduct_id=conduct_id,
                                       status=str(row.get("status")),
                                       errors=[f"只有 DRAFT 能批准,现在是 "
                                               f"{row.get('status')}"])
    spec = _spec_for(row)
    if spec is None:
        response.status_code = http.HTTP_409_CONFLICT
        return ConductApproveResponse(conduct_id=conduct_id, status="draft",
                                       errors=[f"取不到模板 {row.get('spec_id')!r} —— "
                                               f"这台机器上没有它"])

    # ── 谁能点这个头 ────────────────────────────────────────────────
    # 生效档 = min(全局设置, 模板声明的上限)。模板可以把自己钉得更严。
    from mast.conduct.autonomy import (
        ignition_payload, stricter_of, who_may_approve)
    from mast.conduct.settings import get_conduct_knobs

    knobs = get_conduct_knobs()
    level = stricter_of(knobs.autonomy, getattr(spec, "max_autonomy", None))
    verdict = who_may_approve(level, by=body.approved_by,
                              ignition_delay_s=knobs.cd_ignition_delay_s)
    if not verdict.allowed:
        # 403 而不是 409:这不是「状态不对」,是「你没有这个权」。两者的下一步
        # 动作不同 —— 前者等一会儿再试,后者要么换人来批要么改设置。
        response.status_code = http.HTTP_403_FORBIDDEN
        return ConductApproveResponse(conduct_id=conduct_id, status="draft",
                                       errors=[verdict.reason])

    from mast.conduct.validator import validate_spec

    skills, analyses, missing = _skills_and_analyses(request)
    rep = validate_spec(spec, skills=skills, analyses=analyses)
    if not rep.approvable:
        response.status_code = http.HTTP_409_CONFLICT
        return ConductApproveResponse(
            conduct_id=conduct_id, status="draft",
            validation_ok=bool(rep.ok), validation_complete=bool(rep.complete),
            findings=[str(f) for f in rep.findings],
            checks_skipped=list(rep.checks_skipped),
            errors=["模板批不下去" + (f";另外这些没接上: {missing}" if missing else "")])

    # 实测冻结的槽(coord_epoch)。**在渲染快照之前** —— 快照上写的必须是真正
    # 冻进库里的那一组数,不是用户填表时那一组。
    params, frozen, refusal = _freeze_measured_params(
        spec, dict(row.get("params") or {}))
    if refusal:
        response.status_code = http.HTTP_409_CONFLICT
        return ConductApproveResponse(
            conduct_id=conduct_id, status="draft",
            validation_ok=True, validation_complete=True, errors=[refusal])

    # 人读快照。渲染失败**不拦住 approve**(真源是 SQLite),但要如实报出来。
    doc, doc_err = "", ""
    try:
        doc = svc.journal(conduct_id, str(row.get("experiment_id") or "")).sync_spec(
            spec, params)
        if not doc:
            doc_err = svc.journal(conduct_id).status().reason
    except Exception as exc:  # noqa: BLE001
        doc_err = f"快照渲染失败: {exc}"

    changes = {"status": "approved", "approved_by": body.approved_by,
               "approved_at": svc.store.now_iso()}
    payload = {"by": body.approved_by, "spec_doc": doc,
               "spec_version": int(spec.spec_version)}
    # 撤销窗随批准一起冻结（2026-08-27）。一份实现两处调用 —— agent 工具那条
    # 路写的是同一个 ``ignition_payload``，于是两扇门批出来的 conduct 在
    # Director 眼里长得一模一样。
    payload.update(ignition_payload(verdict, svc.store.now_epoch()))
    if frozen:
        # 冻结与状态转移**同一个事务**(``record`` 的唯一一扇门)。分两次写的话,
        # 中间崩一下就是一份 approved 却没冻代次的 conduct —— 而它看起来完全正常。
        changes["params"] = params
        payload["measured_params"] = dict(frozen)
        for name, value in frozen.items():
            payload[name] = value
            payload[f"{name}_frozen_from"] = "measured_at_approve"
    try:
        svc.store.record(conduct_id, "approved", changes=changes, payload=payload)
    except Exception as exc:  # noqa: BLE001
        response.status_code = http.HTTP_409_CONFLICT
        return ConductApproveResponse(conduct_id=conduct_id, status="draft",
                                       validation_ok=True, validation_complete=True,
                                       errors=[str(exc)])
    return ConductApproveResponse(
        ok=True, conduct_id=conduct_id, status="approved",
        spec_doc_path=doc, validation_ok=True, validation_complete=True,
        findings=[str(f) for f in rep.warnings],
        frozen_params=dict(frozen),
        # 「批了就跑」与「批了、N 秒后跑、这段时间能撤回」必须分得开。
        # 只回 ok=True 的话，调用方（面板、agent）没有任何办法知道自己刚才
        # 按下的是哪一种 —— 而这一档存在的全部理由就是后一种。
        deferred=bool(verdict.deferred),
        ignition_delay_s=float(verdict.ignition_delay_s),
        errors=[doc_err] if doc_err else [])


# ── 列表 ────────────────────────────────────────────────────────────

@router.get("/conducts", response_model=ConductListResponse)
def list_conducts(status_filter: Optional[str] = Query(None, alias="status"),
                   limit: int = Query(50, ge=1, le=200)) -> ConductListResponse:
    """列表行。引擎关着时如实说,**不回一个空的正常响应**。"""
    svc = _service()
    if svc is None:
        return ConductListResponse(engine_enabled=_engine_enabled(),
                                    degraded=True, reason=ENGINE_OFF)
    try:
        rows = svc.store.list_conducts(status=status_filter, limit=limit)
        active = svc.store.active() or {}
    except Exception as exc:  # noqa: BLE001
        return ConductListResponse(engine_enabled=True, degraded=True,
                                    reason=f"读库失败: {exc}")
    out: list[ConductListRow] = []
    for r in rows:
        spec = _spec_for(r)
        stage_id, step_id = _position_ids(spec, r)
        out.append(ConductListRow(
            conduct_id=str(r["conduct_id"]),
            title=spec.title if spec is not None else f"(模板 {r['spec_id']} 不在本机)",
            spec_id=str(r["spec_id"]), status=str(r["status"]),
            stage_id=stage_id, step_id=step_id,
            updated_at=str(r.get("updated_at") or "")))
    return ConductListResponse(
        conducts=out,
        active_conduct_id=str(active.get("conduct_id") or ""),
        engine_enabled=True)


# ── 面板(唯一数据源)──────────────────────────────────────────────

@router.get("/conducts/{conduct_id}", response_model=ConductDetail)
def get_conduct(conduct_id: str, response: Response) -> ConductDetail:
    """整个面板的数据,一次拿完。**永不 5xx。**

    读端点的降级方向与写端点相反:引擎没开、库读不到,一律 200 + ``degraded``
    + 一句为什么。用户点开一条旧链接时,页面要能渲染出「引擎未启用」,
    而不是一个错误码 —— 「任何点击最坏只能无反应/提示」是本仓的硬规则。
    写端点该 503 的照样 503:那是**拒绝**,拒绝要响。
    """
    svc = _service()
    if svc is None:
        return ConductDetail(conduct_id=conduct_id, degraded=True,
                              engine_enabled=_engine_enabled(), reason=ENGINE_OFF)
    try:
        row = svc.store.get(conduct_id)
    except Exception as exc:  # noqa: BLE001
        return ConductDetail(conduct_id=conduct_id, degraded=True,
                              reason=f"读库失败: {exc}")
    if row is None:
        response.status_code = http.HTTP_404_NOT_FOUND
        return ConductDetail(conduct_id=conduct_id, reason="没有这份 conduct")
    try:
        return _detail(svc, row)
    except Exception as exc:  # noqa: BLE001 —— 面板端点永不 500
        logger.warning("conduct 面板渲染失败: %s", exc, exc_info=True)
        return ConductDetail(conduct_id=conduct_id, ok=False, degraded=True,
                              status=str(row.get("status") or ""),
                              status_reason=str(row.get("status_reason") or ""),
                              reason=f"面板渲染失败: {type(exc).__name__}: {exc}")


def _position_ids(spec, row: dict) -> tuple[str, str]:
    """``(stage_id, step_id)``。``step_idx = -1`` 表示停在入口闸门上。"""
    if spec is None:
        return "", ""
    from mast.conduct.director import AT_ENTRY_GATE

    si, pi = int(row.get("stage_idx") or 0), int(row.get("step_idx") or 0)
    if si >= len(spec.stages):
        return "", ""
    stage = spec.stages[si]
    if pi == AT_ENTRY_GATE:
        return stage.stage_id, "(入口闸门)"
    steps = stage.all_steps
    if pi >= len(steps):
        return stage.stage_id, "(出口闸门)"
    return stage.stage_id, steps[pi].step_id


def _ignition_view(svc, conduct_id: str, row: dict):
    """撤销窗还开着吗 —— 面板要显示的那个倒计时。

    只在 ``approved`` 态且 ``ignite_at`` 还没到时给出值。已经点火 / 人批 /
    旧格式事件一律回 ``None`` —— 一个恒存在但 ``remaining_s=0`` 的对象会让
    面板不得不自己再判一次「这算不算还在等」，而那就是判据的第二处实现。

    ``remaining_s`` 现算：存下来的剩余秒数会在每次重启时从头再数。
    """
    if str(row.get("status") or "") != "approved":
        return None
    try:
        # 同 ``director._ignition_hold``：``events()`` 升序 + LIMIT，取尾拿到的
        # 是最早那些。要的是**最后一次批准**。
        last = svc.store.last_event(conduct_id, "approved")
    except Exception:  # noqa: BLE001 — 读不到审计流不该让整个 detail 500
        return None
    if last is None:
        return None
    payload = last.get("payload") or {}
    raw = payload.get("ignite_at")
    if raw is None:
        return None
    try:
        ignite_at = float(raw)
    except Exception:  # noqa: BLE001
        return None
    remaining = ignite_at - svc.store.now_epoch()
    if remaining <= 0:
        return None
    return IgnitionView(by=str(payload.get("by") or ""),
                        ignite_at=ignite_at,
                        delay_s=float(payload.get("ignition_delay_s") or 0.0),
                        remaining_s=remaining)


def _detail(svc, row: dict) -> ConductDetail:
    from mast.conduct.director import AT_ENTRY_GATE

    cid = str(row["conduct_id"])
    spec = _spec_for(row)
    na: list[str] = []
    if spec is None:
        na.append(f"模板 {row.get('spec_id')!r} 不在本机 —— 阶段/步骤/预算上限都算不出来")

    si, pi = int(row.get("stage_idx") or 0), int(row.get("step_idx") or 0)
    stage_pos, step_pos = StagePos(idx=si), StepPos(idx=pi)
    timeline: list[TimelineRow] = []
    if spec is not None and si < len(spec.stages):
        stage = spec.stages[si]
        stage_pos = StagePos(idx=si, id=stage.stage_id, title=stage.title)
        steps = stage.all_steps
        if pi == AT_ENTRY_GATE:
            step_pos = StepPos(idx=pi, id="(入口闸门)", kind="gate")
        elif pi < len(steps):
            st = steps[pi]
            step_pos = StepPos(idx=pi, id=st.step_id, title=st.title or st.step_id,
                               kind=st.kind, timeout_s=st.timeout_s)
        else:
            step_pos = StepPos(idx=pi, id="(出口闸门)", kind="gate")
    if spec is not None:
        for i, stage in enumerate(spec.stages):
            total = len(stage.all_steps)
            if i < si:
                done, state = total, "done"
            elif i > si:
                done, state = 0, "pending"
            else:
                done, state = max(0, min(total, pi)), "current"
            timeline.append(TimelineRow(stage_id=stage.stage_id, title=stage.title,
                                        status=state, steps_done=done,
                                        steps_total=total))

    # 当前这一步是什么时候开始的 —— 停滞告警要它。
    #
    # **按 run_id 问**,不是「取最近几条 step_started」:审计流是升序 + LIMIT,
    # 跑过几十步之后那样问到的是很早那一步的时刻,而它会被当成当前步的时刻 ——
    # 一个刚开始的步于是立刻被判成停滞。没有 active_run_id 就是**不在步里**,
    # 这时「当前步的开始时刻」本来就不存在,留 None 而不是找一个旧的顶上。
    started_at: "float | None" = None
    run_id = str(row.get("active_run_id") or "")
    if run_id:
        try:
            for ev in svc.store.events(cid, kind="step_started", run_id=run_id,
                                       limit=5):
                at = (ev.get("payload") or {}).get("at")
                if isinstance(at, (int, float)):
                    started_at = float(at)
                    break
        except Exception:  # noqa: BLE001
            na.append("这一步的开始时刻(审计流读失败)")
    step_pos = step_pos.model_copy(update={"started_at": started_at})

    detour = None
    d = row.get("detour") or None
    if d:
        # 库里存的是**下标**(位置只有两个数是 §4.7 定下的);面板要的是人读的
        # 阶段名,所以在这里翻译一次,而不是往库里加一列。
        back = int(d.get("return_stage_idx") or 0)
        back_id = (spec.stages[back].stage_id
                   if spec is not None and back < len(spec.stages) else str(back))
        detour = DetourView(active=True, return_stage=back_id,
                            return_step=int(d.get("return_step_idx") or 0),
                            reason=str(d.get("reason") or ""),
                            entered_at=d.get("entered_at"))

    wait = _wait_view(row, spec)
    pending = _pending_decision_view(svc, cid, na)
    gates = _gate_history(svc, cid)
    budget = _budget(svc, cid, spec, na)
    heartbeat = _heartbeat(svc, row, step_pos, started_at)
    folder = _folder(svc, cid, str(row.get("experiment_id") or ""))

    return ConductDetail(
        ok=True, conduct_id=cid,
        experiment_id=str(row.get("experiment_id") or ""),
        spec_id=str(row.get("spec_id") or ""),
        spec_version=int(row.get("spec_version") or 0),
        title=spec.title if spec is not None else "",
        status=str(row.get("status") or ""),
        status_reason=str(row.get("status_reason") or ""),
        attended=bool(row.get("attended")),
        params=dict(row.get("params") or {}),
        stage=stage_pos, step=step_pos, detour=detour,
        evidence_epoch=int(row.get("evidence_epoch") or 0),
        active_wait=wait, pending_decision=pending,
        gates_history=gates, budget=budget,
        llm_wakes={str(k): int(v) for k, v in (row.get("llm_wakes") or {}).items()},
        heartbeat=heartbeat, active_run_id=str(row.get("active_run_id") or ""),
        timeline=timeline, folder=folder,
        approved_by=str(row.get("approved_by") or ""),
        approved_at=str(row.get("approved_at") or ""),
        ignition=_ignition_view(svc, cid, row),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        engine_enabled=_engine_enabled(),
        director_running=bool(getattr(svc, "is_running", False)),
        not_available=na)


def _pending_decision_view(svc, cid: str, na: list) -> "PendingDecisionView | None":
    """现在有没有一次等人放行的闸门判定。

    读不到就**说出来**(进 ``not_available``),不静默回 ``None`` —— 「没有可放行的
    判定」与「问不出来」在面板上是两句话:前者按钮该灰,后者该告诉人这一格没答上。
    """
    try:
        pending = svc.director.pending_decision(cid)
    except Exception as exc:  # noqa: BLE001
        na.append(f"等人放行的闸门判定读不到: {exc}")
        return None
    if not pending:
        return None
    return PendingDecisionView(
        decision_id=int(pending.get("decision_id") or 0),
        gate_id=str(pending.get("gate_id") or ""),
        stage_id=str(pending.get("stage_id") or ""),
        which=str(pending.get("which") or ""),
        verdict=str(pending.get("verdict") or ""),
        reason=str(pending.get("reason") or ""))


def _wait_view(row: dict, spec) -> "ActiveWaitView | None":
    w = row.get("active_wait") or None
    if not w:
        return None
    ack_required = bool(w.get("ack_required"))
    ack_at = w.get("ack_at")
    ack_ok = (not ack_required) or ack_at is not None
    cond_view = None
    cond_ok = True
    cond = _wait_condition(spec, row)
    if cond is not None:
        waived = bool(w.get("waived_at"))
        reading = w.get("last_reading")
        age = w.get("last_reading_age_s")
        # 判定用的三个数是**进等待那一刻解出来的**(绑定到 params 的会覆盖模板里的
        # 占位值)。面板必须显示 Director 真正在用的那一组,否则「等到 5 K」在库里
        # 是 5、在屏幕上是 400,而两边都各自没错。
        threshold = float(w.get("cond_value", cond.value))
        stale_after = float(w.get("cond_stale_after_s", cond.stale_after_s))
        # **stale = 读不到 ≠ 没到。** 面板必须分开显示这两件事。
        stale = (reading is None or age is None or float(age) > stale_after)
        met_since = w.get("condition_met_since")
        met = met_since is not None
        cond_ok = waived or (met and not stale)
        cond_view = ConditionView(
            desc=cond.desc or f"{cond.signal} {cond.op} {threshold:g}",
            threshold=threshold, stale_after_s=stale_after,
            current_value=reading if isinstance(reading, (int, float)) else None,
            met=bool(met), met_since=met_since, stale=bool(stale) and not waived,
            reading_age_s=age if isinstance(age, (int, float)) else None,
            waived=waived, waived_by=str(w.get("waived_by") or ""),
            waive_reason=str(w.get("waive_reason") or ""))
    lacking = []
    if not ack_ok:
        lacking.append("人的确认")
    if not cond_ok:
        lacking.append("物理条件")
    return ActiveWaitView(
        wait_id=str(w.get("wait_id") or ""), kind=str(w.get("kind") or ""),
        message=str(w.get("message") or ""),
        ack=AckView(required=ack_required, at=ack_at,
                    by=str(w.get("ack_by") or "")),
        condition=cond_view, lacking=lacking,
        entered_at=w.get("entered_at"), last_notified_at=w.get("last_notified_at"),
        request_id=str(w.get("request_id") or ""))


def _wait_condition(spec, row: dict):
    """当前 wait 步声明的 ConditionSpec(没有就 ``None``)。"""
    if spec is None:
        return None
    try:
        si, pi = int(row.get("stage_idx") or 0), int(row.get("step_idx") or 0)
        steps = spec.stages[si].all_steps
        step = steps[pi]
        return step.wait.condition if step.wait is not None else None
    except Exception:  # noqa: BLE001
        return None


def _gate_history(svc, cid: str) -> list[GateHistoryRow]:
    try:
        evs = svc.store.events(cid, kind="gate_evaluated", limit=200)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for ev in evs[-20:]:
        p = ev.get("payload") or {}
        # ``llm`` 块只在 llm 闸门上有。**不给它一个空 dict 兜底再当成「有」**:
        # 下面每一项拿不到就留空/None,于是「这条不是 LLM 判的」与「是 LLM 判的
        # 但没记到模型名」在面板上是两件不同的事。
        llm = p.get("llm")
        llm = dict(llm) if isinstance(llm, dict) else None
        kind = str(p.get("kind") or ("llm" if llm is not None else ""))
        out.append(GateHistoryRow(
            ts=str(ev.get("ts") or ""),
            gate_id=str(p.get("gate_id") or ""),
            stage_id=str(ev.get("stage_id") or ""),
            verdict=str(p.get("verdict") or ""),
            route=str(p.get("route") or ""),
            note=str(p.get("reason") or ""),
            kind=kind,
            escaped=bool(p.get("escaped")),
            llm_model=str((llm or {}).get("model") or ""),
            llm_parse_path=str((llm or {}).get("parse_path") or ""),
            llm_unavailable=str((llm or {}).get("unavailable") or ""),
            llm_wakes_used=(llm or {}).get("wakes_used"),
            llm_wakes_max=(llm or {}).get("wakes_max"),
        ))
    return out


def _budget(svc, cid: str, spec, na: list) -> BudgetView:
    """预算条。**这条上限今天拦不住任何东西,面板必须说出来。**

    不是「还没接」,是**计价单位对不上**:账本按 provider 原生币种实测,而
    ``usd_max`` 是 USD,合并需要汇率而本仓不自造汇率(单一真源:
    ``conduct.journal.USD_MAX_NOT_ENFORCEABLE``)。

    留一个回 ``None`` 的读数配一个看起来正常的上限,是「装得像在拦的闸」——
    看的人会据此放心,而那正是最坏的一种。
    """
    from mast.conduct.journal import USD_MAX_NOT_ENFORCEABLE

    cap = float(spec.budgets.usd_max) if spec is not None else 0.0
    spent: "float | None" = None
    by_currency: dict = {}
    reason = ""
    detail = None
    getter = getattr(svc, "cost_detail", None)
    if callable(getter):
        try:
            detail = getter(cid)
        except Exception as exc:  # noqa: BLE001
            reason = f"花销读取失败: {exc}"
    if detail is not None:
        # 逐币种读数(``adapters.ConductCost``)。**不折成一个数** ——
        # 折成一个数就得有汇率,而这里没有。
        by_currency = dict(getattr(detail, "by_currency", None) or {})
        extra = str(getattr(detail, "reason", "") or "")
        if extra:
            reason = (reason + ";" + extra) if reason else extra
    reader = getattr(svc, "_cost_reader", None)
    if callable(reader) and not reason:
        try:
            got = reader(cid)
        except Exception as exc:  # noqa: BLE001
            reason = f"花销读取失败: {exc}"
        else:
            if isinstance(got, (int, float)) and not isinstance(got, bool):
                spent = float(got)
    if spent is None and not by_currency and not reason:
        # **读不到 ≠ 花了 0。** 面板上一个空着的预算条会被读成「没花钱」。
        reason = "还没有按 conduct 归集的实测花销口径 —— 读不到,不是 0"
        na.append("USD 花销")
    # **「拦不拦得住」不看这一刻花了多少。**
    #
    # 第一版把它写成 ``spent is not None`` —— 于是一份还没花过钱的 conduct
    # (读数 0.0)会在面板上宣称这条 USD 上限**有效**,而它下一次判决就可能花出
    # 一笔 CNY,同一条上限又变成无效。**上限的可执行性是结构性质,不是当下读数的
    # 性质**:账本按 provider 原生币种记账,而用哪家由调用时的回退链决定。
    #
    # 判据因此是「**已经有记录,而且全都是 USD**」——它自我纠正:哪天口径真的
    # 统一了,这里不用改就会变 True;而零花销时它诚实地说「还不知道」。
    enforceable = bool(by_currency) and set(by_currency) == {"USD"}
    return BudgetView(spent_usd=spent, cap_usd=cap, reason=reason,
                      measured_by_currency=by_currency,
                      enforceable=enforceable,
                      not_enforceable_why=("" if enforceable
                                           else USD_MAX_NOT_ENFORCEABLE))


#: 心跳与停滞判据无关的状态:还没被采纳,或者已经了结。这些状态下心跳本来就
#: 不该更新,报停滞只会制造噪声。
_HEARTBEAT_IRRELEVANT = frozenset({"draft", "approved", "paused", "completed",
                                   "aborted"})


def _heartbeat(svc, row: dict, step: StepPos,
               started_at: "float | None") -> HeartbeatView:
    """心跳年龄 + 停滞判据(设计 §6-1)。

    **两支,不是一支**:

    * 不在步里 —— 心跳每 tick 都该更新,年龄超过 ``3×tick`` 就是决策循环死了;
    * 在步里(``active_run_id`` 非空)—— 心跳**本来就不更新**,那是设计。
      这时要看的是**这一步跑了多久**:超过 ``timeout_s + 余量`` 才算停滞。

    合成一支会让一次正常的两小时扫描天天报警,而报警报久了就没人看 —— 那时
    真的停滞发生了也一样没人看。

    ``timeout_s`` 只喂这个阈值,**不杀步**:卡死的 TCP 事务杀不得(强杀会永久
    损坏 Nanonis 端口)。这是一条写下来的诚实短板,不是待办。
    """
    try:
        from mast.conduct.settings import get_conduct_knobs
        grace = float(get_conduct_knobs().cd_stall_grace_s)
    except Exception:  # noqa: BLE001
        grace = 300.0
    spec = _spec_for(row)
    tick = float(spec.budgets.tick_interval_s) if spec is not None else 15.0
    loop_threshold = 3.0 * tick

    at = row.get("heartbeat_at")
    age: "float | None" = None
    if isinstance(at, (int, float)):
        try:
            age = max(0.0, float(svc.store.now_epoch()) - float(at))
        except Exception:  # noqa: BLE001
            age = None
    in_step = bool(row.get("active_run_id"))
    elapsed: "float | None" = None
    if in_step and isinstance(started_at, (int, float)):
        try:
            elapsed = max(0.0, float(svc.store.now_epoch()) - float(started_at))
        except Exception:  # noqa: BLE001
            elapsed = None

    status = str(row.get("status") or "")
    if status in _HEARTBEAT_IRRELEVANT:
        return HeartbeatView(at=at if isinstance(at, (int, float)) else None,
                             age_s=age, stalled=False, threshold_s=loop_threshold,
                             in_step=in_step, step_elapsed_s=elapsed,
                             reason=f"{status} 状态下心跳与停滞无关")

    if age is None:
        # 一个处在驱动态、却**一次心跳都没有**的 conduct:Director 从来没碰过
        # 它。多半是引擎关着而库里还留着一行 RUNNING —— 这正是要报出来的事,
        # 「没有读数」在这里不是「没问题」。
        return HeartbeatView(at=None, age_s=None, stalled=True,
                             threshold_s=loop_threshold, in_step=in_step,
                             step_elapsed_s=elapsed,
                             reason="处在驱动态却一次心跳都没有 —— "
                                    "指挥线程从来没碰过它(引擎关着?)")

    if in_step:
        threshold = float(step.timeout_s or 0.0) + grace
        if elapsed is None:
            # 步开始时刻读不到 ⇒ **判不了**,退回循环判据并说清楚。
            return HeartbeatView(at=at, age_s=age,
                                 stalled=age > max(loop_threshold, threshold),
                                 threshold_s=max(loop_threshold, threshold),
                                 in_step=True, step_elapsed_s=None,
                                 reason="在步里,但读不到这一步的开始时刻 —— "
                                        "退回按心跳年龄判,结论偏保守")
        return HeartbeatView(
            at=at, age_s=age, stalled=elapsed > threshold,
            threshold_s=threshold, in_step=True, step_elapsed_s=elapsed,
            reason=(f"在步里:心跳不更新是设计,看的是这一步已经跑了 "
                    f"{elapsed / 60:.0f} min(阈值 {threshold / 60:.0f} min)"))

    return HeartbeatView(at=at, age_s=age, stalled=age > loop_threshold,
                         threshold_s=loop_threshold, in_step=False,
                         step_elapsed_s=None,
                         reason=f"不在步里:心跳每 {tick:g} s 该更新一次")


def _folder(svc, cid: str, experiment_id: str) -> FolderView:
    try:
        st = svc.journal(cid, experiment_id).status()
        return FolderView(path=st.path, spec_doc=st.spec_doc,
                          progress_lines=st.progress_lines, reason=st.reason)
    except Exception as exc:  # noqa: BLE001
        return FolderView(reason=f"实验文件夹状态读不到: {exc}")


# ── 意图 ────────────────────────────────────────────────────────────

def _enqueue(conduct_id: str, op: str, response: Response, *,
             args: "dict[str, Any] | None" = None, by: str) -> OpResponse:
    """往意图队列写一条。**不改状态** —— 状态的单写者是 Director。"""
    svc = _service()
    if svc is None:
        response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
        return OpResponse(degraded=True, errors=[ENGINE_OFF])
    if svc.store.get(conduct_id) is None:
        response.status_code = http.HTTP_404_NOT_FOUND
        return OpResponse(errors=["没有这份 conduct"])
    try:
        op_id = svc.store.enqueue_op(conduct_id, op, args=args or {},
                                     requested_by=by)
    except Exception as exc:  # noqa: BLE001
        response.status_code = http.HTTP_400_BAD_REQUEST
        return OpResponse(errors=[str(exc)])
    return OpResponse(ok=True, op_id=op_id, queued=True)


@router.post("/conducts/{conduct_id}/pause", response_model=OpResponse)
def pause_conduct(conduct_id: str, body: OpRequest,
                   response: Response) -> OpResponse:
    """暂停。**不动针** —— 用户按暂停常常正是为了手动干预。"""
    return _enqueue(conduct_id, "pause", response,
                    args={"note": body.note}, by=body.by)


@router.post("/conducts/{conduct_id}/resume", response_model=OpResponse)
def resume_conduct(conduct_id: str, body: OpRequest,
                    response: Response) -> OpResponse:
    """继续。**先走恢复自检** —— 暂停期间世界未知(人可能动过仪器)。"""
    return _enqueue(conduct_id, "resume", response,
                    args={"note": body.note}, by=body.by)


@router.post("/conducts/{conduct_id}/takeover", response_model=OpResponse)
def takeover_conduct(conduct_id: str, body: OpRequest,
                      response: Response) -> OpResponse:
    """人工接管。**必须显式** —— 用户在 Nanonis 界面上的动作不持仪器令牌,
    靠隐式检测去猜「现在是人在开」必然误判。"""
    return _enqueue(conduct_id, "takeover", response,
                    args={"note": body.note}, by=body.by)


@router.post("/conducts/{conduct_id}/attended", response_model=OpResponse)
def set_attended(conduct_id: str, body: AttendedRequest,
                 response: Response) -> OpResponse:
    """有人/无人值守。无人窗口里「判不了」不问人(必然超时),直接走保守分支。"""
    return _enqueue(conduct_id, "set_attended", response,
                    args={"attended": bool(body.attended)}, by=body.by)


@router.post("/conducts/{conduct_id}/ack", response_model=OpResponse)
def ack_wait(conduct_id: str, body: AckRequest, response: Response) -> OpResponse:
    """人确认一次等待。**双闸的一半** —— 人确认 ≠ 物理条件到位。

    ``wait_id`` 不匹配当前等待 → Director 会记 ``op_rejected``;这里先挡一道并回
    409,免得用户按了按钮、什么都没发生、还得去翻审计流才知道按错了地方。
    """
    svc = _service()
    if svc is not None:
        row = svc.store.get(conduct_id) or {}
        cur = (row.get("active_wait") or {}).get("wait_id")
        if cur and cur != body.wait_id:
            response.status_code = http.HTTP_409_CONFLICT
            return OpResponse(errors=[f"wait_id {body.wait_id!r} 与当前等待 "
                                      f"{cur!r} 不匹配 —— 这是对一个旧等待点的确认"])
    out = _enqueue(conduct_id, "ack", response,
                   args={"wait_id": body.wait_id, "note": body.note}, by=body.by)
    if out.ok and svc is not None:
        row = svc.store.get(conduct_id) or {}
        spec = _spec_for(row)
        view = _wait_view(row, spec)
        # 还缺什么,当场回给按按钮的人(下一 tick 才真正解除)。
        out.wait_echo = list(view.lacking) if view is not None else []
    return out


@router.post("/conducts/{conduct_id}/waive-condition", response_model=OpResponse)
def waive_condition(conduct_id: str, body: WaiveRequest,
                    response: Response) -> OpResponse:
    """把 condition 闸标成「证据由人提供」。

    **不是默默放行**:``waived_by`` / ``waive_reason`` 进审计流,面板**持续**显示
    这个标记。它存在的理由是「能停不能解=死锁」—— 温度传感器读不到时 condition
    闸会永远 stale,人必须有一条显式、留痕的解锁路。
    """
    return _enqueue(conduct_id, "waive_condition", response,
                    args={"wait_id": body.wait_id, "reason": body.reason},
                    by=body.by)


@router.post("/conducts/{conduct_id}/override-decision", response_model=OpResponse)
def override_decision(conduct_id: str, body: OverrideDecisionRequest,
                  response: Response) -> OpResponse:
    """用户看过一次闸门判定之后说「继续」。

    在这条路存在之前,一份被闸门停住的 conduct 只有 abort 与 takeover 两条出路
    —— 凌晨闸门停下、早上人看了觉得没问题,唯一的选择是**放弃这份 conduct**
    或**永久接管**。那是「能停不能解」的第二次(第一次是 2026-08-13 的急停闩)。

    **它解的是这一次判定,不是这道闸**:``decision_id`` 是那一次 ``gate_evaluated``
    的 event_id,每判一次换一个。下一次走到同一道闸,照样重新判。

    与 ack 同一条待客之道:``decision_id`` 对不上先在这里回 409,免得用户按了
    按钮、什么都没发生,还得去翻审计流才知道按错了地方。
    """
    svc = _service()
    if svc is not None:
        pending = None
        try:
            pending = svc.director.pending_decision(conduct_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct 待放行判定读取失败: %s", exc)
        if pending is None:
            response.status_code = http.HTTP_409_CONFLICT
            return OpResponse(errors=[
                "当前没有停在一个可放行的闸门判定上 —— "
                "等待步要用 ack / waive-condition,别的停法要用 resume / abort"])
        if str(pending.get("decision_id")) != str(body.decision_id):
            response.status_code = http.HTTP_409_CONFLICT
            return OpResponse(errors=[
                f"decision_id {body.decision_id!r} 与当前判定 "
                f"{pending.get('decision_id')!r} 不匹配 —— 这是对一个旧判定的放行"])
    return _enqueue(conduct_id, "override_decision", response,
                    args={"decision_id": str(body.decision_id),
                          "reason": body.reason}, by=body.by)


@router.post("/conducts/{conduct_id}/abort", response_model=OpResponse)
def abort_conduct(conduct_id: str, body: AbortRequest,
                   response: Response) -> OpResponse:
    """中止:**立刻**置 per-run abort Event,同时把意图入队。

    两件事都要做,而且顺序是「先置位再入队」:Director 可能正卡在一次
    ``executor.run`` 里,那时 tick 不会来 —— 只入队的话,abort 按钮要等到那一步
    自己结束才生效,而那正是最需要它的时刻。
    """
    svc = _service()
    signalled = False
    if svc is not None:
        try:
            signalled = bool(svc.signal_abort(conduct_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct abort 立即置位失败(意图仍会入队): %s", exc)
    out = _enqueue(conduct_id, "abort", response,
                   args={"reason": body.reason, "by": body.by}, by=body.by)
    out.abort_signalled = signalled
    return out


# ── 旧路径别名 /campaign/* (保留一个版本) ─────────────────────────────
#
# 2026-08-20 执行层 campaign → conduct 改名（campaign 一词归还 logging/v2
# 的科研纲领）。前端已经跟着改，但一个仍在运行的旧页面、一个存着旧 URL 的
# 书签、一份贴在交接文档里的 curl —— 让它们直接 404 是把改名的代价转嫁给
# 用户。所以旧路径原样再挂一份，指向同一批 handler。
#
# 这一层**没有自己的逻辑**：路由对象是复制来的，handler 是同一个函数。
# 下一版删掉它时，只删这一段，不会碰到任何业务代码。
legacy_router = APIRouter(tags=["conduct-legacy"], include_in_schema=False)

for _route in list(router.routes):
    _path = getattr(_route, "path", "")
    if not _path.startswith("/conducts"):
        continue
    # /conducts → /campaign, /conducts/{id}/ack → /campaign/{id}/ack
    _legacy_path = "/campaign" + _path[len("/conducts"):]
    legacy_router.add_api_route(
        _legacy_path,
        _route.endpoint,
        methods=list(getattr(_route, "methods", ()) or ()),
        response_model=getattr(_route, "response_model", None),
        name=f"legacy_{getattr(_route, 'name', '')}",
        include_in_schema=False,
    )

del _route, _path, _legacy_path
