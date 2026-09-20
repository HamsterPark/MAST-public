"""将 plan 的绑定块编译为已注册 conduct 模板及其参数。

此编译器不生成新的 ConductSpec。它识别模板并按 params_schema 验证参数，
成功时返回注册表内同一 spec 对象。存储只记录模板 id、版本与冻结参数，
恢复也按同一版本对账，所以不能返回无法持久化和恢复的临时 spec。
等待与仪器步骤由模板定义，不能由草稿猜测。

编译过程只读，不写 plan 或 conduct 库。plan_store 必须显式传入；
读取失败应抛异常，不能误报为 PLAN_NOT_FOUND。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from mast.conduct.spec import ConductSpec
from mast.conduct.templates import TEMPLATES, get_template
from mast.conduct.validator import check_params, skill_index, validate_spec

logger = logging.getLogger(__name__)

#: 编译错误码的闭集，同时覆盖 plan 级前置与槽级失败。
#: 计划不存在、尚未批准、模板无法识别不能冒充槽级错误。
#: 保留完整词汇表使调用方能够稳定地解释各类拒绝。
COMPILE_ERROR_CODES = (
    # plan 级前置(新增 3 个:p3 的闭集只覆盖槽级错误)
    "PLAN_NOT_FOUND", "PLAN_NOT_APPROVED", "TEMPLATE_UNRESOLVED",
    # p3 定死的槽级码(照抄,M1 只触发子集)
    "SLOT_UNFILLED", "SLOT_SOURCE_FORBIDDEN", "SLOT_SOURCE_NEEDS_ACK",
    "SLOT_STALE_CALIBRATION", "SLOT_MAGNITUDE_ABSURD", "SKILL_UNKNOWN",
    "SKILL_CAPABILITY_UNDECLARED", "PREDICATE_UNKNOWN",
    "CRITERION_NO_UNCERTAIN_ROUTE", "MATERIAL_NO_PROFILE",
    "EVIDENCE_NONE_UNACKED", "STAGE_COUNT_EXCEEDED",
)

#: plan 里那个结构化绑定块的键名。**只认这一个名字。**
#:
#: 两个可接受的写法(``template`` / ``spec_id``)就是两个真源,而只有一个会被
#: 读到 —— 另一个写法的人会以为自己写对了。所以下面那条 ``_ALIAS_HINT`` 是
#: 「认出来但拒绝」:说出正确的键名,而不是静默地当没看见。
BLOCK_KEY = "conduct"

#: 绑定块里指模板的那个键。
TEMPLATE_KEY = "template"

_ALIAS_KEYS = ("spec_id", "spec", "template_id", "conduct_template")

#: 绑定块写在哪儿(错误详情里逐字告诉人)。
_WHERE_HINT = (
    f'把绑定块写进 plan.notes 的 JSON:'
    f'{{"{BLOCK_KEY}": {{"{TEMPLATE_KEY}": "<模板 id>", "params": {{...}}}}}};'
    f'或写进 plan.phases[0].steps[0]["{BLOCK_KEY}"](同一个形状)。'
    f'模板 id 见 GET /api/conducts/templates。'
)


# ── 结果类型 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompileError:
    """一条编译错误。``code`` 必在 :data:`COMPILE_ERROR_CODES` 里。

    与 :class:`mast.conduct.validator.Finding` 同一条纪律:写错一个码当场抛,
    而不是让一个谁也没定义过的码流到前端,在那儿被当成「未知情况」静默忽略。
    """

    code: str
    stage_id: str = ""
    slot_name: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.code not in COMPILE_ERROR_CODES:
            raise ValueError(
                f"未知编译错误码 {self.code!r} —— 闭集见 COMPILE_ERROR_CODES")

    def __str__(self) -> str:
        where = "/".join(x for x in (self.stage_id, self.slot_name) if x)
        return f"[{self.code}] {where}: {self.detail}" if where else \
            f"[{self.code}] {self.detail}"


@dataclass(frozen=True)
class CompileResult:
    """一次编译的结论。

    ``ok=True`` 时 ``spec`` **是** ``TEMPLATES[spec_id]`` 同一个对象(不是拷贝、
    更不是合成品)—— 见模块 docstring 的四条理由。真正的可执行产物是
    ``(spec, params)`` 这一对。

    ``ok=False ⇒ spec is None``:**绝不部分编译**(p3 §3.7 逐字)。一个「大部分
    编译好了」的 spec 会被当成可以试着跑一下的东西,而缺的那个槽恰恰是数值。

    ## 三态,不是两态

    ``checks_skipped`` 与 :class:`mast.conduct.validator.ValidationReport` 的同名
    字段一个意思:**该跑的检查没跑全**(典型是没有技能注册表 ⇒ 规则③没跑)。
    它非空时 ``ok`` 也是 False —— 「没检查」不许长得像「检查通过」,这是本仓
    一天出现五次的那族错误。它没有被折进 ``warnings`` 一了百了,是因为折进去
    之后,「注册表读不到」与「此槽会被 approve 覆盖」在结构上就分不开了。
    """

    ok: bool
    spec: "ConductSpec | None"
    params: dict = field(default_factory=dict)
    errors: tuple = ()
    warnings: tuple = ()
    checks_skipped: tuple = ()
    #: 认出来的模板 id(认不出就是空串)。``spec is None`` 时前端也要说得出
    #: 「你指的是哪份模板」——否则一屏错误码没有落点。
    spec_id: str = ""
    spec_version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "checks_skipped", tuple(self.checks_skipped))
        object.__setattr__(self, "params", dict(self.params))
        for e in self.errors:
            if not isinstance(e, CompileError):
                raise TypeError("CompileResult.errors 只收 CompileError")
        if not self.ok and self.spec is not None:
            # 结构上堵死「部分编译」:一个 ok=False 却带着 spec 的结果,调用方
            # 一个 `if res.spec:` 就把它跑起来了。
            raise ValueError("ok=False 时 spec 必须是 None —— 不部分编译")

    def describe(self) -> str:
        lines = [str(e) for e in self.errors]
        lines += [f"[提示] {w}" for w in self.warnings]
        lines += [f"[未检查] {s}" for s in self.checks_skipped]
        return "\n".join(lines) if lines else "编译通过,无发现。"


# ── 校验器发现码 → 编译错误码 ────────────────────────────────────────
#
# p3 的闭集是照**槽体系**写的,而 M1 的校验器管的是**模板结构**。两张表对不齐
# 是事实,不是疏忽 —— 下面这份映射把对得上的接上,对不上的一律归到
# ``TEMPLATE_UNRESOLVED``(「这份模板在这台机器上立不住」),detail 原样带上
# 校验器那一条的全文。**不发明新码**:多一个码就多一条前端要认的分支,而
# 那一条今天没有任何设计文档定义过它该长什么样。

_FINDING_TO_CODE = {
    # 技能这一侧
    "unknown_skill": "SKILL_UNKNOWN",
    "skill_kind_mismatch": "SKILL_UNKNOWN",
    "step_param_unknown": "SKILL_UNKNOWN",
    "recovery_unknown_skill": "SKILL_UNKNOWN",
    # capability / SAFE 模式对账这一侧
    "dangerous_capability_undeclared": "SKILL_CAPABILITY_UNDECLARED",
    "detour_writes_before_retract": "SKILL_CAPABILITY_UNDECLARED",
    "recovery_step_writes": "SKILL_CAPABILITY_UNDECLARED",
    # 判据/分析函数这一侧
    "unknown_analysis_fn": "PREDICATE_UNKNOWN",
    "recovery_rule_field_not_produced": "PREDICATE_UNKNOWN",
    "evidence_field_not_produced": "PREDICATE_UNKNOWN",
}

#: 参数校验的发现码 → 编译错误码。
#:
#: ``param_unknown``(plan 写了一个模板没声明的键)映到 ``TEMPLATE_UNRESOLVED``
#: 而不是任何 ``SLOT_*``:那不是「某个槽有问题」,是**这个绑定块与这份模板对不
#: 上**——它指着的槽在这份模板里根本不存在。p3 的闭集里没有一格装得下它。
_PARAM_FINDING_TO_CODE = {
    "param_missing": "SLOT_UNFILLED",
    "param_out_of_envelope": "SLOT_MAGNITUDE_ABSURD",
    "param_not_in_choices": "SLOT_MAGNITUDE_ABSURD",
    "param_type": "SLOT_MAGNITUDE_ABSURD",
    "param_unknown": "TEMPLATE_UNRESOLVED",
}

#: 由 approve **实测冻结**的槽 —— 编译器对它出一条 warning,不出错误。
#:
#: 单一真源:``api/routes/conducts.py`` 的 ``_freeze_measured_params`` 按同一个
#: 名字去查。两处各写一个字面量就是「一侧改了,另一侧没跟上」的入口。
MEASURED_AT_APPROVE = ("coord_epoch",)

MEASURED_AT_APPROVE_HINT = (
    "此槽由 approve **实测冻结**(那一刻查一次当前坐标代次并写进 params),"
    "手填的值会被覆盖;查不到当前代次时 approve 会拒批 —— 读不到不等于 0。")


# ── 主入口 ────────────────────────────────────────────────────────────

def compile_plan_to_conduct_spec(plan_id: str, *, plan_store=None,
                                 registry=None) -> CompileResult:
    """把一份 **APPROVED** 的 plan 编译成 ``(注册模板, params)``。

    ``plan_store`` 必须显式传(:class:`mast.planning.plan_store.PlanStore` 或
    形状相同的替身)。``registry`` 是技能注册表;不传 ⇒ ``validate_spec`` 的
    规则③跑不了 ⇒ ``checks_skipped`` 非空 ⇒ ``ok=False``。

    **纯读**:不写 plan 库,不写 conduct 库,不取仪器令牌。
    """
    if plan_store is None:
        # 与 ConductStore 需要显式 db_path 同一条纪律。一个「不传就用全局实验库」
        # 的默认值,是测试污染真实数据那五次事故的共同入口。
        raise ValueError(
            "compile_plan_to_conduct_spec 需要显式 plan_store —— 这里没有默认库。"
            "API 传 live runtime 的 _plan_store,测试传 tmp_path 建的那一个。")

    plan = plan_store.load(plan_id)   # 读库失败**抛**,不折叠成 PLAN_NOT_FOUND
    if plan is None:
        return _failed([CompileError(
            "PLAN_NOT_FOUND", detail=f"plan 库里没有 {plan_id!r}")])

    status = str(getattr(plan.status, "value", plan.status) or "")
    if status != "approved":
        return _failed([CompileError(
            "PLAN_NOT_APPROVED",
            detail=f"plan {plan_id!r} 现在是 {status or '(空)'} —— "
                   f"只有 APPROVED 的方案能编译。"
                   f"批准是**人**的动作,不是编译器能替它做的一步")])

    block, where = _binding_block(plan)
    if block is None:
        return _failed([CompileError(
            "TEMPLATE_UNRESOLVED",
            detail=f"plan {plan_id!r} 里没有 conduct 绑定块。{_WHERE_HINT}")])

    name = str(block.get(TEMPLATE_KEY) or "").strip()
    if not name:
        alias = next((k for k in _ALIAS_KEYS if block.get(k)), "")
        extra = (f"(这个块里有 {alias!r},但键名只认 {TEMPLATE_KEY!r} —— "
                 f"两个键名就是两个真源,只有一个会被读到)" if alias else "")
        return _failed([CompileError(
            "TEMPLATE_UNRESOLVED",
            detail=f"{where} 的绑定块没有 {TEMPLATE_KEY!r}{extra}。{_WHERE_HINT}")])

    try:
        spec = get_template(name)
    except KeyError as exc:
        return _failed([CompileError(
            "TEMPLATE_UNRESOLVED", detail=f"{where}: {exc}")])

    raw = block.get("params")
    if raw is not None and not isinstance(raw, Mapping):
        return _failed([CompileError(
            "TEMPLATE_UNRESOLVED",
            detail=f"{where} 的 params 是 {type(raw).__name__},"
                   f"要一个对象(键=模板声明的参数名)。{_WHERE_HINT}")],
            spec_id=spec.spec_id, spec_version=int(spec.spec_version))
    params = dict(raw or {})

    errors: list[CompileError] = []
    warnings: list[str] = []

    # ── 参数逐项:超包络**拒绝,不夹紧**(与 create 路由同款语义)───────
    for f in check_params(spec, params):
        code = _PARAM_FINDING_TO_CODE.get(f.code)
        if code is None:                       # 校验器加了新码而这里没跟上
            code = "TEMPLATE_UNRESOLVED"
        errors.append(CompileError(code, slot_name=str(f.where),
                                   detail=f.message))

    # ── 整份模板:与 approve 走**同一个** validate_spec 调用 ─────────────
    skills = None
    if registry is not None:
        try:
            skills = skill_index(registry)
        except Exception as exc:  # noqa: BLE001
            # 注册表读不到 ⇒ skills 留 None ⇒ 规则③进 checks_skipped ⇒ ok=False。
            # **不当成「没有技能」**:那会把整份模板判成引用了不存在的技能。
            logger.warning("技能注册表不可读(规则③将报「没检查」): %s", exc)
            warnings.append(f"技能注册表不可读: {exc}")
    analyses = None
    try:
        from mast.conduct.analyses import known_names

        analyses = sorted(known_names())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"analysis 注册表不可读: {exc}")

    report = validate_spec(spec, skills=skills, analyses=analyses)
    for f in report.errors:
        errors.append(CompileError(
            _FINDING_TO_CODE.get(f.code, "TEMPLATE_UNRESOLVED"),
            stage_id=str(f.where).split("/")[0] if "/" in str(f.where) else "",
            detail=str(f)))
    warnings.extend(str(f) for f in report.warnings)

    # ── approve 会覆盖的槽:出提示,不出错误 ────────────────────────────
    declared = {p.name for p in spec.params_schema}
    for slot in MEASURED_AT_APPROVE:
        if slot in declared:
            warnings.append(f"{slot}: {MEASURED_AT_APPROVE_HINT}")

    skipped = tuple(report.checks_skipped)
    ok = not errors and not skipped
    return CompileResult(
        ok=ok, spec=spec if ok else None, params=params,
        errors=tuple(errors), warnings=tuple(warnings), checks_skipped=skipped,
        spec_id=spec.spec_id, spec_version=int(spec.spec_version))


# ── 内部 ──────────────────────────────────────────────────────────────

def _failed(errors, *, spec_id: str = "", spec_version: int = 0) -> CompileResult:
    return CompileResult(ok=False, spec=None, params={}, errors=tuple(errors),
                         spec_id=spec_id, spec_version=spec_version)


def _binding_block(plan) -> "tuple[dict | None, str]":
    """从 plan 里取那个结构化绑定块。返回 ``(块, 它在哪儿)``。

    约定的优先级(两处都看,先 notes):

    1. ``plan.notes`` 整体是一段 JSON,里面有 ``"conduct"`` 键;
    2. ``plan.phases[0].steps[0]["conduct"]``。

    ``PlanPhase.steps`` 是**自由 dict**(p3 陷阱 T8),``notes`` 是**自由文本**
    —— 两边都没有 schema 保护,所以这里做的每一步都要能答「读不到怎么办」:
    notes 解不开 JSON **不是错误**(它本来就常常是给人看的一段话),往下看
    phases;两处都没有才报 ``TEMPLATE_UNRESOLVED``。
    """
    notes = getattr(plan, "notes", "") or ""
    if notes.strip().startswith("{"):
        try:
            doc = json.loads(notes)
        except (TypeError, ValueError):
            doc = None
        if isinstance(doc, Mapping):
            blk = doc.get(BLOCK_KEY)
            if isinstance(blk, Mapping):
                return dict(blk), "plan.notes.conduct"

    for pi, phase in enumerate(getattr(plan, "phases", ()) or ()):
        for si, step in enumerate(getattr(phase, "steps", ()) or ()):
            if not isinstance(step, Mapping):
                continue
            blk = step.get(BLOCK_KEY)
            if isinstance(blk, Mapping):
                return dict(blk), f"plan.phases[{pi}].steps[{si}].{BLOCK_KEY}"
    return None, ""


def resolved_template(result: CompileResult) -> "ConductSpec | None":
    """``result`` 认出来的那份**注册**模板(不管 ok 与否)。

    存在的理由是给调用方一条**不经 result.spec** 的路:``spec`` 在 ok=False 时
    刻意是 None(不部分编译),但面板仍然要说得出「你指的是哪份模板、它要填哪
    几个参数」——否则一屏错误码没有落点。
    """
    if not result.spec_id:
        return None
    return TEMPLATES.get(result.spec_id)


__all__ = [
    "COMPILE_ERROR_CODES", "BLOCK_KEY", "TEMPLATE_KEY",
    "MEASURED_AT_APPROVE", "MEASURED_AT_APPROVE_HINT",
    "CompileError", "CompileResult",
    "compile_plan_to_conduct_spec", "resolved_template",
]
