"""conduct 校验器 —— approve 之前跑的那道 lint。

设计:``campaign_director_design.md`` §4.3(三条结构规则)+ §5(绕道首步)+
``stm_capability_vs_sample_layer.md`` §6(模板零逻辑)。

## 为什么校验要在 approve 之前,而不是运行时

一份 spec 的毛病,运行时暴露出来的样子是「凌晨三点某一步取不到参数」。
那时针在表面上、样品在低温里、人在睡觉。**所有能在跑之前问出来的问题,
都必须在跑之前问。**

## 三态,不是两态

:class:`ValidationReport` 有三个属性,刻意分开:

* ``ok`` —— 没发现错误;
* ``complete`` —— 该跑的检查**全跑了**(有些检查需要技能注册表,离线时跑不了);
* ``approvable`` —— 上面两个都成立。

approve 路径只许看 ``approvable``。只看 ``ok`` 的话,一次「没检查」会长得和
一次「检查通过」一模一样 —— 这正是本仓一天出现五次的那族错误。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from mast.conduct.spec import ConductSpec, RuleLeaf, StageSpec, StepSpec

# ── 发现码(闭集)──────────────────────────────────────────────────────

FINDING_CODES = (
    # §4.3 ①
    "wait_without_retract_confirm",
    # §4.3 ②
    "binding_unparseable",
    "binding_forward_reference",
    "binding_not_produced",
    # §4.3 ③
    "dangerous_capability_undeclared",
    # 注册表相关
    "unknown_skill",
    "skill_kind_mismatch",
    "step_param_unknown",
    # §5:绕道首步
    "detour_target_missing",
    # 2026-08-15:``detour_entry_not_retract_confirm`` **退役**,拆成下面两条。
    # 旧码要求绕道目标的**第 0 步**就是退针技能,于是把只读诊断(跨点复测)也
    # 一起拦掉了 —— 而那一步正是「怀疑与处置分开计价」的前提:先只读复查确认
    # 是针坏而不是表面本来就长这样,再付换样品修针的钱。新判据只拦会**改变
    # 表面**的动作(脉冲/扎针/除层/DANGEROUS)。**不留两套码。**
    "detour_writes_before_retract",
    "detour_without_retract",
    "detour_only_stage_unreachable",
    "detour_route_without_target",
    # §8:恢复自检 A3 的复验步(M4-a)。复验步不在任何阶段里 ⇒ 规则②③都够不着
    # 它们,这几条是它们唯一的结构闸。
    "recovery_unknown_skill",
    "recovery_step_writes",
    "recovery_binding_unknown_param",
    "recovery_binding_bad_ref",
    "recovery_binding_not_upstream",
    "recovery_binding_not_produced",
    "recovery_rule_field_not_produced",
    # 规则⑤:证据投影点名了一个那一步不产出的字段(2026-08-15)。运行时它的
    # 样子是「证据缺席 → 转人」—— 一句指向仪器的话,而根因在模板里。
    "evidence_field_not_produced",
    # analysis 步
    "unknown_analysis_fn",
    # 模板层
    "template_logic",
    # 参数包络
    "param_unknown",
    "param_missing",
    "param_type",
    "param_out_of_envelope",
    "param_not_in_choices",
)

#: **确认式**退针技能 —— 「发出即返回」的不算。
#:
#: 判据的实现在 ``core/tip_park.py``:三态 ``parked``/``not_parked``/
#: 读不到,而且「读不到」再分「值得重试」与「本机没声明,等多久都没用」。
#: 这三个技能都会等到确认(或如实说「已下发未确认」)才返回。
#:
#: **往这里加名字之前先问一句**:那个技能是验证了动作生效,还是只验证了命令
#: 发出去了?SafeRetract 在 2026-08-14 之前正是后者 —— 发一条
#: ``ZCtrl_Withdraw`` 就报 ``retracted: True``。名单里混进一个那样的,
#: 「等人时针必已退」这条结构保证就名存实亡。
RETRACT_CONFIRM_SKILLS = frozenset({
    "SafeRetract", "WithdrawTip", "RetractForSampleChange",
})

#: 模板里不许出现的 AST 节点 —— 它们是「判据逻辑」的形状。
#:
#: 允许的是**组合与参数引用**:字面值、参数插值、列表/字典装配、无条件推导式、
#: 单位换算这类算术。不允许分支、循环控制流、比较、布尔组合 ——
#: 判据逻辑一旦进模板,第二个样品就要抄一份(``nth_copy_of_one_action``)。
_TEMPLATE_FORBIDDEN = {
    ast.If: "分支(if)",
    ast.IfExp: "三元表达式",
    ast.For: "for 循环",
    ast.AsyncFor: "for 循环",
    ast.While: "while 循环",
    ast.Try: "try/except",
    ast.TryStar: "try/except*",
    ast.With: "with",
    ast.AsyncWith: "with",
    ast.Match: "match",
    ast.Compare: "比较(阈值判据)",
    ast.BoolOp: "布尔组合(and/or)",
    ast.Assert: "assert",
    ast.Raise: "raise",
}


@dataclass(frozen=True)
class Finding:
    """一条发现。``code`` 在 :data:`FINDING_CODES` 里。"""

    code: str
    where: str
    message: str
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.code not in FINDING_CODES:
            raise ValueError(f"未知发现码 {self.code!r} —— 闭集见 FINDING_CODES")
        if self.severity not in ("error", "warning"):
            raise ValueError("severity 只能是 error/warning")

    def __str__(self) -> str:
        return f"[{self.code}] {self.where}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    findings: tuple = ()
    #: 没能跑的检查 + 为什么。**不是空的就说明这次校验不完整。**
    checks_skipped: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "checks_skipped", tuple(self.checks_skipped))

    @property
    def errors(self) -> tuple:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple:
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def ok(self) -> bool:
        """没发现错误。**注意这不等于「检查过了」** —— 见 :attr:`complete`。"""
        return not self.errors

    @property
    def complete(self) -> bool:
        """该跑的检查都跑了。"""
        return not self.checks_skipped

    @property
    def approvable(self) -> bool:
        """approve 路径唯一该看的那个属性。"""
        return self.ok and self.complete

    def describe(self) -> str:
        lines = [str(f) for f in self.findings]
        lines += [f"[未检查] {s}" for s in self.checks_skipped]
        return "\n".join(lines) if lines else "校验通过,无发现。"


# ── 技能注册表的形状 ──────────────────────────────────────────────────

def skill_index(registry) -> "dict[str, Any]":
    """``{skill_name: SkillMetadata}``,从 ``SkillRegistry`` 取。

    ``list_skills()`` 回的是 ``SkillMetadata`` **对象**不是名字 —— 把它
    ``str()`` 成 ``"SkillMetadata(name='ScanAt', …)"`` 的那个 bug 在真机
    上让 19 个存在的技能被报成缺失(``core/registry.py`` 里有完整记述)。
    这里逐个读 ``.name``,读不出来的**丢弃而不是强转**。
    """
    out: dict[str, Any] = {}
    if registry is None:
        return out
    try:
        entries = registry.list_skills()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"技能注册表不可读: {exc}") from exc
    for meta in entries:
        name = getattr(meta, "name", None)
        if isinstance(name, str) and name:
            out[name] = meta
    return out


# ── 主校验 ────────────────────────────────────────────────────────────

def validate_spec(spec: ConductSpec, *,
                  skills: "Mapping[str, Any] | None" = None,
                  analyses: "Iterable[str] | None" = None) -> ValidationReport:
    """跑完整套模板 lint。

    ``skills`` 是 ``{name: SkillMetadata}``(见 :func:`skill_index`);
    ``analyses`` 是 ``mast.conduct.analyses`` 注册表里的纯函数名。

    **不给就跑不了对应的检查**,它们会进 ``checks_skipped`` —— 离线跑得出
    ``ok=True``,但 ``approvable`` 是 False。approve 路径必须两样都递进来,
    否则「没检查」会长得和「检查通过」一模一样。
    """
    findings: list[Finding] = []
    skipped: list[str] = []

    order = _execution_order(spec)
    produced_by = _produces_index(order)
    param_names = frozenset(p.name for p in spec.params_schema)

    _check_bindings(order, produced_by, param_names, findings)
    _check_condition_refs(spec, findings)
    _check_wait_predecessors(spec, findings)
    _check_detour_target(spec, findings)
    _check_detour_only_stages(spec, findings)
    _check_recovery_bindings(spec, param_names, findings)
    _check_evidence_projection(spec, produced_by, findings)

    if skills is None:
        skipped.append(
            "技能存在性与 DANGEROUS/capability 对账(规则③)—— 没有传 skills;"
            "approve 路径必须传 skill_index(registry)")
        skipped.append(
            "绕道入口「退针之前不得有改变表面的动作」(§5b)—— 判据要读技能的"
            "capabilities/safety_level,没有 skills 就跑不了。"
            "**没检查不等于检查通过**")
        skipped.append(
            "恢复自检 A3 的复验步「只许只读」—— 同一个判据,同样要 skills")
    else:
        _check_skills(spec, skills, findings)
        _check_detour_entry_writes(spec, skills, findings)
        _check_recovery_steps(spec, skills, findings)

    if analyses is None:
        skipped.append(
            "analysis 步的函数存在性 —— 没有传 analyses;"
            "approve 路径必须传 mast.conduct.analyses 的注册表键")
    else:
        _check_analyses(spec, frozenset(analyses), findings)

    return ValidationReport(findings=tuple(findings), checks_skipped=tuple(skipped))


def _execution_order(spec: ConductSpec) -> list[tuple[StageSpec, StepSpec, int]]:
    """(stage, step, 全局序号),按 Director 的执行顺序。"""
    out: list[tuple[StageSpec, StepSpec, int]] = []
    for stage in spec.stages:
        for step in stage.all_steps:
            out.append((stage, step, len(out)))
    return out


def _produces_index(order) -> "dict[str, tuple[str, int]]":
    """``{"steps.<step_id>.<field>": (step_id, 全局序号)}``。"""
    idx: dict[str, tuple[str, int]] = {}
    for _stage, step, pos in order:
        for field_name in step.produces:
            idx[f"steps.{step.step_id}.{field_name}"] = (step.step_id, pos)
    return idx


def _check_bindings(order, produced_by, param_names, findings: list[Finding]) -> None:
    """规则②:bindings 必须命中**上游**步的 produces 白名单。

    解析失败 = 步失败,**没有默认值兜底**(§4.3)。所以这里连「引用写得对不对」
    都要在跑之前问清楚:一个拼错的引用,运行时的样子是凌晨三点某一步取不到参数。

    两个引用命名空间:

    * ``steps.<step_id>.<field>`` —— 上游步的产出(白名单 = 那一步的 produces)。
      step_id 自己带点(``S2.03_scan``),所以**不能**按点切分 ——
      按已知的 produces 全键匹配。
    * ``params.<name>`` —— 用户在 approve 时冻结的 conduct 参数
      (白名单 = ``params_schema``)。没有这条,``params_schema`` 就是个渲染
      表单用的摆设:填了的数一步也到不了 —— 「已经记下来了」是生产方的话,
      要问的是谁读它。
    """
    for _stage, step, pos in order:
        for param, ref in step.bindings.items():
            where = f"{step.step_id}.bindings.{param}"
            if ref.startswith("params."):
                name = ref[len("params."):]
                if name not in param_names:
                    findings.append(Finding(
                        "binding_not_produced", where,
                        f"引用 {ref!r} 指向一个 params_schema 里没有的参数;"
                        f"已声明的是 {sorted(param_names)}"))
                continue
            if not ref.startswith("steps."):
                findings.append(Finding(
                    "binding_unparseable", where,
                    f"引用 {ref!r} 既不是 steps.<step_id>.<produces 里的字段>,"
                    f"也不是 params.<params_schema 里的名字>"))
                continue
            hit = produced_by.get(ref)
            if hit is None:
                findings.append(Finding(
                    "binding_not_produced", where,
                    f"引用 {ref!r} 没有任何一步声明产出它 —— "
                    f"produces 是白名单,不在名单上就取不到,而且不会有默认值兜底"))
                continue
            src_step, src_pos = hit
            if src_pos >= pos:
                findings.append(Finding(
                    "binding_forward_reference", where,
                    f"引用 {ref!r} 来自 {src_step}(第 {src_pos} 步),"
                    f"不在本步(第 {pos} 步)上游 —— 运行到这里时它还没产出"))



def _check_condition_refs(spec: ConductSpec, findings: list[Finding]) -> None:
    """等待条件的数绑到参数上时(``ConditionSpec.*_ref``),名字与**类型**都要对。

    与 ``bindings`` 同一条白名单、同一条纪律。少了这道检查,一个拼错的阈值引用
    要等到凌晨三点进等待步那一刻才现形 —— 而那时它的样子是「这一步失败了」,
    不是「模板写错了」。类型也要查:一个 str 参数绑到温度阈值上,解析时会抛,
    而那同样是在半夜。
    """
    by_name = {p.name: p for p in spec.params_schema}
    for stage in spec.stages:
        for step in stage.all_steps:
            if step.kind != "wait" or step.wait is None:
                continue
            cond = step.wait.condition
            if cond is None:
                continue
            for target, ref in cond.refs.items():
                where = f"{step.step_id}.wait.condition.{target}"
                name = ref[len("params."):]   # 前缀由 ConditionSpec 自己保证
                p = by_name.get(name)
                if p is None:
                    findings.append(Finding(
                        "binding_not_produced", where,
                        f"等待条件的 {target} 引用 {ref!r},而 params_schema 里"
                        f"没有它;已声明的是 {sorted(by_name)}"))
                    continue
                if p.type not in ("float", "int"):
                    findings.append(Finding(
                        "binding_unparseable", where,
                        f"等待条件的 {target} 引用 {ref!r},而那个参数声明的类型是 "
                        f"{p.type!r} —— 阈值/时窗必须是数值"))


def _check_wait_predecessors(spec: ConductSpec, findings: list[Finding]) -> None:
    """规则①:wait 步的前驱链里必须有确认式退针步。

    **等人 = 针必已退**,靠结构保证不靠约定。等待可能是几小时:人会去换样品、
    会碰机器、会开腔。针留在隧道结上等几小时,是把一根修了两周的针交给运气。

    检查范围是**同一阶段内**(entry_actions + 本步之前的 steps)。不跨阶段,
    因为设计 §3-2 已经立了规矩:不信任跨阶段的仪器状态假设,每个阶段入口要重申
    全部前置设置。所以「上一个阶段末尾退过针」在这里不算数 —— 想让它算,就把
    退针步显式写进本阶段。
    """
    for stage in spec.stages:
        steps = stage.all_steps
        for i, step in enumerate(steps):
            if step.kind != "wait":
                continue
            before = steps[:i]
            if any(s.skill in RETRACT_CONFIRM_SKILLS for s in before):
                continue
            findings.append(Finding(
                "wait_without_retract_confirm", f"{stage.stage_id}/{step.step_id}",
                "等待步之前(本阶段内)没有确认式退针步。等人可能是几小时,"
                "针不能留在隧道结上。请在本阶段内加一步 "
                f"{sorted(RETRACT_CONFIRM_SKILLS)} 之一"))


def _all_gates(stage) -> "list":
    """这个阶段的全部闸门:入口、**每一步自己的**、出口。

    步级闸门(``StepSpec.gate``)漏进这个列表的后果是静默的:一个 ``detour``
    去向躲过「无处可去」的检查,跑到那一步才发现绕道没有目标。
    """
    out = [g for g in (stage.entry_gate,) if g is not None]
    out.extend(s.gate for s in stage.all_steps if s.gate is not None)
    if stage.exit_gate is not None:
        out.append(stage.exit_gate)
    return out


def _detour_routes(spec: ConductSpec) -> list[str]:
    """所有指向 ``detour`` 的去向(闸门路由 + 阶段失败策略)。"""
    out: list[str] = []
    for stage in spec.stages:
        if stage.on_fail.then == "detour":
            out.append(f"{stage.stage_id}.on_fail")
        for gate in _all_gates(stage):
            if gate is None:
                continue
            for name, outcome in gate.routes.items():
                if outcome.verdict == "detour":
                    out.append(f"{stage.stage_id}/{gate.gate_id}.routes.{name}")
            if gate.unattended_escape == "detour":
                out.append(f"{stage.stage_id}/{gate.gate_id}.unattended_escape")
    return out


def _check_detour_target(spec: ConductSpec, findings: list[Finding]) -> None:
    """§5(a):绕道要有地方可去,而且那个地方要存在。

    「第一个动作必须是把针拿开」那一半改判据了,见
    :func:`_check_detour_entry_writes` —— 它需要技能元数据,所以离线跑不了,
    单独成一条检查。

    这里只管两件离线就能答的事:**没有修针段却有指向绕道的去向**
    (触发得了、无处可去),以及**目标阶段根本不在 stages 里**。
    """
    if not str(spec.detour.target_stage).strip():
        for where in _detour_routes(spec):
            findings.append(Finding(
                "detour_route_without_target", where,
                "这条去向是 detour,但本 spec 没有修针阶段"
                "(detour.target_stage 为空)。没有可去之处的绕道就是死路 —— "
                "请改成 wait_operator,让人来修针"))
        return

    target = spec.stage(spec.detour.target_stage)
    if target is None:
        findings.append(Finding(
            "detour_target_missing", "detour.target_stage",
            f"绕道目标阶段 {spec.detour.target_stage!r} 不在 stages 里"))
        return
    if not any(s.skill in RETRACT_CONFIRM_SKILLS for s in target.all_steps):
        findings.append(Finding(
            "detour_without_retract", target.stage_id,
            f"绕道目标阶段 {target.stage_id!r} 里一步确认式退针都没有。"
            f"绕道的触发条件就是「针可能坏了」,这一段迟早要把针拿开 —— "
            f"请加一步 {sorted(RETRACT_CONFIRM_SKILLS)} 之一"))


#: 会**改变表面**的能力。绕道进来之后、把针拿开之前,一条都不许出现。
#:
#: 判据从「第一步必须是退针技能」改成这个,是因为前者把**只读诊断**也一起拦掉了
#: (跨点复测要 MoveToXY + 扫图),而那一步恰恰是让修针段能待在流程里的前提:
#: 先花 15 分钟只读复查,确认是针坏而不是表面本来就长这样,再付换样品修针的钱。
#:
#: §5 那条规则真正防的是「别拿一根可能坏了的针**继续往表面上开**」——
#: 开的是脉冲、扎针、除层,不是看一眼。复测那一侧另有两道选点闸兜着
#: (``crash_count == 0``、避开已知脏点)。
SURFACE_CHANGING_CAPABILITIES = frozenset({
    "tip_shaping", "bias_pulse", "layer_removal",
})


def _check_detour_entry_writes(spec: ConductSpec, skills: "Mapping[str, Any]",
                               findings: list[Finding]) -> None:
    """§5(b):绕道目标阶段里,**确认式退针之前不得有会改变表面的动作**。

    绕道的三个触发源之一是 ``recovery_tip_fail`` —— 崩溃重启之后判针坏。
    那条路上第一个动作若是打脉冲/扎针,就是拿一根刚崩过的针再往表面上开一次。
    只读诊断(扫图、移动、正反扫复核)放行:它回答的是「到底是不是针坏了」,
    而那个问题不问清楚,每一次表面异常都要付一轮换样品修针的钱。
    """
    target_id = str(spec.detour.target_stage).strip()
    if not target_id:
        return
    target = spec.stage(target_id)
    if target is None:
        return                      # 目标不存在已由 _check_detour_target 报过
    for step in target.all_steps:
        if step.skill in RETRACT_CONFIRM_SKILLS:
            return                  # 退针到了,后面随便写
        if not step.touches_hardware:
            continue
        meta = skills.get(step.skill)
        if meta is None:
            continue                # 技能不存在已由 _check_skills 报过
        caps = frozenset(getattr(meta, "capabilities", frozenset()) or ())
        level = str(getattr(getattr(meta, "safety_level", None), "value", "")).lower()
        offending = caps & SURFACE_CHANGING_CAPABILITIES
        if offending or level == "dangerous":
            why = (f"声明了 {sorted(offending)}" if offending
                   else "是 DANGEROUS 级")
            findings.append(Finding(
                "detour_writes_before_retract",
                f"{target.stage_id}/{step.step_id}",
                f"绕道进来之后、把针拿开之前就跑 {step.skill!r}(它{why})。"
                f"绕道的触发条件是「针可能坏了」,其中一条还是崩溃重启后的判坏 —— "
                f"在确认式退针之前只允许**只读诊断**,不允许任何会改变表面的动作"))


def _check_evidence_projection(spec: ConductSpec,
                               produced_by: "Mapping[str, Any]",
                               findings: list[Finding]) -> None:
    """规则⑤:``EvidenceSpec.fields`` 点名的字段必须真的被那一步产出。

    **这条要在 approve 时拦下来**,而不是等运行时。运行时它的样子是「证据缺席 →
    转人」—— 一句指向仪器的话,而根因在模板里的一个拼错的字段名。凌晨三点的
    用户会照着那句话去查机器。

    ⚠️ 只查 ``step_data``:别的源(温度、监控、帧指标)的字段名不由 spec 的
    ``produces`` 决定,拿这张表去查会把「这个源本来就有这个字段」判成错的。
    """
    for stage in spec.stages:
        for gate in _all_gates(stage):
            for ev in gate.evidence:
                if ev.source != "step_data" or not ev.fields:
                    continue
                for name in ev.fields:
                    if f"steps.{ev.selector}.{name}" in produced_by:
                        continue
                    findings.append(Finding(
                        "evidence_field_not_produced",
                        f"{stage.stage_id}/{gate.gate_id}",
                        f"闸门点名要 {ev.selector} 的 {name!r},而那一步没有声明"
                        f"产出它 —— 运行时这会以「证据缺席 → 转人」的形态出现,"
                        f"而那句话指向仪器,根因却在这份模板里"))


def _check_recovery_steps(spec: ConductSpec, skills: "Mapping[str, Any]",
                          findings: list[Finding]) -> None:
    """恢复自检 A3 的复验步:**一个会改变表面的动作都不许有**。

    与 §5b 同一个判据、同一个理由,而这里的处境更极端:A3 跑在**进程刚死过一次**
    之后,没有人在场,而且它跑的时候连「针当时是退着还是扎着」都还没确认过。
    一个 DANGEROUS 复验步会让「自检」变成「无人值守地往表面上开一发」。

    另一条:复验步绕开了 pre-flight 的 SAFE 模式对账(那条查的是**下一个流程步**
    落在哪个阶段、阶段声明了哪些 capability;复验步不在任何阶段里,那条查不到
    它)。所以这道闸是它唯一的对账口 —— 少了它,SAFE 模式会在恢复路径上形同
    虚设,而这正是本仓「不填 capability → SAFE 形同虚设」那笔账的第二次。
    """
    for step in spec.recovery.tip_check:
        if not step.touches_hardware:
            continue
        meta = skills.get(step.skill)
        if meta is None:
            findings.append(Finding(
                "recovery_unknown_skill", f"recovery/{step.step_id}",
                f"恢复复验步引用了注册表里没有的技能 {step.skill!r}"))
            continue
        caps = frozenset(getattr(meta, "capabilities", frozenset()) or ())
        level = str(getattr(getattr(meta, "safety_level", None), "value", "")).lower()
        offending = caps & SURFACE_CHANGING_CAPABILITIES
        if offending or level == "dangerous":
            why = (f"声明了 {sorted(offending)}" if offending else "是 DANGEROUS 级")
            findings.append(Finding(
                "recovery_step_writes", f"recovery/{step.step_id}",
                f"恢复自检的针尖复验里跑 {step.skill!r}(它{why})。A3 跑在进程"
                f"刚死过一次之后、没有人在场、针的状态还没确认 —— 复验只许只读。"
                f"而且复验步不在任何阶段里,SAFE 模式的 capability 对账查不到它,"
                f"这道闸是唯一的对账口"))


def _check_recovery_bindings(spec: ConductSpec, param_names: "frozenset[str]",
                             findings: list[Finding]) -> None:
    """复验步的绑定只能指向**兄弟复验步**或 ``params.``。

    指向流程步的产出是一句假话:那些产出是重启**之前**采的,而恢复自检存在的
    理由就是「重启之前的一切都要重新去问」。规则②(bindings 必须命中上游
    produces)在这里管不着 —— 复验步不在 ``_execution_order`` 里。
    """
    known: set = set()
    for step in spec.recovery.tip_check:
        for name, ref in step.bindings.items():
            where = f"recovery/{step.step_id}"
            if ref.startswith("params."):
                key = ref[len("params."):]
                if key not in param_names:
                    findings.append(Finding(
                        "recovery_binding_unknown_param", where,
                        f"绑定 {name}←{ref} 指向一个没有声明的参数"))
                continue
            if not ref.startswith("steps."):
                findings.append(Finding(
                    "recovery_binding_bad_ref", where,
                    f"绑定 {name}←{ref} 既不是 params. 也不是 steps."))
                continue
            body = ref[len("steps."):]
            hit = next((p for p in known if body.startswith(p + ".")), None)
            if hit is None:
                findings.append(Finding(
                    "recovery_binding_not_upstream", where,
                    f"绑定 {name}←{ref} 没有命中**前面某个复验步**的 produces。"
                    f"复验步只许绑兄弟复验步或 params. —— 绑流程步等于拿重启"
                    f"之前采的数当证据,而那正是自检要作废的东西"))
                continue
            field = body[len(hit) + 1:]
            producer = next(s for s in spec.recovery.tip_check if s.step_id == hit)
            if field not in producer.produces:
                findings.append(Finding(
                    "recovery_binding_not_produced", where,
                    f"绑定 {name}←{ref}:{hit} 没有声明产出 {field!r}"))
        known.add(step.step_id)
    rule = spec.recovery.tip_rule
    if rule is not None:
        declared = frozenset(spec.recovery.produces)
        for leaf in _rule_leaves(rule):
            root = leaf.field.split(".")[0]
            if root not in declared:
                findings.append(Finding(
                    "recovery_rule_field_not_produced", "recovery/tip_rule",
                    f"判据读 {leaf.field!r},而复验步一个都没声明产出它"
                    f"(声明了 {sorted(declared)})—— 读不到的字段会让判据永远"
                    f"判不了,而「永远判不了」在面板上长得像「一直没跑」"))


def _rule_leaves(rule):
    """判据树上的所有叶子。"""
    if isinstance(rule, RuleLeaf):
        yield rule
        return
    for child in getattr(rule, "children", ()) or ():
        yield from _rule_leaves(child)


def _check_detour_only_stages(spec: ConductSpec, findings: list[Finding]) -> None:
    """声明成「只有绕道进得来」的阶段,必须真的是绕道目标 —— 否则它谁也进不去。

    正常推进会跳过它(``director._advance_stage``),而绕道只进
    ``detour.target_stage`` 那一个。两者对不上 = 一段永远跑不到的阶段,
    而它在模板里看起来一切正常。
    """
    target_id = str(spec.detour.target_stage).strip()
    for stage in spec.stages:
        if not getattr(stage, "entered_only_by_detour", False):
            continue
        if stage.stage_id != target_id:
            findings.append(Finding(
                "detour_only_stage_unreachable", stage.stage_id,
                f"阶段 {stage.stage_id!r} 声明了 entered_only_by_detour,"
                f"但绕道目标是 {target_id or '(空)'!r} —— 正常流程跳过它、"
                f"绕道又不进它,这一段永远跑不到"))


def _check_skills(spec: ConductSpec, skills: "Mapping[str, Any]",
                  findings: list[Finding]) -> None:
    """规则③ + 技能存在性。

    规则③(SAFE 模式对账):DANGEROUS 步必须落在**声明了对应 capability** 的
    阶段。``StageSpec.capabilities`` 不填 = 一条都不允许 —— 一个不填 capability
    就形同虚设的 SAFE 模式,本仓已经有过一次。
    """
    for stage in spec.stages:
        for step in stage.all_steps:
            if not step.touches_hardware:
                continue
            where = f"{stage.stage_id}/{step.step_id}"
            meta = skills.get(step.skill)
            if meta is None:
                findings.append(Finding(
                    "unknown_skill", where,
                    f"技能 {step.skill!r} 不在注册表里 —— 这台机器上跑不了它"))
                continue
            # ── 规则④:步里写的参数名必须是这个技能真的收的那些 ──────────
            #
            # 2026-08-14 加。起因是模板把 ``SpectroscopyAtPositions`` 的入参写成
            # ``positions_json`` / ``condition_group``,而技能收的是 ``positions``
            # 与一组显式数值 —— 名字对不上,而**校验器当时看不见**:规则②只管
            # binding 的**右**边(引用的产出存不存在),从不问**左**边那个键这个
            # 技能认不认识。于是一份「校验通过」的 spec 会在跑到那一步时才炸,
            # 或者更坏:技能按缺省值跑完,而用户以为自己设的值生效了。
            #
            # 这是「参数名写错」与「参数没生效」长得一样的那一族 —— 一次比较
            # 就能消掉,所以在这里比。
            declared = {getattr(p, "name", "") for p in
                        (getattr(meta, "parameters", ()) or ())}
            if declared:
                used = set(step.params) | set(step.bindings)
                for key in sorted(used - declared):
                    findings.append(Finding(
                        "step_param_unknown", where,
                        f"技能 {step.skill!r} 没有名叫 {key!r} 的参数 —— "
                        f"它收的是 {sorted(declared)}。"
                        f"名字对不上时,这一步要么当场失败,要么按缺省值跑完"
                        f"而没人知道你设的值没进去"))

            level = getattr(getattr(meta, "safety_level", None), "value", "")
            if str(level).lower() != "dangerous":
                continue
            needed = frozenset(getattr(meta, "capabilities", frozenset()) or ())
            missing = needed - stage.capabilities
            if missing or not needed:
                # 两种都拦:①技能声明了 capability 而阶段没声明;②技能是
                # DANGEROUS 却一个 capability 都没声明 —— 后者是
                # `safe_mode_tip_override` 记过的那个洞:SAFE 模式按 capability
                # 过滤,不填就等于谁都拦不住。
                detail = (f"阶段未声明 {sorted(missing)}"
                          if missing else "该技能自己没声明任何 capability")
                findings.append(Finding(
                    "dangerous_capability_undeclared", where,
                    f"DANGEROUS 技能 {step.skill!r}:{detail}。"
                    f"阶段已声明 {sorted(stage.capabilities)}"))


def _check_analyses(spec: ConductSpec, known: frozenset,
                    findings: list[Finding]) -> None:
    """analysis 步引用的纯函数必须真的注册过。

    与技能同理:一个拼错的函数名,运行时的样子是这一步失败;而 analysis 步常常
    是「算出下一步要用的东西」,失败的下一步是一个取不到绑定的执行步。
    """
    for stage in spec.stages:
        for step in stage.all_steps:
            if step.kind != "analysis":
                continue
            if step.analysis_fn not in known:
                findings.append(Finding(
                    "unknown_analysis_fn", f"{stage.stage_id}/{step.step_id}",
                    f"analysis 函数 {step.analysis_fn!r} 不在注册表里;"
                    f"已注册 {sorted(known)}"))


# ── 参数包络 ──────────────────────────────────────────────────────────

def check_params(spec: ConductSpec, params: "Mapping[str, Any]") -> list[Finding]:
    """按 ``params_schema`` 逐字段校验用户填的参数。

    **超包络拒绝,不夹紧。** 夹紧会把一次越界输入变成一次看起来完全正常的运行:
    人以为设了 500 nm,机器跑的是 100 nm,而两者都不会在任何日志里对上。
    针尖登记那条教训的原话就是「超包络要拒绝不要夹紧」。

    返回 findings 列表(空 = 通过)。逐字段返回,不是遇到第一个就停 ——
    让人一次改完。
    """
    out: list[Finding] = []
    by_name = {p.name: p for p in spec.params_schema}
    for name in params:
        if name not in by_name:
            out.append(Finding(
                "param_unknown", name,
                f"spec 没有声明参数 {name!r};已声明的是 {sorted(by_name)}"))
    for p in spec.params_schema:
        if p.name not in params:
            if p.default is None:
                out.append(Finding("param_missing", p.name,
                                   f"参数 {p.name!r} 必填(schema 里没有默认值)"))
            continue
        value = params[p.name]
        if not _type_ok(value, p.type):
            out.append(Finding(
                "param_type", p.name,
                f"参数 {p.name!r} 需要 {p.type},收到 "
                f"{type(value).__name__}({value!r})"))
            continue
        if p.choices:
            if value not in p.choices:
                out.append(Finding(
                    "param_not_in_choices", p.name,
                    f"参数 {p.name!r}={value!r} 不在允许集 {list(p.choices)} 里"))
            continue
        if p.type in ("float", "int"):
            unit = f" {p.unit}" if p.unit else ""
            if p.min_value is not None and float(value) < float(p.min_value):
                out.append(Finding(
                    "param_out_of_envelope", p.name,
                    f"参数 {p.name!r}={value!r}{unit} 低于下限 "
                    f"{p.min_value}{unit} —— 拒绝,不夹紧"))
            if p.max_value is not None and float(value) > float(p.max_value):
                out.append(Finding(
                    "param_out_of_envelope", p.name,
                    f"参数 {p.name!r}={value!r}{unit} 高于上限 "
                    f"{p.max_value}{unit} —— 拒绝,不夹紧"))
    return out


def _type_ok(value: Any, declared: str) -> bool:
    # bool 是 int 的子类:一个 True 混进 int 参数会一路畅通然后变成 1。
    if declared == "bool":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if declared == "int":
        return isinstance(value, int)
    if declared == "float":
        return isinstance(value, (int, float))
    return isinstance(value, str)


# ── 模板零逻辑 lint ───────────────────────────────────────────────────

def lint_template_source(source: str, *, filename: str = "<template>") -> list[Finding]:
    """模板层的那条约束:**只许有组合与参数引用,不许有判据逻辑**。

    出处 ``stm_capability_vs_sample_layer.md`` §6。理由不是洁癖:模板是全仓
    **唯一**允许带样品名的代码位置,判据逻辑一旦写进去,第二个样品就要抄一份
    ——而抄出来的第 N 份里,通常只有一份是对的。

    判据逻辑的可检形状 = 分支 / 循环 / 比较 / 布尔组合 / 带条件的推导式。
    允许的是字面值、参数插值、列表字典装配、无条件推导式、单位换算算术。

    需要判断?写进引擎层(``mast/conduct/`` 的其他模块或 ``vision``/``core``
    的判据机器),模板只引用它的名字和阈值参数。
    """
    out: list[Finding] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        out.append(Finding("template_logic", filename,
                           f"模板解析失败: {exc}"))
        return out
    for node in ast.walk(tree):
        label = _TEMPLATE_FORBIDDEN.get(type(node))
        if label is not None:
            out.append(Finding(
                "template_logic", f"{filename}:{getattr(node, 'lineno', 0)}",
                f"模板里出现了{label} —— 模板只许有组合与参数引用。"
                f"判据逻辑请放引擎层,模板只引用它的名字与阈值参数"))
        if isinstance(node, ast.comprehension) and node.ifs:
            out.append(Finding(
                "template_logic", f"{filename}",
                "推导式里带了 if 过滤 —— 那是判据。要筛选请在引擎层做"))
    return out


def lint_template_module(module) -> list[Finding]:
    """对一个已 import 的模板模块跑 :func:`lint_template_source`。

    读不到源码(冻结包里没有 .py)时**返回一条 finding 说读不到**,而不是空表
    ——「没检查」不许长得像「检查通过」。
    """
    import inspect

    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        return [Finding(
            "template_logic", getattr(module, "__name__", "<module>"),
            f"读不到模板源码,零逻辑 lint 没跑成: {exc}", severity="warning")]
    return lint_template_source(
        source, filename=getattr(module, "__name__", "<template>"))


__all__ = [
    "FINDING_CODES", "RETRACT_CONFIRM_SKILLS", "Finding", "ValidationReport",
    "skill_index", "validate_spec", "check_params", "lint_template_source",
    "lint_template_module",
]
