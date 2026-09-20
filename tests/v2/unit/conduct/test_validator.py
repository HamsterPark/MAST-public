"""conduct 校验器 —— approve 之前那道 lint。

设计:``campaign_director_design.md`` §4.3(三条结构规则)+ §5(绕道首步);
模板零逻辑:``stm_capability_vs_sample_layer.md`` §6。

一份 spec 的毛病,运行时暴露出来的样子是「凌晨三点某一步取不到参数」——
那时针在表面上、样品在低温里、人在睡觉。所以这里测的每一条,都是**在跑之前
就能问出来**的问题。

三条载重规则各配一条变异验证:先证明变异真的应用上了,再看被守卫的行为变红。
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.conduct import validator as V
from mast.conduct.spec import (
    ConductSpec,
    ConditionSpec,
    DetourPolicy,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    StageFailPolicy,
    StageSpec,
    StepSpec,
    WaitSpec,
)
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata

RETRACT = "SafeRetract"


def _meta(name, *, level=SafetyLevel.CONFIRM, caps=frozenset()) -> SkillMetadata:
    return SkillMetadata(name=name, category=SkillCategory.WRITE,
                         safety_level=level, capabilities=frozenset(caps))


def _skills(*metas) -> dict:
    base = {m.name: m for m in metas}
    base.setdefault(RETRACT, _meta(RETRACT))
    return base


def _step(step_id, **kw) -> StepSpec:
    kw.setdefault("kind", "skill")
    kw.setdefault("skill", "ScanAt")
    return StepSpec(step_id=step_id, **kw)


def _retract(step_id="R.00") -> StepSpec:
    return StepSpec(step_id=step_id, kind="skill", skill=RETRACT)


def _wait_step(step_id="W.01") -> StepSpec:
    return StepSpec(step_id=step_id, kind="wait",
                    wait=WaitSpec(kind="operator", message="换样品"))


def _stage(stage_id="S", steps=(), **kw) -> StageSpec:
    kw.setdefault("title", stage_id)
    return StageSpec(stage_id=stage_id, steps=tuple(steps), **kw)


def _spec(stages, *, detour=None, params=()) -> ConductSpec:
    return ConductSpec(
        spec_id="t_v1", spec_version=1, title="t", stages=tuple(stages),
        detour=detour or DetourPolicy(target_stage="", triggers=frozenset()),
        params_schema=tuple(params))


def _codes(report_or_findings) -> list[str]:
    findings = getattr(report_or_findings, "findings", report_or_findings)
    return [f.code for f in findings]


# ── 三态:没检查 ≠ 检查通过 ───────────────────────────────────────────────

def test_a_report_without_a_registry_is_not_approvable():
    """离线 lint 跑得出 ``ok=True``,但 ``approvable`` 必须是 False。

    只看 ``ok`` 的话,一次「没检查」会长得和一次「检查通过」一模一样 ——
    这正是本仓一天出现五次的那族错误。
    """
    spec = _spec([_stage(steps=(_step("a"),))])
    rep = V.validate_spec(spec)
    assert rep.ok is True
    assert rep.complete is False
    assert rep.approvable is False
    assert any("skill_index" in s for s in rep.checks_skipped)
    assert any("analyses" in s for s in rep.checks_skipped)


def test_a_fully_checked_clean_spec_is_approvable():
    spec = _spec([_stage(steps=(_step("a"),))])
    rep = V.validate_spec(spec, skills=_skills(_meta("ScanAt")), analyses=set())
    assert rep.approvable is True
    assert rep.describe() == "校验通过,无发现。"


# ── 规则①:等人 = 针必已退 ───────────────────────────────────────────────

def test_a_wait_without_a_confirmed_retract_is_refused():
    """等待可能是几小时:人会去换样品、会碰机器、会开腔。

    针留在隧道结上等几小时,是把一根修了两周的针交给运气。
    """
    spec = _spec([_stage(steps=(_step("a"), _wait_step()))])
    rep = V.validate_spec(spec)
    assert "wait_without_retract_confirm" in _codes(rep)


def test_a_wait_preceded_by_a_confirmed_retract_passes():
    spec = _spec([_stage(steps=(_retract(), _wait_step()))])
    assert "wait_without_retract_confirm" not in _codes(V.validate_spec(spec))


def test_an_entry_action_can_be_the_retract():
    """入口动作跑在主体之前,所以它算前驱 —— 顺序是承重的。"""
    spec = _spec([_stage(entry_actions=(_retract(),), steps=(_wait_step(),))])
    assert "wait_without_retract_confirm" not in _codes(V.validate_spec(spec))


def test_a_retract_in_an_earlier_stage_does_not_count():
    """不跨阶段。设计 §3-2 立过规矩:不信任跨阶段的仪器状态假设。

    中间可能有人手动接管、别的入口动过、上一次崩在半路 —— 想让它算数,
    就把退针步显式写进本阶段。
    """
    spec = _spec([_stage("S1", steps=(_retract(),)),
                  _stage("S2", steps=(_wait_step(),))])
    assert "wait_without_retract_confirm" in _codes(V.validate_spec(spec))


def test_a_retract_shaped_name_does_not_satisfy_the_rule():
    """名单是**策展的白名单**,不是对 'Retract' 做子串匹配。

    分界线是「验证了动作生效」还是「验证了命令发出去了」。SafeRetract 在
    2026-08-14 之前正是后者:发一条 ZCtrl_Withdraw 就报 retracted=True。
    """
    fake = StepSpec(step_id="F", kind="skill", skill="RetractNow")
    spec = _spec([_stage(steps=(fake, _wait_step()))])
    assert "wait_without_retract_confirm" in _codes(V.validate_spec(spec))


def test_mutation_widening_the_retract_whitelist_disarms_the_rule(monkeypatch):
    """变异验证:把那个「发出即返回」的名字加进白名单,守卫立刻失效。

    ① 先证明变异应用上了(名字进了集合);② 再看被守卫的行为:没有发现了。
    这条同时说明白名单是**载重**的 —— 往里加名字之前要先问那个技能确认了什么。
    """
    fake = StepSpec(step_id="F", kind="skill", skill="RetractNow")
    spec = _spec([_stage(steps=(fake, _wait_step()))])
    assert "wait_without_retract_confirm" in _codes(V.validate_spec(spec))

    monkeypatch.setattr(V, "RETRACT_CONFIRM_SKILLS",
                        V.RETRACT_CONFIRM_SKILLS | {"RetractNow"})
    # ① 变异已应用
    assert "RetractNow" in V.RETRACT_CONFIRM_SKILLS
    # ② 守卫不再报
    assert "wait_without_retract_confirm" not in _codes(V.validate_spec(spec))


# ── 规则②:bindings 必须命中上游 produces ────────────────────────────────

def test_a_binding_to_something_nobody_produces_is_refused():
    """produces 是白名单。不在名单上就取不到,而且**不会有默认值兜底** ——
    一个「取不到就用 0」的绑定,会把一次读失败变成一次看起来正常的运行。"""
    spec = _spec([_stage(steps=(
        _step("a", produces=("x",)),
        _step("b", bindings={"p": "steps.a.y"}),
    ))])
    assert "binding_not_produced" in _codes(V.validate_spec(spec))


def test_a_binding_that_hits_the_whitelist_passes():
    spec = _spec([_stage(steps=(
        _step("a", produces=("x",)),
        _step("b", bindings={"p": "steps.a.x"}),
    ))])
    assert _codes(V.validate_spec(spec)) == []


def test_a_forward_reference_is_refused():
    """运行到这一步时,被引用的那一步还没产出任何东西。"""
    spec = _spec([_stage(steps=(
        _step("a", bindings={"p": "steps.b.x"}),
        _step("b", produces=("x",)),
    ))])
    assert "binding_forward_reference" in _codes(V.validate_spec(spec))


def test_a_step_cannot_bind_to_its_own_output():
    spec = _spec([_stage(steps=(
        _step("a", produces=("x",), bindings={"p": "steps.a.x"}),
    ))])
    assert "binding_forward_reference" in _codes(V.validate_spec(spec))


def test_a_step_id_containing_dots_still_resolves():
    """真实 step_id 长这样:``S2.03_scan_500mV``。

    按点切分会把它切碎 —— 所以解析是按已知 produces 全键匹配,不是 split('.')。
    """
    spec = _spec([_stage(steps=(
        _step("S2.02_plan", produces=("plan_json",)),
        _step("S2.03_run", bindings={"plan_json": "steps.S2.02_plan.plan_json"}),
    ))])
    assert _codes(V.validate_spec(spec)) == []


def test_an_unparseable_reference_is_refused():
    spec = _spec([_stage(steps=(_step("a", bindings={"p": "上一步的那个数"}),))])
    assert "binding_unparseable" in _codes(V.validate_spec(spec))


def test_a_step_can_bind_to_a_conduct_parameter():
    """没有这条,``params_schema`` 就是个渲染表单用的摆设:填了的数一步也到不了。

    「已经记下来了」是生产方的话 —— 要问的是谁读它。
    """
    spec = _spec([_stage(steps=(_step("a", bindings={"setpoint_a":
                                                     "params.setpoint_a"}),))],
                 params=(ParamSpec(name="setpoint_a", type="float"),))
    assert _codes(V.validate_spec(spec)) == []


def test_a_binding_to_an_undeclared_parameter_is_refused():
    spec = _spec([_stage(steps=(_step("a", bindings={"p": "params.nope"}),))],
                 params=(ParamSpec(name="setpoint_a", type="float"),))
    assert "binding_not_produced" in _codes(V.validate_spec(spec))


# ── 规则③:DANGEROUS 落在声明了 capability 的阶段 ────────────────────────

def test_a_dangerous_step_in_a_stage_that_declared_nothing_is_refused():
    """SAFE 模式按 capability 过滤 —— 阶段不声明,就等于谁都拦不住。"""
    spec = _spec([_stage(steps=(_step("a", skill="TipPulse"),))])
    rep = V.validate_spec(spec, skills=_skills(
        _meta("TipPulse", level=SafetyLevel.DANGEROUS, caps={"bias_pulse"})),
        analyses=set())
    assert "dangerous_capability_undeclared" in _codes(rep)


def test_a_dangerous_step_in_a_stage_that_declared_it_passes():
    spec = _spec([_stage(steps=(_step("a", skill="TipPulse"),),
                         capabilities=frozenset({"bias_pulse"}))])
    rep = V.validate_spec(spec, skills=_skills(
        _meta("TipPulse", level=SafetyLevel.DANGEROUS, caps={"bias_pulse"})),
        analyses=set())
    assert rep.approvable is True


def test_a_dangerous_skill_declaring_no_capability_is_still_caught():
    """技能自己不填 capability 时,SAFE 模式形同虚设 —— 这一条本仓记过一次。

    所以「阶段声明了什么」拦不住它,得由这里拦:DANGEROUS 而没有任何
    capability 声明,一律报。
    """
    spec = _spec([_stage(steps=(_step("a", skill="Mystery"),),
                         capabilities=frozenset({"bias_pulse"}))])
    rep = V.validate_spec(spec, skills=_skills(
        _meta("Mystery", level=SafetyLevel.DANGEROUS)), analyses=set())
    assert "dangerous_capability_undeclared" in _codes(rep)


def test_mutation_dropping_the_capability_check_lets_it_through(monkeypatch):
    """变异验证:把规则③摘掉,上面那条守卫必须变哑。"""
    spec = _spec([_stage(steps=(_step("a", skill="TipPulse"),))])
    skills = _skills(_meta("TipPulse", level=SafetyLevel.DANGEROUS,
                           caps={"bias_pulse"}))
    assert "dangerous_capability_undeclared" in _codes(
        V.validate_spec(spec, skills=skills, analyses=set()))

    monkeypatch.setattr(V, "_check_skills", lambda *a, **k: None)
    # ① 变异已应用(检查函数被换成空操作)
    assert V._check_skills(spec, skills, []) is None
    # ② 守卫不再报
    assert _codes(V.validate_spec(spec, skills=skills, analyses=set())) == []


def test_a_skill_the_machine_does_not_have_is_reported():
    spec = _spec([_stage(steps=(_step("a", skill="NotInstalled"),))])
    rep = V.validate_spec(spec, skills=_skills(), analyses=set())
    assert "unknown_skill" in _codes(rep)
    assert rep.approvable is False


def test_analysis_and_wait_steps_are_not_looked_up_as_skills():
    spec = _spec([_stage(steps=(
        _retract(),
        StepSpec(step_id="an", kind="analysis", analysis_fn="f"),
        _wait_step(),
    ))])
    rep = V.validate_spec(spec, skills=_skills(), analyses={"f"})
    assert _codes(rep) == []


def test_an_unregistered_analysis_function_is_reported():
    spec = _spec([_stage(steps=(
        StepSpec(step_id="an", kind="analysis", analysis_fn="nope"),))])
    rep = V.validate_spec(spec, skills=_skills(), analyses={"f"})
    assert "unknown_analysis_fn" in _codes(rep)


# ── 规则④:步里写的参数名必须是这个技能真的收的 ──────────────────────────
#
# 2026-08-14 加。规则②只管 binding 的**右**边(引用的产出存不存在),
# 从不问**左**边那个键这个技能认不认识 —— 于是一份「校验通过」的 spec 可以
# 把参数喂给一个不存在的名字。

def _meta_with_params(name, *param_names) -> SkillMetadata:
    from mast.core.types import ParameterSpec
    return SkillMetadata(
        name=name, category=SkillCategory.WRITE,
        safety_level=SafetyLevel.CONFIRM,
        parameters=[ParameterSpec(name=p, type="float", description=p)
                    for p in param_names])


def test_a_param_name_the_skill_does_not_take_is_reported():
    """真实起因:模板把取谱引擎的入参写成 ``positions_json``,而它收的是
    ``positions``。运行前没有任何东西会说这句话。"""
    spec = _spec([_stage(steps=(
        _step("a", skill="Sp", params={"positions_json": "x"}),))])
    rep = V.validate_spec(spec, skills=_skills(_meta_with_params("Sp", "positions")),
                          analyses=set())
    assert "step_param_unknown" in _codes(rep)
    assert rep.approvable is False
    # 报告里要说得出它到底收哪些 —— 否则改一次要去翻源码
    assert any("positions" in f.message for f in rep.findings)


def test_a_binding_key_is_checked_too_not_just_the_reference():
    """bindings 的**左**边同样是参数名。规则②看右边,这条看左边。"""
    spec = _spec([_stage(steps=(
        _step("a", skill="Sp", produces=("out",)),
        _step("b", skill="Sp", bindings={"nope": "steps.a.out"}),))])
    rep = V.validate_spec(spec, skills=_skills(_meta_with_params("Sp", "positions")),
                          analyses=set())
    assert "step_param_unknown" in _codes(rep)


def test_the_right_param_names_pass():
    spec = _spec([_stage(steps=(
        _step("a", skill="Sp", params={"positions": "x"}),))])
    rep = V.validate_spec(spec, skills=_skills(_meta_with_params("Sp", "positions")),
                          analyses=set())
    assert "step_param_unknown" not in _codes(rep)


def test_a_skill_that_declares_no_parameters_is_not_second_guessed():
    """没声明参数表的技能不判 —— 「它没声明」与「它不收」是两件事,
    而把前者读成后者会把一批老技能全判成错的。"""
    spec = _spec([_stage(steps=(
        _step("a", skill="Bare", params={"whatever": 1}),))])
    rep = V.validate_spec(spec, skills=_skills(_meta("Bare")), analyses=set())
    assert "step_param_unknown" not in _codes(rep)


def test_mutation_dropping_the_param_name_check_lets_it_through(monkeypatch):
    """先证明变异落到了守卫上,再证明测试红 —— 否则「守卫在」只是一句声称。

    ## 前提检查为什么不用 ``inspect.getsource``(2026-08-15 换的)

    它按**import 那一刻**记下的 ``co_firstlineno`` 去切**当前磁盘上**的文件。
    别人在 ``validator.py`` 上方插几行,切出来的就是错位片段 —— 这里是一条
    **正向**断言,错位的表现是**假红**:一次真回归的样子,而根因完全在别处。
    (共用树上这条真的响过:20:1x 那一轮全量里它红了,单跑绿。)

    改成 ``co_consts``:字面量取自**已加载的 code object**,完全不碰磁盘。
    这是正向断言,没有空转风险 —— 常量表里没有它就红。
    """
    from mast.conduct import validator as mod

    literals = [c for c in mod._check_skills.__code__.co_consts
                if isinstance(c, str)]
    assert any("step_param_unknown" in c for c in literals), (
        f"变异前提不成立:守卫不在这个函数里。它的字面量:{literals[:8]}")

    real = mod._check_skills

    def _no_param_check(spec, skills, findings):
        out: list = []
        real(spec, skills, out)
        findings.extend(f for f in out if f.code != "step_param_unknown")

    monkeypatch.setattr(mod, "_check_skills", _no_param_check)
    spec = _spec([_stage(steps=(
        _step("a", skill="Sp", params={"positions_json": "x"}),))])
    rep = mod.validate_spec(
        spec, skills=_skills(_meta_with_params("Sp", "positions")), analyses=set())
    assert "step_param_unknown" not in _codes(rep), "变异没生效"


# ── §5:绕道首步 ─────────────────────────────────────────────────────────

def test_the_detour_target_must_exist():
    spec = _spec([_stage("S2", steps=(_step("a"),))],
                 detour=DetourPolicy(target_stage="S1"))
    assert "detour_target_missing" in _codes(V.validate_spec(spec))


def test_the_detour_target_must_take_the_tip_away_somewhere():
    """绕道的触发条件就是「针可能坏了」,这一段迟早要把针拿开。

    2026-08-15:判据从「**第 0 步**必须是退针技能」放宽成「这一段里得有」——
    旧判据把只读诊断(跨点复测)也一起拦掉了,而那一步正是「先确认真是针坏了、
    再付换样品修针的钱」的前提。**退针之前不许干什么**由
    ``detour_writes_before_retract`` 管,判据更精确而不是更松。
    """
    spec = _spec([_stage("S1", steps=(_step("a", skill="ApproachTip"),)),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="S1"))
    assert "detour_without_retract" in _codes(V.validate_spec(spec))


class _Meta:
    """最小 skill 元数据替身。``capabilities`` 与 ``safety_level`` 是新判据
    唯一要读的两样。"""

    class _Level:
        def __init__(self, v): self.value = v

    def __init__(self, level="auto", capabilities=()):
        self.safety_level = self._Level(level)
        self.capabilities = frozenset(capabilities)
        self.parameters = ()


_SKILLS = {
    "SafeRetract": _Meta(),
    "CrossPointTipCheck": _Meta(level="confirm"),     # 只读诊断:扫图 + 移动
    "MoveToXY": _Meta(),
    "TipShape": _Meta(level="dangerous", capabilities={"tip_shaping"}),
    "BiasPulse": _Meta(level="confirm", capabilities={"bias_pulse"}),
    "ScanAt": _Meta(),
    "ApproachTip": _Meta(),
}


def test_a_read_only_diagnosis_may_run_before_the_retract():
    """**这条是新判据存在的理由。**

    跨点复测要 MoveToXY + 扫图 —— 那也是硬件动作,但它不改变表面。拦掉它等于
    让每一次表面异常都要付一轮换样品修针的钱(两次人工换样品、几小时等待),
    而它本来能在 15 分钟只读复查里被排除掉。
    """
    spec = _spec([_stage("S1", steps=(_step("d", skill="CrossPointTipCheck"),
                                      _retract(), _step("f", skill="TipShape"))),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="S1"))
    codes = _codes(V.validate_spec(spec, skills=_SKILLS,
                                   analyses=()))
    assert "detour_writes_before_retract" not in codes


@pytest.mark.parametrize("skill", ["TipShape", "BiasPulse"])
def test_a_surface_changing_action_before_the_retract_is_refused(skill):
    """脉冲、扎针、除层 —— 这些才是「拿一根可能坏了的针继续往表面上开」。

    绕道的触发源之一是 ``recovery_tip_fail``(崩溃重启后判针坏),那条路上
    第一个动作若是打脉冲,就是拿一根刚崩过的针再开一次。
    """
    spec = _spec([_stage("S1", steps=(_step("w", skill=skill), _retract())),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="S1"))
    assert "detour_writes_before_retract" in _codes(
        V.validate_spec(spec, skills=_SKILLS, analyses=()))


def test_the_write_before_retract_check_says_when_it_could_not_run():
    """判据要读技能元数据 ⇒ 离线跑不了。**「没检查」不许长得像「检查通过」。**"""
    spec = _spec([_stage("S1", steps=(_step("w", skill="TipShape"), _retract())),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="S1"))
    rep = V.validate_spec(spec)
    assert "detour_writes_before_retract" not in _codes(rep)
    assert any("退针之前" in s for s in rep.checks_skipped)
    assert rep.approvable is False


def test_a_detour_only_stage_that_is_not_the_target_can_never_run():
    """正常推进跳过它、绕道又不进它 —— 一段永远跑不到的阶段,而它在模板里
    看起来一切正常。"""
    spec = _spec([_stage("R", steps=(_retract(),), entered_only_by_detour=True),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="", triggers=frozenset()))
    assert "detour_only_stage_unreachable" in _codes(V.validate_spec(spec))


def test_a_step_level_gate_routing_to_a_missing_detour_is_caught():
    """步级闸门也要进「无处可去」的清查 —— 漏掉它的后果是静默的:
    一个 detour 去向躲过检查,跑到那一步才发现绕道没有目标。"""
    gate = GateSpec(gate_id="g", kind="rule", rule=RuleLeaf("x", "exists"),
                    routes={"ok": GateOutcome("pass"),
                            "bad": GateOutcome("detour")})
    spec = _spec([_stage("S2", steps=(_step("a", gate=gate), _step("b")))])
    assert "detour_route_without_target" in _codes(V.validate_spec(spec))


def test_a_proper_detour_target_passes():
    spec = _spec([_stage("S1", steps=(_retract(), _step("a"))),
                  _stage("S2", steps=(_step("b"),))],
                 detour=DetourPolicy(target_stage="S1"))
    assert _codes(V.validate_spec(spec)) == []


def test_a_route_to_detour_without_a_repair_stage_is_refused():
    """触发得了、无处可去 —— 又一个「能挂不能解」。

    没有修针段时正确的去向是 wait_operator:人来修针,而不是假装有地方可去。
    """
    gate = GateSpec(gate_id="g", kind="rule", rule=RuleLeaf("x", "exists"),
                    routes={"ok": GateOutcome("pass"),
                            "bad": GateOutcome("detour")})
    spec = _spec([_stage("S2", steps=(_step("a"),), exit_gate=gate)])
    assert "detour_route_without_target" in _codes(V.validate_spec(spec))


def test_a_stage_fail_policy_pointing_at_a_missing_detour_is_refused():
    spec = _spec([_stage("S2", steps=(_step("a"),),
                         on_fail=StageFailPolicy(then="detour"))])
    assert "detour_route_without_target" in _codes(V.validate_spec(spec))


# ── 参数包络:拒绝,不夹紧 ───────────────────────────────────────────────

def _pspec(**kw) -> ConductSpec:
    return _spec([_stage(steps=(_step("a"),))],
                 params=(ParamSpec(name="setpoint_a", type="float", unit="A",
                                   min_value=1e-12, max_value=100e-9),
                         ParamSpec(name="mode", type="str",
                                   choices=("fast", "slow"), default="fast"),
                         ParamSpec(name="n", type="int", default=1)))


def test_an_out_of_envelope_value_is_rejected_not_clamped():
    """明显超出量级的电流输入必须拒绝，不能夹紧成看似正常的运行参数。"""
    findings = V.check_params(_pspec(), {"setpoint_a": 2.0})
    assert [f.code for f in findings] == ["param_out_of_envelope"]
    assert "不夹紧" in findings[0].message
    # 而且真的没有产出一个「修正后的值」——校验器只回发现,不回参数
    assert not hasattr(findings[0], "corrected")


def test_every_field_is_reported_not_just_the_first():
    """让人一次改完,而不是改一个跑一次。"""
    findings = V.check_params(_pspec(), {"setpoint_a": 1.5, "mode": "medium"})
    assert set(f.code for f in findings) == {"param_out_of_envelope",
                                             "param_not_in_choices"}


def test_a_required_parameter_with_no_default_must_be_filled():
    assert [f.code for f in V.check_params(_pspec(), {})] == ["param_missing"]


def test_an_unknown_parameter_is_reported_not_ignored():
    findings = V.check_params(_pspec(), {"setpoint_a": 5e-11, "typo": 1})
    assert "param_unknown" in [f.code for f in findings]


def test_a_bool_never_passes_as_an_int():
    """``True`` 是 int 的子类。混进来会一路畅通然后变成 1。"""
    findings = V.check_params(_pspec(), {"setpoint_a": 5e-11, "n": True})
    assert "param_type" in [f.code for f in findings]


def test_an_int_is_accepted_where_a_float_is_declared():
    """Nanonis 两种都收,别为一个 ``3`` 拦下一个本意是 ``3.0`` 的填写。"""
    spec = _spec([_stage(steps=(_step("a"),))],
                 params=(ParamSpec(name="hold_s", type="float"),))
    assert V.check_params(spec, {"hold_s": 3}) == []


def test_zero_is_judged_against_the_envelope_like_any_other_number():
    """0 不是「没填」。填了 0 而下限是 1 pA,那就是超包络。"""
    findings = V.check_params(_pspec(), {"setpoint_a": 0})
    assert [f.code for f in findings] == ["param_out_of_envelope"]


# ── 模板零逻辑 lint ─────────────────────────────────────────────────────

CLEAN = '''
"""一个只做装配的模板。"""
from mast.conduct.spec import StepSpec

SIZE_NM = 5.0
STEPS = tuple(StepSpec(step_id=f"S.{i}", kind="skill", skill="ScanAt",
                       params={"size_m": SIZE_NM * 1e-9})
              for i in range(3))
'''


def test_a_pure_assembly_template_passes():
    """字面值、参数插值、装配、单位换算、无条件推导式 —— 全部允许。"""
    assert V.lint_template_source(CLEAN) == []


@pytest.mark.parametrize("snippet,why", [
    ("X = 1\nif X:\n    Y = 2\n", "分支"),
    ("def f(a):\n    return 1 if a else 2\n", "三元"),
    ("for i in range(3):\n    pass\n", "循环"),
    ("def f(v):\n    return v > 0.8\n", "阈值比较"),
    ("def f(a, b):\n    return a and b\n", "布尔组合"),
    ("Y = [x for x in range(3) if x]\n", "带条件的推导式"),
    ("def f():\n    raise ValueError('x')\n", "raise"),
    ("try:\n    X = 1\nexcept Exception:\n    X = 2\n", "try"),
])
def test_judgement_logic_in_a_template_is_refused(snippet, why):
    """判据逻辑一进模板,第二个样品就要抄一份 —— 而抄出来的第 N 份里,
    通常只有一份是对的。

    需要判断?写进引擎层,模板只引用它的名字和阈值参数。
    """
    findings = V.lint_template_source(snippet, filename="t.py")
    assert findings, f"{why}没有被拦住"
    assert all(f.code == "template_logic" for f in findings)


def test_an_unreadable_template_source_is_not_silently_clean():
    """读不到源码 ⇒ 说读不到,而不是回一张空表。

    「没检查」不许长得像「检查通过」。
    """
    class NoSource:
        __name__ = "fake_template"

    findings = V.lint_template_module(NoSource())
    assert findings and findings[0].code == "template_logic"
    assert findings[0].severity == "warning"


def test_a_broken_template_reports_instead_of_raising():
    findings = V.lint_template_source("def f(:\n", filename="t.py")
    assert [f.code for f in findings] == ["template_logic"]


# ── 发现码是闭集 ────────────────────────────────────────────────────────

def test_every_finding_code_is_declared():
    with pytest.raises(ValueError):
        V.Finding("made_up_code", "x", "y")


def test_skill_index_reads_names_not_str_of_the_dataclass():
    """技能存在性检查必须提取 SkillMetadata.name，而不是将整份对象字符串化后当作名称。"""
    class Reg:
        def list_skills(self):
            return [_meta("ScanAt"), "NotAMeta", 42]

    idx = V.skill_index(Reg())
    assert set(idx) == {"ScanAt"}


def test_an_unreadable_registry_raises_instead_of_looking_empty():
    """注册表读不出来 ≠ 一个技能都没有。后者会让每个步都报 unknown_skill。"""
    class Broken:
        def list_skills(self):
            raise RuntimeError("db down")

    with pytest.raises(ValueError):
        V.skill_index(Broken())


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
