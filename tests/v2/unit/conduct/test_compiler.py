"""plan → conduct 编译器的契约(``mast.conduct.compiler``)。

这里钉的每一条,都是「合起来就出事」的一个区分:

1. **闭集是闭集**:``COMPILE_ERROR_CODES`` 逐字快照。改一个名字就要有人来改这条
   测试,而改的那个人会顺手看见前端按码分支的那张表;
2. **plan 级前置说得出话**:没这份方案 / 还没批 / 认不出模板,各是一句不同的话
   ——它们各自要人去做的事完全不同;
3. **每缺一个槽一条 ``SLOT_UNFILLED``**(p3-T5:零语料路径是**主用例**不是边界),
   **超包络逐字段**且拒绝不夹紧;
4. **不部分编译**:``ok=False ⇒ spec is None``;
5. **``spec`` 是注册模板那个对象本身**(``is``,不是 ``==``)—— 附一条变异测试
   证明这道断言真的抓得住合成品;
6. **纯读**:编译前后 conducts 表一行不多、plan 一个字不改。

全程 ``tmp_path``:PlanStore 与 ConductStore 都落 tmp 的库,plan 不带
``experiment_id``(``plan_store._doc_target`` 因此退回本 store 目录)——
**绝不碰真实 experiments**。测试污染真实数据在本仓已经五次。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct.compiler import (  # noqa: E402
    COMPILE_ERROR_CODES,
    CompileError,
    CompileResult,
    compile_plan_to_conduct_spec,
    resolved_template,
)
from mast.conduct.spec import ConductSpec  # noqa: E402
from mast.conduct.store import ConductStore  # noqa: E402
from mast.conduct.templates import TEMPLATES  # noqa: E402
from mast.planning.plan_store import (  # noqa: E402
    ExperimentPlan,
    PlanPhase,
    PlanStatus,
    PlanStore,
)

SMOKE_PARAMS = {"target_temperature_k": 5.0,
                "temperature_stale_after_s": 600.0,
                "probe_positions_m": "1e-8,2e-8; -3e-8,0",
                "probe_points_max": 4}


# ── 夹具 ────────────────────────────────────────────────────────────

@pytest.fixture
def plans(tmp_path):
    """只往 ``tmp_path`` 写的 PlanStore。"""
    return PlanStore(tmp_path / "plans.db", plans_dir=tmp_path / "plans")


@pytest.fixture(scope="module")
def registry():
    """真的技能注册表 —— 不给的话规则③进 ``checks_skipped``,``ok`` 恒为 False。

    那正是第 6 组要单独钉的东西,所以别的用例必须拿一份真的,免得所有断言都被
    「没检查」这一条盖住。
    """
    from mast.core.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    return reg


def _save(plans, *, plan_id="p1", status=PlanStatus.APPROVED, block=None,
          in_phase=False, notes=None):
    """存一份 plan;绑定块默认写进 ``notes`` 的 JSON。"""
    phases = []
    if in_phase and block is not None:
        phases = [PlanPhase(id="ph0", name="第一段", steps=[{"conduct": block}])]
    if notes is None:
        notes = json.dumps({"conduct": block}, ensure_ascii=False) if (
            block is not None and not in_phase) else ""
    plans.save(ExperimentPlan(
        plan_id=plan_id, experiment_id="", name="测试方案",
        goal="把编译器走一遍", phases=phases, status=status, notes=notes))
    return plan_id


def _smoke_block(**overrides):
    params = dict(SMOKE_PARAMS)
    params.update(overrides)
    return {"template": "_smoke_v1", "params": params}


def _codes(result) -> list[str]:
    return [e.code for e in result.errors]


def _is_registered_template(spec) -> bool:
    """这个 spec 是不是**注册表里那一个对象本身**。

    ``is`` 不是 ``==``:``ConductSpec`` 是 frozen dataclass,一个逐字段重建出来的
    合成品与注册模板**内容相等**。用 ``==`` 写这道断言,合成器上线的那天它照样
    绿着 —— 见下面的变异测试。
    """
    return any(spec is t for t in TEMPLATES.values())


# ── 1. 闭集快照 ─────────────────────────────────────────────────────

def test_error_codes_are_a_pinned_closed_set():
    """逐字快照。改名/增删都要有人来动这一行。

    前 3 个是本仓新增的 plan 级前置(p3 §3.7 那张表只覆盖**槽级**错误:它假定
    plan 已经取到手、已经批过、模板已经认出来了);后 12 个逐字照抄 p3,顺序
    都不动 —— 即使 M1 只触发得了其中一个子集。一张写好了的闭集是下一轮的接线图;
    删掉「今天还不会发生的」,下一个人就会另起一套名字。
    """
    assert COMPILE_ERROR_CODES == (
        "PLAN_NOT_FOUND", "PLAN_NOT_APPROVED", "TEMPLATE_UNRESOLVED",
        "SLOT_UNFILLED", "SLOT_SOURCE_FORBIDDEN", "SLOT_SOURCE_NEEDS_ACK",
        "SLOT_STALE_CALIBRATION", "SLOT_MAGNITUDE_ABSURD", "SKILL_UNKNOWN",
        "SKILL_CAPABILITY_UNDECLARED", "PREDICATE_UNKNOWN",
        "CRITERION_NO_UNCERTAIN_ROUTE", "MATERIAL_NO_PROFILE",
        "EVIDENCE_NONE_UNACKED", "STAGE_COUNT_EXCEEDED",
    )
    assert len(set(COMPILE_ERROR_CODES)) == len(COMPILE_ERROR_CODES)


def test_an_unknown_code_is_refused_at_construction():
    """闭集外的码**当场抛**,不是流到前端被当成「未知情况」静默忽略。"""
    with pytest.raises(ValueError, match="COMPILE_ERROR_CODES"):
        CompileError("SLOT_LOOKS_FINE")


# ── 2. plan 级前置 ──────────────────────────────────────────────────

def test_a_missing_plan_is_not_found(plans, registry):
    r = compile_plan_to_conduct_spec("nope", plan_store=plans, registry=registry)
    assert r.ok is False and _codes(r) == ["PLAN_NOT_FOUND"]
    assert r.spec is None


def test_a_draft_plan_is_refused_as_not_approved(plans, registry):
    """批准是**人**的动作。编译器不替它做,也不假装没看见。"""
    pid = _save(plans, status=PlanStatus.DRAFT, block=_smoke_block())
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["PLAN_NOT_APPROVED"]
    assert "draft" in r.errors[0].detail


def test_a_plan_without_a_binding_block_says_where_to_write_it(plans, registry):
    """认不出模板 ⇒ 一条 ``TEMPLATE_UNRESOLVED``,而且**说出块该写在哪儿**。

    「plan 里没有 conduct 块」这句话本身没有任何指向 —— 读到它的人下一步要么
    去翻代码,要么放弃。
    """
    pid = _save(plans, notes="这是一段给人看的话,不是 JSON")
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["TEMPLATE_UNRESOLVED"]
    detail = r.errors[0].detail
    assert "plan.notes" in detail and "phases[0].steps[0]" in detail


def test_an_unregistered_template_lists_what_is_registered(plans, registry):
    pid = _save(plans, block={"template": "no_such_template_v9", "params": {}})
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["TEMPLATE_UNRESOLVED"]
    assert "_smoke_v1" in r.errors[0].detail, "认不出来时要说出有哪些"


def test_the_template_key_has_exactly_one_name(plans, registry):
    """写成 ``spec_id`` 的块被**认出来并拒绝**,不是静默当没看见。

    两个可接受的键名就是两个真源,而只有一个会被读到 —— 另一个写法的人会以为
    自己写对了,然后去别处找原因。
    """
    pid = _save(plans, block={"spec_id": "_smoke_v1", "params": {}})
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["TEMPLATE_UNRESOLVED"]
    assert "spec_id" in r.errors[0].detail and "template" in r.errors[0].detail


def test_the_block_can_also_live_in_the_first_phase_step(plans, registry):
    """第二条约定的路径:``plan.phases[0].steps[0]["conduct"]``。"""
    pid = _save(plans, block=_smoke_block(), in_phase=True)
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert r.ok is True, r.describe()
    assert r.spec_id == "_smoke_v1"


# ── 3. 槽:每缺一个一条,超包络逐字段 ────────────────────────────────

def test_every_unfilled_slot_gets_its_own_error(plans, registry):
    """p3-T5:**零语料路径是主用例**。四个参数一个都没填 ⇒ 四条 SLOT_UNFILLED。

    「有几个槽没填」这个数字本身就是方案页要显示的东西(T12:``已填 N / 共 M``)。
    合并成一条「参数不完整」的话,人只能一个一个试。
    """
    pid = _save(plans, block={"template": "_smoke_v1", "params": {}})
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert r.ok is False and r.spec is None
    assert _codes(r) == ["SLOT_UNFILLED"] * 4
    assert sorted(e.slot_name for e in r.errors) == sorted(SMOKE_PARAMS)


def test_one_missing_slot_is_one_error(plans, registry):
    pid = _save(plans, block={"template": "_smoke_v1",
                              "params": {k: v for k, v in SMOKE_PARAMS.items()
                                         if k != "probe_points_max"}})
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["SLOT_UNFILLED"]
    assert r.errors[0].slot_name == "probe_points_max"


def test_out_of_envelope_slots_are_refused_field_by_field(plans, registry):
    """**拒绝,不夹紧**,而且**逐字段** —— 让人一次改完,不是改一个再撞一个。"""
    pid = _save(plans, block=_smoke_block(target_temperature_k=9999.0,
                                          probe_points_max=999))
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["SLOT_MAGNITUDE_ABSURD"] * 2
    assert sorted(e.slot_name for e in r.errors) == [
        "probe_points_max", "target_temperature_k"]
    assert all("不夹紧" in e.detail for e in r.errors)


def test_a_wrong_type_is_a_bad_value_not_an_empty_slot(plans, registry):
    """填了一个 str 到 float 槽:是「值本身错了」,**不是**「没填」。

    两句话要人做的事不同:一个去改那个值,一个去找它为什么没填上。
    """
    pid = _save(plans, block=_smoke_block(target_temperature_k="五开尔文"))
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert _codes(r) == ["SLOT_MAGNITUDE_ABSURD"]


def test_a_key_the_template_never_declared_is_a_template_mismatch(plans, registry):
    """块里写了模板没声明的键 ⇒ ``TEMPLATE_UNRESOLVED``,不是任何 ``SLOT_*``。

    那不是「某个槽有问题」,是**这个块与这份模板对不上**:它指着的槽在这份模板
    里根本不存在。一个悄悄被忽略的参数,填的人会一直以为它生效了。
    """
    pid = _save(plans, block=_smoke_block(bias_v=-2.0))
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert "TEMPLATE_UNRESOLVED" in _codes(r)
    assert any(e.slot_name == "bias_v" for e in r.errors)


# ── 4. 「没检查」不许长得像「检查通过」 ─────────────────────────────

def test_without_a_registry_the_result_is_not_ok(plans):
    """规则③跑不了 ⇒ ``checks_skipped`` 非空 ⇒ ``ok=False``。

    与 approve 只看 ``approvable`` 是同一条:一次「什么都没检查」不许长得和
    「检查全过」一模一样。
    """
    pid = _save(plans, block=_smoke_block())
    r = compile_plan_to_conduct_spec(pid, plan_store=plans)      # 没有 registry
    assert r.ok is False and r.spec is None
    assert r.errors == (), "没检查不是「发现了错误」——它是第三态"
    assert r.checks_skipped, "跳过了哪些检查要说出来"
    assert any("规则③" in s for s in r.checks_skipped)


# ── 5. 不部分编译 + spec 是注册模板本身 ─────────────────────────────

def test_a_failed_compile_never_carries_a_spec(plans, registry):
    """``ok=False ⇒ spec is None``(p3 §3.7 逐字)。

    一个「大部分编译好了」的 spec 会被当成可以试着跑一下的东西,而缺的那个槽
    恰恰是数值。这里连**构造**都堵死:见 ``CompileResult.__post_init__``。
    """
    pid = _save(plans, block={"template": "_smoke_v1", "params": {}})
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert r.ok is False and r.spec is None
    with pytest.raises(ValueError, match="不部分编译"):
        CompileResult(ok=False, spec=TEMPLATES["_smoke_v1"])


def test_a_good_compile_returns_the_registered_template_object_itself(plans,
                                                                     registry):
    """``result.spec is TEMPLATES[spec_id]`` —— **同一个对象**,不是拷贝。

    M1 的编译器**不合成 ConductSpec**,四条代码事实各自都足以否掉那条路
    (见 ``compiler.py`` 的模块 docstring):spec 根本不入库;重启自检 A5 按
    ``spec_version`` 去 ``TEMPLATES`` 对账;等待步只可能来自模板(p3 §3.7 逐字
    「草稿不产出等待步」);槽体系还不存在。

    所以真正的可执行产物是 ``(注册模板, params)`` 这一对。哪天真要开始合成,
    这条测试会先红 —— 它逼着改的人先回答「合成出来的 spec 存在哪儿、重启之后
    谁认得它」。
    """
    pid = _save(plans, block=_smoke_block())
    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert r.ok is True, r.describe()
    assert r.spec is TEMPLATES["_smoke_v1"]
    assert _is_registered_template(r.spec)
    assert r.params == SMOKE_PARAMS
    assert resolved_template(r) is TEMPLATES["_smoke_v1"]


def test_mutation_a_synthesized_spec_would_be_caught():
    """**变异验证**:先证明上面那道断言真的会动手。

    ``ConductSpec`` 是 frozen dataclass ⇒ 一个逐字段重建出来的合成品与注册模板
    **内容相等**(``==`` 是 True)。也就是说,把上面那条写成 ``==`` 的话,
    合成器上线的那天它照样绿着,而「等待步从哪来」「重启后谁认得它」两个问题
    一个都没被问过。

    这里构造正是那个合成品,断言 ``==`` 抓不到、``is`` 抓得到。
    """
    original = TEMPLATES["_smoke_v1"]
    synthesized = dataclasses.replace(original)      # 逐字段重建 = 合成器的产物

    assert isinstance(synthesized, ConductSpec)
    assert synthesized == original, "前提:内容相等 —— 所以 == 这道闸是空的"
    assert synthesized is not original
    assert _is_registered_template(synthesized) is False, (
        "这道断言抓不住合成品 —— 那么它对真实的合成器也一样抓不住")
    with pytest.raises(AssertionError):
        assert synthesized is TEMPLATES["_smoke_v1"]


# ── 6. 纯读 ─────────────────────────────────────────────────────────

def test_compiling_writes_nothing_anywhere(tmp_path, plans, registry):
    """编译是一次**干跑**:conduct 库一行不多,plan 一个字不改。

    写发生在 ``POST /api/conducts``(建草稿)与 ``approve``(冻参数)那两个人的
    动作里。编译器要是顺手建了点什么,用户就会在还没决定做不做之前,先撞上
    单活跃不变式。
    """
    store = ConductStore(tmp_path / "conduct.db")
    store.create(experiment_id="e1", spec_id="_smoke_v1", spec_version=1,
                 params=dict(SMOKE_PARAMS))
    before_rows = store.list_conducts()
    before_events = store.events(before_rows[0]["conduct_id"], limit=500)

    pid = _save(plans, block=_smoke_block())
    before_plan = plans.load(pid).to_dict()

    r = compile_plan_to_conduct_spec(pid, plan_store=plans, registry=registry)
    assert r.ok is True, r.describe()

    after_rows = store.list_conducts()
    assert len(after_rows) == len(before_rows) == 1
    assert after_rows[0]["updated_at"] == before_rows[0]["updated_at"]
    assert len(store.events(after_rows[0]["conduct_id"], limit=500)) == \
        len(before_events)

    after_plan = plans.load(pid).to_dict()
    assert after_plan == before_plan, "编译器改了 plan —— 它应该纯读"
    assert plans.doc_id_for(pid) is None, (
        "编译不该给 plan 建文档;这份 plan 没有 experiment_id,"
        "任何文档写入都意味着碰到了 tmp_path 之外的地方")


def test_a_missing_plan_store_is_refused_loudly(registry):
    """没有默认库 —— 与 ``ConductStore`` 需要显式 ``db_path`` 同一条纪律。

    一个「不传就用全局实验库」的默认值,是测试污染真实数据那五次事故的共同入口。
    """
    with pytest.raises(ValueError, match="显式 plan_store"):
        compile_plan_to_conduct_spec("p1", registry=registry)


def test_a_broken_plan_store_raises_instead_of_saying_not_found(registry):
    """读库失败**抛**,不折叠成 ``PLAN_NOT_FOUND``。

    「这份方案不存在」与「plan 库读不到」指向的地方完全不同:一个去建方案,
    一个去查磁盘。本仓「读不到被折叠成一个具体的值」那族缺陷一天出现过五次。
    """
    class Broken:
        def load(self, plan_id):
            raise OSError("database is locked")

    with pytest.raises(OSError):
        compile_plan_to_conduct_spec("p1", plan_store=Broken(), registry=registry)
