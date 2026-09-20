"""起草期的拒绝与建议 —— 「官方技能优先」在结构上的那一半。

提示词里那句「官方覆盖得了的用官方的」是说给模型听的,而说服模型是本仓已经失败
过四次的做法。这一组钉的是**不靠说服**的那几条:

* **纯别名硬拒** —— 给官方技能套个壳不是新能力,直接指回本体;
* **重合出 hint 但不拒** —— 覆盖不等于等价,硬拒会把「我确实要个不一样的」也拒掉。
  照搬 DP 的 ``_py_hint``:路由,不阻拦。这条测试同时防它退化成硬拦。
* 节点白名单 / 循环封顶 / 关闭名单 / 撞名 —— agent 轨专有的四条。

每一条负例都断言**具体那句话**,不是「ok 是 False」:一个因为别的原因失败的草稿
也能让 ``ok is False`` 恒真,而那样的负例测试是空的。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from _forge_fixtures import (  # noqa: E402
    StubCtx,
    call,
    fresh_registry,
    spec_with,
    store,
    tools,
    two_step_spec,
)


def _draft(spec, st=None, reg=None):
    return call(tools(reg=reg, ctx=StubCtx(), st=st)["draft_composite"],
                spec_json=__import__("json").dumps(spec))


def _problems(rep) -> str:
    return "\n".join(rep.get("problems") or [])


# ── 基线:一份好草稿要真的通过 ────────────────────────────────────────
#
# 先钉正例。全是负例的一组测试里,一个「什么都拒」的实现同样全绿。

def test_a_real_two_step_composite_validates_clean(tmp_path):
    rep = _draft(two_step_spec(), st=store(tmp_path))
    assert rep["ok"] is True, _problems(rep)
    assert rep["problems"] == []


def test_required_params_supplied_by_expr_are_not_reported_missing(tmp_path):
    """必填参数用 ``$expr`` 给,不该被读成「没给」。

    ``check_parameter_bounds`` 末尾自带一份必填检查,而设计期只能把**字面量**喂
    给它 —— 于是一个步骤只要同时有字面量和 $expr 参数就凭空报缺参(全 $expr 时
    反而不报,因为调用被 ``if literal`` 短路了)。那是一条硬错,直接拦下保存。
    这条测试钉住修复:``two_step_spec`` 的 ScanAt 正是这个混合形状。
    """
    rep = _draft(two_step_spec(), st=store(tmp_path))
    assert "Required parameter" not in _problems(rep)
    assert "缺少必填参数" not in _problems(rep)


# ── 官方优先:硬拒的那一条 ──────────────────────────────────────────

def test_a_single_step_spec_is_refused_as_a_wrapper(tmp_path):
    spec = spec_with([{"type": "step", "id": "a", "skill": "GetBias", "params": {}}],
                     name="JustGetBias")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "套壳" in _problems(rep)
    assert "GetBias" in _problems(rep), "拒绝时要指回本体的名字,不然模型不知道改调谁"


def test_a_two_step_spec_is_not_a_wrapper(tmp_path):
    """负例要有边界:多一步就不是套壳了,别把正当的组合一起拒掉。"""
    rep = _draft(two_step_spec(), st=store(tmp_path))
    assert "套壳" not in _problems(rep)


def test_a_single_step_wrapped_in_a_loop_is_not_a_wrapper(tmp_path):
    """一个 step + 循环 = 比本体多做了事,不是套壳。"""
    spec = spec_with([{"type": "loop", "id": "lp", "mode": "repeat", "count": "3",
                       "max_iter": 3, "var": "i", "body": [
                           {"type": "step", "id": "a", "skill": "GetBias",
                            "params": {}}]}], name="PollBias")
    rep = _draft(spec, st=store(tmp_path))
    assert "套壳" not in _problems(rep)
    assert rep["ok"] is True, _problems(rep)


# ── 官方优先:建议的那一条(**不**拒) ──────────────────────────────

def test_overlap_with_an_existing_composite_is_a_hint_not_a_refusal(tmp_path):
    """已经有个现成的在做这件事 → 提醒,但仍然放行。

    这条测试有两半,第二半更重要:``ok`` 必须仍然是 True。DP 的 ``_py_hint`` 第
    一版就是命中即拒,后来撤销了 —— 覆盖不等于等价(参数、判据、失败处置都可能
    不同),硬拒会把「我确实需要一个不一样的」也一起拒掉。
    """
    st = store(tmp_path)
    reg = fresh_registry()
    ts = tools(reg=reg, ctx=StubCtx(), st=st)
    import json
    first = call(ts["save_composite"], spec_json=json.dumps(two_step_spec("Existing")))
    assert first["ok"] is True, first

    same_steps = two_step_spec("MyOwnVersion")
    rep = call(ts["draft_composite"], spec_json=json.dumps(same_steps))
    assert rep["ok"] is True, _problems(rep)          # ← 放行
    assert any("Existing" in h for h in rep["hints"]), rep["hints"]   # ← 但提醒了


def test_a_novel_composite_gets_no_overlap_hint(tmp_path):
    """建议也要有边界:没有重合就别乱提示,否则提示会被当噪声忽略。"""
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "GetCurrent", "params": {}},
    ], name="ReadTwoThings")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["hints"] == []


# ── agent 轨白名单 ────────────────────────────────────────────────

# 每个节点的形状本身都是**合法的**(GUI 轨接受它们,见下一条测试),这样负例隔离
# 的是白名单那一条,而不是顺带被一个形状错误拒掉 —— 后者的话把白名单整段删掉,
# 这几个用例照样绿。
@pytest.mark.parametrize("node,why", [
    ({"type": "human", "id": "h", "message": "?", "routes": {"go": []}},
     "human"),
    ({"type": "llm", "id": "l", "mode": "route", "responsibility": "r",
      "routes": {"go": []}, "escape": "go"}, "llm"),
    ({"type": "agent", "id": "g", "agent": "data_processing", "task": "t"},
     "agent"),
])
def test_llm_human_and_agent_nodes_are_refused_on_the_agent_track(node, why, tmp_path):
    spec = two_step_spec("WithExotic")
    spec["nodes"] = spec["nodes"] + [node]
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "agent 轨不允许这些节点类型" in _problems(rep)
    assert why in _problems(rep)


def test_the_gui_track_still_accepts_the_nodes_the_agent_track_refuses(tmp_path):
    """白名单只加在 agent 轨。编辑器里人是在场的,human 节点是功能不是死锁。"""
    from mast.skills.composite.spec import CompositeSpec
    spec = two_step_spec("WithHuman")
    spec["nodes"] = spec["nodes"] + [
        {"type": "human", "id": "h", "message": "继续?", "routes": {"go": []}}]
    assert CompositeSpec.from_dict(spec).validate() == []


# ── 触硬件的循环必须封顶 ──────────────────────────────────────────

def test_a_hardware_loop_without_max_iter_is_refused(tmp_path):
    spec = spec_with([{"type": "loop", "id": "lp", "mode": "repeat", "count": "3",
                       "var": "i", "body": [
                           {"type": "step", "id": "a", "skill": "GetBias",
                            "params": {}}]}], name="Unbounded")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "max_iter" in _problems(rep)


def test_a_hardware_loop_above_the_cap_is_refused(tmp_path):
    spec = spec_with([{"type": "loop", "id": "lp", "mode": "repeat", "count": "3",
                       "max_iter": 5000, "var": "i", "body": [
                           {"type": "step", "id": "a", "skill": "GetBias",
                            "params": {}}]}], name="TooMany")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "5000" in _problems(rep)


def test_a_pure_computation_loop_needs_no_cap(tmp_path):
    """封顶管的是**打到硬件**的循环。纯计算循环没有那个代价,别一起管。"""
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "GetCurrent", "params": {}},
        {"type": "loop", "id": "lp", "mode": "repeat", "count": "3", "var": "i",
         "body": [{"type": "set", "id": "s", "var": "n", "value": "0"}]},
    ], name="CountOnly")
    rep = _draft(spec, st=store(tmp_path))
    assert "max_iter" not in _problems(rep)


# ── 关闭名单不能靠包一层洗白 ──────────────────────────────────────

def test_a_disabled_skill_cannot_be_laundered_through_a_composite(tmp_path, monkeypatch):
    """被关掉的高级能力包进 spec 也打不开。

    ``build_instrument_skill_tools`` 只按名字把技能从工具表里摘掉(「看不见 =
    用不了」),而 spec 的子步走 ``ExecutionContext.run`` —— 那条路**不查这张
    名单**。所以包一层就是一条现成的洗白通道,必须在起草期拦住。
    """
    import mast.agents._shared.skill_forge_tools as mod
    monkeypatch.setattr(mod, "_disabled_names", lambda: frozenset({"GetCurrent"}))
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "GetCurrent", "params": {}},
    ], name="Laundering")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "GetCurrent" in _problems(rep)
    assert "关闭" in _problems(rep)


def test_a_disabled_skill_nested_in_a_try_body_is_still_caught(tmp_path, monkeypatch):
    """藏在 try 体里也要抓到 —— 这正是那次遍历漏洞的形状。"""
    import mast.agents._shared.skill_forge_tools as mod
    monkeypatch.setattr(mod, "_disabled_names", lambda: frozenset({"GetCurrent"}))
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "try", "id": "t",
         "body": [{"type": "step", "id": "b", "skill": "GetCurrent", "params": {}}],
         "finally": []},
    ], name="HiddenInTry")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "GetCurrent" in _problems(rep)


# ── 撞名 ─────────────────────────────────────────────────────────

def test_colliding_with_a_builtin_skill_name_is_refused(tmp_path):
    """同名会让后续的删除动作把内置技能一起卸掉(这是它的原始事故形状)。"""
    spec = two_step_spec("SetBias")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "SetBias" in _problems(rep)
    assert "占用" in _problems(rep)


def test_colliding_with_a_seed_template_name_is_refused(tmp_path):
    """种子是 `if not exists: save` —— 删掉之后下次启动它会带着模板内容复活。"""
    from mast.skills.composite.templates import builtin_templates
    spec = two_step_spec(builtin_templates()[0].name)
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "模板种子" in _problems(rep)


# ── 未知技能 / z-approach 仍由官方内核负责 ────────────────────────

def test_an_unknown_skill_is_refused(tmp_path):
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "NoSuchSkillAtAll", "params": {}},
    ], name="Ghost")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "NoSuchSkillAtAll" in _problems(rep)


def test_coarse_z_approach_is_a_hard_design_time_error(tmp_path):
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "MotorMove",
         "params": {"direction": "z-approach", "steps": 10}},
    ], name="SneakyApproach")
    rep = _draft(spec, st=store(tmp_path))
    assert rep["ok"] is False
    assert "z-approach" in _problems(rep)


def test_a_spec_with_no_steps_is_refused(tmp_path):
    """「成功地什么都没做」是最坏的一种成功。

    `CompositeSpec.validate` 对空 nodes 是满意的(结构上确实没毛病),所以一份什么
    都不做的 spec 本来能一路存进去、注册成技能、被调用、返回成功。
    """
    rep = _draft(spec_with([], name="DoesNothing"), st=store(tmp_path))
    assert rep["ok"] is False
    assert "一个 step 都没有" in _problems(rep)


def test_the_syntax_reference_is_sent_on_demand_not_in_the_schema(tmp_path):
    """语法速查跟着**失败**走,不跟着 schema 走。

    核心包工具的 schema 每一次模型调用都在上下文里,而这段东西只在「真的要造技能、
    而且草稿还没写对」时有用。放进 docstring 等于让每一轮都替那件偶尔发生的事付钱。
    """
    import json as _json
    from mast.agents._shared.skill_forge_tools import SPEC_SYNTAX
    ts = tools(ctx=StubCtx(), st=store(tmp_path))

    asked = call(ts["draft_composite"], spec_json="?")
    assert asked["syntax"] == SPEC_SYNTAX

    bad = call(ts["draft_composite"], spec_json=_json.dumps(
        spec_with([{"type": "step", "id": "a", "skill": "Ghost", "params": {}}])))
    assert bad["ok"] is False and "syntax" in bad

    good = call(ts["draft_composite"], spec_json=_json.dumps(two_step_spec()))
    assert good["ok"] is True and "syntax" not in good, "成功回执还在付语法的钱"

    # 而它**不在** schema 里 —— 这是省下来的那一半,没有它上面三条都成立但没意义。
    assert SPEC_SYNTAX not in (ts["draft_composite"].description or "")


def test_bad_json_is_a_problem_not_a_crash(tmp_path):
    rep = call(tools(ctx=StubCtx(), st=store(tmp_path))["draft_composite"],
               spec_json="{not json")
    assert rep["ok"] is False
    assert "JSON" in _problems(rep)
