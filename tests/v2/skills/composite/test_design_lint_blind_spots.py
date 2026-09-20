"""设计期 lint 的两个盲点(2026-08-25 修)——GUI 与 agent 两条轨共用同一份内核。

## 盲点一:``try`` 体里的 step 是隐形的

``validate_spec_payload`` 的三个遍历(``_walk_steps`` / ``_collect_names`` /
``_lint_exprs``)各自手写 if / loop / llm / human / agent 五种容器,**漏了 try**。
于是藏在 ``try`` 的 body / finally 里的步骤对整个设计期检查是不存在的:未知技能、
超包络的字面量、z-approach 硬错,一条都不报 —— 而运行期它照样打到硬件上。

``spec.walk_nodes`` 早就是「遍历一棵 spec 树」的唯一真源(loader 的缺失技能检查
已经在用它),它自己的注释就写着这件事。这一组钉住三个遍历都委托过去了。

## 盲点二:``$expr`` 提供的必填参数被读成「没给」

``SafetyGuard.check_parameter_bounds`` 末尾自带一份**必填参数在不在**的检查,而
设计期只能把**字面量**喂给它。于是一个步骤只要**同时**有字面量和 $expr 参数就凭空
报「Required parameter … is missing」——一条硬错,直接把保存拦下来。
(全 $expr 时反而不报,因为调用被 ``if literal`` 短路了。这种「多给一个参数反而
更糟」的非单调性正是它难被发现的原因。)
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

import pytest  # noqa: E402

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.webui.builder_api import validate_spec_payload  # noqa: E402


@pytest.fixture(scope="module")
def reg():
    r = SkillRegistry()
    r.discover("mast.skills.builtins", "mast.skills.composite")
    return r


def _spec(nodes, params=None):
    return {"name": "T", "description": "d", "safety_level": "confirm",
            "params": params or [], "nodes": nodes}


def _all_errors(rep) -> str:
    return "\n".join(list(rep.get("problems") or [])
                     + [e for s in rep.get("steps") or [] for e in (s.get("errors") or [])])


# ── 盲点一:try ────────────────────────────────────────────────────

@pytest.mark.parametrize("branch", ["body", "finally"])
def test_an_unknown_skill_inside_a_try_branch_is_caught(reg, branch):
    node = {"type": "try", "id": "t", "body": [], "finally": []}
    node[branch] = [{"type": "step", "id": "ghost", "skill": "NoSuchSkill",
                     "params": {}}]
    rep = validate_spec_payload(_spec([node]), registry=reg)
    assert rep["ok"] is False
    assert "NoSuchSkill" in _all_errors(rep)


def test_a_coarse_z_approach_inside_a_try_body_is_caught(reg):
    """这是最贵的那一个:它在运行期也会被 ctx.run 拒,但设计期不说的话,模型是
    在**做完前面几步硬件动作之后**才知道的。"""
    node = {"type": "try", "id": "t", "body": [
        {"type": "step", "id": "m", "skill": "MotorMove",
         "params": {"direction": "z-approach", "steps": 10}}], "finally": []}
    rep = validate_spec_payload(_spec([node]), registry=reg)
    assert rep["ok"] is False
    assert "z-approach" in _all_errors(rep)


def test_a_node_id_declared_inside_a_try_resolves_for_later_expressions(reg):
    """遍历漏掉 try 还有第二个症状:try 里声明的名字在别处被判成「悬空引用」。"""
    nodes = [
        {"type": "try", "id": "t",
         "body": [{"type": "step", "id": "probe", "skill": "GetBias", "params": {}}],
         "finally": []},
        {"type": "if", "id": "c", "cond": "probe['bias_v'] > 0", "then": []},
    ]
    rep = validate_spec_payload(_spec(nodes), registry=reg)
    assert "未知名字 'probe'" not in "\n".join(rep["problems"]), rep["problems"]


def test_the_walk_still_sees_the_containers_it_always_saw(reg):
    """修法是「委托到唯一真源」,不是「加一个 try 分支」——别把旧的丢了。"""
    for wrap in (
        {"type": "if", "id": "i", "cond": "True",
         "then": [{"type": "step", "id": "g", "skill": "NoSuchSkill", "params": {}}]},
        {"type": "loop", "id": "l", "mode": "repeat", "count": "1", "max_iter": 1,
         "body": [{"type": "step", "id": "g", "skill": "NoSuchSkill", "params": {}}]},
        {"type": "human", "id": "h", "message": "?",
         "routes": {"go": [{"type": "step", "id": "g", "skill": "NoSuchSkill",
                            "params": {}}]}},
    ):
        rep = validate_spec_payload(_spec([wrap]), registry=reg)
        assert "NoSuchSkill" in _all_errors(rep), wrap["type"]


# ── 盲点二:$expr 提供的必填参数 ──────────────────────────────────

def test_expr_supplied_required_params_do_not_read_as_missing(reg):
    """混合字面量 + $expr 是最常见的写法,它必须能过。

    这条测试同时是那个「精确剔除」过滤的绊线:``check_parameter_bounds`` 那句
    «Required parameter '…' is missing» 的措辞一旦改动,过滤会失配,这里就红。
    """
    nodes = [{"type": "step", "id": "scan", "skill": "ScanAt",
              "params": {"center_x_m": {"$expr": "x_m"},
                         "center_y_m": {"$expr": "y_m"},
                         "size_m": 1e-7}}]
    params = [{"name": "x_m", "type": "number", "required": True},
              {"name": "y_m", "type": "number", "required": True}]
    rep = validate_spec_payload(_spec(nodes, params), registry=reg)
    assert rep["ok"] is True, _all_errors(rep)
    assert "Required parameter" not in _all_errors(rep)


def test_a_genuinely_missing_required_param_is_still_reported(reg):
    """剔除要有边界:真没给就必须报,否则这个修复把一道检查整个关掉了。"""
    nodes = [{"type": "step", "id": "scan", "skill": "ScanAt",
              "params": {"center_x_m": 0.0}}]
    rep = validate_spec_payload(_spec(nodes), registry=reg)
    assert rep["ok"] is False
    errs = _all_errors(rep)
    assert "center_y_m" in errs and "size_m" in errs


def test_literal_bounds_are_still_enforced(reg):
    """第二条边界:剔除的只是「必填在不在」,**包络**一个字都不能松。"""
    nodes = [{"type": "step", "id": "b", "skill": "SetBias",
              "params": {"bias_v": 500.0}}]
    rep = validate_spec_payload(_spec(nodes), registry=reg)
    assert rep["ok"] is False, "500 V 的偏压被放过去了"


# ── registry 形参 ─────────────────────────────────────────────────

def test_an_explicit_registry_is_used_instead_of_the_module_global(reg):
    """agent 侧持有的是被注入的注册表 —— 校验和执行必须问同一个对象。"""
    empty = SkillRegistry()
    nodes = [{"type": "step", "id": "s", "skill": "GetBias", "params": {}}]
    assert validate_spec_payload(_spec(nodes), registry=reg)["ok"] is True
    rep = validate_spec_payload(_spec(nodes), registry=empty)
    assert rep["ok"] is False
    assert "GetBias" in _all_errors(rep)
