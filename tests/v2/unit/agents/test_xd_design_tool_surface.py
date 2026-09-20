"""experiment_design 的工具面 + 它的输出契约。

## 为什么这两件事在同一个文件里

因为它们是同一次断链的两半。2026-07-27 给 XD 补上 `create_plan` 修的是**工具**;
而它的系统提示词到 2026-08-14 还在教它「把 ExperimentPlan 形状的 JSON 放进对话,
supervisor 会解析进 `experiment_plan` 字段」—— 那个解析器**不存在**
(`agents/orchestrator/` 全目录零命中)。模型跟的是提示词,不是工具清单,所以工具
通了而方案照样只活在对话里、并在 2000 字符处被截断。

一半没落地的修复看起来和落地了一模一样。所以这里同时钉:
  * 授出去的**是**哪些工具(遍历真实构造,不是读一份名单),
  * 提示词**说的**是不是同一件事。

## 判据的选法

工具面**不**按「等于某个硬编码全集」判 —— 那种断言每次加工具都要改,改的人会顺手
把期望值抄成现状,于是它只会证明「今天等于今天」。这里钉的是**关键成员与关键排除**:
能不能落库(create_plan)、有没有知识可用(D6 七件)、能不能给自己盖章(approve_plan)、
会不会拿到一个长得像可以直接填的数(get_literature_parameters)。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/agents/test_xd_design_tool_surface.py -q
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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents._shared.meta_tools import (  # noqa: E402
    DESIGN_TOOL_NAMES,
    META_TOOL_NAMES,
    make_meta_tools,
)

#: D6 增授的知识工具。「要出主意的那个 agent 恰好是知识工具最少的那个」是这一族
#: 存在的理由 —— 少一件就退回那个不对称。
_KNOWLEDGE_TOOLS = (
    "query_knowledge", "get_workflow_advice", "get_measurement_template",
    "get_skill_guidance", "search_deep_reference", "read_reference_section",
    "get_fault_diagnosis",
)


def _xd_tools() -> dict:
    """XD 实际拿到的 meta 工具对象,按 runtime 的过滤方式**真的构造一遍**。

    只读名单会漏掉一整类错误:名单里写了一个工厂根本不产的名字。这里遍历真实构造,
    所以那种名字会以「不在结果里」的形式当场暴露。
    """
    names = set(DESIGN_TOOL_NAMES)
    return {t.name: t for t in make_meta_tools(lambda: {}) if t.name in names}


# ── 名单本身 ────────────────────────────────────────────────────────────────

def test_every_design_name_is_a_real_meta_tool() -> None:
    """名单里的每个名字都要能被 make_meta_tools() 真的造出来。

    否则「授予」是一句空话:过滤器按名字取,取不到就静默少一件,谁也不会知道。
    """
    missing = sorted(set(DESIGN_TOOL_NAMES) - {t.name for t in make_meta_tools(lambda: {})})
    assert not missing, f"DESIGN_TOOL_NAMES 里这些名字没有对应的工具: {missing}"
    assert set(DESIGN_TOOL_NAMES) <= set(META_TOOL_NAMES), (
        "DESIGN_TOOL_NAMES 必须是 META_TOOL_NAMES 的子集 —— "
        "XD 的工具是从 IC 的 meta 全集里过滤出来的,不在全集里的名字永远取不到")


def test_no_duplicate_names_in_the_constant() -> None:
    """名单是 tuple 拼出来的,重复不会报错、只会让人数错数。"""
    assert len(DESIGN_TOOL_NAMES) == len(set(DESIGN_TOOL_NAMES))


# ── 关键成员:能落库、有知识可用 ────────────────────────────────────────────

def test_design_agent_can_persist_a_plan() -> None:
    """没有 create_plan,XD 的全部产出就是对话文本:plans 表 0 行,
    「实验方案」永远是空的 —— 2026-07-27 取证的那份 28 步方案就是这么丢的。"""
    tools = _xd_tools()
    assert "create_plan" in tools
    for readback in ("list_plans", "get_plan_progress"):
        assert readback in tools, f"{readback} 不在:XD 存完就再也看不见自己存了什么"


@pytest.mark.parametrize("name", _KNOWLEDGE_TOOLS)
def test_design_agent_has_the_knowledge_tool(name: str) -> None:
    tools = _xd_tools()
    assert name in tools, (
        f"{name} 没给 XD —— 要给实验出主意的 agent 又回到了知识工具最少的那个")


def test_design_agent_knows_which_tip_is_installed() -> None:
    """方案要按针尖定(钨腐蚀针和 qPlus 能承受的处理完全不同)。"""
    tools = _xd_tools()
    assert "get_current_tip" in tools and "list_tips" in tools


# ── 关键排除:不批自己的方案、不碰表面、不拿现成的数 ────────────────────────

def test_the_designer_may_approve_but_the_gate_moved_rather_than_vanished() -> None:
    """**这条断言 2026-08-20 被翻了过来 —— 翻过来比删掉重要。**

    从前：`approve_plan` 不给 XD，理由是「一个 agent 批准自己写的方案 = 审批
    这道闸门被取消了，而且看不出来」。那个担心是对的；**它选的实现方式不对**。

    「不给工具」把一道安全边界建在了「模型想不到去做」上面：同一个能力在 IC
    手里有、在 XD 手里没有，而两者都是模型；真要绕，换一句话交给 IC 就行了。
    更要紧的是它顺带封死了另一件事——夜里没有人在场时，什么也批不了，
    而那正是全自动实验室要解决的问题。

    现在闸门在服务端，不在工具面：一份 plan 被批准之后要真的开跑，还得过
    conduct 的 `approve`，那里由**自主度策略**裁决谁点头算数（attended 档下
    agent 批不动，supervised 档下批了也要过撤销窗）。所以这条测试改成钉
    **闸门还在**，而不是钉**工具不在**。

    要推翻这次改动，需要回答的是：conduct 的自主度默认档还是不是 attended，
    以及 approve 路由是不是仍然按 `by` 署名分流。任何一条不成立，这里就该
    翻回去。
    """
    assert "approve_plan" in _xd_tools(), (
        "approve_plan 又被从 XD 手里拿走了。如果这是有意的，请先回答："
        "把关到底靠什么——如果答案还是「不给工具」，那它防不住换个 agent 去调。")

    # 闸门本体：自主度默认必须是最严档，且 approve 要按署名分流。
    from mast.conduct.autonomy import DEFAULT_LEVEL, who_may_approve

    assert DEFAULT_LEVEL == "attended", (
        "conduct 自主度的默认档不再是 attended —— 那道真正的闸门松了，"
        "而工具面这一侧已经按「服务端把关」放开了。两边同时放开等于没有闸门。")
    assert not who_may_approve("attended", by="agent:experiment_design").allowed
    assert who_may_approve("attended", by="用户").allowed


def test_the_designer_may_drive_a_plan_but_not_the_instrument() -> None:
    """执行工具同样不再按角色裁（同上一条的理由）。

    真正的边界在别处，而且**没有**跟着放开：推进一份 plan 是往记录里写一行，
    动仪器要经 ExecutionContext 的完整管道，而 XD 手里一个硬件技能都没有。
    这条断言钉的就是后者 —— 放开的是账本，不是仪器。
    """
    tools = _xd_tools()
    for name in ("advance_plan", "pause_plan", "resume_plan"):
        assert name in tools, f"{name} 又被按角色裁掉了"
    # 边界仍在：设计 agent 手里没有任何会驱动仪器的东西。
    for name in ("mark_area_used", "record_coarse_move", "register_tip"):
        assert name not in tools, f"{name} 会改变别人读到的世界，不该在设计 agent 手里"


def test_literature_prior_numbers_are_not_handed_to_the_designer() -> None:
    """`get_literature_parameters` 返回 p25/p50/p75 —— 一个**长得像可以直接填**的
    数字,而「这不是权威默认」只写在返回文本里。那是说服,不是结构。

    要拦住「顺手把 p50 抄进 params」,靠的是**手里没有这个工具**,不是提示词里多写
    一句别抄。文献数值应经 literature agent 的证据包带着出处进来。
    """
    assert "get_literature_parameters" not in _xd_tools()


def test_design_agent_holds_no_tool_that_writes_the_surface() -> None:
    """设计 agent 不碰表面:`mark_area_used` 写实验记录,
    `record_coarse_move` 推进坐标代次 —— 两者都会改变别人读到的世界。
    `register_tip` 会清掉学习标定,而换针是仪器旁的物理事件。
    """
    tools = _xd_tools()
    for name in ("mark_area_used", "record_coarse_move", "register_tip",
                 "update_tip", "remove_current_tip", "spawn_background_task"):
        assert name not in tools, f"{name} 落到了设计 agent 手里"


# ── 输出契约:提示词说的和工具做的必须是同一件事 ────────────────────────────

def _prompt() -> str:
    from mast.agents.experiment_design.prompts import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _prompt_source() -> str:
    return (Path(_MASTV2_ROOT) / "mast" / "agents" / "experiment_design"
            / "prompts.py").read_text(encoding="utf-8")


def test_the_dead_parser_is_no_longer_advertised() -> None:
    """`experiment_plan` 字段活着,但类型是 DocRef 指针(create_plan 写的);
    **没有任何 supervisor 代码解析它**。教模型「写进对话等人解析」= 教它把方案扔掉。
    """
    p = _prompt()
    for dead in ("supervisor will parse", "state field"):
        assert dead not in p, f"提示词里还留着那个不存在的解析器: {dead!r}"


def test_the_prompt_names_the_only_way_to_produce_a_plan() -> None:
    p = _prompt()
    assert "create_plan" in p
    assert "唯一" in p, "没有一句话说清「产出方案只有这一条路」"
    # 说明「写在对话里会怎样」,而不只是「请调用工具」—— 后者是要求,前者是理由。
    assert "2000" in p, "没说长内容会被 2000 字符截断 —— 那正是当年丢方案的机制"
    assert "压缩" in p, "没说会被上下文压缩摘掉"


def test_the_prompt_leaves_a_refutable_note() -> None:
    """删一条防护/改一句契约时,要留下「要推翻我需要回答什么」。

    今天「supervisor 会解析」是假话;哪天真有人写了解析器,这行注记会告诉他
    该回来改哪一段 —— 而不是让他对着一段没有来由的措辞猜。
    """
    src = _prompt_source()
    assert "要推翻本条" in src, "prompts.py 里没有可反驳注记"
    assert "解析 `experiment_plan` 的代码" in src, (
        "注记没说清推翻它需要拿出什么证据(supervisor 里的解析代码 + 行号)")
    # 注记必须在**注释**里,不能进模型上下文:它是给下一个改这段的人看的。
    assert "要推翻本条" not in _prompt(), "可反驳注记漏进了 SYSTEM_PROMPT"


def test_the_prompt_advertises_the_knowledge_surface() -> None:
    """工具给了而提示词不说,等于没给 —— 「明明有却用不上」的老形状。"""
    p = _prompt()
    for name in _KNOWLEDGE_TOOLS:
        assert name in p, f"{name} 授了但提示词里没提,模型不会去用"


def test_the_prompt_does_not_describe_an_envelope_nothing_accepts() -> None:
    """`create_plan(name, goal, phases)` 收的就这三个参数。

    提示词若还在教一个 `sample_id / research_question / steps / material_context`
    的信封,模型会照着拼一个工具收不下的东西 —— 那是把一句假话换成另一句。
    """
    p = _prompt()
    assert "material_context" not in p
    for key in ("phases", "success_criteria", "on_fail"):
        assert key in p, f"契约里没写 {key},而 PlanPhase 有这个字段"
