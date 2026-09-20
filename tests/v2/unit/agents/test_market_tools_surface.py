"""市场工具族：搜得到全集，但改不了自己的工具面。

这一组钉的同样是一个**权限判断**，不是一份名单：

* **放开的一半** —— 全部 agent 都能搜市场全集（订阅收窄了工具面，如果 agent 连
  「本机有没有这个能力」都查不到，它只会说「我没有工具」而不是「有一个能干这事
  的技能不在我手上」）。
* **接住的一半** —— agent 能到达的写路径只有 pending 推荐。改订阅是人面上的动作。

第二半比第一半重要得多：一份只测「大家都能搜」的测试，会在写工具被加上去之后
继续绿着。所以这里有一条 co_names 结构断言 + 一条行为断言，两条一起。

还有第三件事：**两条入口必须给同一族**。本仓在 ask-the-operator 上栽过 —— 同一个
能力群聊有、私聊没有，于是 agent 收得到回复却问不出问题。
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

import inspect  # noqa: E402

import pytest  # noqa: E402

from mast.agents._shared.market_tools import (  # noqa: E402
    MARKET_TOOL_NAMES,
    make_market_tools,
)
from mast.skills import subscription as sub  # noqa: E402

AGENTS = ("research_director", "literature", "experiment_design",
          "instrument_control", "data_processing", "paper_writing",
          "paper_review")

#: 一切能改订阅面的函数名。工具的字节码里出现任何一个，就是自我扩权通道。
_WRITE_ENTRYPOINTS = ("set_subscribed", "subscribe", "unsubscribe",
                      "reset_to_default", "resolve_recommendation")


def _names(agent: str) -> tuple[str, ...]:
    return tuple(t.name for t in make_market_tools(agent))


def _tool(agent: str, name: str):
    return next(t for t in make_market_tools(agent) if t.name == name)


# ─────────────────────────────────────────────────────────────────────────────
# 放开的一半
# ─────────────────────────────────────────────────────────────────────────────

def test_the_name_list_matches_what_the_factory_builds():
    """名单与实物对不上是本仓踩过的形状（清单说有、图上没有）。"""
    assert _names("test") == MARKET_TOOL_NAMES


@pytest.mark.parametrize("agent", AGENTS)
def test_every_agent_gets_the_same_market_surface(agent):
    assert _names(agent) == MARKET_TOOL_NAMES


def test_both_entry_points_wire_this_family():
    """群聊与私聊必须给同一族 —— 否则「你能不能做 X」的答案取决于他点开了哪个页面。"""
    from mast.agents.orchestrator import graph as orch

    orch_src = inspect.getsource(orch)
    assert "make_market_tools" in orch_src, "群聊那条线没挂市场工具"

    runtime_src = (Path(_MASTV2_ROOT) / "mast" / "core" / "runtime.py").read_text(
        encoding="utf-8")
    assert "make_market_tools" in runtime_src, "私聊那条线没挂市场工具"
    # 自检：扫描器确实看得见这种接线（conduct 是已知接了的那一族）
    assert "make_conduct_tools" in orch_src and "make_conduct_tools" in runtime_src


# ─────────────────────────────────────────────────────────────────────────────
# 接住的一半：agent 写不进订阅面
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool_name", MARKET_TOOL_NAMES)
def test_no_market_tool_can_write_the_subscription(tool_name):
    """结构断言：工具的字节码里不该出现任何订阅写入口。"""
    fn = _tool("ic", tool_name).func
    names = set(fn.__code__.co_names)
    offenders = sorted(names & set(_WRITE_ENTRYPOINTS))
    assert not offenders, (
        f"{tool_name} 直接调用了订阅写入口 {offenders} —— 一个能改自己工具面的 "
        "agent，在任何自主度档位下都等于没有工具面约束。")


def test_the_bytecode_scanner_can_see_the_pending_write():
    """闸门自检：扫描器必须在推荐工具里看见 ``add_recommendation``。

    看不见的话，上面那条「没有写入口」可能只是因为它什么都没看见。
    """
    fn = _tool("ic", "recommend_skill_subscription").func
    assert "add_recommendation" in set(fn.__code__.co_names), (
        "扫描器在推荐工具里连 add_recommendation 都看不见 —— 它坏了")


def test_no_write_shaped_tool_exists_in_the_family():
    assert not [n for n in MARKET_TOOL_NAMES
                if "subscribe" in n and n != "recommend_skill_subscription"]


def test_this_family_does_not_reach_the_workflow_menu():
    """结构性排除：不在任何导出表里 ⇒ register_workflow_tool_skills 够不到。

    一个 composite 步骤去「推荐订阅」会把一次工作流执行变成一串待人裁决的提示。
    """
    from mast.skills.composite.tool_skills import WORKFLOW_TOOL_EXPORTS

    for agent, attrs in WORKFLOW_TOOL_EXPORTS.items():
        import importlib
        mod = importlib.import_module(agent if "." in agent else
                                      f"mast.agents.{agent}.tools")
        for attr in attrs:
            exported = {getattr(t, "name", "") for t in (getattr(mod, attr, None) or [])}
            assert not (exported & set(MARKET_TOOL_NAMES)), (
                f"{agent}.{attr} 导出了市场工具 —— 它会被桥接成一个注册表技能")


# ─────────────────────────────────────────────────────────────────────────────
# 行为
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def market(subscription_store):
    """一个装了真目录的市场 + 干净订阅 store。"""
    from mast.agents.instrument_control.tools import discover_instrument_skills
    from mast.webui import builder_api

    reg = discover_instrument_skills()
    builder_api.set_live_registry(reg)
    builder_api.invalidate_catalog()
    yield reg
    builder_api.set_live_registry(None)
    builder_api.invalidate_catalog()


def test_search_sees_skills_the_agent_does_not_have(market):
    """市场是全集 —— 看不见未订阅的技能，这个工具就没有存在的理由。"""
    sub.set_subscribed({m.name for m in market.list_skills()} - {"SetBias"})
    out = _tool("ic", "search_skill_market").invoke({"query": "SetBias"})
    assert "SetBias" in out
    assert '"subscribed": false' in out.lower().replace("'", '"')


def test_search_marks_what_the_agent_already_has(market):
    out = _tool("ic", "search_skill_market").invoke({"query": "SetBias"})
    assert '"subscribed": true' in out.lower().replace("'", '"')


def test_search_says_when_it_truncated(market):
    """静默截断会被读成「市场里只有这些」。"""
    out = _tool("ic", "search_skill_market").invoke({"query": ""})
    assert "未列出" in out


def test_recommend_writes_only_pending(market):
    sub.set_subscribed({m.name for m in market.list_skills()} - {"SetBias"})
    before = sub.subscribed_names()

    out = _tool("ic", "recommend_skill_subscription").invoke(
        {"skill_name": "SetBias", "reason": "要调偏压"})

    assert "rec-" in out
    assert sub.subscribed_names() == before, "推荐就把订阅改了 —— 用户确认是摆设"
    assert [r["skill"] for r in sub.pending_recommendations()] == ["SetBias"]
    assert sub.pending_recommendations()[0]["by_agent"] == "ic", "署名没落上"


def test_recommend_refuses_a_skill_that_does_not_exist(market):
    out = _tool("ic", "recommend_skill_subscription").invoke(
        {"skill_name": "NoSuchSkillHere", "reason": "随便"})
    assert "没有" in out
    assert sub.pending_recommendations() == []


def test_recommend_says_so_when_already_subscribed(market):
    out = _tool("ic", "recommend_skill_subscription").invoke(
        {"skill_name": "SetBias", "reason": "随便"})
    assert "已经" in out
    assert sub.pending_recommendations() == []


def test_recommending_twice_is_merged_not_stacked(market):
    sub.set_subscribed({m.name for m in market.list_skills()} - {"SetBias"})
    first = _tool("ic", "recommend_skill_subscription").invoke(
        {"skill_name": "SetBias", "reason": "一次"})
    second = _tool("ic", "recommend_skill_subscription").invoke(
        {"skill_name": "SetBias", "reason": "又一次"})
    assert "已经有一条" in second
    assert len(sub.pending_recommendations()) == 1
    rec_id = sub.pending_recommendations()[0]["id"]
    assert rec_id in first and rec_id in second


def test_tools_degrade_instead_of_raising(subscription_store, monkeypatch):
    """目录不可用时要能诚实作答 —— 工具环不能因为一个查询崩掉。"""
    from mast.webui import builder_api

    monkeypatch.setattr(builder_api, "get_catalog",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _tool("ic", "search_skill_market").invoke({"query": "x"})
    assert "读不到" in out
