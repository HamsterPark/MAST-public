"""conduct 工具族：全部 agent 都有，把关在服务端。

## 这一组钉的是一个**权限哲学**，不是一份名单

2026-08-20 之前，本仓靠「按角色裁工具面」把关：谁能批准方案、谁能推进计划，
在建图时决定。这一组测试钉的是那次改变的两半——缺了任何一半，改动就变成了
纯粹的放松：

* **放开的一半**：六个 agent 拿到的 conduct 工具**完全一样**。工具面不再是
  安全边界，所以它不该长得像一个。
* **接住的一半**：把关真的在服务端。自主度默认最严、工具本身不驱动仪器、
  引擎关着时只读不写。

第二半比第一半重要得多。一份只测「大家都有」的测试，会在闸门被悄悄拆掉之后
继续绿着。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.agents._shared.conduct_tools import (  # noqa: E402
    CONDUCT_TOOL_NAMES,
    make_conduct_tools,
)

AGENTS = ("research_director", "literature", "experiment_design",
          "instrument_control", "data_processing", "paper_writing",
          "paper_review")


def _names(agent: str) -> tuple[str, ...]:
    return tuple(t.name for t in make_conduct_tools(agent))


# ── 放开的一半 ────────────────────────────────────────────────────

def test_the_name_list_matches_what_the_factory_actually_builds():
    """名单与实物对不上，是本仓踩过的形状（清单说有、图上没有）。"""
    assert _names("test") == CONDUCT_TOOL_NAMES


@pytest.mark.parametrize("agent", AGENTS)
def test_every_agent_gets_the_same_conduct_surface(agent):
    assert _names(agent) == CONDUCT_TOOL_NAMES, (
        f"{agent} 拿到的 conduct 工具与别的 agent 不同。"
        "按角色裁工具面这件事已经被裁决掉了 —— 把关在服务端，"
        "不同的工具面只会让「谁能做什么」重新变成一份要人肉维护的清单。")


def test_the_orchestrator_hands_them_to_every_wired_agent():
    """图那一侧也要真的发出去 —— 名单对了而没接线，是另一个已知形状。

    数目从 ``_AGENT_NAMES`` 派生，不再写死一个 6：写死的那个数，在 2026-08-21 加
    research_director 时正好把这条测试变成了「加一个 agent 就红一次」的噪音，
    而它本来要拦的是「某个 agent 没接线」。派生之后两件事都还在管。
    """
    from mast.agents.orchestrator.graph import _AGENT_NAMES

    assert set(AGENTS) == set(_AGENT_NAMES), (
        "这份名单和编排器的 _AGENT_NAMES 不一致 —— 先对齐再谈接线")
    src = (Path(_MASTV2_ROOT) / "mast" / "agents" / "orchestrator"
           / "graph.py").read_text(encoding="utf-8")
    assert src.count('extra_tools=_shared(') == len(_AGENT_NAMES), (
        f"orchestrator 里不是全部 {len(_AGENT_NAMES)} 个 agent 都走 "
        "_shared(...) 拿 conduct 工具了")
    for agent in AGENTS:
        assert f'_shared("{agent}")' in src, f"{agent} 没接上 conduct 工具"


def test_the_private_chat_path_gets_them_too():
    """两条入口给的工具面必须一样。

    否则「能不能看多天实验跑到哪了」取决于用户点开的是哪个页面 —— 本仓刚在
    ask-the-operator 上栽过这个形状（群聊有、私聊没有，于是 agent 收得到回复
    却问不出问题）。
    """
    src = (Path(_MASTV2_ROOT) / "mast" / "core" / "runtime.py").read_text(encoding="utf-8")
    i = src.find("def _chat_agent_extra_tools")
    assert i > 0
    body = src[i: i + 4000]
    assert "make_conduct_tools" in body, "私聊那条线没有 conduct 工具"


# ── 接住的一半（真正的闸门） ──────────────────────────────────────

def test_the_default_autonomy_is_still_the_strictest():
    """工具面放开的前提是这道闸没松。两边同时放开等于没有闸门。"""
    from mast.conduct.autonomy import DEFAULT_LEVEL

    assert DEFAULT_LEVEL == "attended"


def test_no_conduct_tool_drives_the_instrument_directly():
    """结构断言：这一族不 import 任何执行面。

    工具只往意图队列里放东西，Director 在自己的 tick 里消费。一旦这里出现
    ExecutionContext / hold_for_skill / 技能注册表，就等于开了第二条驱动仪器
    的路 —— 而那正是 2026-07-28 审计里「共用一个 ConnectionPool 却没有仲裁」
    的形状。
    """
    import ast
    import inspect

    from mast.agents._shared import conduct_tools as CT

    tree = ast.parse(inspect.getsource(CT))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    blob = " ".join(sorted(imported))
    for forbidden in ("execution_context", "ExecutionContext", "instrument_lock",
                      "hold_for_skill", "connection"):
        assert forbidden not in blob, (
            f"conduct 工具里出现了 {forbidden!r} —— 这一族只递意图，不驱动仪器。")


def test_reads_work_with_the_engine_off_and_writes_say_why_they_do_not():
    """引擎关着时：能读，不能写，而且**说得出为什么**。

    设置页那句「关掉它不会关掉读端点」在此之前只是一句承诺。这条测试是它的
    实现的证据 —— 同时也钉住另一半：写操作在引擎关着时必须**明说**原因，
    而不是回一个看起来正常的空结果。
    """
    from mast.conduct import service as svc_mod

    old = svc_mod.get_service()
    svc_mod.set_service_for_test(None)
    try:
        tools = {t.name: t for t in make_conduct_tools("test")}
        r = json.loads(tools["conduct_status"].invoke({"conduct_id": ""}))
        assert r["ok"] is True and r["engine_running"] is False

        w = json.loads(tools["conduct_post_op"].invoke(
            {"conduct_id": "x", "op": "start"}))
        assert w["ok"] is False
        assert "cd_enabled" in w["reason"] or "没在跑" in w["reason"], (
            "写操作在引擎关着时没说清为什么 —— 一个「什么都没发生」的成功响应"
            "比一次失败更难查。")
    finally:
        svc_mod.set_service_for_test(old)


def test_post_op_only_accepts_a_closed_set():
    tools = {t.name: t for t in make_conduct_tools("test")}
    r = json.loads(tools["conduct_post_op"].invoke(
        {"conduct_id": "x", "op": "自己发明的操作"}))
    assert r["ok"] is False


def test_the_tools_sign_who_they_are():
    """署名必须带 agent 名 —— 审计流要答得出「这一步是谁点的头」。"""
    import inspect

    from mast.agents._shared import conduct_tools as CT

    src = inspect.getsource(CT.make_conduct_tools)
    assert 'f"agent:{agent_name' in src, (
        "工具不再按 agent 署名。自主度策略靠 by 的形状分流，"
        "署名丢了之后 agent 的批准会被当成人的批准。")
