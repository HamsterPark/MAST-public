"""科研策划 RD（research_director）—— 决策链上半环的接线与边界。

## 这个文件钉的是三件事，不是一件

一个新 agent 的失败模式在本仓有固定形状，三种都不一样：

1. **零件都对，线没接。** 建得出图、工具也齐，但编排器路由不到它 / 产物没人读 /
   提示词没进注册表。这一类看起来完全正常，直到有人问「为什么它从来不出现」。
2. **边界只写在提示词里。** 「它不碰仪器」如果只是一句话，那它就只是一句话。
   RD 能进后台白名单，靠的是**工具面里没有执行面**这个结构事实 —— 所以这里用
   结构断言钉它，而不是相信 docstring。
3. **判据被折叠成一个值。** campaign 层的读操作必须分得开「库打不开」「还没有
   纲领」「有但你没给 id」—— 这三件事在本仓已经被折叠过很多次。

## 跑法

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/agents/test_research_director.py -q
"""
from __future__ import annotations

import ast
import inspect
import json
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
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from mast.agents._shared import artifact_channel as ac  # noqa: E402
from mast.agents._shared.campaign_tools import (  # noqa: E402
    CAMPAIGN_TOOL_NAMES,
    make_campaign_tools,
)
from mast.agents._shared.conduct_tools import CONDUCT_TOOL_NAMES  # noqa: E402
from mast.agents.research_director import tools as rd_tools  # noqa: E402
from mast.agents.research_director.graph import build as build_rd  # noqa: E402
from mast.agents.research_director.prompts import SYSTEM_PROMPT  # noqa: E402
from tests.v2.toolcall import tool_call, tool_text  # noqa: E402

AGENT = "research_director"


def _fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _names(tools) -> tuple[str, ...]:
    return tuple(str(getattr(t, "name", "")) for t in tools)


@pytest.fixture()
def v2_store(tmp_path, monkeypatch):
    """把 v2 记录库指到 tmp。

    ``v2_experiment_db_path`` 先看 ``MAST_DATA_DIR`` 再看 ``project_root()``，
    所以设一个就够 —— 而**必须**设：不设的话这些测试会读写用户真实的
    ``experiments/mast_experiments_v2.db``，而本仓「测试污染真实数据」已经犯过五次。
    """
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.agents._shared.data_paths import v2_experiment_db_path

    assert str(tmp_path) in str(v2_experiment_db_path()), "重定向没生效，不许继续"
    return tmp_path


# ════════════════════════════════════════════════════════════════════
# 1. 图建得出来，工具面与实物一致
# ════════════════════════════════════════════════════════════════════

def test_the_graph_builds_with_a_fake_model():
    agent = build_rd(None, model=_fake_model())
    assert agent.name == AGENT


def test_the_graph_builds_without_a_buffer_and_without_a_context_provider():
    """RD 与 instrument_control 不同类：它不需要 buf，也不需要 context_provider。

    编排器对 IC 会在缺 context_provider 时直接 raise；对 RD 不该有任何这类前置，
    否则「加一个不碰仪器的 agent」会连带把硬件依赖搬进来。
    """
    assert build_rd(model=_fake_model()) is not None


def test_the_tool_name_list_matches_what_build_tools_actually_returns():
    """名单说有、图上没有 —— 本仓的固定形状，所以名单要对着实物核。"""
    built = _names(rd_tools.build_tools(None))
    assert set(rd_tools.AGENT_TOOL_NAMES) <= set(built)
    assert set(rd_tools.AGENT_TOOL_NAMES) == set(CAMPAIGN_TOOL_NAMES)
    # handoff 是工具面的一部分，但不属于「领域工具」这份名单
    assert {n for n in built if n.startswith("handoff_to_")} == {
        "handoff_to_experiment_design", "handoff_to_literature",
        "handoff_to_supervisor"}


def test_the_campaign_family_matches_its_own_name_list():
    assert _names(make_campaign_tools("test")) == CAMPAIGN_TOOL_NAMES


def _compiled_tool_names(agent) -> set[str]:
    """编译好的子图里 ToolNode 真正持有的工具名。

    读的是编译产物，不是 ``build_tools`` 的返回 —— 后者只能证明「装配函数打算给
    什么」，而 standalone 过滤发生在装配之后。用一个软兜底（拿不到就跳过断言）会
    把这条测试变成「看着在防护其实没有」，所以这里拿不到就直接失败。
    """
    node = getattr(agent, "nodes", {}).get("tools")
    assert node is not None, "编译出来的图里没有 tools 节点"
    by_name = getattr(getattr(node, "bound", None), "tools_by_name", None)
    assert by_name, "拿不到 ToolNode 的工具表 —— 断言无从谈起，不许软过"
    return set(by_name)


def test_standalone_strips_the_handoffs_and_keeps_the_domain_tools():
    """私聊里没有可交接的对象；剥掉 handoff 不该顺手剥掉本职工作。"""
    full = _compiled_tool_names(build_rd(None, model=_fake_model()))
    lone = _compiled_tool_names(build_rd(None, model=_fake_model(), standalone=True))

    assert {n for n in full if n.startswith("handoff_to_")}, "群聊那份就没有 handoff"
    assert not {n for n in lone if n.startswith("handoff_to_")}
    assert set(CAMPAIGN_TOOL_NAMES) <= lone
    assert full - lone == {n for n in full if n.startswith("handoff_to_")}


def test_it_gets_the_conduct_family_like_every_other_agent():
    """conduct 工具由编排器经 extra_tools 发下来 —— RD 与其它 agent 一视同仁。

    2026-08-20 的裁决是「工具面全开 + 服务端裁决」。新加一个 agent 时最容易
    走回头路的地方就是这里：因为它「只是策划」就少发一族，于是「谁能看多天实验
    跑到哪了」又变回一份要人肉维护的清单。
    """
    from mast.agents._shared.conduct_tools import make_conduct_tools

    agent = build_rd(None, model=_fake_model(),
                     extra_tools=make_conduct_tools(AGENT))
    src = (Path(_MASTV2_ROOT) / "mast" / "agents" / "orchestrator"
           / "graph.py").read_text(encoding="utf-8")
    assert f'_shared("{AGENT}")' in src, "编排器没给 RD 发共享工具族"
    assert agent is not None
    # 名单侧：这一族对 RD 与对别人完全一样
    assert _names(make_conduct_tools(AGENT)) == CONDUCT_TOOL_NAMES


# ════════════════════════════════════════════════════════════════════
# 2. 边界：它手上没有任何驱动仪器的东西（结构断言）
# ════════════════════════════════════════════════════════════════════

_FORBIDDEN_IMPORTS = (
    "execution_context", "ExecutionContext", "instrument_lock", "hold_for_skill",
    "skill_adapter", "wrap_skill", "SkillRegistry", "discover_instrument_skills",
    "nanonis", "safety_mw", "SafetyGateMiddleware",
)


@pytest.mark.parametrize("mod_path", [
    "mast.agents.research_director.tools",
    "mast.agents.research_director.graph",
    "mast.agents._shared.campaign_tools",
])
def test_no_module_of_this_agent_imports_an_execution_plane(mod_path):
    """结构断言，不是承诺。

    RD 能进 BACKGROUNDABLE，靠的就是「无人盯着时它不可能让模型动仪器」——
    而那句话只有在工具面里确实没有执行面时才成立。一旦这里出现技能适配器 /
    ExecutionContext / Nanonis，那条准入理由当场失效，这条测试就是它的守卫。
    """
    import importlib

    mod = importlib.import_module(mod_path)
    tree = ast.parse(inspect.getsource(mod))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    blob = " ".join(sorted(imported))
    for forbidden in _FORBIDDEN_IMPORTS:
        assert forbidden not in blob, (
            f"{mod_path} import 了 {forbidden!r} —— RD 不驱动仪器，"
            "它的产出是一份委托，动手的是下游三层。")


def test_it_imports_no_sibling_agent():
    """不跨 agent 导入 —— 而这条**在这个 agent 上没有 hook 保护**。

    agent_boundary 钩子（不随仓） 靠一份写死的 ``AGENT_NAMES`` 判断「这个
    文件属于哪个 agent」，名字不在里面就直接 ``return 0``。research_director 加
    进来时那份名单没跟上，于是这个包的每一次写入都从闸门旁边走了过去 ——
    「看着在防护其实没有」的标准形状。

    在那一行补上之前（也在那之后），这条测试是这条不变式在本包上的真实守卫。
    """
    import importlib
    import pkgutil

    import mast.agents.research_director as pkg

    siblings = {"literature", "experiment_design", "instrument_control",
                "data_processing", "paper_writing", "paper_review",
                "orchestrator", "brainstorm", "buffer_summarizer"}
    seen = 0
    for info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        tree = ast.parse(inspect.getsource(mod))
        seen += 1
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                parts = n.split(".")
                if parts[:2] == ["mast", "agents"] and len(parts) > 2:
                    assert parts[2] not in siblings, (
                        f"{info.name}.py 直接 import 了兄弟 agent {parts[2]!r}；"
                        "要共享的东西搬到 agents/_shared/")
    assert seen >= 3, "一个模块都没扫到 —— 这条闸门在空转"


def test_it_holds_no_instrument_skill():
    """行为侧的同一件事：它的工具名与真实技能注册表零交集。

    比读 import 更硬 —— 技能可以经工厂函数进来而不留下一个可 grep 的名字。
    """
    from mast.agents.instrument_control.tools import discover_instrument_skills

    registry = discover_instrument_skills()
    skills = {m.name for m in registry.list_skills()}
    assert skills, "技能注册表是空的，这条闸门在空转"
    held = set(_names(rd_tools.build_tools(None)))
    assert not (held & skills), f"RD 拿到了硬件技能：{sorted(held & skills)}"


def test_it_writes_no_artifact_that_belongs_to_the_instrument():
    """产物图侧：RD 不该出现在扫描数据 / 视觉缓冲的任何一边。"""
    from mast.agents._shared.artifacts import derive_flow

    for flow in derive_flow():
        if flow.artifact.id in ("scan_files", "vision_buffer"):
            assert AGENT not in flow.writers, f"RD 被推导成 {flow.artifact.id} 的写者"
            assert AGENT not in flow.readers, f"RD 被推导成 {flow.artifact.id} 的读者"


def test_the_prompt_forbids_inventing_instrument_numbers():
    """提示词里必须写死这一条。

    「LLM 丢指数 / 发明数字」在本仓有单独一条经验教训，而移除诱因的第一步是
    让这一层根本不认为填数是它的活。
    """
    assert "不填任何仪器数值" in SYSTEM_PROMPT or "不填任何仪器参数" in SYSTEM_PROMPT
    assert "不设计具体步骤" in SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════════
# 3. 后台白名单：它进得来，而且走的是判据
# ════════════════════════════════════════════════════════════════════

def test_it_is_backgroundable_and_spawn_accepts_it():
    from mast.core.background_runs import BackgroundRunManager

    assert AGENT in BackgroundRunManager.BACKGROUNDABLE
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: None)
    rec = mgr.spawn(instruction="回顾一下 synthetic_sample 这条线还值不值得做",
                    agents=(AGENT,))
    assert tuple(rec["agents"]) == (AGENT,)


def test_the_whitelist_comment_says_why_this_one_qualifies():
    """注释会漂，所以把「它是照判据进来的」这句话本身也钉住。

    没有它，下一个人看到表里多了一个名字，只会读成「白名单又松了一格」。
    """
    from mast.core.background_runs import BackgroundRunManager

    src = inspect.getsource(BackgroundRunManager)
    head = src[: src.index("def __init__")]
    assert AGENT in head, "BACKGROUNDABLE 旁边没有说明 RD 为什么进得来"
    assert "不是例外" in head or "not an exception" in head


# ════════════════════════════════════════════════════════════════════
# 4. 接线：编排器 / 产物通道 / 提示词注册表
# ════════════════════════════════════════════════════════════════════

def test_the_orchestrator_knows_about_it():
    from mast.agents.orchestrator.graph import (
        _AGENT_NAMES, _ROUTER_JSON_HINT, _ROUTER_PROMPT,
    )

    assert AGENT in _AGENT_NAMES
    assert AGENT in _ROUTER_PROMPT, "路由提示词里没有它 —— 模型不会派它"
    assert AGENT in _ROUTER_JSON_HINT, (
        "文本-JSON 那一档的枚举里没有它。这一档不是备用路径：always-on reasoning "
        "的 provider 上它才是真正干活的那一档。")


def test_the_orchestrator_can_wire_it_alone():
    """include_agents 只给它一个也要建得出来（不需要 context_provider）。"""
    from mast.agents.orchestrator.graph import build as build_orch

    app = build_orch(None, supervisor_model=None,
                     agent_model_overrides={"__supervisor_no_model__": True,
                                            AGENT: _fake_model()},
                     include_agents=(AGENT,))
    assert AGENT in app.get_graph().nodes


def test_the_live_runtime_wires_the_whole_roster_not_a_copy_of_it():
    """接线的那一侧最容易漏，而漏了**不报错**。

    编排器的 supervisor 拿 ``include_agents`` 当路由白名单：不在里面的 agent，
    路由器点了名也会被 ``_coerce_targets`` 静静丢掉，这一轮就以一句
    「目标已完成」收场。没有日志、没有异常 —— 和「它判断不需要派」完全一样。

    2026-08-21 加第七个 agent 时，前台 runtime 里正好躺着一份写死的六元组
    （和 build() 的默认值逐字节相同，所以它只带来了漂移，没带来任何信息）。
    这条测试钉的是「那份副本不许回来」。
    """
    import re

    src = (Path(_MASTV2_ROOT) / "mast" / "core" / "runtime.py").read_text(
        encoding="utf-8")
    from mast.agents.orchestrator.graph import _AGENT_NAMES

    sites = [m.start() for m in re.finditer(r"include_agents\s*=", src)]
    assert sites, "runtime 里一个 include_agents 都没有 —— 这条闸门在空转"
    for i in sites:
        block = src[i: i + 400]
        listed = {a for a in _AGENT_NAMES if f'"{a}"' in block}
        if not listed:
            continue          # 派生的（传变量），正是我们想要的形状
        assert AGENT in listed, (
            "runtime 里又出现了一份写死的 agent 名单，而且漏了 "
            f"{AGENT}：{sorted(listed)}。写死的副本和 build() 的默认值逐字节"
            "相同，所以它只会带来漂移。")
    # 后台那一份必须是**派生**的（BACKGROUNDABLE ∩ 全名单），不是抄的
    assert "BackgroundRunManager.BACKGROUNDABLE)" in src


def test_the_non_hardware_rosters_all_know_about_it():
    """「除了仪器控制之外的所有 agent」在树里有好几份名单。

    它们表达的是**同一条判据**（不给会驱动仪器的那一个），所以少了任何一份，
    都会变成「这个 agent 在这一页能用、在那一页不能用」—— 而用户看到的是
    「功能时有时无」。这条测试把「人肉找不齐」换成一道闸门。
    """
    from mast.agents._shared.environment_tools import _ACTIVATABLE
    from mast.core.background_runs import BackgroundRunManager
    from mast.skills.composite.agent_node import DELEGATABLE_AGENTS

    for label, roster in (
        ("BACKGROUNDABLE", BackgroundRunManager.BACKGROUNDABLE),
        ("_ACTIVATABLE", set(_ACTIVATABLE)),
        ("DELEGATABLE_AGENTS", set(DELEGATABLE_AGENTS)),
    ):
        assert "instrument_control" not in roster, (
            f"{label} 里出现了 instrument_control —— 这几份名单的共同判据就是排除它")
        assert AGENT in roster, f"{label} 少了 {AGENT}"


def test_the_artifact_channel_carries_the_campaign():
    """产物矩阵是**连线机制**：写了没人读，等于没写。"""
    assert "research_campaign" in ac.CARRIED_FIELDS
    assert AGENT in ac.CONSUMES, "RD 不在 CONSUMES 里 —— 它看不到任何上游产物"
    # 它读得到文献报告（这是它的输入）
    assert "literature_report" in ac.CONSUMES[AGENT]
    # 它的产出有人读：实验设计（委托的收件人）与调度器
    assert "research_campaign" in ac.CONSUMES["experiment_design"]
    assert "research_campaign" in ac.CONSUMES["supervisor"]
    assert ac.producer_of("research_campaign") == AGENT
    assert ac.field_label("research_campaign") != "research_campaign"


def test_the_carried_field_has_a_reducer():
    """没有 reducer 的 carried field 会让**每一次** fan-out 崩，不是偶尔崩。"""
    import typing

    from mast.agents.state import AgentSubState, MASTState

    for schema in (MASTState, AgentSubState):
        hints = typing.get_type_hints(schema, include_extras=True)
        ann = hints["research_campaign"]
        assert "last_wins" in str(ann), f"{schema.__name__} 里这个通道没有 reducer"


def test_the_prompt_is_in_the_registry():
    """不在注册表里 = 用户在「上下文注入」页看不到也改不了它。"""
    from mast.prompts.registry import get_entry, render_default

    entry = get_entry(f"agent.{AGENT}.system")
    assert entry is not None, "提示词没有进注册表"
    assert entry.overridable
    assert entry.agent == AGENT
    # 注册的 loader 必须真的指向这份提示词。**注册 ≠ 跑过** —— 一个指错模块的
    # loader 在页面上长得完全正常，只是显示的是别人的文本。
    text, err = render_default(entry)
    assert not err, f"提示词 loader 读不出来：{err}"
    assert text == SYSTEM_PROMPT
    # 而图那一侧确实经注册表取（不是直接读常量），否则覆写永远不生效。
    src = (Path(_MASTV2_ROOT) / "mast" / "agents" / AGENT
           / "graph.py").read_text(encoding="utf-8")
    assert f'resolve_prompt("agent.{AGENT}.system"' in src


def test_it_has_a_default_model():
    """没有条目时 make_chat_model 会退回一个不是给它挑的模型。"""
    from mast.agents._shared.models import AGENT_MODEL

    assert AGENT in AGENT_MODEL


# ════════════════════════════════════════════════════════════════════
# 5. campaign 工具：读得出、写得进、错的当场拒
# ════════════════════════════════════════════════════════════════════

def _tools():
    return {t.name: t for t in make_campaign_tools("test")}


def test_an_empty_store_says_there_is_nothing_not_that_it_failed(v2_store):
    """「还没有纲领」和「库打不开」必须是两句不同的话。"""
    out = json.loads(_tools()["campaign_list"].invoke({}))
    assert out["ok"] is True
    assert out["count"] == 0
    assert "没有" in (out["note"] or "")


def test_a_status_filter_scans_past_the_page_limit(v2_store):
    """「最近 N 条里没有」不能被答成「没有」。

    with_stats 没有 status 参数，过滤是客户端做的。取数窗口若等于 limit，
    一个 limit=1 的 running 查询就会在第二条纲领上答错 —— 而它答的是一句
    完全正常的「没有符合条件的纲领」。
    """
    t = _tools()
    old = json.loads(t["campaign_create"].invoke(
        {"title": "旧的", "hypothesis": "h"}))["campaign_id"]
    t["campaign_update"].invoke({"campaign_id": old, "status": "running"})
    t["campaign_create"].invoke({"title": "新的", "hypothesis": "h"})  # draft，更新

    out = json.loads(t["campaign_list"].invoke({"status": "running", "limit": 1}))
    assert out["count"] == 1, "按状态筛只看了第一页 —— 「没扫到」被答成了「没有」"
    assert out["campaigns"][0]["campaign_id"] == old


def test_create_then_get_round_trips(v2_store):
    t = _tools()
    made = json.loads(t["campaign_create"].invoke({
        "title": "synthetic_sample 条纹相",
        "hypothesis": "一维条纹来自 CDW 而非表面重构",
        "hypothesis_kind": "confirmatory",
        "goal_json": json.dumps({"question": "条纹周期是否随温度变化"}),
    }))
    assert made["ok"] is True
    cid = made["campaign_id"]

    got = json.loads(t["campaign_get"].invoke({"campaign_id": cid}))
    assert got["ok"] is True
    c = got["campaign"]
    assert c["hypothesis_kind"] == "confirmatory"
    assert c["status"] == "draft"
    assert c["goal"]["question"].startswith("条纹周期")


def test_an_off_enum_kind_is_refused_not_coerced(v2_store):
    """拒绝，不夹紧。夹紧会让「填错了」看起来像「填对了」。"""
    out = json.loads(_tools()["campaign_create"].invoke({
        "title": "x", "hypothesis": "y", "hypothesis_kind": "随便写的",
    }))
    assert out["ok"] is False
    assert "hypothesis_kind" in out["reason"]
    # 而且确实没有偷偷建一条
    listed = json.loads(_tools()["campaign_list"].invoke({}))
    assert listed["count"] == 0


def test_an_off_enum_status_is_refused(v2_store):
    t = _tools()
    cid = json.loads(t["campaign_create"].invoke(
        {"title": "x", "hypothesis": "y"}))["campaign_id"]
    out = json.loads(t["campaign_update"].invoke(
        {"campaign_id": cid, "status": "差不多完成了"}))
    assert out["ok"] is False
    assert "status" in out["reason"]


def test_a_dangling_parent_is_refused(v2_store):
    """谱系指向一份不存在的纲领，比没有谱系更糟 —— 它看起来像有出处。"""
    out = json.loads(_tools()["campaign_create"].invoke({
        "title": "x", "hypothesis": "y", "parent_campaign_id": "01NOPE",
    }))
    assert out["ok"] is False


def test_update_revises_the_hypothesis_and_reports_what_changed(v2_store):
    t = _tools()
    cid = json.loads(t["campaign_create"].invoke(
        {"title": "x", "hypothesis": "旧假设"}))["campaign_id"]
    out = json.loads(t["campaign_update"].invoke(
        {"campaign_id": cid, "hypothesis": "新假设", "status": "running"}))
    assert out["ok"] is True
    assert set(out["changed"]) == {"hypothesis", "status"}

    got = json.loads(t["campaign_get"].invoke({"campaign_id": cid}))
    assert got["campaign"]["hypothesis"] == "新假设"
    assert got["campaign"]["status"] == "running"


def test_update_with_nothing_to_change_says_so(v2_store):
    """「什么都没改」不能长得像一次成功的修订。"""
    t = _tools()
    cid = json.loads(t["campaign_create"].invoke(
        {"title": "x", "hypothesis": "y"}))["campaign_id"]
    out = json.loads(t["campaign_update"].invoke({"campaign_id": cid}))
    assert out["changed"] == []
    assert out["note"]


def test_update_refuses_to_guess_which_campaign(v2_store):
    """改错一份纲领比不改更糟，所以这里不替你猜。"""
    out = json.loads(_tools()["campaign_update"].invoke(
        {"campaign_id": "", "hypothesis": "z"}))
    assert out["ok"] is False


def test_get_without_an_id_says_it_picked_one_for_you(v2_store):
    t = _tools()
    t["campaign_create"].invoke({"title": "a", "hypothesis": "h"})
    out = json.loads(t["campaign_get"].invoke({}))
    assert out["ok"] is True
    assert out["note"], "替你挑了一份却没说 —— 「当前纲领」会被当成事实用下去"


def test_a_missing_campaign_is_not_a_read_failure(v2_store):
    out = json.loads(_tools()["campaign_get"].invoke({"campaign_id": "01NOPE"}))
    assert out["ok"] is True and out["campaign"] is None
    assert "没有" in out["note"]


def test_request_plan_lands_in_the_store_and_on_the_channel(v2_store):
    """RD 的主要产出：委托既要落库（几周后查得到），又要进产物通道（下游读得到）。

    只做前者 = 下游看不见；只做后者 = 重启就没了。
    """
    t = _tools()
    cid = json.loads(t["campaign_create"].invoke(
        {"title": "x", "hypothesis": "h"}))["campaign_id"]
    # 走完整 ToolCall 信封，就像 ToolNode 在运行时那样 —— 绕过它（用 .func）
    # 正是 2026-05-30 那次「每个技能的 state 写入都被静默丢弃」能在全绿测试下
    # 存活的原因。
    ret = t["campaign_request_plan"].invoke(tool_call(
        t["campaign_request_plan"],
        {"campaign_id": cid,
         "plan_request": "设计一次能区分 CDW 与表面重构的测量；说明什么算答完了。"}))
    # 落库那一半
    got = json.loads(t["campaign_get"].invoke({"campaign_id": cid}))
    assert "CDW" in got["campaign"]["goal"]["plan_request"]
    # 通道那一半 —— 必须是真的 Command，否则 langgraph 会把 update 静默丢掉
    assert isinstance(ret, ac.ArtifactToolReturn), (
        "返回的不是 ArtifactToolReturn：普通字符串的 state 写入会被静默丢弃")
    ref = ret.update["research_campaign"]
    assert ref.campaign_id == cid
    assert "CDW" in ref.plan_request


def test_request_plan_refuses_an_unknown_campaign(v2_store):
    t = _tools()["campaign_request_plan"]
    out = json.loads(tool_text(t.invoke(tool_call(
        t, {"campaign_id": "01NOPE", "plan_request": "做点什么"}))))
    assert out["ok"] is False


def test_the_rendered_campaign_block_names_the_commission(v2_store):
    """渲染出来的那一段要让下游一眼看到「要你做的事」。

    渲染成 DocRef 的形状会印一个 ``doc_id=<ULID>`` 和一句 load_document ——
    那个 id 喂给那个工具永远失败，而模型会把失败当成信息继续推理。
    """
    ref = ac.campaign_ref(campaign_id="01ABC", title="T", hypothesis="H",
                          hypothesis_kind="confirmatory", status="running",
                          plan_request="区分 A 与 B")
    line = ac.render_field("research_campaign", ref, {"campaign_get"})
    assert "01ABC" in line and "区分 A 与 B" in line
    assert "campaign_get" in line
    assert "doc_id" not in line

    # 不持有 campaign_get 的 agent 不该被指向一个它调不了的工具
    line2 = ac.render_field("research_campaign", ref, set())
    assert "campaign_get" not in line2
    assert "01ABC" in line2


def test_an_empty_ref_renders_nothing_rather_than_a_placeholder():
    """空产物必须**不出现**。一行「科研纲领：（无）」会教会模型跳过整段。"""
    assert ac.render_field("research_campaign", None, set()) is None
    assert ac.render_field(
        "research_campaign", ac.campaign_ref(campaign_id=""), set()) is None


def test_experiments_and_claims_read_without_a_campaign(v2_store):
    """留空 = 全库最近 —— 用来回答「这件事是不是已经有人做过了」。"""
    t = _tools()
    for name in ("campaign_experiments", "campaign_claims"):
        out = json.loads(t[name].invoke({}))
        assert out["ok"] is True and out["count"] == 0
        assert out["note"], f"{name} 空结果没有说明是「没有」而不是「读不到」"
