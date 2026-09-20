"""技能工坊经**真实 IC 图**跑一遍:查 → 起草 → 保存 → 执行。

单测能证明每个工具自己对,证明不了**它们在图上够得着**。这一条走的是模型真正
走的那条路:`build()` → tool packs 可见性 → 中间件链 → ToolNode → Command 落进
state。本仓踩过的形状正是这一类——「工具建好了、挂上了,而这个 agent 从来没有
看见过它」。

⚠️ 版本库**必须**重定向到 tmp:`composite_store()` 默认指向
`project_root()/config/composite_skills`,那是用户真实的技能库。
「测试污染真实数据」在这个仓已经记过五次。
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import json  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mast.agents.instrument_control.graph import build  # noqa: E402
from mast.agents.instrument_control.tools import (  # noqa: E402
    discover_instrument_skills,
)
from mast.core.types import NanonisCallRecord, SkillResult  # noqa: E402


#: 模型**实际收到**的那张工具表。写进模块级列表而不是实例属性 —— 上游的
#: GenericFakeChatModel 是 pydantic 模型,给它设未声明的字段会抛。
BOUND_TOOLS: list[str] = []


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        BOUND_TOOLS[:] = [getattr(t, "name", "") for t in tools]
        return self


class _Ctx:
    """ExecutionContext 替身:``run`` 是子步的唯一入口,记下它就等于记下会发生什么。"""

    def __init__(self):
        self.ran: list[str] = []
        self.state = None

    def check_abort(self) -> bool:
        return False

    def run(self, skill_name, params=None, **kw):
        self.ran.append(skill_name)
        return SkillResult(skill_name=skill_name, success=True,
                           data={"fft_quality": 0.9})

    def safe_call(self, method, *args, role: str = "main"):
        return NanonisCallRecord(method=method, args=args, return_value=("", b"", [0.0]))


SPEC = {
    "name": "E2EScanThenCheck", "description": "扫一张并判读质量",
    "safety_level": "confirm",
    "params": [{"name": "x_m", "type": "number", "required": True},
               {"name": "y_m", "type": "number", "required": True}],
    "nodes": [
        {"type": "step", "id": "scan", "skill": "ScanAt",
         "params": {"center_x_m": {"$expr": "x_m"}, "center_y_m": {"$expr": "y_m"},
                    "size_m": 1e-7}},
        {"type": "step", "id": "q", "skill": "AssessImageQuality", "params": {}},
    ],
}


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """把工坊的版本库钉在 tmp 上 —— 见模块 docstring 的那条警告。"""
    import mast.agents._shared.skill_forge_tools as forge
    from mast.skills.composite.version_store import CompositeVersionStore
    store = CompositeVersionStore(tmp_path / "composites")
    monkeypatch.setattr(forge, "_default_store", lambda: store)
    return store


def _agent(model, ctx, registry):
    return build(
        buf=None,
        context_provider=(lambda: ctx),
        registry=registry,
        model=model,
        checkpointer=InMemorySaver(),
        enable_hitl=False,
        standalone=True,
    )


def _turn(agent, text, thread):
    return agent.invoke({"messages": [("user", text)]},
                        config={"configurable": {"thread_id": thread}})


def _joined(result) -> str:
    return "\n".join(str(m.content) for m in result["messages"]
                     if hasattr(m, "content"))


def test_the_forge_tools_are_visible_to_the_model_by_default(tmp_store):
    """tool packs 默认开着,而工坊必须仍然在模型看得见的那一份里。

    「一个模型从没被告知的工具等于不存在」—— 这条走的是 bind_tools,也就是模型
    真正收到的那张表,而不是 build_tools 的返回值。
    """
    from mast.agents._shared.skill_forge_tools import FORGE_TOOL_NAMES
    reg = discover_instrument_skills()
    BOUND_TOOLS.clear()
    model = _FakeChatModel(messages=iter([AIMessage(content="ok")]))
    agent = _agent(model, _Ctx(), reg)
    _turn(agent, "你好", "e2e-visible")
    bound = set(BOUND_TOOLS)
    assert bound, "没抓到 bind_tools —— 这条断言什么都没在验"
    assert set(FORGE_TOOL_NAMES) <= bound, (
        f"模型收到的工具表里缺:{sorted(set(FORGE_TOOL_NAMES) - bound)}")
    # 边界:tool packs 确实在起作用(不是「全都给了」所以碰巧包含工坊)。
    assert len(bound) < 80, f"可见工具 {len(bound)} 个 —— tool packs 没生效?"


def test_catalog_draft_save_run_all_the_way_through_the_graph(tmp_store):
    """四步全链,经真实图。最后一步的判据是**子步真的跑了**,不是回执好看。"""
    reg = discover_instrument_skills()
    ctx = _Ctx()
    model = _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "skill_catalog", "args": {"query": "ScanAt"},
            "id": "tc-1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{
            "name": "draft_composite", "args": {"spec_json": json.dumps(SPEC)},
            "id": "tc-2", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{
            "name": "save_composite", "args": {"spec_json": json.dumps(SPEC)},
            "id": "tc-3", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{
            "name": "run_composite",
            "args": {"name": "E2EScanThenCheck",
                     "params_json": json.dumps({"x_m": 0.0, "y_m": 0.0})},
            "id": "tc-4", "type": "tool_call"}]),
        AIMessage(content="造了 E2EScanThenCheck 并跑了一次。"),
    ]))
    agent = _agent(model, ctx, reg)
    result = _turn(agent, "把扫图和判读串成一个技能,然后跑一次", "e2e-chain")
    joined = _joined(result)

    assert "ScanAt" in joined, "catalog 没把官方技能列出来"
    assert "E2EScanThenCheck" in joined
    assert tmp_store.exists("E2EScanThenCheck"), "没有落进版本库"
    assert reg.has("E2EScanThenCheck"), "没有热注册进注册表"
    # 真正的判据:composite 的两个子步经 ctx.run 打出去了。
    assert ctx.ran == ["ScanAt", "AssessImageQuality"], ctx.ran
    # Command 的 state 副作用落进了 state(tool_call_id 对不上就不会有这一条)。
    assert "E2EScanThenCheck" in (result.get("executed_skills") or [])


def test_a_wrapper_spec_is_refused_on_the_graph_path_too(tmp_store):
    """「官方优先」那道硬拒在图上也生效,而且拒绝语指回本体。"""
    reg = discover_instrument_skills()
    ctx = _Ctx()
    wrapper = {"name": "JustGetBias", "description": "d", "safety_level": "auto",
               "params": [], "nodes": [
                   {"type": "step", "id": "a", "skill": "GetBias", "params": {}}]}
    model = _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "save_composite", "args": {"spec_json": json.dumps(wrapper)},
            "id": "tc-1", "type": "tool_call"}]),
        AIMessage(content="被拒了,直接调 GetBias。"),
    ]))
    agent = _agent(model, ctx, reg)
    joined = _joined(_turn(agent, "给 GetBias 包一层", "e2e-wrapper"))
    assert "套壳" in joined
    assert not tmp_store.exists("JustGetBias"), "被拒了却还是存进去了"
    assert not reg.has("JustGetBias")


def test_proposing_python_does_not_register_anything(tmp_store, tmp_path, monkeypatch):
    """第三级在图上也只落盘:注册表不动、白名单不动。"""
    import mast.llm.skill_author as sa
    monkeypatch.setattr(sa, "_CUSTOM_SKILLS_DIR", tmp_path / "custom")
    reg = discover_instrument_skills()
    code = (
        "from mast.skills.base import BaseSkill\n"
        "from mast.core.types import SkillMetadata, SkillResult\n"
        "class ReadWidget(BaseSkill):\n"
        "    def metadata(self):\n"
        "        return SkillMetadata(name='ReadWidget', description='d')\n"
        "    def execute(self, context, params):\n"
        "        return SkillResult(skill_name='ReadWidget', success=True)\n"
    )
    model = _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "propose_python_skill",
            "args": {"name": "ReadWidget", "code": code,
                     "rationale": "现有技能读不到这个通道"},
            "id": "tc-1", "type": "tool_call"}]),
        AIMessage(content="提议了 ReadWidget,等你审。"),
    ]))
    agent = _agent(model, _Ctx(), reg)
    joined = _joined(_turn(agent, "我们缺一个读 widget 的技能", "e2e-propose"))
    assert "没有启用" in joined
    assert (tmp_path / "custom" / "ReadWidget.py").exists()
    assert not (tmp_path / "custom" / "enabled.json").exists()
    assert not reg.has("ReadWidget")
