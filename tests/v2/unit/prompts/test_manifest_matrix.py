"""注入矩阵的**双向**闸门：栈里有的必须在册，在册的必须在栈里。

## 为什么两个方向都要

2026-08-24 之前只有一个方向的闸门（``test_prompt_overrides.py``：标了
``overridable`` 的 id 必须在源码里被引用）。漏的那个方向是「**在注入但没登记**」，
而它的症状是：上下文注入页少画了一块，用户看到的构成图是错的 —— 而且看不出来。
反向清点当场找出 7 个：alert_delivery、skill_image ×2、IC 的 Nanonis 速查尾巴、
LIT 的工具可用性尾巴、compaction、tool_refine。

同一次清点还发现**四条归属标错**：``mw.mode_belief.*`` / ``mw.experiment_prefs``
/ ``mw.instrument_profile`` 在册上没有 agent 字段（UI 渲染成「全局」），实际
只挂在 IC(+XD) 上。改覆写的人会以为自己在影响全部七家。

## 判据从哪来

真源是**中间件自己的模块级 ``AGENTS`` 常量**，登记表的 ``agents_from`` 与
共享栈的 ``applies_to`` 都读它。这里做的是：真的建一次图，把 ``create_agent``
收到的 ``middleware=`` 截下来，和登记表算出来的名单对账。

**自报是不够的。** ``builds.record_build`` 是 build() 自己填的；这里用
monkeypatch 截获真正传进去的那个列表，两者必须逐项相等 —— 一份没人核的自报
记录和没有记录一样。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/prompts/test_manifest_matrix.py -q
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2 = str(_REPO / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from mast.agents._shared import shared_stack  # noqa: E402
from mast.prompts import builds, manifest, registry  # noqa: E402

_SHARED = _REPO / "MASTv2" / "mast" / "agents" / "_shared"


class _Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def _model():
    return _Fake(messages=iter([AIMessage(content="ok")] * 20))


def _summarizer():
    return _model()


class _Cognition:
    """记忆服务的最小替身。

    **必须有它**，不是为了方便：``build_shared_middleware`` 在 ``cognition is
    None`` 时根本不挂 ``MemoryRecallMiddleware``，于是不给替身就等于在测一个
    少了一项的栈 —— 而闸门要钉的恰恰是「登记表说会到的都真的挂上了」。
    「替身太残让正例恒假」与「替身太顺让负例恒绿」是同一枚硬币。
    """

    def memory_index(self, **kw):
        return "## MEMORY.md\n- (empty)"

    def recall(self, query, **kw):
        return []


def _shared_deps() -> set[str]:
    """本次建栈**满足了**哪些可选依赖（对应登记表的 ``requires``）。"""
    return {"cognition"}


def _build(agent: str, spy=None):
    """真的建一次图，返回 ``(agent_obj, 中间件类名列表)``。

    共享表用**真的** ``build_shared_middleware`` 加一个假 summarizer —— 走
    ``CoreRuntime._chat_agent_middleware`` 时 ``make_chat_model`` 没有 key 会抛、
    被 except 吞掉，测试里于是永远拿到缺了 compaction 与 tool_refine 的**残表**，
    而断言照样绿。那是「替身太残让正例恒假」，闸门因此钉不住真实的挂载顺序。
    """
    extra = shared_stack.build_shared_middleware(
        agent, model_id="kimi-k3", summarizer=_summarizer(),
        cognition=_Cognition(), namespace_provider=lambda: None, settings=None)
    mod = importlib.import_module(f"mast.agents.{agent}.graph")
    if spy is not None:
        mod.create_agent = spy(mod.create_agent)          # type: ignore[attr-defined]
    try:
        if agent == "instrument_control":
            obj = mod.build(None, lambda: SimpleNamespace(), model=_model(),
                            enable_hitl=False, extra_middleware=extra)
        else:
            obj = mod.build(None, model=_model(), extra_middleware=extra)
    finally:
        if spy is not None:
            importlib.reload(mod)
    return obj, list(builds.last_build(agent).middleware)


# ── 0. 自检：闸门确实扫到了东西 ─────────────────────────────────────────

def test_the_registry_is_not_empty_and_declares_middleware():
    entries = registry.entries()
    assert len(entries) >= 30, f"登记表只有 {len(entries)} 条，多半没加载全"
    with_mw = [e for e in entries if e.middleware]
    assert len(with_mw) >= 12, f"只有 {len(with_mw)} 条声明了 middleware"


def test_manifest_agents_match_the_orchestrator_list():
    """名单**派生**不抄。抄一份的代价是加第七个 agent 时漏掉一侧。"""
    from mast.agents.orchestrator.graph import _AGENT_NAMES

    assert set(manifest.AGENTS) == set(_AGENT_NAMES), (
        f"manifest.AGENTS={manifest.AGENTS} 与 orchestrator._AGENT_NAMES 对不上")


# ── 1. 正向：栈里的每个注入器都在册，且归属对得上 ───────────────────────

@pytest.mark.parametrize("agent", manifest.AGENTS)
def test_mounted_stack_matches_the_manifest(agent):
    _, mounted = _build(agent)
    by_mw = manifest.entries_by_middleware()

    # (a) 栈里每个**注入型**中间件都在册
    unregistered = [n for n in mounted
                    if n not in by_mw and n in _injector_class_names()]
    assert not unregistered, (
        f"{agent} 的栈里这些注入器不在登记表里：{unregistered}\n"
        "在册的才会出现在「上下文注入」页上 —— 漏掉的那一块，用户看到的构成图"
        "是错的，而且看不出来。")

    # (b) 在册且声明了这个 agent 的中间件，必须真的挂上了
    for cls, entry in by_mw.items():
        if cls not in _injector_class_names():
            continue
        if not registry.applies_to(entry, agent):
            continue
        if "group" not in (entry.paths or ()):
            continue
        if entry.requires and entry.requires not in _shared_deps():
            continue        # 这个可选子系统这次没起来，它不挂是对的
        assert cls in mounted, (
            f"登记表说 {entry.id}（{cls}）会到 {agent}，但它的栈里没有这个中间件。\n"
            f"栈：{mounted}")

    # (c) 反过来：没声明这个 agent 的定向中间件，不许出现在它的栈里
    for cls, entry in by_mw.items():
        if cls not in _injector_class_names():
            continue
        if registry.applies_to(entry, agent):
            continue
        assert cls not in mounted, (
            f"{agent} 挂了 {cls}，但登记表说它只给 "
            f"{registry.agents_of(entry)} —— 两边有一边是错的。")


def _injector_class_names() -> set[str]:
    """``_shared/`` 下**真的往上下文里写字**的 AgentMiddleware 子类。"""
    return set(_scan_injectors())


def _scan_injectors() -> dict[str, str]:
    """AST 扫出「类名 → 文件」，判据 = 类体里调了注入 helper 或改了消息。"""
    calls = {"append_system_block", "append_human_block", "append_new_human"}
    out: dict[str, str] = {}
    for path in sorted(_SHARED.glob("*.py")):
        if path.name in ("inject.py", "shared_stack.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            hit = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if nm in calls:
                        hit = True
                if isinstance(sub, ast.Attribute) and sub.attr == "system_message":
                    parent_is_store = isinstance(getattr(sub, "ctx", None), ast.Store)
                    if parent_is_store:
                        hit = True
                if isinstance(sub, ast.keyword) and sub.arg in (
                        "system_message", "messages"):
                    fn = getattr(getattr(sub, "value", None), "func", None)
                    hit = hit or (getattr(fn, "attr", "") == "override")
            if hit:
                out[node.name] = path.name
    return out


# ── 2. 反向：AST 扫出来的注入器，一个都不许缺登记 ───────────────────────

def test_the_reverse_gate_actually_found_injectors():
    """一个扫不到任何东西的反向闸门永远绿。"""
    found = _scan_injectors()
    assert len(found) >= 8, f"只扫到 {len(found)} 个注入器：{found}"
    for must in ("TipContextMiddleware", "LiveStateMiddleware",
                 "AlertDeliveryMiddleware", "SkillImageMiddleware"):
        assert must in found, f"反向闸门漏掉了 {must}"


def test_every_injector_in_shared_is_registered():
    found = _scan_injectors()
    by_mw = manifest.entries_by_middleware()
    missing = {cls: mod for cls, mod in found.items() if cls not in by_mw}
    assert not missing, (
        "这些中间件往上下文里写东西，但登记表里没有它们：\n  "
        + "\n  ".join(f"{c}  ({m})" for c, m in sorted(missing.items()))
        + "\n每一个都要在 mast/prompts/registry.py 的 _SPEC 里加一条 —— "
        "不在册 = 上下文注入页少画一块。")


def test_the_reverse_gate_catches_a_planted_unregistered_injector():
    """变异自检：塞一个没登记的注入器进去，闸门必须逮到。"""
    src = (
        "class PlantedMiddleware(AgentMiddleware):\n"
        "    def _apply(self, request):\n"
        "        return append_system_block(request, 'mw.planted', 'x')\n"
    )
    tmp = _SHARED / "_planted_probe_mw.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        assert "PlantedMiddleware" in _scan_injectors(), "反向闸门是摆设"
    finally:
        tmp.unlink(missing_ok=True)


# ── 3. 自报的建图记录必须与真正传进去的那份相等 ─────────────────────────

def test_the_build_record_matches_what_create_agent_actually_received():
    """``record_build`` 是 build() 自己填的。这里截真正的实参对账。"""
    seen: dict = {}

    def spy(orig):
        def wrapper(*a, **kw):
            seen["middleware"] = [type(m).__name__ for m in (kw.get("middleware") or [])]
            seen["system_prompt"] = kw.get("system_prompt") or ""
            seen["tools"] = [getattr(t, "name", "?") for t in (kw.get("tools") or [])]
            return orig(*a, **kw)
        return wrapper

    _, recorded = _build("data_processing", spy=spy)
    assert seen, "spy 没被调用 —— 对账等于没做"
    assert recorded == seen["middleware"], (
        f"自报的中间件链与实际传进去的不一致：\n  自报 {recorded}\n  实际 "
        f"{seen['middleware']}")

    rec = builds.last_build("data_processing")
    assert rec.system_chars == len(seen["system_prompt"])
    assert rec.tool_surface is not None
    assert rec.tool_surface.count == len(seen["tools"])


def test_build_time_tails_are_registered_blocks_not_anonymous_text():
    """IC 的 Nanonis 速查、LIT 的工具可用性说明都要各占一条登记条目。

    它们是**建图时拼进系统提示**的两段独立文本；不拆开的话，抓包侧只能把整段
    system 记成一整块，「这 19 k 字符里哪 1.2 k 是手册索引」就答不出来。
    """
    seen: dict = {}

    def spy(orig):
        def wrapper(*a, **kw):
            seen["system_prompt"] = kw.get("system_prompt") or ""
            return orig(*a, **kw)
        return wrapper

    _build("instrument_control", spy=spy)
    rec = builds.last_build("instrument_control")
    ids = [i for i, _ in rec.system_blocks]
    assert ids == ["agent.instrument_control.system",
                   "agent.instrument_control.manual_index"], ids
    assert sum(c for _, c in rec.system_blocks) == len(seen["system_prompt"]), (
        "分解出来的字符数与真正传给 create_agent 的系统提示对不上 —— "
        "中间少了一段没登记的尾巴。")
    for pid in ids:
        assert registry.get_entry(pid) is not None, f"{pid} 不在登记表里"


# ── 4. 定向注入真的生效了吗（不是只在册上写着）──────────────────────────

def test_tip_context_reaches_the_three_agents_that_read_spectra_and_no_others():
    from mast.agents._shared.tip_context_mw import AGENTS as TIP_AGENTS

    for agent in manifest.AGENTS:
        mws = shared_stack.build_shared_middleware(
            agent, model_id="kimi-k3", summarizer=_summarizer(), settings=None)
        names = [type(m).__name__ for m in mws]
        expected = agent in TIP_AGENTS
        assert ("TipContextMiddleware" in names) is expected, (
            f"{agent}: 针尖块 {'该有却没有' if expected else '不该有却有'}")


def test_a_conditional_block_says_what_it_depends_on():
    """「全员」与「凡是有这个子系统的全员」是两件事。

    记忆召回只在 cognition 起得来时才挂。登记成无条件的「全员」，矩阵就会在
    一台没起 cognition 的机器上说谎 —— 而那正是矩阵最该说实话的场合。
    """
    entry = registry.get_entry("mw.memory_recall")
    assert entry.requires == "cognition"

    without = shared_stack.build_shared_middleware(
        "paper_review", model_id="kimi-k3", summarizer=_summarizer(),
        cognition=None, settings=None)
    assert "MemoryRecallMiddleware" not in [type(m).__name__ for m in without]

    with_cog = shared_stack.build_shared_middleware(
        "paper_review", model_id="kimi-k3", summarizer=_summarizer(),
        cognition=_Cognition(), namespace_provider=lambda: None, settings=None)
    assert "MemoryRecallMiddleware" in [type(m).__name__ for m in with_cog]


def test_the_registry_derives_attribution_from_the_middleware_not_a_copy():
    """登记表的归属必须是**派生**的。抄一份的症状是改了中间件、册上没变。"""
    from mast.agents._shared.tip_context_mw import AGENTS as TIP_AGENTS

    entry = registry.get_entry("mw.tip_context")
    assert entry.agents_from, "mw.tip_context 没有声明 agents_from —— 归属是抄的"
    assert registry.agents_of(entry) == tuple(TIP_AGENTS)


# ── 5. 矩阵本身 ─────────────────────────────────────────────────────────

def test_matrix_covers_every_entry_and_every_agent():
    m = manifest.matrix()
    assert len(m["rows"]) == len(registry.entries())
    for row in m["rows"]:
        assert set(row["cells"]) == set(m["agents"])


def test_manifest_for_orders_static_prompt_first():
    for agent in ("instrument_control", "paper_review"):
        blocks = manifest.manifest_for(agent)["blocks"]
        assert blocks[0].id.startswith(f"agent.{agent}."), (
            f"{agent} 的第一块不是它自己的系统提示：{blocks[0].id}")


def test_manifest_says_whether_the_order_is_real_or_declared():
    """「声明顺序」被当成「实际顺序」读，会让人以为自己在看运行时事实。"""
    builds.reset()
    assert manifest.manifest_for("paper_review")["order_source"] == "declared"
    _build("paper_review")
    assert manifest.manifest_for("paper_review")["order_source"] == "build"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
