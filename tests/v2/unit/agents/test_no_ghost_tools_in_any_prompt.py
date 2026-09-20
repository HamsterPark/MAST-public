"""每个 agent 的提示词点名的工具，都必须真的存在。

## 为什么要有这一条

**这个 bug 在本仓发生过。** ``core/runtime.py:4795`` 那段注释写着：

    for one day it named a `load_document` tool that existed nowhere in the tree

模型不会怀疑提示词。它读到「用 ``load_document(doc_id)`` 把正文读出来」就会去调，
然后拿到一个 unknown tool，然后要么重试要么放弃 —— 一整轮烧掉，而日志里只留下
一条看不出根因的工具错误。

data_processing 早就有这一条（``test_fft_and_prompt.py::test_prompt_names_no_
ghost_tools``），**另外五家一直没有**。守卫只覆盖 N 家里的一家，等于漏掉 N-1 家。
这个文件补上其余五家。

## 一个 agent 拿到的工具，远不止 build_tools() 那些

这条闸门最容易踩的坑有两个：

* ``build_tools(buf, …)`` —— **必须传一个真的 buf**。传 None，缓冲工具
  （``read_latest_tip_status`` / ``get_scan_progress`` /
  ``get_tip_history_since``）根本不会被建出来，于是三个真实存在的工具被报成幽灵。
* ``_shared/meta_tools.py`` 里由工厂建的一大族（实验/样品/方案/知识/针尖登记…），
  经 ``extra_tools`` 挂上，不进任何 ``build_tools``。
* ``DOCUMENT_TOOLS`` / ``ENVIRONMENT_TOOLS`` / ``ASK_USER_TOOLS`` / ``FIGURE_TOOLS``
  —— 在 ``runtime.py:4806/4819/4832`` 挂给**每一个** agent。

**名字一律从生产方取**：``@tool("...")`` 装饰器与 ``*_TOOLS`` 常量。抄一份，
等它加工具时这份抄件就成了假话。

## 这条闸门的**精度边界**（写明，别让它假装更强）

它回答的是「这个名字在本仓**存在吗**」，不是「这个 agent **拿得到吗**」。
后者要把真图建出来（需要一个真模型），成本不成比例。**发生过的那个 bug 是前者**
—— 提示词点了一个树里根本没有的名字。按 agent 精确归属留给
``test_manifest_matrix.py`` 那条线。

## 它不检查什么

反方向（「工具存在却从没被介绍过」）不在这里 —— 那要按 agent 判断哪些该进提示词，
DP 有自己的那一条（``test_prompt_mentions_every_real_tool``）。
"""

from __future__ import annotations

import ast
import inspect
import re
import tempfile
from pathlib import Path

import pytest

#: 要检查的 agent。instrument_control 不在这里 —— 它走按需加载，可见工具面
#: 逐调用变化，由 ``tests/v2/unit/prompts/test_tool_packs.py`` 那一族负责。
AGENTS = [
    "research_director",
    "literature",
    "experiment_design",
    "data_processing",
    "paper_writing",
    "paper_review",
]

#: 提示词里长得像调用、但**不是工具**的名字。
#: 每条都要写清它是什么 —— 否则这张表会变成「报红就往里加」的垃圾桶。
NOT_TOOLS = {
    # py_run 沙箱内的辅助函数：`mastdata.save_result(...)` / `npy_load(path)`
    "save_result", "savefig", "npy_load",
    # 提示词里举例用到的三方库函数
    "curve_fit",
}


def _buf():
    """真的 BufferService。**不要换成替身** —— 缓冲工具挂在它上面。"""
    from mast.buffer.service import BufferService

    d = Path(tempfile.mkdtemp())
    return BufferService(wal_path=d / "buf.sqlite", wal_enabled=False)


def _prompt(agent: str) -> str:
    mod = __import__(f"mast.agents.{agent}.prompts", fromlist=["SYSTEM_PROMPT"])
    return getattr(mod, "SYSTEM_PROMPT", "")


def _own_tools(agent: str, buf) -> set[str]:
    """``build_tools`` 各家签名不同，按**参数名**喂，别按位置猜。

    硬编码调用方式会在下一次谁加个参数时静默 skip 掉一整个 agent ——
    那正是这条闸门要防的形状。
    """
    mod = __import__(f"mast.agents.{agent}.tools", fromlist=["build_tools"])
    build = getattr(mod, "build_tools", None)
    if build is None:
        pytest.skip(f"{agent} 没有 build_tools")

    kwargs = {}
    for name, param in inspect.signature(build).parameters.items():
        if name == "buf":
            kwargs["buf"] = buf
        elif name == "registry":
            from mast.agents.instrument_control.tools import (
                discover_instrument_skills,
            )
            kwargs["registry"] = discover_instrument_skills()
        elif param.default is inspect.Parameter.empty:
            raise AssertionError(
                f"{agent}.build_tools 有个这条闸门不认识的必填参数 {name!r}；"
                "补上它的构造方式，不要让这个 agent 被静默跳过")
    return {getattr(t, "name", getattr(t, "__name__", ""))
            for t in build(**kwargs)}


def _shared_tool_names() -> set[str]:
    """``agents/`` 下所有工具的名字 —— 从**装饰器与常量**取，不手抄。

    两个来源：

    * ``@tool("name")`` 装饰器的字面量（含工厂函数里嵌套定义的那一大族）。
    * ``*_TOOLS = [...]`` 常量里的对象（它们在模块顶层，import 得到）。
    """
    names: set[str] = set()
    base = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "agents"
    assert base.is_dir(), f"找不到 agents/：{base}"

    for src in base.rglob("*.py"):
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fn_name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if fn_name != "tool" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)

    # 顶层常量里的（有些用 @tool 无参形式，名字取自函数名）
    for mod_name, const in [
        ("document_tools", "DOCUMENT_TOOLS"),
        ("environment_tools", "ENVIRONMENT_TOOLS"),
        ("ask_tools", "ASK_USER_TOOLS"),
        ("figure_tools", "FIGURE_TOOLS"),
    ]:
        try:
            mod = __import__(f"mast.agents._shared.{mod_name}", fromlist=[const])
            names |= {getattr(t, "name", getattr(t, "__name__", ""))
                      for t in getattr(mod, const)}
        except Exception:                                        # noqa: BLE001
            pass

    assert len(names) > 40, (
        f"只从 _shared/ 认出 {len(names)} 个工具名 —— 提取规则多半失效了。"
        "这条断言存在的理由：一个认不出任何名字的提取器会让整条闸门恒绿。")
    return names


def _handoff_names() -> set[str]:
    from mast.agents.orchestrator.graph import _AGENT_NAMES

    return {f"handoff_to_{a}" for a in _AGENT_NAMES} | {"handoff_to_supervisor"}


#: 提示词里**明确当成调用写**的名字：``name(`` 或 ``` `name(` ```。
#:
#: 刻意**不**认光秃秃的反引号名（`` `doc_id` ``）—— 那多半是参数名或字段名，
#: 认它会把一堆参数报成幽灵工具（第一版就是这么错的）。
_CALL = re.compile(r"\b([a-z_][a-z0-9_]{3,})\s*\(")


def _named_in(prompt: str) -> set[str]:
    return set(_CALL.findall(prompt)) - NOT_TOOLS


@pytest.fixture(scope="module")
def universe():
    """本仓里**存在**的工具名全集 + 一个真 buf。"""
    return _buf(), _shared_tool_names() | _handoff_names() | _skill_names()


def _skill_names() -> set[str]:
    """全部仪器技能名 —— IC 的提示词点的多半是这些。"""
    from mast.agents.instrument_control.tools import discover_instrument_skills

    return {m.name for m in discover_instrument_skills().list_skills()}


def test_instrument_control_prompt_names_no_ghost_tools(universe):
    """IC 单独一条 —— 它不在上面的 parametrize 里。

    原因是 IC 走**按需加载**：可见工具面逐调用变化，``build_tools`` 那一套对它
    不成立。但「提示词点了一个树里没有的名字」这个错**它照样会犯**，而且它的
    提示词最长、点名的工具最多。

    加这一条之前 IC 靠**手工**核 —— 手工核过等于没有闸门。
    """
    _, universe_names = universe
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT

    ghosts = sorted(n for n in _named_in(SYSTEM_PROMPT)
                    if n not in universe_names)
    assert not ghosts, (
        f"[instrument_control] 提示词点名了本仓不存在的工具：{ghosts}")


def test_the_lookup_family_is_named_in_the_ic_prompt(universe):
    """IC 的十三个「去查」工具必须在提示词里被点到名。

    这一条钉的是 2026-08-24 查出来的事：这一族**全部存在、全部挂载，而 IC 的
    提示词一个都没提过**。一个从没被介绍过的工具等于不存在 ——
    ``test_background_offload_prompt.py`` 的 docstring 早就写下过这句判词。
    """
    _, universe_names = universe
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT

    lookup = [
        "get_workflow_advice", "get_measurement_template",
        "get_literature_parameters", "get_skill_guidance",
        "get_fault_diagnosis", "get_noise_reference", "query_knowledge",
        "search_deep_reference", "read_reference_section",
        "get_map_analysis", "get_next_scan_position", "get_markers_near",
        "get_coarse_map",
    ]
    gone = [t for t in lookup if t not in universe_names]
    assert not gone, (
        f"这些查询工具不存在了：{gone}。要么改名了（同步这份清单），"
        "要么被删了（那就把提示词里那一节也删掉）")

    unmentioned = [t for t in lookup if t not in SYSTEM_PROMPT]
    assert not unmentioned, (
        f"这些「去查」工具存在、挂载了，但 IC 的提示词一次都没提：{unmentioned}\n"
        "模型不会去调一个它不知道存在的工具 —— 在「# 动手之前先查」那一节里"
        "给它加一行「什么时候用它」。")


@pytest.mark.parametrize("agent", AGENTS)
def test_prompt_names_no_ghost_tools(agent, universe):
    """提示词点名的每个工具都必须在本仓真的存在。"""
    buf, shared = universe
    prompt = _prompt(agent)
    assert prompt, f"{agent} 的 SYSTEM_PROMPT 是空的"

    have = _own_tools(agent, buf) | shared
    ghosts = sorted(n for n in _named_in(prompt) if n not in have)
    assert not ghosts, (
        f"[{agent}] 提示词点名了本仓不存在的工具：{ghosts}\n"
        "模型会照着调，然后拿到 unknown tool，一整轮就烧在这上面。\n"
        "要么把工具接上，要么把提示词里那一句删掉 —— "
        "不要往 NOT_TOOLS 里加，那张表只放「本来就不是工具」的名字。"
    )


def test_this_gate_would_actually_catch_a_ghost(universe):
    """变异验证：植一个不存在的工具名，这条闸门必须报红。

    没有这一条，上面那条可能只是因为正则一个都没匹配到而恒绿 ——
    「闸门在检查形状，没在问它到底说不说得出话」。
    """
    buf, shared = universe
    have = _own_tools("paper_review", buf) | shared
    planted = _prompt("paper_review") + "\n收尾时调 `summon_a_tool_that_is_not_real()`。\n"

    ghosts = sorted(n for n in _named_in(planted) if n not in have)
    assert "summon_a_tool_that_is_not_real" in ghosts, (
        "植入的幽灵工具没被抓到 —— 这条闸门是空的，它的绿灯不算数")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_regex_finds_real_tools_in_each_prompt(agent, universe):
    """反向自检：正则必须在每家提示词里认出**若干**个真工具。

    只验「没有幽灵」不够 —— 一个匹配不到任何名字的正则永远没有幽灵。
    """
    buf, shared = universe
    have = _own_tools(agent, buf) | shared
    hits = _named_in(_prompt(agent)) & have
    assert len(hits) >= 3, (
        f"[{agent}] 正则只认出 {len(hits)} 个真工具（{sorted(hits)}）—— "
        "多半是提取规则失效了，而不是这份提示词真的不提工具")
