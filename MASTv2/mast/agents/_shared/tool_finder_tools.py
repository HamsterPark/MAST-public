"""`search_tools` / `load_tool_pack` —— 让 agent 自己把需要的工具调出来。

配套 :mod:`mast.agents._shared.tool_visibility_mw`（收窄）与
:mod:`mast.agents._shared.tool_packs`（分包）。三者的分工：

* ``tool_packs``  分包与检索的**纯逻辑**（没有 langchain 依赖，好测）；
* ``tool_visibility_mw``  每次调用收窄可见集、贴目录；
* 本模块  两个元工具，把「取出来」这件事写进 state。

## 为什么 search 要**自动加载**

一个只回「有这个工具，名字叫 X」的检索工具，会让模型多走一整轮（搜 → 读 →
再 load → 再调）。而它每多走一轮就多付一次全上下文的钱，那正是这套机制要省的
东西。所以 ``search_tools`` 命中之后**直接把相关包写进 state**：下一次模型调用
里那些工具的 schema 就在了，可以直接调。

## 找不到的时候说找不到

返回「没有匹配」而不是返回一个最接近的包 —— 「读不到被折叠成一个具体的值」是
这个仓反复栽过的形状。模型看到「没找到」会去问用户或换个说法，看到一个错的
包只会照着错的用。
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from mast.agents._shared import tool_packs as _tp
from mast.agents._shared.tool_visibility_mw import STATE_KEY

logger = logging.getLogger(__name__)


def _result(tool_call_id: str, name: str, text: str,
            packs: list[str] | None = None) -> Command:
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=text, tool_call_id=tool_call_id, name=name)],
    }
    if packs:
        update[STATE_KEY] = list(packs)
    return Command(update=update)


def make_tool_finder_tools(catalog_provider: Callable[[], _tp.Catalog | None]) -> list:
    """两个元工具，绑在一个返回当前 Catalog 的闭包上。

    用 provider 而不是直接传 Catalog，是因为工具在建图时就要造出来，而目录也在
    建图时算 —— 顺序上先有工具后有目录，闭包把这个先后解开。
    """

    @tool("search_tools")
    def search_tools(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """按你想做的事查找工具，并**自动把相关工具包调出来**。

        你手上的工具远多于这一次发给你的那些。想做的事在核心工具里没有对应的
        入口时，用它：写你想做什么（中文英文都行，例如「设置锁相放大器的调制
        幅度」「读共振频率」「打开数据记录」），命中的工具包会自动加载，
        **下一步就能直接调那些工具**。

        找不到就会如实说找不到 —— 那时候请换个说法再找一次，或者问用户。

        Args:
            query: 你想做的事，或工具名的一部分。
        """
        catalog = catalog_provider()
        if catalog is None:
            return _result(tool_call_id, "search_tools",
                           "工具目录不可用（这个 agent 没有分包）—— 你看到的就是"
                           "全部工具。")
        hits = _tp.search(catalog, query)
        if not hits:
            # 兜底：撞不到具体工具，不等于没有这个能力 —— 工具名与描述几乎全是
            # 英文，而问问题的是中文。拿查询去撞包的中文标签，把对的那一包取出来
            # 让模型自己看 schema。仍然撞不到才说找不到。
            guess = _tp.search_packs(catalog, query)
            if guess:
                labels = "、".join(f"{p}（{_tp.pack_label(p)}）" for p in guess)
                return _result(
                    tool_call_id, "search_tools",
                    f"没有工具的名字/描述直接匹配「{query}」，但按类别看最可能在这里："
                    f"{labels}。已经为你加载，**请在下一步里直接看这些工具的参数**；"
                    "如果里面确实没有你要的能力，如实告诉用户，不要猜一个名字去调。",
                    guess)
            packs = ", ".join(catalog.known_packs())
            return _result(
                tool_call_id, "search_tools",
                f"没有匹配「{query}」的工具。**不要照着猜一个名字去调。**\n"
                f"可以换个说法再找一次，或者直接取整包：{packs}。\n"
                "确实没有这个能力时，如实告诉用户。")
        packs = _tp.packs_of(hits)
        lines = [f"匹配「{query}」的工具（最相关在前）："]
        for h in hits:
            where = "、".join(p for p in h["packs"] if p != _tp.CORE) or "核心"
            summary = h["summary"] or ""
            lines.append(f"  · {h['name']}  [{where}]  {summary}")
        if packs:
            labels = "、".join(f"{p}（{_tp.pack_label(p)}）" for p in packs)
            lines.append("")
            lines.append(f"已为你加载：{labels}。这些工具的完整参数会出现在你的"
                         "下一步里，**直接调用即可，不用再搜一次**。")
        else:
            lines.append("")
            lines.append("这些都在核心包里，现在就能直接调。")
        return _result(tool_call_id, "search_tools", "\n".join(lines), packs)

    @tool("load_tool_pack")
    def load_tool_pack(
        pack: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """把一整包工具调出来（已经知道要哪一类时用它，比搜索快一步）。

        包名见系统提示里的「工具目录」一节。取过的包不会再收回去。

        Args:
            pack: 包名，例如 "spectroscopy" / "pll" / "motion"。
        """
        catalog = catalog_provider()
        if catalog is None:
            return _result(tool_call_id, "load_tool_pack",
                           "工具目录不可用（这个 agent 没有分包）—— 你看到的就是"
                           "全部工具。")
        name = (pack or "").strip()
        available = catalog.known_packs()
        if name not in available:
            return _result(
                tool_call_id, "load_tool_pack",
                f"没有名为「{name}」的工具包。可取的是：{', '.join(available)}。\n"
                "不确定该取哪个就用 search_tools 描述你要做的事。")
        n = len(catalog.tools_by_pack.get(name, ()))
        return _result(
            tool_call_id, "load_tool_pack",
            f"已加载 `{name}`（{_tp.pack_label(name)}，{n} 个工具）。"
            "它们的完整参数会出现在你的下一步里。",
            [name])

    return [search_tools, load_tool_pack]


__all__ = ["make_tool_finder_tools"]
