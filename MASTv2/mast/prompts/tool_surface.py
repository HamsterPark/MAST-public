"""工具面体量：一次模型调用里**最大的那一块**，而它不在消息列表里。

## 为什么单开一个模块

2026-08-24 的实测（把每个 agent 的工具转成 provider 格式再量字节）：

===================  ======  ==============  ==================================
agent                工具数  schema 总字符   最大的几个
===================  ======  ==============  ==================================
instrument_control      394         354 027  SearchDomainBoundary 7 933、
                                             LineSTSAcrossWall 6 299、
                                             PrepareNobleTip 5 670
data_processing          39          28 513  py_run 1 530、record_analysis 1 248
literature               26          19 047  lib_add 1 359
===================  ======  ==============  ==================================

IC 的静态系统提示词是 18 183 字符。**工具 schema 是它的十九倍。** 在这个数字
出现之前，「上下文冗长」这件事一直被当成提示词问题在治 —— 把 IC 提示词从 18 k
砍到 9 k，对单次请求体积的影响不到 3%。

所以：任何「这一次调用由什么构成」的呈现，缺了工具面就是把最大的那块画成了
不存在。:mod:`mast.prompts.capture` 从 provider **实收**的 ``invocation_params``
里量（那是真的那一份）；这里量的是**建图时**的估计，用于「还没发生过任何请求
时也能回答这个 agent 的工具面有多大」。两者口径不同，各自标注来源。

## 边界

``convert_to_openai_tool`` 是 provider 无关的规范化形式。Anthropic 的实际编码
略有不同（``input_schema`` vs ``function.parameters``），所以这里的字节数是
**同一量级的估计**，不是账单。要精确值看抓包里的 ``tools_chars``。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: top-N 明细留多少条。394 个的全表约 12 kB，没必要背着。
TOP_N = 12


@dataclass(frozen=True)
class ToolSurface:
    """一个 agent 的工具面。``fmt`` 说明字节数是按哪种编码量的。"""

    count: int
    schema_chars: int
    top: tuple[tuple[str, int], ...] = ()
    fmt: str = "openai_tool"
    #: 量不出来时的说明（没有 langchain、工具对象不认识……）。空 = 量出来了。
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "schema_chars": self.schema_chars,
                "top": [{"name": n, "chars": c} for n, c in self.top],
                "fmt": self.fmt, "note": self.note}


def _one(tool: Any) -> tuple[str, int]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    spec = convert_to_openai_tool(tool)
    name = ""
    if isinstance(spec, dict):
        fn = spec.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "")
        name = name or str(spec.get("name") or "")
    name = name or str(getattr(tool, "name", "") or "?")
    return name, len(json.dumps(spec, ensure_ascii=False, default=str))


def measure(tools: Any) -> ToolSurface:
    """量一组工具的 schema 体量。**永不抛** —— 量不出来就如实说量不出来。"""
    items = list(tools or [])
    if not items:
        return ToolSurface(count=0, schema_chars=0, note="这个 agent 没有工具。")
    sizes: list[tuple[str, int]] = []
    failed = 0
    for t in items:
        try:
            sizes.append(_one(t))
        except Exception:  # noqa: BLE001
            failed += 1
    if not sizes:
        return ToolSurface(count=len(items), schema_chars=0,
                           note=f"{len(items)} 个工具的 schema 都转不出来，无法计量。")
    sizes.sort(key=lambda kv: kv[1], reverse=True)
    note = "" if not failed else f"另有 {failed} 个工具的 schema 转不出来，未计入。"
    return ToolSurface(count=len(items),
                       schema_chars=sum(c for _, c in sizes),
                       top=tuple(sizes[:TOP_N]), note=note)


# ── 建图时记下来，供 API 在「还没跑过任何请求」时也答得出 ────────────────

_LOCK = threading.Lock()
_BY_AGENT: dict[str, ToolSurface] = {}


def record(agent: str, tools: Any) -> ToolSurface:
    surface = measure(tools)
    with _LOCK:
        _BY_AGENT[agent] = surface
    return surface


def get(agent: str) -> ToolSurface | None:
    with _LOCK:
        return _BY_AGENT.get(agent)


def all_surfaces() -> dict[str, ToolSurface]:
    with _LOCK:
        return dict(_BY_AGENT)


def reset() -> None:
    with _LOCK:
        _BY_AGENT.clear()


__all__ = ["TOP_N", "ToolSurface", "all_surfaces", "get", "measure", "record", "reset"]
