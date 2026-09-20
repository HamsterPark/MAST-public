"""「上下文注入」页的读侧契约（只读；编辑仍走 /api/admin/prompts）。

与 ``schemas_admin`` 的 Prompt* 的分工：那边回答「这段话是什么、能不能改」，
这边回答「**哪个 agent 每次调用收到哪些块、各占多大**」——后者在 2026-08-24
之前根本没有真源，只能靠人读七份 graph.py 加一张共享表推。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from mast.api.schemas_admin import PromptCaptureDetail, PromptSummary

Availability = Literal["static", "live", "needs_hardware", "needs_request"]
When = Literal["always", "when_set", "on_event", "on_mode", "build_time"]
Position = Literal["system", "last_human", "new_human", "state_messages",
                   "tool_result", "tools"]


class InjectionBlock(PromptSummary):
    """一条注入，加上「谁收到 / 什么时候 / 落在哪」。"""

    #: ``["*"]`` = 全员。空列表不会出现（解析不出来时退回 ``["*"]`` 并记日志）。
    agents: list[str] = Field(default_factory=list)
    when: When = "always"
    position: Position = "system"
    #: 实现它的中间件类名（空 = 不是中间件，比如 agent 自己的系统提示）。
    middleware: str = ""
    #: 哪几条建图路径上有它：group / standalone / workflow。
    paths: list[str] = Field(default_factory=list)
    #: 依赖哪个可选子系统才会挂上（空 = 无条件）。
    #: 「全员」与「凡是有这个子系统的全员」是两件事。
    requires: str = ""
    #: 这个块只到一个 agent 手上。矩阵里用它回答「有没有针对性」。
    exclusive: bool = False


class ToolPackInfo(BaseModel):
    name: str
    label: str = ""
    tools: int = 0
    chars: int = 0


class ToolSurfaceInfo(BaseModel):
    """工具面 —— 一次调用里最大的一块，而它不在消息列表里。

    ``schema_chars`` 是**建图时**按 provider 无关的 openai-tool 形式量的估计；
    真正发出去的那一份在抓包的 ``tools_chars`` 里。两个口径不同，各自标注。
    """

    count: int = 0
    schema_chars: int = 0
    top: list[dict] = Field(default_factory=list)
    fmt: str = "openai_tool"
    #: 按需加载开着时：核心可见多少、省了多少。None = 这个 agent 没分包。
    core_tools: Optional[int] = None
    core_chars: Optional[int] = None
    packs: list[ToolPackInfo] = Field(default_factory=list)


class AgentInfo(BaseModel):
    id: str
    label: str = ""
    #: 建过图才有；没有就是这个进程还没跑过它。
    tool_count: Optional[int] = None
    tool_chars: Optional[int] = None
    system_chars: Optional[int] = None


class InjectionMatrixResponse(BaseModel):
    agents: list[AgentInfo] = Field(default_factory=list)
    blocks: list[InjectionBlock] = Field(default_factory=list)
    #: 每个块对每个 agent：``true`` = 会收到。行 = 块，列 = agent。
    cells: dict[str, dict[str, bool]] = Field(default_factory=dict)
    degraded: bool = False
    note: str = ""


class AgentInjectionResponse(BaseModel):
    agent: str
    label: str = ""
    #: 按**实际挂载顺序**。
    blocks: list[InjectionBlock] = Field(default_factory=list)
    #: ``build`` = 顺序来自这个进程真的建过的那次图；``declared`` = 登记表的
    #: 声明顺序。把后者当成前者读，会让人以为自己在看运行时事实。
    order_source: Literal["build", "declared"] = "declared"
    tool_surface: Optional[ToolSurfaceInfo] = None
    tool_surface_note: str = ""
    middleware: list[str] = Field(default_factory=list)
    degraded: bool = False


class CaptureBlockRef(BaseModel):
    """快照里某条消息的一段来自哪个块。``start``/``end`` 是**码点**下标。"""

    id: str
    start: int = 0
    end: int = 0
    chars: int = 0
    position: str = "system"


class LatestCaptureResponse(PromptCaptureDetail):
    """某个 agent 最近一次**主模型调用**。

    「最近一次」指主模型节点那一次 —— 中间件内部的压缩 / 精炼子调用也用同一个
    source，不筛的话这里会返回一条摘要请求。
    """

    found: bool = True
    reason_code: str = ""
    reason: str = ""
    #: provider 上报的；取不到就是 None（这里不做字符数折算）。
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    tokens_source: Literal["provider", "unavailable"] = "unavailable"
    #: 工具 schema 的实发体量（**不在 messages 里**）。
    tool_count: Optional[int] = None
    tools_chars: Optional[int] = None
    tools_top: list[dict] = Field(default_factory=list)
    tools_source: Literal["invocation_params", "unavailable"] = "unavailable"
    #: 块归属是从哪儿查到的。
    blocks_source: Literal["ledger", "additional_kwargs", "none"] = "none"
    node: str = ""
    thread_id: str = ""


__all__ = [
    "AgentInfo", "AgentInjectionResponse", "CaptureBlockRef", "InjectionBlock",
    "InjectionMatrixResponse", "LatestCaptureResponse", "ToolPackInfo",
    "ToolSurfaceInfo",
]
