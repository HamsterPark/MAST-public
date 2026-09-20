"""建图台账：这个 agent 这次**实际**挂了哪些中间件、带了多大的工具面。

## 为什么必须记下来

``create_agent`` 把 ``middleware=`` 拆成五类钩子列表之后只存进闭包
（``langchain/agents/factory.py``），编译出来的 ``CompiledStateGraph`` 上不保留
原列表；而只实现 ``wrap_model_call`` 的注入器**不产生图节点**，所以从
``graph.nodes`` 也反推不出来。于是「哪个 agent 收到哪些注入块」这个问题，在
2026-08-24 之前只能靠人读七份 ``graph.py`` 加一张共享表推出来 —— 没有任何文档
或测试固化它，而登记表里有四条的归属就是这么标错的。

这里让每次 ``build()`` 自报家门。**自报是不够的**，所以配一道结构闸门：测试里
monkeypatch 掉 ``create_agent`` 截获真正传进去的 ``middleware=``，与这份记录
逐项对账（``tests/v2/unit/prompts/test_manifest_matrix.py``）。一份没人核的
自报记录和没有记录一样。

## 顺带做的一件事

``system_blocks`` 把 agent 的静态系统提示拆成**登记表里的条目**（IC = 系统提示
＋ Nanonis 模块速查两条），登记进 :mod:`mast.prompts.ledger` 的钉住区。之后每
一次 model call 的 system 都是从这一段长出来的，抓包侧因此能把最前面那一大块
也标上来源，而不是记成一整块 ``system.base``。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from mast.prompts import ledger as _ledger
from mast.prompts import tool_surface as _ts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildRecord:
    """一次 ``build()`` 的快照。"""

    agent: str
    #: 中间件类名，**按挂载顺序**（外层在前，与传给 create_agent 的顺序一致）。
    middleware: tuple[str, ...] = ()
    #: 静态系统提示的分解：[(prompt_id, chars), …]。
    system_blocks: tuple[tuple[str, int], ...] = ()
    system_chars: int = 0
    tool_surface: _ts.ToolSurface | None = None
    #: 按需加载的目录（只有分了包的 agent 才有）。
    tool_catalog: Any = None
    #: 私聊（standalone，无 handoff 工具）还是群聊。
    standalone: bool = False
    #: 哪条建图路径：group / standalone / workflow。
    path: str = "group"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "middleware": list(self.middleware),
            "system_blocks": [{"id": i, "chars": c} for i, c in self.system_blocks],
            "system_chars": self.system_chars,
            "tool_surface": self.tool_surface.to_dict() if self.tool_surface else None,
            "tool_packs": (
                {"core": len(self.tool_catalog.core),
                 "core_chars": self.tool_catalog.core_chars,
                 "total_chars": self.tool_catalog.total_chars,
                 "packs": [{"name": p,
                            "tools": len(self.tool_catalog.tools_by_pack.get(p, ())),
                            "chars": self.tool_catalog.chars_by_pack.get(p, 0)}
                           for p in self.tool_catalog.known_packs()]}
                if self.tool_catalog is not None else None),
            "standalone": self.standalone,
            "path": self.path,
        }


_LOCK = threading.Lock()
_BY_AGENT: dict[str, BuildRecord] = {}


def record_build(agent: str, *, middleware: Any = (), tools: Any = (),
                 system_blocks: Any = (), standalone: bool = False,
                 path: str = "group", tool_catalog: Any = None) -> BuildRecord:
    """记下这次建图。**永不抛** —— 台账坏了不能挡住建图。

    Args:
        agent: agent id（与 capture 的 ``source``、登记表的 ``agent`` 同一套）。
        middleware: 传给 ``create_agent`` 的那个列表，原样。
        tools: 传给 ``create_agent`` 的工具列表，原样。
        system_blocks: ``[(prompt_id, text), …]``，按拼接顺序。
        standalone: 私聊路径（handoff 工具被剥掉）。
        path: group / standalone / workflow。
    """
    try:
        names = tuple(type(m).__name__ for m in (middleware or ()))
        parts = [(str(i), str(t or "")) for i, t in (system_blocks or ())]
        _ledger.register_base(agent, parts)
        blocks = tuple((i, len(t)) for i, t in parts)
        rec = BuildRecord(
            agent=agent, middleware=names, system_blocks=blocks,
            system_chars=sum(c for _, c in blocks),
            tool_surface=_ts.record(agent, tools),
            tool_catalog=tool_catalog,
            standalone=bool(standalone),
            path=("standalone" if standalone else path),
        )
        with _LOCK:
            _BY_AGENT[agent] = rec
        logger.debug("build recorded: %s — %d mw, %d tools (%d chars schema), "
                     "system %d chars", agent, len(names),
                     rec.tool_surface.count if rec.tool_surface else 0,
                     rec.tool_surface.schema_chars if rec.tool_surface else 0,
                     rec.system_chars)
        return rec
    except Exception:  # noqa: BLE001
        logger.debug("record_build(%s) failed (swallowed)", agent, exc_info=True)
        return BuildRecord(agent=agent)


def last_build(agent: str) -> BuildRecord | None:
    with _LOCK:
        return _BY_AGENT.get(agent)


def all_builds() -> dict[str, BuildRecord]:
    with _LOCK:
        return dict(_BY_AGENT)


def reset() -> None:
    with _LOCK:
        _BY_AGENT.clear()


__all__ = ["BuildRecord", "all_builds", "last_build", "record_build", "reset"]
