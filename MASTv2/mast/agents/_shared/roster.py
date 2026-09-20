"""这套系统有哪几个 agent —— 一份名单，一个真源。

为什么它不该住在 ``orchestrator/graph.py`` 里（2026-08-27 抽出）
---------------------------------------------------------------
「有哪几个 agent」是**系统的组成**，不是某个执行引擎的实现细节。它住在 supervisor
图里，只是因为那张图是第一个需要它的人。代价在退出 langgraph 时才现形：
``api/routes/orchestrator.py`` 为了拿这份名单，得 import 一个即将被删的图模块。

抄第二遍不是选项 —— 名单抄第二遍就会漂，而漂的症状是「某个 agent 在一个界面里在、
另一个界面里不在」，没人会想到去对两份常量。

顺序有意义
----------
``research_director`` 在最前，因为它是**战役层**（为什么做），排在执行层之前；其余
六个大致按一次实验的自然流程排。路由提示词里的名单按这个顺序渲染给模型看，所以顺序
是**给人和模型读的**，不是集合语义 —— 用 tuple 不用 frozenset 正是为此。
"""
from __future__ import annotations

#: 全部七个 agent，按「战役层在前、执行流程在后」排列。
AGENT_NAMES: tuple[str, ...] = (
    "research_director",
    "literature",
    "experiment_design",
    "instrument_control",
    "data_processing",
    "paper_writing",
    "paper_review",
)

#: 会驱动真实仪器的那些。今天只有一个，但写成集合是因为**判据是「碰不碰硬件」**，
#: 不是「是不是叫 instrument_control」—— 将来加一个光学 agent 时，漏掉它的形态会是
#: 「两个 agent 同时动机器」，而那种漏法在代码里长得和「就是没加」一模一样。
INSTRUMENT_CAPABLE: frozenset[str] = frozenset({"instrument_control"})


def non_instrument_agents() -> tuple[str, ...]:
    """不碰硬件的那些，保持 :data:`AGENT_NAMES` 的顺序。"""
    return tuple(a for a in AGENT_NAMES if a not in INSTRUMENT_CAPABLE)


__all__ = ["AGENT_NAMES", "INSTRUMENT_CAPABLE", "non_instrument_agents"]
