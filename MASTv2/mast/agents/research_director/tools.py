"""科研策划 RD 的工具装配。

工具面很短，而**短是结论不是省事**。RD 只做 Campaign 层：

  * 科研纲领的读写 —— 来自 :mod:`mast.agents._shared.campaign_tools`，
    经 ``logging/v2`` 的 repo，不自己写 SQL；
  * 既往实验记录 / 主张的只读查询 —— 同一族里的两个只读工具；
  * 交接 —— 交给文献、交给实验设计、交回编排器。

## 三样**故意没有**的东西

**1. 没有任何硬件技能。** 与 literature 一样，这个 agent 不装 SafetyGate、不装
HITL、不 import 技能注册表。它不是「暂时还没接上仪器」，是**这一层根本不动仪器**：
campaign 层的产出是一个委托，plan 层写方案，conduct 层管执行，真正动手的只有
instrument_control。结构上保证这件事，比在提示词里叮嘱它便宜得多，也可靠得多
（有测试钉着这一条）。

**2. 没有文献检索工具。** 这是一次权衡，理由值得写下来：语义检索那条路上有一整套
**降级诚实性**逻辑（嵌入器挂掉时结果退化成关键词匹配，排序不再有意义，于是工具
必须拒绝把它包装成「相关文献」—— 见 ``literature/tools.py`` 的）。
把那段逻辑在这里再实现一遍，就是本仓「同一个动作 N 份实现、往往只有一份对」的
标准形状，而错的那一份会安静地把降级结果当成检索结果用。
所以 RD 读的是 LIT **已经写好的报告**（上下文里带 doc_id，用 ``load_document``
读全文），需要新的调研就 ``handoff_to_literature``。
要给 RD 真正的检索能力，正确的做法是把 ``search_local_corpus`` 连同它的降级门
一起搬进 ``_shared/``、让 LIT 从那里 import —— 一份实现两个持有者，而不是两份。

**3. 没有 ``load_document`` / ``ask_user`` 的本地副本。** 它们由运行时统一挂在
每个 agent 的 ``extra_tools`` 上（``core/runtime.py`` 里 memory + documents +
environment + ask_user 那一段），编排器再经 ``_shared(agent_name)`` 发下来。
在这里再声明一遍会撞名。
"""

from __future__ import annotations

import logging

from mast.agents._shared.campaign_tools import (
    CAMPAIGN_TOOL_NAMES,
    make_campaign_tools,
)
from mast.agents._shared.handoff import make_handoff

logger = logging.getLogger(__name__)

#: 交接目标，按「最常走的那条路」排序。RD 的默认下游是实验设计 —— 它的产出是一份
#: 委托，而委托要有人接。
_HANDOFFS: tuple[tuple[str, str], ...] = (
    (
        "experiment_design",
        "把科研纲领的委托（plan_request）交给实验设计 agent，由它起草具体方案。"
        "纲领与委托已在上下文里，reason 只写一句路由说明即可。",
    ),
    (
        "literature",
        "需要新的文献调研时交给文献 agent —— 检索/取文/写综述的工具在它那里，"
        "不在这里。说清楚要查什么、为哪条假设服务。",
    ),
    (
        "supervisor",
        "交回编排器。纲领已经写好或修订完、且这一轮不需要立刻起草方案时用它。",
    ),
)


def build_tools(buf=None) -> list:
    """RD 的工具列表。

    ``buf`` 只为签名对齐而存在（``agents._shared.artifacts.agent_tool_names`` 用
    ``build_tools(None)`` 探测每个 agent 的真实工具名），**这个 agent 不读视觉
    缓冲**：针尖状态、扫描进度都不是 campaign 层的输入，而挂一组自己不用的工具
    只会让产物图上多出一条不存在的读边。
    """
    tools = list(make_campaign_tools("research_director"))
    tools += [make_handoff(target, desc) for target, desc in _HANDOFFS]
    logger.info("research_director: built %d tools", len(tools))
    return tools


#: 领域工具名（不含 handoff、不含运行时统一下发的那几族）。测试用它对账。
AGENT_TOOL_NAMES: tuple[str, ...] = CAMPAIGN_TOOL_NAMES

__all__ = ["build_tools", "AGENT_TOOL_NAMES"]
