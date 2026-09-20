"""目标终止判据 —— 「什么算答完了」的**代码求值**那一半。

## 为什么有这个包（2026-08-27）

在它之前，MAST 的「目标」有两种形态，都不被代码读：

* 一次 run-task 的目标是 ``messages[0]`` 那句自然语言（``_goal_text``），
  「做完了没有」写在路由提示词里由模型自己判；
* 一条科研纲领的目标是 ``campaigns.goal_json``，其中 ``success_criteria``
  全仓**只被渲染成文本给模型看**，没有一处代码解析或求值它。

如果没有代码求值的终止条件，模型可能过早宣称完成，也可能在每次唤醒都重置
per-run 熔断的情况下重复成功动作。仅按重复失败签名停止无法发现后一种循环。

## 这个包**不**做什么

**不写第二个三态求值器。** 仓里已经有一个：:func:`mast.conduct.rules.evaluate`
（``TRUE`` / ``FALSE`` / ``UNDECIDABLE``，缺席用哨兵不用 ``None``，``all`` 先看
假再看判不了）。本包只做两件事——**谓词目录**（每条谓词从哪读证据）与**编译**
（把 ``done_when`` 编成 ``RuleTree``），然后调那个核。与 Director 的分工完全同构：
Director 收证据（``director._collect_evidence``）→ 核判；这里收证据 → 同一个核判。

三态映射固定：``TRUE`` → done、``FALSE`` → not_done、``UNDECIDABLE`` → unknown。

## 三条纪律

1. **done 当且仅当核返回 TRUE。** 没有第二条通往 done 的路。
2. **读不到 ⇒ 证据里不写这个字段**（让 ``lookup`` 回哨兵 → UNDECIDABLE）。
   不许写 ``None`` / ``0`` / ``False`` 冒充——那会把「读不到」变成一个答案。
3. **闭集**。谓词只能从 :data:`CATALOG` 里选；两个写入口（REST 与 agent 工具）
   都是**整体拒绝**，不做部分丢弃：丢掉 ``all`` 里一个合取项会让目标被削弱，
   于是更早「达成」，于是错误地抑制唤醒——那是 fail-open 方向。

## 依赖方向

顶层只 import 标准库 + :mod:`mast.conduct.rules` / :mod:`mast.conduct.spec`。
一切路径与库句柄由调用方注入（见 :mod:`mast.goals.sources`），模块内不解析
``project_root()``——否则就成了路径的第二真源。``mast/core/wake_scheduler.py``
**不**直接 import 本包，由 runtime 注入回调（有结构测试钉着）。
"""
from __future__ import annotations

from mast.goals.spec import (
    CATALOG,
    Combo,
    DoneWhen,
    GoalSpecError,
    Predicate,
    describe_done_when,
    done_when_to_json,
    fingerprint,
    normalise_done_when,
)
from mast.goals.verdict import (
    DONE,
    NOT_DONE,
    UNKNOWN,
    GoalVerdict,
    ItemVerdict,
    evaluate_done_when,
    render_goal_block,
)

__all__ = [
    "CATALOG",
    "Combo",
    "DoneWhen",
    "GoalSpecError",
    "Predicate",
    "describe_done_when",
    "done_when_to_json",
    "fingerprint",
    "normalise_done_when",
    "DONE",
    "NOT_DONE",
    "UNKNOWN",
    "GoalVerdict",
    "ItemVerdict",
    "evaluate_done_when",
    "render_goal_block",
]
