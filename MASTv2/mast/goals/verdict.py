"""求值 —— 收证据、编译、**调 conduct 那个核**、把三态翻译成三个词。

这里没有一行比较逻辑。比较、组合、「缺席不是假」全在
:func:`mast.conduct.rules.evaluate`；本模块只负责把证据摆好、把结论翻译成
调用方能用的形状（未满足清单、判不了的原因）。

**done 当且仅当核返回 TRUE。** 没有第二条通往 done 的路 —— 有一条结构测试
（patch 掉那个核 ⇒ 所有结论变 unknown）钉着这句话。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from mast.conduct import rules as _rules
from mast.goals.spec import (
    CATALOG,
    DoneWhen,
    Predicate,
    compile_rule,
    describe_done_when,
    iter_predicates,
)

logger = logging.getLogger(__name__)

#: 三个词。与 conduct 三态一一对应，且**故意不叫同一个名字** —— 那边说的是
#: 「判据成不成立」，这边说的是「目标做完没做完」，混用会让日志读起来像在说
#: 同一件事。
DONE = "done"
NOT_DONE = "not_done"
UNKNOWN = "unknown"

_FROM_TRISTATE = {
    _rules.TRUE: DONE,
    _rules.FALSE: NOT_DONE,
    _rules.UNDECIDABLE: UNKNOWN,
}


@dataclass(frozen=True)
class ItemVerdict:
    """一条谓词的答案 + 为什么。"""

    kind: str
    args: dict
    state: str                      # done / not_done / unknown
    text: str = ""                  # 人读的判据描述
    reason: str = ""                # 只在 unknown 时有内容（读不到什么）

    def as_dict(self) -> dict:
        return {"kind": self.kind, "args": dict(self.args), "state": self.state,
                "text": self.text, "reason": self.reason}


@dataclass(frozen=True)
class GoalVerdict:
    """一次求值的完整答案。``verdict`` 是唯一该被拿去做决定的字段。"""

    verdict: str = UNKNOWN
    reason: str = ""
    satisfied: int = 0
    total: int = 0
    per_predicate: "tuple[ItemVerdict, ...]" = ()
    checked_at: float = 0.0

    @property
    def is_done(self) -> bool:
        """**只有这一个属性能用来「结束」。**

        写成属性而不是让调用方各自比较字符串，是因为 ``!= "not_done"``
        这种写法会把 unknown 悄悄读成 done —— 而这两者驱动的下一步相反。
        """
        return self.verdict == DONE

    @property
    def unmet(self) -> "list[ItemVerdict]":
        return [i for i in self.per_predicate if i.state == NOT_DONE]

    @property
    def unknowns(self) -> "list[ItemVerdict]":
        return [i for i in self.per_predicate if i.state == UNKNOWN]

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason,
                "satisfied": self.satisfied, "total": self.total,
                "per_predicate": [i.as_dict() for i in self.per_predicate],
                "checked_at": self.checked_at}


def _now() -> float:
    return time.time()


def evaluate_done_when(
    spec: "DoneWhen | None",
    collect: "Callable[[str, dict], Mapping[str, Any] | None]",
    *,
    reason_when_absent: str = "没有判据",
) -> GoalVerdict:
    """求值一份 ``done_when``。**永不抛。**

    ``collect(kind, args)`` 回一个证据字典，或 ``None`` / 抛异常表示读不到。
    读不到的**字段**必须**不出现**在返回的字典里 —— 写成 ``None``/``0``/
    ``False`` 会把「读不到」变成一个答案，而这正是本仓一天犯四次的那个错。

    没有判据 ⇒ ``unknown``（**不是 done**）。「没人写过什么算答完」和「答完了」
    是两件事，而把前者读成后者会让每一个没写判据的目标立刻「达成」。
    """
    if spec is None:
        return GoalVerdict(verdict=UNKNOWN, reason=reason_when_absent,
                           checked_at=_now())

    preds = iter_predicates(spec)
    evidence: dict[str, dict] = {}
    notes: dict[int, str] = {}
    for idx, pred in enumerate(preds):
        ns = f"p{idx}"
        want = CATALOG[pred.kind].evidence
        try:
            got = collect(pred.kind, {k: v for k, v in pred.args})
        except Exception as exc:  # noqa: BLE001 — 收集器坏了是 unknown，不是 done
            logger.debug("goal collector %s failed: %s", pred.kind, exc)
            notes[idx] = f"证据读不到：{type(exc).__name__}: {exc}"
            evidence[ns] = {}
            continue
        if got is None:
            notes[idx] = "证据读不到（收集器说不知道）"
            evidence[ns] = {}
            continue
        # 只搬目录声明过的字段。收集器多给的东西不进证据包 —— 否则一个手滑
        # 多写的键会让判据读到一个它本不该看见的值。
        bucket = {k: got[k] for k in want if k in got}
        missing = [k for k in want if k not in got]
        if missing:
            notes[idx] = f"证据里缺 {missing}"
        # 收集器可以捎带解释（比如「best_frames 文件读不懂」）。
        if isinstance(got, Mapping) and got.get("_why"):
            notes[idx] = str(got["_why"])
        evidence[ns] = bucket

    rule = compile_rule(spec)
    overall = _FROM_TRISTATE.get(_rules.evaluate(rule, evidence), UNKNOWN)

    items: list[ItemVerdict] = []
    for idx, pred in enumerate(preds):
        leaf = CATALOG[pred.kind].leaf(pred)
        from mast.conduct.spec import RuleLeaf

        one = RuleLeaf(field=f"p{idx}.{leaf.field}", op=leaf.op, value=leaf.value)
        state = _FROM_TRISTATE.get(_rules.evaluate(one, evidence), UNKNOWN)
        items.append(ItemVerdict(
            kind=pred.kind, args={k: v for k, v in pred.args}, state=state,
            text=CATALOG[pred.kind].describe(pred),
            reason=(notes.get(idx, "") if state == UNKNOWN else ""),
        ))

    satisfied = sum(1 for i in items if i.state == DONE)
    if overall == DONE:
        reason = f"判据全部满足（{satisfied}/{len(items)}）"
    elif overall == NOT_DONE:
        unmet = [i.text for i in items if i.state == NOT_DONE]
        reason = f"还差：{'；'.join(unmet)}"
    else:
        why = [f"{i.text}（{i.reason or '判不了'}）"
               for i in items if i.state == UNKNOWN]
        reason = f"判不了：{'；'.join(why)}"
    return GoalVerdict(verdict=overall, reason=reason, satisfied=satisfied,
                       total=len(items), per_predicate=tuple(items),
                       checked_at=_now())


def render_goal_block(verdict: GoalVerdict, goal_text: str = "") -> str:
    """给路由模型看的判据块。

    **这是事实，不是说服。** 它告诉模型「代码这一刻怎么看这个目标」，与
    ``park_block`` 同一性质；判断仍然由代码做，模型看不看得懂都不改变结论。
    """
    if verdict.total <= 0 and not goal_text:
        return ""
    lines = ["# 本次任务的目标判据（由代码求值，不是由你判断）"]
    if goal_text:
        lines.append(f"目标：{goal_text}")
    mark = {DONE: "✓", NOT_DONE: "✗", UNKNOWN: "?"}
    for i in verdict.per_predicate:
        suffix = f" —— {i.reason}" if i.reason else ""
        lines.append(f"  {mark.get(i.state, '?')} {i.text}{suffix}")
    lines.append(f"目前 {verdict.satisfied}/{verdict.total} 满足。")
    if verdict.verdict == NOT_DONE:
        lines.append(
            "**判据没满足就不要 __end__。** 还没做的阶段自己派下去；"
            "确实卡住了（等人 / 等硬件）就说清楚卡在哪，那不是完成。")
    elif verdict.verdict == UNKNOWN:
        lines.append("判据这一刻读不到，按你自己的判断走 —— 但把读不到这件事说出来。")
    return "\n".join(lines)


__all__ = ["DONE", "NOT_DONE", "UNKNOWN", "GoalVerdict", "ItemVerdict",
           "evaluate_done_when", "render_goal_block"]
