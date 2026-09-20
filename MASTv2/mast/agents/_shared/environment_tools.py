"""Let an agent look around before it decides to work.

Why this module exists (2026-07-30, docs/v2/design/wakeup_scheduling.md)
-----------------------------------------------------------------------
The artifact channel (2026-07-29) made "what has been produced" machine-readable
for the first time, but only as a PUSH: each agent is shown the fields its own
``CONSUMES`` row lists, at the moment it is dispatched. Nothing could ask the
question the other way round — *what is in this experiment right now?* — which is
the question a scheduling decision actually needs:

    "现在环境中有没有一些已有的资料可用?要不要等?"

The data all existed. ``_shared.artifacts.list_existing()`` walks every artifact
class on disk and ``class_status()`` reports per-class counts, and both were
consumed by exactly one caller: the topology REST endpoint. No agent could reach
them. This module is that reach.

The one hard rule here
----------------------
``ClassStatus`` distinguishes ``known=False`` (the store could not be read) from
``count == 0`` (the store was read and is empty). That distinction is the entire
reason the dataclass carries a ``known`` flag, and it must survive into the text
the model sees, with a DIFFERENT symbol and a DIFFERENT sentence. Collapsing them
into "没有" tells the model a database outage is an empty experiment — after which
it will confidently write a report about having found nothing. Empty is a fact to
act on; unreadable is a fact to report.

And the corollary: an empty listing is rendered as empty. Never padded with an
example, never illustrated with a plausible-looking doc_id. The 2026-07-27
coordinate incident was exactly this — an example in a prompt read as data.
"""
from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

logger = logging.getLogger(__name__)

#: Cap on the per-class recent-item detail. A survey is for deciding whether to
#: start, not for reading the archive — ``list_documents`` / ``load_document`` are
#: for that, and every line here competes with the agent's real instructions.
_RECENT_PER_CLASS = 3


def _age(ts: float | None) -> str:
    """"3 分钟前" / "2 天前". Empty string when there is no usable timestamp —
    an invented age is worse than a missing one."""
    if not ts:
        return ""
    import time

    delta = time.time() - float(ts)
    if delta < 0:
        return ""
    if delta < 90:
        return "刚刚"
    if delta < 5400:
        return f"{int(delta // 60)} 分钟前"
    if delta < 172800:
        return f"{int(delta // 3600)} 小时前"
    return f"{int(delta // 86400)} 天前"


#: How to obtain an id for each class, so every line's pointer is redeemable. A
#: line that says a document exists but not how to open it is a dead end, and the
#: channel already paid for that once (``load_document`` was advertised for a day
#: before it existed).
_HOW_TO_GET = {
    "literature_report": "list_documents(kind='literature_report') → load_document(doc_id)",
    "draft": "list_documents(kind='paper_draft') 或 kind='experiment_report' → load_document(doc_id)",
    "review": "list_documents(kind='review') → load_document(doc_id)",
    "experiment_plan": "list_documents(kind='plan') → load_document(doc_id)",
    "scan_files": "find_scans() / load_scan(path 或 scan_id)",
    "figures": "（文件路径已在上面给出）",
}


def _pointer(item: dict, artifact_id: str) -> str:
    """The redeemable pointer for one item: a real ``doc_id``, else its path.

    ``list_existing()`` guarantees a ``doc_id`` key but SYNTHESISES it as
    ``"<artifact_id>:<file stem>"`` for classes that are not documents (scan files,
    figures). Printing that would hand the model an id ``load_document`` cannot
    resolve — the same defect the readback table shipped with for a day, where an
    agent was told to call a tool that did not exist. So a synthesised id is
    detected and the file path is offered instead, which IS redeemable (via
    ``load_scan`` for scans, or directly for figures).
    """
    doc_id = str(item.get("doc_id") or "")
    if doc_id and not doc_id.startswith(f"{artifact_id}:"):
        return f"  `doc_id={doc_id}`"
    path = item.get("path")
    return f"  `{path}`" if path else ""


@tool("survey_environment")
def survey_environment() -> str:
    """看一眼**本实验现在有哪些资料可用**(动手之前先探查环境)。

    回答的是「环境里已经有什么」这个问题:文献报告、实验方案、扫描数据、分析图、
    草稿、评审各有没有、有几份、最新的是多久以前。用于判断:现在就能干活,还是缺的
    东西会让产出变成猜测。

    什么时候用:
      * 拿到一个任务但不确定别人是否已经做过其中一部分;
      * 要判断「该等一等还是现在开工」;
      * 上下文里的「上游产物」块是空的,但你怀疑磁盘上其实有东西
        (那个块只显示交接过来的指针,不显示实验文件夹里的全部资料)。

    Returns:
        每类资料一行。**「❌ 尚无」和「⚠️ 读不到」是两件不同的事**,分别标注 ——
        前者是确认过的空(可以据此决定等或不等),后者是这次查询失败(必须如实说明,
        不能当成空)。空清单就是空,不会用示例填充。
    """
    try:
        from mast.agents._shared.artifacts import class_status, list_existing
    except Exception as e:  # noqa: BLE001 — a tool never raises at the model
        return (f"survey_environment: 探查模块不可用({type(e).__name__}: {e})。"
                "这**不等于**环境是空的 —— 是这次查询失败了,请如实说明。")

    try:
        existing = list_existing()
        rows = class_status(existing)
    except Exception as e:  # noqa: BLE001
        return (f"survey_environment: 读不到环境状态({type(e).__name__}: {e})。"
                "这**不等于**环境是空的 —— 是这次查询失败了,请如实说明。")

    # Group the concrete items by class so each produced line can name its newest
    # few. list_existing() rows are the SAME source class_status() counted, so the
    # detail and the count can never disagree.
    by_class: dict[str, list[dict]] = {}
    for r in existing or []:
        by_class.setdefault(r.get("artifact_id") or "", []).append(r)
    for items in by_class.values():
        items.sort(key=lambda d: d.get("modified_at") or 0, reverse=True)

    lines: list[str] = ["本实验现有资料:"]
    unreadable: list[str] = []
    produced = 0
    for cs in rows:
        art = cs.artifact
        label = art.label
        if not cs.known:
            # DIFFERENT symbol, DIFFERENT wording — see the module docstring.
            unreadable.append(f"  ⚠️ {label} —— 读不到({cs.detail})")
            continue
        if cs.count <= 0:
            lines.append(f"  ❌ {label} —— 尚无(已确认,不是查询失败)")
            continue
        produced += 1
        newest = by_class.get(art.id) or []
        head = f"  ✅ {label} —— {cs.count} 项"
        when = _age((newest[0].get("modified_at") if newest else None))
        if when:
            head += f",最新 {when}"
        lines.append(head)
        for item in newest[:_RECENT_PER_CLASS]:
            name = item.get("name") or "(未命名)"
            lines.append(f"       · {name}{_pointer(item, art.id)}")
        if len(newest) > _RECENT_PER_CLASS:
            lines.append(f"       · …还有 {len(newest) - _RECENT_PER_CLASS} 项")
        how = _HOW_TO_GET.get(art.id)
        if how:
            lines.append(f"       取用:{how}")

    if unreadable:
        lines.append("")
        lines.append("**以下几类这次读不到**(不是「没有」,是查询失败;"
                     "如果你的判断依赖它们,请把这一点写进结论):")
        lines.extend(unreadable)

    if produced == 0 and not unreadable:
        lines.append("")
        lines.append("→ 本实验目前**确实**什么产物都还没有(已逐类确认)。"
                     "如果你的任务需要上游资料,应当如实说明缺什么,"
                     "**不要凭空编造内容**。")
    return "\n".join(lines)


@tool("request_activation")
def request_activation(
    agent: str,
    why: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    waiting_for: str = "",
    instruction: str = "",
) -> Command:
    """**建议**让另一个智能体接着干(提候选,不是下命令)。

    你在自己的工作里发现「这件事应该由 X 来做」时用它 —— 比如你写完了草稿,觉得
    该让评审看一遍;或者你发现缺一份文献综述。**这不会直接启动对方**:请求会交给
    编排器,由它统一决定和调度(所有循环/预算熔断照常生效)。

    Args:
        agent: 建议激活谁。只能是非仪器智能体:literature / experiment_design /
               data_processing / paper_writing / paper_review。
               **不能是 instrument_control** —— 仪器只在前台由用户在场时驱动。
        why:   为什么现在该让它做(一句话,会给编排器看)。
        waiting_for: 可选。如果你觉得它还得等某样东西,写产物字段名
               (literature_report / experiment_plan / last_scan / analysis /
               draft / review)。留空 = 现在就能开始。
        instruction: 可选。你建议它具体做什么。留空则由编排器根据上下文决定。

    Returns:
        一句确认。**不代表对方已经开始** —— 由编排器决定。
    """
    target = (agent or "").strip()
    reason = (why or "").strip()
    if target not in _ACTIVATABLE:
        return _plain(
            tool_call_id, "request_activation",
            f"不能请求激活 {target!r}。可选:{'、'.join(sorted(_ACTIVATABLE))}。"
            "(instrument_control 不在其中 —— 仪器只在前台由用户在场时驱动。)")
    if not reason:
        return _plain(tool_call_id, "request_activation",
                      "request_activation 需要 why:说明为什么现在该让它做。")

    waits = [w.strip() for w in (waiting_for or "").replace("，", ",").split(",")
             if w.strip()]
    bad = [w for w in waits if w not in _WAITABLE]
    if bad:
        return _plain(
            tool_call_id, "request_activation",
            f"waiting_for 里有无法识别的名字:{'、'.join(bad)}。"
            f"只能用:{'、'.join(_WAITABLE)} —— 自由文本没法在资料到位时被匹配上,"
            "等于永远不会被叫醒。")

    note = f"[REQUEST → {target}] {reason}"
    if instruction.strip():
        note += f"（建议任务:{instruction.strip()[:200]}）"
    # routing_hints, not a direct jump. The 2026-05-30 removal of agent→agent
    # dispatch stands: an A→B→A ping-pong through a direct edge is bounded by
    # nothing but recursion_limit, and this is the same request wearing a different
    # hat. Going through the channel the supervisor already consumes means every
    # loop guard, the budget gate and the activation check all still run.
    return Command(update={
        "routing_hints": [target],
        "messages": [ToolMessage(content=note, tool_call_id=tool_call_id,
                                 name="request_activation")],
    })


def _plain(tool_call_id: str, name: str, text: str) -> Command:
    """A tool result with no state write — for refusals and guidance."""
    return Command(update={
        "messages": [ToolMessage(content=text, tool_call_id=tool_call_id, name=name)],
    })


#: Who may be requested. instrument_control is excluded for the same reason it can
#: never be backgrounded: it is the sole hardware agent and stays foreground and
#: interactive, so no automatic mechanism can ever cause a hardware command.
_ACTIVATABLE = frozenset({
    "research_director",
    "literature", "experiment_design", "data_processing",
    "paper_writing", "paper_review",
})


def _waitable() -> tuple[str, ...]:
    try:
        from mast.agents._shared.artifact_channel import WAITABLE_FIELDS

        return WAITABLE_FIELDS
    except Exception:  # noqa: BLE001
        return ()


_WAITABLE = _waitable()

#: Handed to EVERY agent, same argument as ``DOCUMENT_TOOLS``: read-only (or, for
#: ``request_activation``, advisory), side-effect free, and "what is in this
#: experiment / who should go next" is not a per-agent privilege — it is the
#: precondition for any agent deciding whether its own work is possible yet.
#:
#: Note ``spawn_background_task`` already exists but reaches only instrument_control
#: (it is a meta-tool). That asymmetry is why five of the six agents had no way at
#: all to say "someone else should do this next".
ENVIRONMENT_TOOLS: list = [survey_environment, request_activation]

__all__ = ["survey_environment", "request_activation", "ENVIRONMENT_TOOLS"]
