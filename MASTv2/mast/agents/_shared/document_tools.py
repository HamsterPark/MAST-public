"""Read-only document tools every agent can hold.

Why this module exists (2026-07-30, found by audit the day after shipping)
-------------------------------------------------------------------------
The artifact channel landed on 2026-07-29 and told agents, in their own context,
things like:

    - **文献报告**：《NiI2 综述》 v2  `doc_id=rpt__abc123`
        读取：load_document(doc_id) 读全文

``load_document`` **did not exist.** Not on literature, not on
experiment_design, not anywhere in the tree. The channel handed every consumer a
document id and an instruction to call a tool that was not in its tool table.

That is the worst shape a prompt bug can take in this system, and the repo has
paid for it before: an instruction to use a capability the agent does not have is
strictly worse than no instruction, because the model spends turns trying,
"fails", and then reasons from the failure. Two more of the same kind were found
in the same audit:

  * ``paper_writing`` is shown its OWN draft's ``doc_id`` (so a revision can
    continue the same document) but had no reader at all — ``load_draft`` belongs
    to paper_review's tool table, not PW's;
  * ``data_processing`` is shown ``experiment_plan`` whose readback names
    ``get_plan_progress()``, which lives in the meta-tool set that only
    instrument_control and experiment_design receive.

Two fixes, and the second is the one that matters
-------------------------------------------------
1. This module: a real ``load_document`` / ``list_documents`` pair, read-only, on
   every agent. Documents are the substrate the whole artifact channel points at;
   being able to read one is not a per-agent privilege.
2. ``upstream_mw`` now passes the agent's ACTUAL tool names into the renderer, and
   a readback line is emitted only when the tool it names is really there
   (``artifact_channel.render_upstream_block(..., available_tools=…)``). That is
   what makes this class of drift structurally impossible rather than merely
   fixed once: add a field, forget the tool, and the block simply omits the
   promise instead of making a false one.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

#: Cap on returned body text. Same order as ``load_draft``'s 4000 — long enough
#: for a survey's substance, short enough not to blow the turn's budget on a
#: document the agent may only need to skim.
_BODY_MAX = 4000


def _truncate(s: str, n: int = _BODY_MAX) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n…（正文共 {len(s)} 字，此处截断；需要某一节就说明章节名）"


@tool("load_document")
def load_document(doc_id: str, version: int = 0) -> str:
    """按 doc_id 读回一份文档的全文（文献报告 / 实验报告 / 论文草稿 / 评审 / 计划定义）。

    上下文里「上游产物」那一块给出的每个 `doc_id` 都可以用这个工具读回来。**指针给你
    的是标题和要点，正文要读才有** —— 要引用具体数值、复核别人的结论、或者接着改一份
    已有文档，都必须先读。

    Args:
        doc_id:  产物块里给出的 doc_id（原样传，不要改写、不要凭标题猜）。
        version: 留 0 = 最新版本。传具体版本号可以读历史版本（版本永不覆盖）。

    Returns:
        身份行（doc_id / kind / 标题 / 版本）+ 正文。找不到时给出**诚实的**说明和
        下一步动作，不返回空字符串、也不编造内容。
    """
    ref = (doc_id or "").strip()
    if not ref:
        return ("load_document: 需要一个 doc_id。上下文「上游产物」块里每条都带 "
                "`doc_id=…`；也可以先用 list_documents() 看本实验现有哪些文档。")
    try:
        from mast.documents import store
        s = store()
        entry = s.get(ref) or s.resolve_ref(ref)
    except Exception as e:  # noqa: BLE001 — a tool never raises at the model
        return f"load_document failed: {type(e).__name__}: {e}"
    if entry is None:
        return (f"load_document: 找不到 doc_id={ref!r} 对应的文档。"
                "不要凭标题猜 id —— 用 list_documents() 看现有文档，或向上游确认。")
    try:
        v = int(version or 0) or None
        text = entry.read_text(v)
    except Exception as e:  # noqa: BLE001
        return f"load_document: 读取失败（{type(e).__name__}: {e}）"
    if not text:
        return (f"load_document: 文档 {entry.doc_id} 存在，但 "
                f"{'v' + str(version) if version else '最新版本'} 读不到正文。"
                "这是异常情况（版本文件缺失或为空），请如实报告，不要当作空文档处理。")
    meta = getattr(entry, "meta", None)
    title = getattr(meta, "title", "") or "(未命名)"
    kind = getattr(meta, "kind", "") or "?"
    shown = v or entry.latest_version
    head = (f"doc_id = {entry.doc_id}\n"
            f"  kind:  {kind}\n"
            f"  标题:  {title}\n"
            f"  版本:  v{shown}（最新 v{entry.latest_version}）\n")
    return _truncate(head + "\n---\n" + text)


@tool("list_documents")
def list_documents(kind: str = "") -> str:
    """列出**本实验现有哪些文档**（谁产出了什么、到哪个版本、多久以前）。

    用来回答「环境里已经有什么资料了」——比如动手写报告前先看有没有别人已经存过的
    文献综述，或者判断某一步是不是已经有人做过了。

    Args:
        kind: 留空 = 全部。也可以只看一类：`literature_report` / `experiment_report`
              / `paper_draft` / `review` / `plan`。

    Returns:
        每行一份文档：doc_id / kind / 标题 / 最新版本 / 修改时间。**空清单就说空**，
        不会用示例填充 —— 「还没有任何文档」和「读不到」是两件不同的事，会分别说明。
    """
    k = (kind or "").strip() or None
    try:
        from mast.documents import store
        rows = store().list(kind=k)
    except Exception as e:  # noqa: BLE001
        return (f"list_documents: 读不到文档清单（{type(e).__name__}: {e}）。"
                "这不等于『没有文档』—— 是这次查询失败了。")
    if not rows:
        return (f"list_documents: 本实验目前没有{('「' + k + '」类') if k else ''}文档。"
                "（这是确认过的空，不是查询失败。）")
    lines = [f"本实验现有文档（{len(rows)} 份）："]
    for e in rows[:60]:
        meta = getattr(e, "meta", None)
        title = getattr(meta, "title", "") or "(未命名)"
        ekind = getattr(meta, "kind", "") or "?"
        lines.append(f"  [{ekind}] {title} — v{e.latest_version}  doc_id={e.doc_id}")
    if len(rows) > 60:
        lines.append(f"  …还有 {len(rows) - 60} 份（用 kind= 收窄）")
    lines.append("用 load_document(doc_id) 读全文。")
    return "\n".join(lines)


#: Handed to EVERY agent. Read-only and side-effect free, so there is no reason
#: to make reading a document a per-agent privilege — the artifact channel points
#: every one of them at documents by design.
DOCUMENT_TOOLS: list = [load_document, list_documents]

__all__ = ["load_document", "list_documents", "DOCUMENT_TOOLS"]
