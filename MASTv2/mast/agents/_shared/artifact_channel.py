"""The inter-agent product channel — declared once, in one place.

Before this module (see ``docs/v2/design/agent_communication_context_redesign.md``)
the ONLY thing that crossed from one agent to the next was a single free-text
``reason`` string on the handoff tool. Everything an agent actually produced —
its prose, every tool call, every tool result — stayed in the subgraph's own
namespace and evaporated the moment it handed control back, because
``Command(graph=Command.PARENT)`` short-circuits out of the subgraph and only
``command.update`` survives. The next agent (and the SAME agent on its next hop)
was re-seeded from a parent transcript that held nothing but routing notes.

Three symptoms of that one fact, all of which this channel addresses:

  * ``paper_writing``'s prompt had to tell the model **not to trust the handoff
    message** and to re-read the review off disk instead;
  * ``experiment_design``'s prompt claimed a LIT summary would be "in the
    conversation" — it structurally could not be;
  * a document's identity (``doc_id``) survived only as a sentence the model was
    asked to copy forward by hand, on exactly the channel that gets compacted.

What crosses now, and what does not
-----------------------------------
POINTERS cross. BODIES do not. A survey, a draft, a review and a plan are
versioned documents in the experiment folder (``mast.documents`` — ``vNNN.md``,
never overwritten); state carries a :class:`~mast.agents.state.DocRef` with the
id, the version, the path and a BOUNDED summary. That keeps the checkpoint small
(it is rewritten every super-step) and keeps one authority for the text: the
file. A copy in state could only drift from it.

Two tables, and why they are separate
-------------------------------------
``CARRIED_FIELDS`` is what the handoff customs desk copies across the boundary —
a *transport* concern, deliberately generous: carrying a field nobody reads costs
a few dozen bytes, while failing to carry one loses work.

``CONSUMES`` is what each agent is SHOWN — a *context* concern, deliberately
narrow: every rendered line competes with the agent's real instructions, so an
agent is only told about products it can act on.

Both live here rather than in the agents so that adding a product does not
require touching six packages, and — the hard constraint — so no agent package
ever imports another (agent_boundary 钩子（不随仓）).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from mast.agents._shared.artifact_types import (
    AnalysisResult,
    CampaignRef,
    DocRef,
    ScanResult,
)

logger = logging.getLogger(__name__)


class ArtifactToolReturn(Command):
    """A tool return that is BOTH a human summary and a state write.

    Same shape (and same reason) as ``skill_adapter._SkillToolReturn``: langgraph's
    ToolNode only applies ``command.update`` when the tool returns a real
    ``Command``; a plain string — even a str subclass with extra attributes — is
    converted to a bare ToolMessage and every state side-effect is discarded.

    ``str(x)`` and ``"y" in x`` still see the summary so the direct-call unit
    tests (``tool.invoke({...})`` and assert on the text) keep working. A caller
    that needs a real ``str`` (a regex, ``.splitlines()``) must say ``str(x)`` —
    that is a genuine type change, not something to paper over.
    """

    def __init__(self, summary: str, update: dict | None = None, *,
                 tool_call_id: str = "", name: str = ""):
        upd = dict(update or {})
        if tool_call_id:
            upd.setdefault("messages", [ToolMessage(content=summary,
                                                    tool_call_id=tool_call_id,
                                                    name=name or "")])
        super().__init__(update=upd)
        object.__setattr__(self, "_summary", summary)

    def __str__(self) -> str:
        return getattr(self, "_summary", "")

    def __contains__(self, item) -> bool:
        return str(item) in getattr(self, "_summary", "")

    def __len__(self) -> int:
        return len(getattr(self, "_summary", ""))

#: Every artifact key the handoff tool copies from the agent subgraph into the
#: parent graph. **Every name here MUST have a reducer in MASTState** — under a
#: fan-out two branches can write the same key in one super-step and a bare
#: LangGraph channel raises ``InvalidUpdateError`` on that concurrent write.
#: ``test_artifact_channel.py`` asserts this rather than trusting the comment.
CARRIED_FIELDS: tuple[str, ...] = (
    "research_campaign",
    "literature_report",
    "experiment_plan",
    "draft",
    "review",
    "analysis",
    "last_scan",
    "scan_id",
)

#: What each agent is shown about upstream work, in render order.
#:
#: Each agent also sees its OWN latest product. That is not redundancy: an agent
#: re-entered on a later hop gets a fresh subgraph namespace and has no memory of
#: its previous visit, so without this a reviser cannot tell which document it
#: was revising. It is the cheapest fix for "the agent forgets its own last hop".
CONSUMES: dict[str, tuple[str, ...]] = {
    # RD (2026-08-21): the Campaign layer reasons FROM what has already been
    # produced — its own current campaign (so an iteration continues the same
    # programme rather than founding a new one), the survey, what the last plan
    # actually was, and what the analysis found. It is the one role whose input is
    # "everything downstream has done so far".
    "research_director": ("research_campaign", "literature_report",
                          "experiment_plan", "analysis"),
    # LIT: its own report (so a revision continues the same document), and the
    # plan when one exists (a survey aimed at a live plan beats a generic one).
    # The campaign tells it what the survey is FOR — a search aimed at a live
    # hypothesis beats one aimed at a topic.
    "literature": ("research_campaign", "literature_report", "experiment_plan"),
    # XD: the survey it is supposed to design against — the prompt has always
    # claimed this was available; until now it was not. The campaign carries the
    # COMMISSION (plan_request): what this design is supposed to discriminate.
    "experiment_design": ("research_campaign", "literature_report",
                          "experiment_plan", "last_scan"),
    # IC: the plan to execute, and what was measured last.
    "instrument_control": ("experiment_plan", "last_scan", "scan_id"),
    # DP: what to analyse, plus its own last analysis.
    "data_processing": ("last_scan", "scan_id", "analysis", "experiment_plan"),
    # PW: everything a manuscript is assembled from, plus the review to address.
    "paper_writing": ("draft", "review", "analysis", "literature_report", "last_scan"),
    # PR: the draft under review and the evidence behind it.
    "paper_review": ("draft", "analysis", "literature_report"),
    # The SCHEDULER (2026-07-30). Not an agent — the supervisor node is a bare
    # function outside the middleware stack — but it is the role that decides who
    # runs next, and it was the only one in the system that could not see what had
    # already been produced. It gets EVERYTHING, because "has this stage already
    # produced its artifact?" is precisely the question it needs to answer and
    # previously had to guess from a one-line handoff reason.
    "supervisor": CARRIED_FIELDS,
}

#: Header of the injected block. Static + overridable, so an operator can retune
#: the wording in 高级管理 → 上下文注入 without a rebuild (registered in
#: ``mast.prompts.registry`` as ``mw.upstream_artifacts.header``).
UPSTREAM_BLOCK_HEADER = (
    "## 上游产物（其他智能体已完成的工作）\n"
    "下面每一条都是**真实存在**的产物，不是示例。正文不在这里 —— 需要全文时用括号里"
    "给出的工具去读；**修订同一份文档时把 doc_id 原样传回去**，否则会另立新文档。"
)

#: How to read each document kind back, as ``(tool_name, rendered_hint)`` in
#: preference order. The FIRST entry whose tool the agent actually holds is the
#: one rendered; if it holds none of them, no readback line is emitted at all.
#:
#: That check is not defensive coding, it is the fix for a real defect (found
#: 2026-07-30, one day after this channel shipped): the table used to name
#: ``load_document`` unconditionally and **that tool did not exist anywhere in the
#: tree**, so every consumer was told to call something it could not call. Two
#: neighbours had the same shape — ``paper_writing`` is shown its own draft but
#: ``load_draft`` is paper_review's tool, and ``data_processing`` is shown the
#: plan but the plan tools live in the meta-tool set it does not receive.
#:
#: An instruction to use a capability the agent does not have is worse than no
#: instruction: the model spends turns trying, treats the failure as information,
#: and reasons on from it. Gating on the real tool list makes this class of drift
#: structurally impossible instead of merely fixed once.
_READBACK: dict[str, tuple[tuple[str, str], ...]] = {
    "literature_report": (("load_document", "load_document(doc_id) 读全文"),),
    "experiment_report": (("load_document", "load_document(doc_id) 读全文"),),
    "paper_draft": (("load_draft", "load_draft(doc_id) 读全文"),
                    ("load_document", "load_document(doc_id) 读全文")),
    "review": (("load_review", "load_review(doc_id) 读全文"),
               ("load_document", "load_document(doc_id) 读全文")),
    # NB the plan's id is a planning-DB plan_id, NOT a documents-store doc_id
    # (create_plan sets doc_id=plan_id), so load_document is deliberately NOT an
    # option here — it would look plausible and fail.
    "plan": (("get_plan_progress", "get_plan_progress() 看进度"),
             ("list_plans", "list_plans() 看计划清单")),
}


def _readback_hint(kind: str, available: "set[str] | None") -> str | None:
    """The readback instruction for ``kind``, or None when the agent lacks the tool.

    ``available is None`` means "the caller could not determine the tool list" —
    we then fall back to the first (canonical) hint rather than silently dropping
    every instruction, because a missing tool list is a caller bug, not evidence
    that the agent has no tools.
    """
    options = _READBACK.get(kind or "")
    if not options:
        return None
    if available is None:
        return options[0][1]
    for tool_name, hint in options:
        if tool_name in available:
            return hint
    return None


def _doc_line(label: str, ref: Any, available: "set[str] | None" = None) -> str | None:
    """One rendered line for a DocRef-shaped value, or None when there is none.

    Accepts a plain dict as well as a ``DocRef`` because a value round-tripped
    through the SQLite checkpointer can come back as either.
    """
    doc_id = _get(ref, "doc_id")
    if not doc_id:
        return None
    version = _get(ref, "version") or 0
    title = _get(ref, "title") or "(未命名)"
    kind = _get(ref, "kind") or ""
    by = _get(ref, "produced_by") or ""
    summary = (_get(ref, "summary") or "").strip()
    path = _get(ref, "path") or ""

    head = f"- **{label}**：《{title}》 v{version}  `doc_id={doc_id}`"
    if by:
        head += f"  —— 由 {by} 产出"
    parts = [head]
    how = _readback_hint(kind, available)
    if how:
        parts.append(f"    读取：{how}")
    elif path:
        # No tool to read it with — name the file so the operator (and a human
        # reading the transcript) can still find it. Do NOT invent a tool.
        parts.append(f"    文件：{path}")
    if summary:
        parts.append(f"    要点：{summary}")
    return "\n".join(parts)


def _campaign_line(ref: Any, available: "set[str] | None" = None) -> str | None:
    """One rendered line for a :class:`~mast.agents.state.CampaignRef`.

    Deliberately NOT routed through :func:`_doc_line`: a campaign is a DB row, not
    a versioned document, and rendering it as one would print ``doc_id=<ULID>``
    next to a readback hint (``load_document``) that cannot resolve it. An
    identifier shown under the wrong name is worse than none — the model will
    dutifully pass it to the wrong tool and read the failure as information.

    ``plan_request`` is rendered LAST and labelled as a commission, because that
    is the half a downstream designer must act on; the hypothesis is context.
    """
    cid = _get(ref, "campaign_id")
    if not cid:
        return None
    title = _get(ref, "title") or "(未命名纲领)"
    kind = _get(ref, "hypothesis_kind") or ""
    status = _get(ref, "status") or ""
    hypothesis = (_get(ref, "hypothesis") or "").strip()
    request = (_get(ref, "plan_request") or "").strip()
    parent = _get(ref, "parent_campaign_id") or ""

    head = f"- **科研纲领**：《{title}》  `campaign_id={cid}`"
    bits = [b for b in (kind, status) if b]
    if bits:
        head += "  —— " + " · ".join(bits)
    parts = [head]
    if parent:
        parts.append(f"    承接自：`{parent}`（这是一次迭代，不是新开一个纲领）")
    # Same rule as _readback_hint: never name a tool the reader does not hold.
    if available is None or "campaign_get" in available:
        parts.append(f"    读取：campaign_get(\"{cid}\") 看完整目标/谱系/既往实验")
    if hypothesis:
        parts.append(f"    假设：{hypothesis}")
    done_when = (_get(ref, "done_when_brief") or "").strip()
    if done_when:
        # 「什么算答完了」的**机器可判**那半。渲染它是为了让下游知道这条线什么
        # 时候会自己停 —— 判断仍然由代码做（``mast.goals``），这里只是让读到
        # 上游产物的人看得见那条线的终点长什么样。
        parts.append(f"    什么算答完（机器判）：{done_when}")
    if request:
        parts.append(f"    **委托（要你做的事）**：{request}")
    return "\n".join(parts)


def _scan_line(ref: Any) -> str | None:
    path = _get(ref, "sxm_path")
    status = _get(ref, "status") or ""
    if not path and not status:
        return None
    bits = []
    if path:
        bits.append(f"`{path}`")
    if status:
        bits.append(f"状态={status}")
    dur = _get(ref, "duration_s")
    if dur:
        bits.append(f"耗时={dur:.0f}s")
    warns = _get(ref, "warnings") or []
    line = "- **最近一次扫描**：" + "，".join(bits)
    if warns:
        line += "\n    告警：" + "；".join(str(w) for w in warns[:5])
    return line


def _analysis_line(ref: Any) -> str | None:
    metrics = _get(ref, "metrics") or {}
    figures = _get(ref, "figures") or []
    anomalies = _get(ref, "anomalies") or []
    summary = (_get(ref, "summary") or "").strip()
    if not (metrics or figures or anomalies or summary):
        return None
    parts = ["- **数据分析结果**："]
    if summary:
        parts[0] += summary
    if metrics:
        shown = list(metrics.items())[:8]
        parts.append("    指标：" + "，".join(f"{k}={v:g}" for k, v in shown))
    if figures:
        parts.append(f"    图表（{len(figures)} 张，路径可直接引用）：\n"
                     + "\n".join(f"      · {p}" for p in figures[:8]))
    if anomalies:
        parts.append("    异常：" + "；".join(str(a) for a in anomalies[:5]))
    return "\n".join(parts)


_LABELS = {
    "literature_report": "文献报告",
    "experiment_plan": "实验方案",
    "draft": "实验报告 / 论文草稿",
    "review": "评审报告",
}


def _get(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def render_field(field: str, value: Any,
                 available_tools: "set[str] | None" = None) -> str | None:
    """Render ONE artifact field, or return None when it carries nothing.

    Returning None (rather than a placeholder) is deliberate: an empty product
    must produce NO line at all. A block that says "文献报告：（无）" trains the
    model to skim past the whole section, and a fabricated example would be
    worse still — that is the lesson the 2026-07-27 coordinate incident already
    charged us for.

    ``available_tools`` is the consuming agent's real tool names; it gates the
    readback instruction so the block never names a tool the agent lacks.
    """
    try:
        if field == "research_campaign":
            return _campaign_line(value, available_tools)
        if field in _LABELS:
            return _doc_line(_LABELS[field], value, available_tools)
        if field == "last_scan":
            return _scan_line(value)
        if field == "analysis":
            return _analysis_line(value)
        if field == "scan_id":
            return f"- **当前 scan_id**：`{value}`" if value else None
    except Exception as exc:  # noqa: BLE001 — a render glitch must not break a turn
        logger.debug("artifact render failed for %s: %s", field, exc)
    return None


def render_upstream_block(state: Any, agent_id: str, *,
                          header: str | None = None,
                          available_tools: "set[str] | None" = None) -> str:
    """The block injected into ``agent_id``'s system prompt, or "" when empty.

    ``state`` may be a dict or any mapping-ish object; unknown agents get "".

    ``available_tools``: the tool names this agent really holds, so a readback
    instruction is only rendered when it can actually be followed. Pass None only
    when the list genuinely cannot be determined — the renderer then falls back to
    the canonical hint, which is the pre-2026-07-30 behaviour and the reason a
    nonexistent ``load_document`` was advertised to every consumer for a day.
    """
    fields = CONSUMES.get(agent_id or "", ())
    if not fields or state is None:
        return ""
    lines: list[str] = []
    for f in fields:
        val = _get(state, f)
        if val is None:
            continue
        rendered = render_field(f, val, available_tools)
        if rendered:
            lines.append(rendered)
    if not lines:
        return ""
    head = header if header is not None else UPSTREAM_BLOCK_HEADER
    return head + "\n" + "\n".join(lines)


def carried_from(state: Any) -> dict[str, Any]:
    """The subset of ``CARRIED_FIELDS`` present in ``state``, for the handoff.

    Only non-empty values are returned: writing ``None`` into a ``last_wins``
    channel is a no-op by that reducer's contract, but sending it would still
    make every handoff update look like it carries seven products.
    """
    out: dict[str, Any] = {}
    if state is None:
        return out
    for f in CARRIED_FIELDS:
        val = _get(state, f)
        if val is None or val == "" or val == {} or val == []:
            continue
        out[f] = val
    return out


def campaign_ref(*, campaign_id: str, title: str = "", hypothesis: str = "",
                 hypothesis_kind: str = "", status: str = "",
                 plan_request: str = "", parent_campaign_id: str = "",
                 done_when_brief: str = "") -> CampaignRef:
    """Build a :class:`CampaignRef` from a raw ``campaigns`` row's fields.

    Kept beside :func:`doc_ref` for the same reason it exists: the one place that
    knows the channel's shape should be the one place that builds its values, so
    a tool never has to import the state module and get the field set slightly
    wrong. Text bounds are enforced by the model, not here.
    """
    return CampaignRef(
        campaign_id=str(campaign_id or ""), title=title or "",
        hypothesis=hypothesis or "", hypothesis_kind=hypothesis_kind or "",
        status=status or "", plan_request=plan_request or "",
        parent_campaign_id=str(parent_campaign_id or ""),
        done_when_brief=done_when_brief or "")


def doc_ref(*, doc_id: str, version: int = 0, kind: str = "", title: str = "",
            path: str = "", summary: str = "", produced_by: str = "") -> DocRef:
    """Build a :class:`DocRef`. The summary bound is enforced by the model."""
    return DocRef(doc_id=str(doc_id or ""), version=int(version or 0),
                  kind=kind or "", title=title or "", path=str(path or ""),
                  summary=summary or "", produced_by=produced_by or "")


# ═══════════════════════════════════════════════════════════════════════════
# Readiness — "does this agent have what it needs to produce anything real?"
# (2026-07-30, docs/v2/design/wakeup_scheduling.md)
# ═══════════════════════════════════════════════════════════════════════════
#
# A third table, and the same separation-of-concerns argument as the first two:
# CARRIED_FIELDS is transport, CONSUMES is presentation, and these are
# ADMISSION — whether dispatching an agent right now can produce anything but
# fiction. Kept here for the same two reasons: adding a product must not require
# touching six packages, and no agent package may import another.

#: Without it, this agent can only produce garbage. Checked with NO model call —
#: the answer is not a judgement.
#:
#: **Deliberately only two entries.** A hard dependency has to hold
#: UNCONDITIONALLY. "paper_writing hard-needs a draft" is false on a first draft;
#: "experiment_design hard-needs a survey" is false when the operator dictates the
#: plan. Those are conditional, and conditional is exactly what the model is asked
#: about instead. Putting a doubtful entry here builds a rule that cannot express
#: the rule it is standing in for, and then enforces the wrong one — the failure
#: mode is a permanently parked agent, not a visible mistake.
REQUIRES: dict[str, tuple[str, ...]] = {
    # Reviewing a manuscript that does not exist cannot produce a review; it can
    # only produce an invented one.
    "paper_review": ("draft",),
    # Likewise "analysis" with nothing measured.
    "data_processing": ("last_scan",),
}

#: Better with it, workable without. These go to the MODEL: "you are missing X —
#: start anyway with the limitation stated, or wait for it?" A soft miss is the
#: common case, not an exception (experiment_design without a survey is an
#: ordinary way to start), so nothing here may ever block on its own.
PREFERS: dict[str, tuple[str, ...]] = {
    "experiment_design": ("literature_report",),
    "paper_writing": ("analysis", "last_scan", "literature_report"),
    "paper_review": ("analysis", "literature_report"),
    "instrument_control": ("experiment_plan",),
}

#: The closed set an agent may say it is WAITING FOR. Derived from
#: ``CARRIED_FIELDS`` rather than written out again: a fourth hand-maintained list
#: of the same names is a list that will drift, and a ``waiting_for`` value that
#: does not match a real channel can never be matched against a future change —
#: i.e. it is a park that never wakes. Adding a product to CARRIED_FIELDS makes it
#: waitable automatically.
#:
#: ``scan_id`` is excluded: it is an identifier that travels WITH ``last_scan``,
#: never an arrival in its own right, so waiting on it would be waiting for
#: something that never independently "arrives".
WAITABLE_FIELDS: tuple[str, ...] = tuple(
    f for f in CARRIED_FIELDS if f != "scan_id")


def _present_in_state(state: Any, field: str) -> bool:
    """Is ``field`` carrying a real value in ``state``?

    Same emptiness test as :func:`carried_from`, so "carried" and "present" can
    never disagree about what counts as a product.
    """
    val = _get(state, field)
    return not (val is None or val == "" or val == {} or val == [])


#: Which artifact CLASS id in ``_shared.artifacts.ARTIFACTS`` backs each channel
#: field, for the disk-based answer. Ids are that registry's, verified against it —
#: a guessed id would silently resolve to "class not found" and report every field
#: as unknown, which reads like a broken store rather than a typo.
#:
#: ``analysis`` maps to ``figures`` because that is what a completed analysis
#: actually leaves on disk (DP renders into data/figures/ and the mosaic/montage
#: dirs). It is the loosest entry here: figures can exist without the analysis this
#: field means. Deliberately kept anyway — for a WAIT decision the question is
#: "has anything downstream-usable appeared", and the alternative is permanent
#: ``unknown``, i.e. an agent that can never be told its wait is over.
_FIELD_TO_ARTIFACT_CLASS: dict[str, tuple[str, ...]] = {
    "research_campaign": ("research_campaign",),
    "literature_report": ("literature_report",),
    "draft": ("draft",),
    "review": ("review",),
    "experiment_plan": ("experiment_plan",),
    "last_scan": ("scan_files",),
    "analysis": ("figures",),
}


def _present_on_disk(field: str) -> "bool | None":
    """Disk-side answer for one field: True / False / ``None`` = cannot tell.

    The idle path has no ``state`` — no run is awake, so nothing holds one — and
    disk is the authority anyway (state only ever carried pointers INTO it). The
    three-valued return is the whole point and mirrors ``ClassStatus.known``:
    "there is no analysis yet" and "the analysis store could not be read" must not
    collapse into the same answer, because the first is a reason to wait and the
    second is a reason to say so out loud.
    """
    classes = _FIELD_TO_ARTIFACT_CLASS.get(field)
    if not classes:
        return None
    try:
        from mast.agents._shared.artifacts import class_status

        by_id = {cs.artifact.id: cs for cs in class_status()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("readiness: class_status unavailable (%s)", field, exc_info=exc)
        return None
    any_known = False
    for cid in classes:
        cs = by_id.get(cid)
        if cs is None:
            continue
        if cs.known:
            any_known = True
            if cs.count > 0:
                return True
    return False if any_known else None


def readiness(agent_id: str, state: Any = None) -> dict:
    """What ``agent_id`` has, and what it is missing, right now.

    Returns ``{"present", "missing_hard", "missing_soft", "unknown"}`` — all lists
    of field names, in declaration order.

    ⚠️ **State-present is authoritative; state-ABSENT is not.** A field carried in
    ``state`` is proof the product exists. Its absence proves nothing, because the
    channel only populates state when an agent hands off WITHIN this run — a fresh
    run over an experiment folder full of drafts starts with an empty state, and
    concluding "there is no draft" from that would park the reviewer while its input
    sat on disk. So a state miss falls through to disk, and only disk can turn a
    miss into a confirmed absence. This is the same authority order the rest of the
    system uses: the file is the truth, state carries pointers into it.

    Consequently ``state=None`` is not a different mode, just the case where the
    fast path has nothing to offer.

    ``unknown`` is NOT folded into ``missing_*``. A store that cannot be read is
    not an absent product, and the difference decides whether the honest move is
    to wait or to say "I could not check". Callers must surface it rather than
    treating it as either.
    """
    agent = agent_id or ""
    hard = REQUIRES.get(agent, ())
    soft = tuple(f for f in PREFERS.get(agent, ()) if f not in hard)

    present: list[str] = []
    missing_hard: list[str] = []
    missing_soft: list[str] = []
    unknown: list[str] = []

    for field, bucket in [(f, missing_hard) for f in hard] + \
                         [(f, missing_soft) for f in soft]:
        if state is not None and _present_in_state(state, field):
            present.append(field)
            continue
        got = _present_on_disk(field)
        if got is True:
            present.append(field)
        elif got is False:
            bucket.append(field)
        else:
            unknown.append(field)

    return {"present": present, "missing_hard": missing_hard,
            "missing_soft": missing_soft, "unknown": unknown}


def producer_of(field: str) -> str:
    """Which agent normally produces ``field`` — for naming it in a question.

    "You are missing some information" gets an equally vague answer back; "you are
    missing the analysis, which data_processing produces" gets a decision. Derived
    from ``CONSUMES``-adjacent knowledge kept in one place rather than spelled out
    in the prompt text, so a wiring change cannot leave the prompt lying.
    """
    return {
        "research_campaign": "research_director",
        "literature_report": "literature",
        "experiment_plan": "experiment_design",
        "last_scan": "instrument_control",
        "scan_id": "instrument_control",
        "analysis": "data_processing",
        "draft": "paper_writing",
        "review": "paper_review",
    }.get(field, "")


def field_label(field: str) -> str:
    """Human label for a channel field, reusing the render labels where they exist."""
    return _LABELS.get(field, {
        "research_campaign": "科研纲领",
        "last_scan": "扫描数据",
        "analysis": "分析结果",
        "scan_id": "scan_id",
    }.get(field, field))


def doc_ref_from_save(res: Any, *, kind: str = "", produced_by: str = "",
                      summary: str = "") -> DocRef | None:
    """Build a :class:`DocRef` from a ``mast.documents.store().save()`` result.

    Returns None when the save did not succeed — a pointer to a document that
    was not written would be worse than no pointer at all.
    """
    try:
        if not getattr(res, "ok", False):
            return None
        return doc_ref(
            doc_id=getattr(res, "doc_id", "") or "",
            version=getattr(res, "version", 0) or 0,
            kind=kind or getattr(res, "kind", "") or "",
            title=getattr(res, "title", "") or "",
            path=str(getattr(res, "path", "") or ""),
            summary=summary,
            produced_by=produced_by,
        )
    except Exception as exc:  # noqa: BLE001 — never break a save over its pointer
        logger.debug("doc_ref_from_save failed: %s", exc)
        return None


__all__ = [
    "CARRIED_FIELDS", "CONSUMES", "UPSTREAM_BLOCK_HEADER",
    "ArtifactToolReturn",
    "render_field", "render_upstream_block", "carried_from",
    "doc_ref", "doc_ref_from_save", "campaign_ref",
    "AnalysisResult", "CampaignRef", "DocRef", "ScanResult",
]

