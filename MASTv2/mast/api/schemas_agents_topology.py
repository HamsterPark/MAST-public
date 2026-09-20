"""Pydantic models for the *agents_topology* domain (parity rebuild Wave A).

This is the read-only inspection seam the old rich Gradio Agents inspector
(``gui/static/agents/agents-ui.jsx`` → ``/agents/snapshot``, ``/artifacts``,
the live HITL interrupt host) rendered, whose LOGIC still lives in the Python
core but lost its endpoint when the UI was rewritten Gradio→TS SPA.

It is ADDITIVE to ``schemas_agents.py`` (tool catalog / model table /
conversation CRUD already covered there): here we add the *topology snapshot*
(per-agent status, threads index, handoff-event timeline, buffer ticker,
capabilities), the *pending HITL interrupts* feed, and the *workspace artifact*
topology + R/W permission matrix.

House rule: every response carries ``degraded: bool`` so the SPA renders an
empty-but-not-broken state when the live agents runtime is not wired into this
API process (standalone dev). The API layer holds NO business logic / safety
checks — these models only mirror the shapes the kept core handlers produce
(``mast.webui.agents_api`` + the live ``MASTApp`` agents-state hooks exposed on
``ctx``). Reasoning/safety stays in the core.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── topology: per-agent registry node ──────────────────────────────────


class AgentNode(BaseModel):
    """One node in the 7-agent topology the inspector draws.

    Mirrors the old JSX ``AGENTS`` registry entry overlaid with live status
    derived from ``/agents/snapshot`` (active / held / model / thinking /
    thread depth). Everything beyond ``id`` is best-effort; missing live data
    just leaves the honest defaults so the graph still renders."""

    id: str
    model: Optional[str] = None
    thinking: Optional[str] = None
    active: bool = False
    held: bool = False
    thread_count: int = 0


class HandoffEvent(BaseModel):
    """One entry in the supervisor→agent / agent→agent hand-off timeline.

    Shape mirrors ``/agents/snapshot.handoff_events`` rows:
    ``{t, kind, text, targets}`` (text already truncated by the core handler).
    ``kind="dispatch"`` with several ``targets`` marks a PARALLEL fan-out — the
    supervisor started all of those agents at once."""

    t: Optional[float] = None
    kind: Optional[str] = None
    text: str = ""
    targets: list[str] = Field(default_factory=list)


class BufferEvent(BaseModel):
    """One DINOv3 vision / scan summary tick (``snapshot.buffer_events``)."""

    t: Optional[float] = None
    kind: Optional[str] = None
    text: str = ""


class ActiveTask(BaseModel):
    """The in-flight SUP task surfaced in the snapshot (None when idle)."""

    id: Optional[str] = None
    description: str = ""
    active: bool = False
    final_text: str = ""
    error: Optional[str] = None


class AgentsCapabilities(BaseModel):
    """Honest capability flags — GATED on the live orchestrator backend.

    The old snapshot reflected reality: operator controls (interject / hold /
    live model switch / tool visibility) are only real on the orchestrator
    path; the MissionPlanner fallback honours none of them. Defaults are the
    conservative all-False so a degraded snapshot never advertises a control
    the backend can't service."""

    interject: bool = False
    hold: bool = False
    model_switch_live: bool = False
    buffer_summarizer_active: bool = False
    abort: bool = False
    tool_visibility: bool = False


class AgentsSnapshotResponse(BaseModel):
    """The Agents-tab topology snapshot (parity with the old
    ``/agents/snapshot`` payload the JSX inspector polled).

    Combines the static registry/model/thinking (always available via
    ``webui.agents_api``, even standalone) with the live session state (threads
    index, handoff timeline, holds, interject count, active task, capabilities)
    when an agents runtime is wired onto ``ctx``. ``backend`` reports which path
    is live (``orchestrator`` | ``mission_planner`` | ``none``)."""

    agents: list[AgentNode] = Field(default_factory=list)
    pipeline: list[str] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    thinking: dict[str, str] = Field(default_factory=dict)
    holds: dict[str, bool] = Field(default_factory=dict)
    threads_index: dict[str, int] = Field(default_factory=dict)
    handoff_events: list[HandoffEvent] = Field(default_factory=list)
    buffer_events: list[BufferEvent] = Field(default_factory=list)
    interject_count: int = 0
    pending_interrupt_count: int = 0
    interrupt_gating: bool = False
    session_active: bool = False
    active_agent_id: Optional[str] = None
    # Every agent currently running. Parallel fan-out makes "the active agent"
    # plural; ``active_agent_id`` keeps the old single-slot contract (it carries
    # a joined label like "instrument_control ‖ literature" when several run).
    active_agents: list[str] = Field(default_factory=list)
    artifacts_count: int = 0
    backend: str = "none"
    capabilities: AgentsCapabilities = Field(default_factory=AgentsCapabilities)
    active_task: Optional[ActiveTask] = None
    server_time: Optional[str] = None
    degraded: bool = False


# ── pending HITL interrupts ────────────────────────────────────────────


class PendingInterrupt(BaseModel):
    """One pending DANGEROUS-skill / workflow-human approval awaiting the
    operator.

    Mirrors ``MASTApp._snapshot_pending_interrupts()`` rows (the rich-object
    shape the old ``AGInterruptHost`` normalised). Fields beyond ``event_id``
    are best-effort because the two interrupt kinds (skill-gate vs
    workflow-human) carry slightly different payloads — extras land in
    ``extra`` so nothing is dropped."""

    event_id: str
    kind: Optional[str] = None
    agent_id: Optional[str] = None
    skill: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    allowed_decisions: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    thread_id: Optional[str] = None
    ask: Optional[dict[str, Any]] = Field(
        default=None,
        description="``kind='ask_user'`` only: the structured question "
        "(question / header / options[{label,description}] / multi_select / "
        "allow_custom / timeout_action) the choice UI renders. Typed rather "
        "than left in ``extra`` because answering one is a first-class action, "
        "not an unrecognised payload.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class InterruptsResponse(BaseModel):
    """Pending HITL interrupts for one agent (or all when ``agent_id`` is the
    roster). ``interrupt_gating`` advertises the live gate is real so the SPA
    surfaces the approve/edit/reject host."""

    agent_id: str
    interrupts: list[PendingInterrupt] = Field(default_factory=list)
    count: int = 0
    interrupt_gating: bool = False
    degraded: bool = False


# ── workspace artifacts + permission matrix ────────────────────────────


class ArtifactEntry(BaseModel):
    """One artifact that ACTUALLY EXISTS on disk.

    Was a typed-MASTState slot read out of ``task["artifacts"]`` — a dict nothing
    ever wrote, so the list could only ever be empty. It now describes a real
    file: which artifact class it belongs to, where it is, how big, when it
    changed, and whether the operator may edit it (file-backed only — an artifact
    the operator cannot meaningfully edit must not pretend to be editable, which
    is what made the old editor a write-only black hole)."""

    id: str
    kind: Optional[str] = None
    producer: Optional[str] = None
    edited: bool = False
    preview: str = ""
    path: str = ""
    bytes: int = 0
    modified_at: Optional[float] = None
    editable: bool = False


class ArtifactGroup(BaseModel):
    """One artifact CLASS and whether anything is in it (/ #29).

    Two things this exists to stop:

      * **#16 — "编辑/填充这些东西还是不会自动生产".** All nine classes showed
        待产出 forever. Half of that was a UI id mismatch; the other half was
        that four classes (文献库 / 实验记录 / 长期记忆 / 视觉缓冲) live in SQLite
        and a JSON registry, and NOTHING enumerated them — so they could not
        report themselves as produced no matter how full they got. ``count``
        comes from each class's own store.
      * **#29 — a flat list buried everything under .sxm rows.** ``high_volume``
        marks the class that arrives in bulk so the UI can start it collapsed.

    ``known=False`` means the store could not be read. It is NOT ``count=0``:
    "I cannot tell" and "there is nothing" are different answers and only one of
    them is a lie about a locked database."""

    artifact_id: str
    label: str = ""
    kind: str = ""
    store: str = ""
    editable: bool = False
    #: Items in this class. Meaningful only when ``known``.
    count: int = 0
    #: False when the class's store could not be probed at all.
    known: bool = True
    #: True iff ``known and count > 0`` — the flag the UI reads to decide
    #: 已产出 vs 待产出. Precomputed so no caller re-derives it wrongly.
    produced: bool = False
    #: One honest line ("77 条实验记录" / "无法读取 <path>").
    detail: str = ""
    #: This class arrives in bulk (scans). UI hint only.
    high_volume: bool = False


class ArtifactsResponse(BaseModel):
    """Workspace artifact topology — best-effort from the live session's typed
    artifacts / MASTState. Degrades to an empty list (``degraded=True``) when no
    schema / live session is available."""

    artifacts: list[ArtifactEntry] = Field(default_factory=list)
    #: Per-CLASS status for ALL nine artifacts, in registry order — including
    #: the ones with zero produced files, so the UI never has to guess a class's
    #: state by intersecting id spaces (which is exactly how #16 happened).
    groups: list[ArtifactGroup] = Field(default_factory=list)
    count: int = 0
    session_active: bool = False
    degraded: bool = False


class ArtifactPermission(BaseModel):
    """One artifact in the research pipeline: WHO writes it, WHO reads it, and
    WHERE it actually lives.

    Rebuilt 2026-07-11 ("访问拓扑不科学"). The old matrix was a
    single-writer fiction derived from a one-to-one {agent: artifact} table:
    every artifact had exactly one writer and EVERY other agent was listed as a
    reader. Both halves were wrong —

      * artifacts really do have SEVERAL writers (figures are produced by
        data_processing AND pulled into the manuscript by paper_writing; the
        long-term memory is written by all six agents), and
      * "everyone else reads it" is false — paper_review never opens a scan
        file, literature never reads an analysis.

    So this is now a real DIRECTED data-flow edge set, derived from what the
    agents' tools actually do, with ``store`` naming the concrete location so a
    claim here can be checked against the disk.
    """

    artifact_id: str
    label: str = ""            # human-readable name (Chinese UI)
    kind: str = "state"        # state | file | db | index
    store: str = ""            # where it physically lives
    writers: list[str] = Field(default_factory=list)
    readers: list[str] = Field(default_factory=list)
    # True when more than one agent legitimately writes it — the case the old
    # single-writer model could not express at all.
    multi_writer: bool = False


class ArtifactPermissionsResponse(BaseModel):
    """The artifact data-flow graph (who writes / reads what).

    ``degraded=True`` (empty) only if the roster itself is unavailable — the
    derivation is static, so it does not depend on a live run."""

    permissions: list[ArtifactPermission] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ── Pending activations (parked agents) — wakeup scheduling W3, 2026-07-30 ───
# An agent the supervisor declined to dispatch because its upstream inputs do not
# exist yet. These rows are the ANSWER to the design's own biggest worry: a parked
# agent that nobody can see is indistinguishable from a hung system, and this repo
# has already paid for that confusion once (621 fail_silent_end events).

class GoalCheckView(BaseModel):
    """一份 park 最近一次目标判据求值。

    与 ``GoalProgressView``（纲领列表那一个）**故意不是同一个模型**：那边答的是
    「这条纲领做到哪了」（要 satisfied/total），这边答的是「这份等待还该不该醒」
    （要 checked_at —— 一个三小时前的结论和一个刚算出来的结论不是一回事）。
    """

    verdict: str = ""
    reason: str = ""
    checked_at: float = 0.0


class PendingActivation(BaseModel):
    """One parked agent, as the UI needs to render it in a single row."""

    park_id: str
    agent: str
    experiment_id: str = ""
    #: Closed-set channel field names — never free text, or it could never be matched
    #: against a future arrival.
    waiting_for: list[str] = Field(default_factory=list)
    #: Human labels for the same fields, so the UI need not know the channel names.
    waiting_for_labels: list[str] = Field(default_factory=list)
    #: Which agent normally produces the first thing it waits for ("" if unknown).
    blocked_by: str = ""
    reason: str = ""
    instruction: str = ""
    #: True when a HARD dependency was missing (decided with no model call at all).
    hard: bool = False
    status: str = "waiting"      # waiting|woken|done|done_by_goal|expired|cancelled
    created_at: float = 0.0
    deadline_at: float = 0.0
    #: Seconds until the deadline; NEGATIVE once overdue. Pre-computed so the row
    #: cannot disagree with the backend about whether a wait has expired.
    seconds_left: float = 0.0
    waited_human: str = ""
    #: How many times the agent was asked to wake and said no. Shown because a wait
    #: that keeps declining is a different situation from one nobody has asked yet.
    declines: int = 0
    woken_run_id: str = ""
    #: Expired and NOT yet acknowledged → the UI must keep this row demanding
    #: attention. A one-shot toast is not a delivery when nobody is watching.
    needs_attention: bool = False
    #: 这份等待是为哪条科研纲领等的 —— 建 park 时**冻结**（2026-08-27）。空串是
    #: 诚实的答案：空闲进程里没有任何东西能重新推导它，而猜错等于把 A 纲领的
    #: 判据套到 B 的等待上，静默地不唤醒。
    campaign_id: str = ""
    #: 最近一次目标判据求值：``{"verdict", "reason", "checked_at"}``。
    #: **只在结论变化时更新** —— 调度器每 60 s 走一趟，每趟落一次盘只为记下
    #: 「还是没满足」，那是无谓写入，也会让 ``updated_at`` 失去信息量。
    #: ``verdict == "unknown"`` 时 ``reason`` 说的是**读不到什么** —— 判不了必须
    #: 可见，否则它和「一切正常」长得一样。
    goal_check: GoalCheckView = Field(default_factory=GoalCheckView)


class PendingActivationsResponse(BaseModel):
    parks: list[PendingActivation] = Field(default_factory=list)
    count: int = 0
    #: Count of rows with needs_attention — lets a badge render without a scan.
    attention_count: int = 0
    degraded: bool = False


class AcknowledgeParkResponse(BaseModel):
    ok: bool = False
    park_id: str = ""
    message: str = ""
