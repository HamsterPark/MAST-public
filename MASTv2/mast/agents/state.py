"""Shared state schema across all 6 agents + Orchestrator.

Communication contract (plan §架构; WIRED 2026-07-29, see
``docs/v2/design/agent_communication_context_redesign.md``):
  - Typed artifacts live as MASTState fields, NOT as chat message content.
  - Agents surface artifacts into their own context via a `wrap_model_call`
    middleware that templates the relevant fields into their system prompt
    (``agents/_shared/upstream_mw.py``; which agent consumes which field is
    declared once in ``agents/_shared/artifact_channel.py``).
  - Handoff: `Command(goto="supervisor", graph=Command.PARENT, update={...})`.
    Because that short-circuits out of the agent subgraph, the ONLY way an
    artifact crosses into the parent graph is by riding in that ``update`` —
    the handoff tool acts as the "customs desk" and copies the non-empty
    artifact fields across (``agents/_shared/handoff.py``).
  - Every key with Annotated[..., reducer] must document its reducer here.

POINTERS, NEVER BODIES (2026-07-29). The checkpointer writes the FULL channel
value every super-step, so a field holding an 8000-word survey would be
rewritten into SQLite on every hop. Document-backed artifacts therefore travel
as :class:`DocRef` — doc_id + version + path + a BOUNDED summary — and the body
stays in the experiment folder (``mast.documents``), which is versioned and
survives a power cut. Same discipline as ``ScanResult.sxm_path`` (path only),
``libraries`` (work_id pointers into the one big index) and ``event_refs``
(event ids, payloads stay in the buffer journal). ``DocRef`` enforces the
summary bound in a field validator rather than by convention — a convention
would be re-broken by the first caller in a hurry.

Invariants (hook-enforced by block_scan_tensors_in_checkpointer 钩子（不随仓）):
  - No field may hold torch tensors, numpy arrays, file handles, sockets, or
    Nanonis client objects. Only JSON-serializable types.
  - Paths (strings) are OK; arrays are NOT.
  - Per-thread_id budget enforcement via `budget_remaining_usd` — CALLER-SEEDED:
    the orchestrator ENDs at <=0 but does NOT meter cost / decrement this itself.

Reducer conventions:
  - add_messages:        LangGraph's built-in, handles RemoveMessage + de-dupe by id
  - dedupe_append:       list[str] preserving order, skipping existing values
  - dedupe_event_refs:   list[str] of event_ids, monotonic-grow (events never retract)
  - merge_dicts:         shallow dict merge, right-side wins
  - sum_int_dicts:       per-key ADDING merge for int dicts (visit_count) — every
                         writer contributes a delta that accumulates; None CLEARS
                         (an empty dict is a NO-OP, not a reset)
  - operator.add:        list concatenation (scan_paths, error_log — append-only logs)
  - last_wins:           single-valued but concurrency-TOLERANT (active_agent)
  - merge_routing_hints: append-dedupe, None CLEARS (routing_hints)

PARALLEL-SAFETY INVARIANT (2026-07-11; TIGHTENED 2026-07-29): the supervisor can
fan out to several agents in ONE super-step, so several agent nodes may write the
SAME key concurrently. Any key without a reducer is a LangGraph ``LastValue``
channel and raises ``InvalidUpdateError`` on that concurrent write. Therefore:
**every key an agent handoff writes MUST have a reducer.**

  ⚠️ This used to carry an exemption — "keys only ever written by ONE agent
  (``draft`` by paper_writing, ``analysis`` by data_processing) need no reducer,
  no two branches contend". **That exemption is void.** It held only while
  nothing wrote artifact fields on the handoff path. Since 2026-07-29 the
  handoff tool copies EVERY non-empty artifact field across the parent boundary,
  so under a fan-out two branches genuinely can write the same key in one
  super-step (IC and DP both touching ``last_scan``/``scan_id`` is the easy
  case). Every artifact field is therefore annotated with ``last_wins``, and a
  test asserts that no carried field is left bare. Adding a field to
  ``artifact_channel.CARRIED_FIELDS`` without a reducer crashes every parallel
  run, not just an unlucky one.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────────────
# Reducers (pure functions, no side effects)
# ─────────────────────────────────────────────────────────────────────

def dedupe_append(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append right to left, preserving order, skipping items already in left.

    Used for `executed_skills` — we want a chronological audit trail without
    duplicates when the same skill is re-executed (e.g., after rollback).
    """
    left = list(left or [])
    right = list(right or [])
    seen = set(left)
    for x in right:
        if x not in seen:
            seen.add(x)
            left.append(x)
    return left


def dedupe_event_refs(
    left: list[str] | None, right: list[str] | None
) -> list[str]:
    """Append-only, dedup-by-value reducer for buffer event_id strings.

    Semantically distinct from ``dedupe_append``:
      - inputs MUST be event_id strings (uuid hex) — opaque ids, not skill names;
      - the channel is monotonic — once an event_id is observed it never retracts
        (the buffer's event_journal is the authoritative payload store; state
        only keeps the pointers, per compass §7.2 "Don't store event payloads
        in state — checkpoints write full channel value each super-step");
      - ordering is preserved (chronological by first-seen at reducer time).

    This is a separate function (not a thin alias) so the contract is greppable:
    any future code that reads ``MASTState["event_refs"]`` knows the list is
    deduped event_ids only, never skill names or arbitrary strings.

    Properties (covered by tests/v2/agents/_shared/test_buffer_hitl.py):
      - identity:       dedupe_event_refs(x, []) == x  (and  ([], x) == x)
      - idempotency:    dedupe_event_refs(x, x) == x
      - associativity:  dedupe_event_refs(dedupe_event_refs(a, b), c)
                          == dedupe_event_refs(a, dedupe_event_refs(b, c))
    """
    left = list(left or [])
    right = list(right or [])
    if not right:
        return left
    seen = set(left)
    for ev_id in right:
        if ev_id not in seen:
            seen.add(ev_id)
            left.append(ev_id)
    return left


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow dict merge, right-side wins on key collision.

    Used for `scan_metadata`, `pending_approvals`.
    """
    return {**(left or {}), **(right or {})}


def sum_int_dicts(
    left: dict[str, int] | None, right: dict[str, int] | None
) -> dict[str, int]:
    """Per-key ADDING reducer for int-valued dicts.

    Used for ``visit_count``. Unlike ``merge_dicts`` (right-side overwrites),
    this ACCUMULATES counts per key so every visit to an agent increments its
    running total::

        sum_int_dicts({"ic": 3}, {"ic": 1}) == {"ic": 4}

    Why this matters: the supervisor's per-agent loop
    guard reads ``max(visit_count.values()) > 8``. The handoff tool bumps the
    target's count by writing ``{target: 1}`` every hop. With the old
    ``merge_dicts`` reducer the right side OVERWROTE the left, so a repeatedly
    re-entered agent stayed pinned at 1 forever and the per-agent guard could
    never trip — the most common "stuck bouncing into the same agent" loop went
    completely unbounded. A summing reducer makes ``{target: 1}`` deltas add up,
    so each writer contributes ``+1`` and the guard sees the real visit total.

    Contract for writers: every producer of a ``visit_count`` update must write a
    DELTA (e.g. ``{"supervisor": 1}``), NOT an absolute snapshot, since the
    reducer adds them. Negative / zero deltas are permitted (and simply leave the
    running total unchanged or decremented). Missing keys default to 0.

    ``None`` on the right CLEARS the channel (2026-07-29) — the same escape hatch
    ``merge_routing_hints`` has, and for the same reason: an append/add-only
    reducer can never be reset, so without it the channel only grows.

    Why that mattered: ``run_task`` seeded ``{"visit_count": {}}`` at the start of
    every task believing it was a reset. Under an ADDING reducer an empty dict is
    a NO-OP. So hops accumulated for the LIFETIME of a group-chat thread, and once
    the running total crossed the loop guard (40) EVERY subsequent task ended
    immediately — the conversation was bricked, and nothing said why. The same
    bug was diagnosed once before, on 2026-06-29, and answered by deleting the
    per-agent cap instead of fixing the reset; the identical defect simply stayed
    on the total. Seed ``None`` to start a task cleanly.
    """
    if right is None:
        return {}
    out: dict[str, int] = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0) + int(v)
    return out


def last_wins(left: Any, right: Any) -> Any:
    """Single-valued channel that TOLERATES concurrent writers (parallel fan-out).

    A plain (un-annotated) TypedDict key compiles to a LangGraph ``LastValue``
    channel, which raises ``InvalidUpdateError: can receive only one value per
    step`` the moment two nodes write it in the SAME super-step. That is exactly
    what happens once the supervisor fans out to several agents and they all
    hand back at once (each handoff writes ``active_agent``) — spike-verified
    2026-07-11.

    This reducer keeps the "one value" semantics but makes a concurrent write a
    no-crash last-writer-wins instead of a hard failure. ``None`` on the right is
    treated as "no opinion" so a writer can leave the value untouched.

    Only use it for channels nothing *depends* on for correctness — ``active_agent``
    is a narration/debug hint (the UI tracks the live agent through
    ``_agents_api_state["task"]["active_agent_id"]``, not through state), so an
    arbitrary winner among concurrent branches is harmless. For a channel where
    every branch's value MATTERS, collect them in a list instead (see
    :func:`merge_routing_hints`).
    """
    return right if right is not None else left


def merge_routing_hints(
    left: list[str] | None, right: list[str] | None
) -> list[str]:
    """Collect every branch's "who should run next" intent; ``None`` CLEARS.

    Parallel fan-out makes routing intent inherently plural: when the supervisor
    dispatches literature + data_processing at once, BOTH may come back with a
    next-hop request (→ paper_writing, → paper_writing). The old scalar
    ``routing_hint`` could not represent that (and, being un-annotated, crashed
    on the concurrent write).

    Semantics:
      * ``right`` is a list  → append-dedupe (order preserved, first-seen wins),
        so N parallel handoffs accumulate into N hints.
      * ``right is None``    → CLEAR to ``[]``. The supervisor writes ``None``
        after consuming the hints, which is what keeps this channel from growing
        forever (an append-only reducer alone could never be reset).

    Identity / idempotency:
      merge_routing_hints(x, [])   == x
      merge_routing_hints(x, x)    == x
      merge_routing_hints(x, None) == []
    """
    if right is None:
        return []
    out = list(left or [])
    for x in right:
        if x and x not in out:
            out.append(x)
    return out


def merge_libraries(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Incremental, history-preserving merge for the ``libraries`` channel.

    The ``libraries`` channel is a LIGHTWEIGHT, JSON-only mirror of the
    authoritative ``mast.knowledge.libraries`` registry (the real persistence
    lives in ``artifacts/literature_libs/registry.json``). State only carries
    a snapshot so a checkpoint / handoff can ship "which libraries exist + which
    is active + each library's member work_id pointers" without re-reading the
    registry. **It never stores paper content** — a library is a set of
    ``work_id`` string pointers into the single big OpenAlex index; the big index
    is the one true library and every other library is a pointer set into it
    (owner design G: 大库是唯一真实的库,其他库都是大库的指针).

    Shape (pure JSON — no tensors/handles)::

        {
          "active_id": "<library_id>",
          "items": {
            "<library_id>": {
              "name":    "<display name>",
              "scope":   "global|custom|experiment",
              "members": ["W100", "W200", ...]   # work_id pointers, strings
            },
            ...
          }
        }

    Merge semantics (incremental, never drops history):
      - ``active_id``: right-side wins when present & non-empty (the latest
        writer's active library); else keep the left's.
      - ``items``: per-library-id merge. A library present only on one side is
        kept. For a library present on both sides:
          * ``name`` / ``scope``: right-side wins when present (rename / re-scope).
          * ``members``: UNION (order-preserving, left first) — additive, so a
            stale snapshot that lost a member never erases it from the merged
            view. Removals are reconciled against the authoritative registry on
            the next full snapshot write, not via this reducer (additive-only
            keeps concurrent super-step writes monotonic, like
            ``dedupe_event_refs``).

    Identity / idempotency:
      - ``merge_libraries(x, None) == merge_libraries(None, x) == x``
      - ``merge_libraries(x, x) == x``
    """
    left = left or {}
    right = right or {}
    if not right:
        return {**left} if left else {}
    if not left:
        return {**right}

    out: dict[str, Any] = {}

    # active_id: latest non-empty writer wins.
    r_active = right.get("active_id")
    l_active = left.get("active_id")
    out["active_id"] = r_active if r_active else l_active

    l_items: dict[str, Any] = dict(left.get("items") or {})
    r_items: dict[str, Any] = dict(right.get("items") or {})
    merged_items: dict[str, Any] = {}
    for lib_id in list(l_items.keys()) + [k for k in r_items if k not in l_items]:
        l_rec = l_items.get(lib_id) or {}
        r_rec = r_items.get(lib_id) or {}
        if not r_rec:
            merged_items[lib_id] = dict(l_rec)
            continue
        if not l_rec:
            merged_items[lib_id] = dict(r_rec)
            continue
        # name / scope: right-side wins when present.
        name = r_rec.get("name") if r_rec.get("name") is not None else l_rec.get("name")
        scope = r_rec.get("scope") if r_rec.get("scope") is not None else l_rec.get("scope")
        # members: order-preserving union (additive — never drops a member).
        members: list[str] = list(l_rec.get("members") or [])
        seen = set(members)
        for m in (r_rec.get("members") or []):
            if m not in seen:
                seen.add(m)
                members.append(m)
        merged_items[lib_id] = {"name": name, "scope": scope, "members": members}

    out["items"] = merged_items
    return out


# ─────────────────────────────────────────────────────────────────────
# Typed artifacts —— 定义已搬到 ``_shared/artifact_types.py``（2026-08-27）
# ─────────────────────────────────────────────────────────────────────
#
# 这四个 pydantic 模型是**引擎无关**的：它们描述「一份文档指针长什么样」，与谁在
# 驱动 agent 无关。留在这个文件里只是因为下面的通道表要拿它们当注解类型 —— 而那笔
# 账在退出 langgraph 时才结：删除清单把本文件整个划掉，于是 ``artifact_channel.py``
# （663 行，绝大部分与 langgraph 无关）当场 ImportError。**一个文件里住着两种职责，
# 静态分析看不出来**，是在隔离副本上真删一次才看见的。
#
# 下面这行是**再导出**，不是第二份定义 —— 既有的 ``from mast.agents.state import
# DocRef`` 一个都不用改。本文件随图一起删除时，这行跟着走，新家不受影响。
from mast.agents._shared.artifact_types import (  # noqa: E402,F401
    LIST_MAX_ITEMS,
    SUMMARY_MAX_CHARS,
    AnalysisResult,
    CampaignRef,
    DocRef,
    ScanResult,
)


# ─────────────────────────────────────────────────────────────────────
# The shared graph state
# ─────────────────────────────────────────────────────────────────────

class MASTState(TypedDict):
    """Top-level LangGraph state. Every agent reads from / writes to this."""

    # --- chat transcript (supports RemoveMessage + id-dedupe) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- routing ---
    # last_wins reducer, NOT a bare LastValue channel: with parallel fan-out
    # several agents hand back in the SAME super-step and each handoff writes
    # this key — a bare channel raises InvalidUpdateError on that concurrent
    # write. Nothing reads this field for correctness (it is narration; the UI
    # tracks the live agent via _agents_api_state), so an arbitrary winner among
    # concurrent branches is fine — not crashing is the point.
    active_agent: NotRequired[Annotated[str, last_wins]]

    # --- typed artifacts: the inter-agent product channel (WIRED 2026-07-29) --
    # These are what an agent's work leaves behind for the NEXT agent. They are
    # copied across the subgraph→parent boundary by the handoff tool (see
    # agents/_shared/handoff.py) and rendered into the consumer's system prompt
    # by UpstreamArtifactMiddleware. Which agent consumes which field is declared
    # once in agents/_shared/artifact_channel.py.
    #
    # EVERY one carries ``last_wins``: under a fan-out two branches can write the
    # same key in one super-step and a bare channel would raise
    # InvalidUpdateError (see the PARALLEL-SAFETY INVARIANT above). "Only one
    # agent writes this" is NOT a safe reason to omit the reducer any more.
    #
    # Newest wins and there is no history here on purpose: every version is
    # already immortal on disk (vNNN.md / the .sxm files), so keeping a chain in
    # state would duplicate the archive AND grow without bound.
    # The Campaign layer's product (2026-08-21): the hypothesis this run exists to
    # test, plus the commission it hands experiment_design. Upstream of everything
    # else here — it answers 为什么做, the rest answer 做什么 / 怎么做 / 做出了什么.
    research_campaign: NotRequired[Annotated[CampaignRef, last_wins]]
    literature_report: NotRequired[Annotated[DocRef, last_wins]]
    experiment_plan: NotRequired[Annotated[DocRef, last_wins]]
    draft: NotRequired[Annotated[DocRef, last_wins]]
    review: NotRequired[Annotated[DocRef, last_wins]]
    analysis: NotRequired[Annotated[AnalysisResult, last_wins]]
    last_scan: NotRequired[Annotated[ScanResult, last_wins]]
    scan_id: NotRequired[Annotated[str, last_wins]]
    # Removed 2026-07-29 (zero reads, zero writes, models never instantiated):
    # pending_scan, tip_status, last_vision_event. Tip state and vision events
    # are authoritative in the BufferService — a copy here could only go stale.

    # --- audit / append-only logs ---
    # ALL of these are idempotent under a re-merge, and that is a REQUIREMENT, not
    # a nicety: an agent that ends without handing off writes its whole final state
    # back into these channels, and its copy was seeded from this one. Under a bare
    # ``operator.add`` that write-back duplicates every entry — measured 2026-07-30.
    # ``scan_paths`` and ``error_log`` were bare ``add`` until then.
    #
    # The cost of de-duping ``error_log`` is that two byte-identical failures read
    # as one. That is acceptable because it is NOT the authoritative record: every
    # skill call is written to the records DB unconditionally by ``_emit_record``
    # (both success and failure branches, since the 2026-07-27 forensics), and that
    # is where a count belongs. This channel has zero readers in the tree.
    executed_skills: Annotated[list[str], dedupe_append]
    #: 本轮已取出的工具包（按需加载）。父图也要有它，否则子图不交接直接结束时
    #: 这一条会被父图的 schema 丢掉 —— 症状是「刚取出来的包下一跳又没了」。
    loaded_tool_packs: Annotated[list[str], dedupe_append]
    scan_paths: Annotated[list[str], dedupe_append]   # artifact PATHS, never raw data
    scan_metadata: Annotated[dict[str, Any], merge_dicts]
    error_log: Annotated[list[str], dedupe_append]

    # --- buffer event pointers (RISK A.5 / compass §7.2) ---
    # event_id strings only — the full VisionEvent payload stays in the
    # BufferService event_journal (aiosqlite WAL) to keep checkpoints small.
    # Append-only, dedup-by-value via dedupe_event_refs reducer.
    event_refs: Annotated[list[str], dedupe_event_refs]

    # --- loop / budget guards ---
    # ADDING reducer (sum_int_dicts): every hop writes a +1 DELTA for the agent
    # being entered, so the supervisor's per-agent guard sees the real running
    # total. (Was merge_dicts, which overwrote and pinned every agent at 1 —
    # 审查.)
    visit_count: Annotated[dict[str, int], sum_int_dicts]
    # Hard gate (): the supervisor ENDs the run when this is <= 0.
    # The graph still does no billing of its own — the value is SEEDED by the host
    # at run start and REFRESHED every hop from a ``budget_probe`` callable
    # (mast.billing.run_meter). Until 2026-07-30 it was seeded by nobody at all,
    # so the gate had never once fired; see run_meter's docstring for both that
    # history and the attribution caveat (ledger rows carry no run_id, so the
    # probe reports system-wide spend since run start — an over-estimate, which
    # is the safe direction for a ceiling).
    #
    # ⚠️ DELIBERATELY has NO reducer, and this is the one field where the
    # PARALLEL-SAFETY INVARIANT above must NOT be applied mechanically.
    #
    # Annotating it ``Annotated[float, last_wins]`` was tried on 2026-07-30 and
    # ends every run on its FIRST hop. A reducer turns a NotRequired channel from
    # "absent" into "initialised", and a numeric channel initialises to the type's
    # zero — verified directly::
    #
    #     x: NotRequired[Annotated[float, last_wins]]  → state.get("x") == 0.0
    #     y: NotRequired[float]                        → state.get("y") is None
    #
    # For a gate whose whole contract is ``<= 0 → END``, 0.0 is not a harmless
    # default, it is "budget exhausted". So the reducer would convert the honest
    # inert default into an unconditional kill switch.
    #
    # Omitting it is safe here for a reason that does NOT generalise to the
    # artifact fields: the sole writer is the ``supervisor`` node, which is never a
    # ``Send`` target and so cannot run concurrently with itself, and
    # ``AgentSubState`` deliberately excludes this key, so no subgraph can write it
    # back either. If a second writer is ever added, the fix is a reducer whose
    # identity is not zero — not last_wins.
    budget_remaining_usd: NotRequired[float]       # hard gate at 0; see above

    # --- handoff routing hints ---
    # Every agent handoff routes through the 'supervisor' node so the loop /
    # budget guard runs on EVERY inter-agent hop (direct sibling→sibling hops
    # used to bypass it entirely, letting an A→B→A ping-pong run unbounded).
    # When an agent wants a specific next agent it records that intent here; the
    # supervisor honours it (after running its guards) instead of re-asking the
    # LLM router.
    #
    # PLURAL because routing intent is plural under parallel fan-out: when the
    # supervisor dispatches several agents at once they all hand back in the same
    # super-step, each with its own next-hop request. The scalar `routing_hint`
    # could neither represent that nor survive the concurrent write (bare
    # LastValue → InvalidUpdateError). Reducer: merge_routing_hints —
    # append-dedupe, and `None` CLEARS (the supervisor writes None once it has
    # consumed them, which is what bounds this channel).
    #
    # The supervisor dispatches ALL valid hints: several agents asking for a next
    # hop IS a fan-out request, and it is honoured as one.
    routing_hints: NotRequired[Annotated[list[str], merge_routing_hints]]

    # --- parked activations (2026-07-30, wakeup scheduling W2) ------------------
    # Agents the supervisor did NOT dispatch because dispatching them right now
    # could only produce fiction (a hard dependency is missing), or because the
    # agent itself was asked and chose to wait for something.
    #
    #   {agent: {"waiting_for": [field...], "reason": str, "at": epoch,
    #            "park_id": str, "asked": bool}}
    #
    # This is the FAST PATH, not the authority. The authority is the on-disk board
    # (``core/park_board.py``), because a park has to outlive the run: the whole
    # point is to be woken later, and when the system is idle no run exists to hold
    # a state. Every run rebuilds this from the board at its start, and every WRITE
    # goes to the board — two writers on two threads (the graph parks; the wake
    # scheduler counts declines) would otherwise drift apart. Same
    # authority/cache split as versions.jsonl vs the documents DB index.
    #
    # ``merge_dicts`` so a fan-out where two branches park is not a concurrent-write
    # crash, and so a no-handoff write-back is idempotent.
    pending_activations: NotRequired[Annotated[dict[str, Any], merge_dicts]]

    # --- supervisor → operator question (2026-08-01) ----------------------------
    # The goal was too ambiguous to route, so instead of guessing an agent or
    # quietly ending, the supervisor hands the question to the ``ask_operator``
    # node, which pauses the graph on it.
    #
    #   {"question": str, "options": [str, ...]}
    #
    # It lives in state rather than being asked inline in ``supervisor_node``
    # because a resume REPLAYS its node from the top: the supervisor's LLM route
    # call would run a second time and might not reach the interrupt again,
    # silently discarding the operator's answer. ``ask_operator`` makes no model
    # call before interrupting, so its replay is deterministic.
    #
    # Bare LastValue (no reducer) is correct and deliberate: only the supervisor
    # writes it, one at a time, and ``None`` must CLEAR it — a reducer would turn
    # "no question pending" into a value that never goes away (see the
    # 「reducer changes the default」 note at the top of this file).
    pending_user_question: NotRequired[dict[str, Any] | None]

    # --- 目标终止判据 (2026-08-27) -----------------------------------------
    # 「什么算做完了」的**代码可求值**那一半。在它之前，一次 run 的目标只是
    # ``messages[0]`` 那句自然语言，而「做完了没有」写在路由提示词里由模型自
    # 己判 —— 图里没有任何一条边在检查它，代码层的终止只有跳数/预算/递归三个
    # 熔断。两个老病同根：**太早停**（模型说「完了」——fail_silent 那 621 条）
    # 与**停不下来**（每次唤醒是新 run，per-run 熔断全归零）。
    #
    #   {"text": str,                     # 判据（_goal_text 优先读它）
    #    "done_when": <闭集判据树的 JSON>,  # mast.goals.normalise_done_when 校验过
    #    "baseline": {field: [count, mtime]},   # 目标设定那一刻的产物版本
    #    "operator_confirmed": {"answer": "yes"|"no", "at": float} | None,
    #    "last_verdict": {...},           # 最近一次求值，给 UI/SSE 看
    #    "asked": bool}                   # 本 run 问过用户没有（每 run 一次）
    #
    # **裸 LastValue，没有 reducer** —— 与 ``pending_user_question`` 同一个理由，
    # 而且更要紧：dict 一旦加了 reducer，缺省就从「没有」变成 ``{}``，于是
    # 「这次没给目标」和「给了个空目标」再也分不开，而前者必须逐字节保持今天的
    # 行为。写者只有 supervisor 与 ask_operator（都不是 Send 的目标，且不在同一
    # 个 super-step），所以并发写不会发生。
    #
    # **不进 AgentSubState**（子图写不回来）、**不进 CARRIED_FIELDS**（那张表是
    # 产物运输，goal 是控制面）、**不塞进 CampaignRef**（那是跨 run 的 DB 指针、
    # 由 research_director 的工具写 —— 把终止判据放进去等于让 run 里的模型改写
    # 「自己何时该停」，正是要移除的诱因）。
    goal: NotRequired[dict[str, Any] | None]

    # --- HITL ---
    pending_approvals: Annotated[dict[str, Any], merge_dicts]

    # --- literature libraries snapshot (read-only mirror, P1) ---
    # Lightweight JSON mirror of the mast.knowledge.libraries registry so a
    # checkpoint / handoff can carry "which libraries exist + active + each
    # library's member work_id pointers". The AUTHORITATIVE store is still
    # artifacts/literature_libs/registry.json — this channel is a snapshot for
    # checkpoint/handoff portability only. A library is a SET OF work_id string
    # pointers into the one big OpenAlex index (the only real library); no paper
    # content is ever stored here. Reducer: merge_libraries (incremental,
    # member union, never drops history). Pure JSON — never tensors/handles.
    #   {"active_id": str, "items": {lib_id: {"name", "scope", "members": [work_id...]}}}
    libraries: NotRequired[Annotated[dict[str, Any], merge_libraries]]

    # --- Composite skill progress (Phase 7 graph framework) ---
    # Key: composite skill name; Value: CompositeProgress.to_dict() snapshot.
    # On checkpoint flush after each sub-step the merge_dicts reducer
    # replaces the entry for that skill with the latest snapshot so
    # resume reads the freshest state.
    composite_progress: Annotated[dict[str, dict[str, Any]], merge_dicts]

    # --- session scoping (thread_id = experiment_id in SqliteSaver) ---
    experiment_id: NotRequired[str]
    sample_id: NotRequired[str]
    run_id: NotRequired[str]


class AgentSubState(TypedDict):
    """The state schema an AGENT SUBGRAPH is compiled with — NOT ``MASTState``.

    Why this exists (2026-07-30, found by spike after shipping)
    ----------------------------------------------------------
    On 2026-07-29 the six agents were switched to ``state_schema=MASTState`` so
    their tools' artifact writes would stop being silently dropped. That was
    necessary but too wide: it also gave every subgraph its own copy of the
    PARENT'S CONTROL PLANE, and one exit path writes that copy back.

    An agent that hands off leaves via ``Command(graph=Command.PARENT)``, which
    short-circuits — only ``command.update`` crosses. But an agent that ends
    WITHOUT handing off (``ModelCallLimit``'s ``jump_to:end``, a StallGuard forced
    stop, or the model simply answering in prose) returns its full final state,
    and that state is merged into the parent through the parent's reducers.
    ``visit_count``'s reducer ADDS, so the subgraph's copy — seeded from the
    parent at dispatch — was added on top of the parent's own::

        hands off      → visit_count {'supervisor': 2, 'literature': 1}   ✓
        ends silently  → visit_count {'supervisor': 2, 'literature': 2}   ✗ inflated

    Measured, not deduced. And that path is the common one: the diagnostics ledger
    records 621 ``fail_silent_end`` events. Every one of them was charging the loop
    guard twice for one hop — the guard both caps read.

    The rule this encodes: **a channel that only the parent's control plane uses
    does not belong in a subgraph's schema.** ``visit_count`` / ``routing_hints`` /
    ``active_agent`` / ``budget_remaining_usd`` are written by the handoff tool via
    ``Command.PARENT``, which addresses the PARENT's channels — so leaving them out
    here costs nothing and removes the whole class of write-back pollution.

    What stays: everything a subgraph's own tools genuinely read or write. All of
    those reducers are idempotent (``add_messages`` de-dupes by id,
    ``dedupe_append`` / ``dedupe_event_refs`` by value, ``merge_dicts`` and
    ``last_wins`` overwrite), so a no-handoff write-back is a no-op rather than a
    duplication. That is the property to preserve when adding a field here —
    a bare ``operator.add`` channel would silently double on this path.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # --- the artifact channel (tools write these; upstream_mw reads them) ---
    research_campaign: NotRequired[Annotated[CampaignRef, last_wins]]
    literature_report: NotRequired[Annotated[DocRef, last_wins]]
    experiment_plan: NotRequired[Annotated[DocRef, last_wins]]
    draft: NotRequired[Annotated[DocRef, last_wins]]
    review: NotRequired[Annotated[DocRef, last_wins]]
    analysis: NotRequired[Annotated[AnalysisResult, last_wins]]
    last_scan: NotRequired[Annotated[ScanResult, last_wins]]
    scan_id: NotRequired[Annotated[str, last_wins]]

    # --- audit / progress the skill adapter writes ---
    executed_skills: Annotated[list[str], dedupe_append]
    scan_paths: Annotated[list[str], dedupe_append]
    scan_metadata: Annotated[dict[str, Any], merge_dicts]
    error_log: Annotated[list[str], dedupe_append]
    event_refs: Annotated[list[str], dedupe_event_refs]
    composite_progress: Annotated[dict[str, dict[str, Any]], merge_dicts]

    # --- 本轮已取出的工具包（按需加载；reducer 只增不减）---
    #
    # 单调增长是设计的一部分，不是省事：可见集一旦回缩，模型刚看见的工具下一步
    # 就没了；而对 prompt cache 来说，只增不减才让工具前缀保持稳定。
    # dedupe_append 天然幂等，所以 fail_silent 那条「不交接就把整份 state 写回
    # 父图」的路径上它是 no-op（见本类 docstring 里那条纪律）。
    loaded_tool_packs: Annotated[list[str], dedupe_append]

    # --- session scoping (read-only for agents; carried so tools can see it) ---
    experiment_id: NotRequired[str]
    sample_id: NotRequired[str]
    run_id: NotRequired[str]


__all__ = [
    # Reducers
    "dedupe_append", "dedupe_event_refs", "merge_dicts", "sum_int_dicts",
    "merge_libraries", "last_wins", "merge_routing_hints",
    # Artifacts
    "DocRef", "ScanResult", "AnalysisResult", "CampaignRef",
    "SUMMARY_MAX_CHARS", "LIST_MAX_ITEMS",
    # State
    "MASTState", "AgentSubState",
]
