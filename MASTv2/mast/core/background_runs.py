"""Background-run manager — true parallelism WITHOUT changing LangGraph semantics.

The multi-agent orchestrator is one checkpointed graph, so LangGraph's super-step
(BSP) barrier serialises it: a fan-out batch must FULLY finish before the
supervisor dispatches again. That is correct BSP semantics, not a defect — but it
means a long, independent task (a literature survey) run inside the foreground
batch blocks the operator from interacting with instrument_control until it ends
("文献 agent 查文献时仪器 agent 收不到消息", ).

This manager breaks that WITHOUT touching the graph: a long/independent task is
run as a SEPARATE orchestrator invocation — its OWN ``thread_id`` and its OWN
checkpointer — on a daemon thread, so it executes CONCURRENTLY with the live
foreground chat. Its transcript streams back into the SAME durable conversation
(via an injected sink onto ``ConversationStore``), tagged as background, so the
operator sees one merged conversation while the foreground (supervisor +
instrument_control) stays responsive.

Why this is SAFE to run concurrently (the same premise that makes fan-out safe):
instrument_control is the SOLE agent that touches hardware and it stays in the
FOREGROUND; every backgroundable agent (literature / data_processing /
paper_writing / paper_review) is pure compute/network and holds no instrument
skill (pinned by tests/…/test_parallel_real_graphs.py). Two background runs never
contend for hardware because neither is instrument_control.

Resource governance:
  A condition-variable ADMISSION CONTROLLER (not a bare semaphore) gates when a
  queued run may start, honouring three limits together:
    * a global concurrency cap (``max_concurrent``);
    * a per-agent-type quota (``per_type_quota`` — e.g. at most 2 literature runs
      at once, so one type can't starve the others);
    * PRIORITY — a ``high`` run is admitted before ``normal`` waiters.
  A run reports fine-grained progress (``steps`` / ``progress`` / ``last_activity``)
  so the UI shows a live bar, not just start/end.

Extension hooks (wired by the runtime for the smarter-decision + cross-run
features; no-ops when unset, so this module stays unit-testable with fakes):
  * ``duration_sink(agent_type, duration_s)`` — records each run's real duration
    so auto-background can learn which task types are consistently slow;
  * ``completion_sink(run)`` — fired once when a run reaches ``done`` so the
    runtime can inject its output + a "X done, dependent Y can proceed" hint into
    the foreground (NEVER auto-executing anything — the supervisor/HITL decides).

Isolation contract (no shared mutable graph state):
  * each run gets ``thread_id = "bg-<run_id>"`` and the caller's ``run_fn`` gives
    it a DEDICATED checkpointer — the foreground's checkpoint state is never read
    or written, so a background run can never pollute the live MASTState;
  * the ONE shared resource is ``ConversationStore`` and its ``append_message`` is
    atomic + busy-timeout-serialised (safe for concurrent writers by design);
  * ``run_fn`` is INJECTED (not imported), so this module pulls in nothing heavy
    and is unit-testable with a fake runner.

Dependency-light + boundary-clean: imports only the stdlib.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Tag every background transcript row so the operator (and the UI) can tell a
# background contribution from a foreground one at a glance.
BG_TAG = "「后台」"

_TERMINAL = frozenset({"done", "failed", "aborted"})
_PRIORITIES = ("normal", "high")


@dataclass
class BackgroundRun:
    run_id: str
    conversation_id: str
    instruction: str
    agents: tuple[str, ...]
    title: str = ""
    priority: str = "normal"        # normal | high — high admits before normal
    status: str = "queued"          # queued | running | done | failed | aborted
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    final_text: str = ""
    error: str = ""
    thread_id: str = ""
    # ── fine-grained progress (resource-governance ①) ──
    steps: int = 0                  # streamed messages so far
    progress: int | None = None     # 0..100; soft-derived from steps, or set by run_fn
    last_activity: str = ""         # short preview of the latest streamed message
    # ── cross-run dependency (controlled ③) — an EXPLICIT, operator/agent-declared
    #    follow-up. {"next_agent": str?, "note": str}. NEVER auto-executed. ──
    on_done: dict[str, Any] | None = None
    # ── seeded product snapshot (2026-07-30) — an IMMUTABLE, by-value copy of the
    #    foreground's artifact channel, merged into this run's initial state. Only
    #    pointers (DocRefs); bodies stay on disk, which is shared and authoritative.
    #    ``snapshot_at`` travels with it because a snapshot goes STALE between spawn
    #    and execution, and the woken agent is told so rather than being allowed to
    #    assume it is live. ──
    seed_artifacts: dict[str, Any] | None = None
    snapshot_at: float = 0.0
    # ── the park this run was woken FOR ("" if not a wake). Lets the board close the
    #    park when its products come back, instead of assuming a spawn is a success. ──
    park_id: str = ""
    _abort: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def agent_type(self) -> str:
        """The quota / duration key for this run — its primary (first) agent."""
        return self.agents[0] if self.agents else ""

    def public(self) -> dict:
        """JSON-safe snapshot for API / UI (no Event, no thread handle)."""
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "instruction": self.instruction,
            "agents": list(self.agents),
            "title": self.title,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "final_text": self.final_text,
            "error": self.error,
            "thread_id": self.thread_id,
            "steps": self.steps,
            "progress": self.progress,
            "last_activity": self.last_activity,
            # Counts + timestamp only, never the snapshot itself: this dict goes to
            # the UI and over the wire, and the pointers are an internal detail.
            "seeded_artifacts": sorted(self.seed_artifacts or {}),
            "snapshot_at": self.snapshot_at,
            "park_id": self.park_id,
        }


# run_fn(instruction, thread_id, agents, emit, abort_event) -> final_text
#   emit(agent_id: str, role: str, text: str) is called per streamed message;
#   emit.progress(pct: int, note: str = "") optionally reports an explicit percent;
#   emit.compaction(agent_id: str, text: str) optionally marks a context
#     compaction — both are OPTIONAL attributes, so a run_fn
#     must getattr them rather than assume they exist.
RunFn = Callable[[str, str, "tuple[str, ...]", Callable[[str, str, str], None], threading.Event], str]
# transcript_sink(conversation_id, kind, agent_id, role, text) -> None
SinkFn = Callable[[str, str, str, str, str], None]
# duration_sink(agent_type, duration_s) -> None
DurationSink = Callable[[str, float], None]
# completion_sink(run) -> None
CompletionSink = Callable[["BackgroundRun"], None]


class BackgroundRunManager:
    """Spawn, track, and abort detached background orchestrator runs.

    Thread-safe. ``spawn`` returns IMMEDIATELY (a daemon thread does the work) so
    the caller — the foreground request thread — is never blocked. An admission
    controller (below) gates when a queued run may START, so a burst of requests
    cannot exhaust threads / provider rate limits, one task type cannot starve the
    others, and a ``high``-priority run jumps the queue.
    """

    # 判据（2026-08-20 写清楚，行为不变）：**驱动这一步的是不是一次模型判断，
    # 而当时有没有人盯着。** 排除的实质是「无人盯着的、由模型判断驱动仪器的
    # run」——不是「后台线程不许碰硬件」。后者早有在产先例：SafetyWatchdog
    # 与 conduct 的 ConductDirector 都是确定性代码线程，都会驱动技能，都在
    # 后台跑。它们与 agent run 的区别不在线程，在**下一步由什么决定**：
    # 前者由代码和 spec 决定，后者由一次 LLM 采样决定。
    #
    # 所以一个新 agent 能不能进这张表，问题不是「它想不想后台跑」，而是
    # 「它会不会在没人看着的时候让模型决定动仪器」。research_director 这类
    # 只读文献/记录、产出假设的角色可以进；instrument_control 不能。
    #
    # 要给无人值守的仪器动作找路，正确的方向是 conduct 那一层（确定性状态机
    # + 闸门 + 授权包络 + 全程留痕），不是把 IC 塞进这张表。
    #
    # research_director 就是照这条判据进来的（2026-08-21，**不是例外**）：它的
    # 工具面里没有任何执行面 —— campaign 表的读写、既往记录的只读查询、交接，
    # 就这三样；仪器动作在它下游隔着 plan 与 conduct 两层。所以「无人盯着时模型
    # 会不会驱动仪器」这个问题在它身上是结构性的「不会」，而不是一句承诺。
    # （``agents/research_director/tools.py`` 的模块文档说明为什么工具面这么短，
    # 并有一条结构断言钉着它不 import 执行面。）
    BACKGROUNDABLE = frozenset({
        "research_director",
        "literature", "experiment_design", "data_processing",
        "paper_writing", "paper_review",
    })

    def __init__(
        self,
        *,
        run_fn: RunFn,
        transcript_sink: SinkFn | None = None,
        duration_sink: DurationSink | None = None,
        completion_sink: CompletionSink | None = None,
        max_concurrent: int = 3,
        per_type_quota: int = 2,
        max_history: int = 100,
    ) -> None:
        self._run_fn = run_fn
        self._sink = transcript_sink
        self._duration_sink = duration_sink
        self._completion_sink = completion_sink
        self._max_concurrent = max(1, int(max_concurrent))
        self._per_type_quota = max(1, int(per_type_quota))
        self._max_history = max(8, int(max_history))
        self._runs: dict[str, BackgroundRun] = {}
        self._threads: dict[str, threading.Thread] = {}
        # ONE lock guards all mutable state; the Condition wakes queued workers
        # each time a slot / quota frees so they re-check admission.
        self._cond = threading.Condition(threading.Lock())
        self._running_ids: set[str] = set()
        self._running_by_type: dict[str, int] = {}

    # ── public API ──────────────────────────────────────────────────────────
    def spawn(
        self,
        *,
        instruction: str,
        conversation_id: str = "",
        agents: "tuple[str, ...] | list[str]" = ("literature",),
        title: str = "",
        priority: str = "normal",
        on_done: dict[str, Any] | None = None,
        seed_artifacts: dict[str, Any] | None = None,
        park_id: str = "",
    ) -> dict:
        """Start a background run and return its record IMMEDIATELY (non-blocking).

        ``agents`` is filtered to the backgroundable set (instrument_control can
        never be detached). ``priority`` ('high'|'normal') decides queue order when
        a slot is contended. ``on_done`` is an EXPLICIT, caller-declared follow-up
        (never auto-guessed) surfaced when the run completes. Raises ValueError on
        an empty instruction or a request naming no backgroundable agent.

        ``seed_artifacts`` (2026-07-30) is an immutable SNAPSHOT of the foreground's
        product channel, merged into the run's initial state. It exists because a
        background run is state-ISOLATED — its own ``thread_id``, its own
        ``InMemorySaver`` — which is what makes it safe and also what made it blind:
        a detached agent could not see anything the foreground had produced.

        By VALUE, never shared, and that is forced rather than chosen: LangGraph
        checkpoints are per-``thread_id``, so two runs writing one thread would
        corrupt each other. Passing pointers works anyway because the channel already
        carries only ``DocRef``s — the bodies are on disk, which is shared AND
        authoritative, so ``load_document(doc_id)`` in the woken run reads the real
        latest text. The "pointer, not payload" rule earning its keep a second time.

        ``park_id`` links the run back to the board entry that caused it, so a woken
        park can be closed when its products return instead of being assumed finished.
        """
        instruction = (instruction or "").strip()
        if not instruction:
            raise ValueError("empty instruction")
        agents_t = tuple(a for a in (agents or ()) if a in self.BACKGROUNDABLE)
        if not agents_t:
            raise ValueError(
                "no backgroundable agent requested (instrument_control cannot be "
                "detached: a detached run is one where a model decides the next "
                "instrument action with nobody watching. Unattended hardware work "
                "goes through the conduct layer — a deterministic state machine "
                "with gates and a declared envelope — not through here.)")
        prio = priority if priority in _PRIORITIES else "normal"

        run_id = uuid.uuid4().hex[:12]
        rec = BackgroundRun(
            run_id=run_id, conversation_id=conversation_id or "",
            instruction=instruction, agents=agents_t, priority=prio,
            title=(title or instruction[:40]), thread_id=f"bg-{run_id}",
            on_done=(dict(on_done) if isinstance(on_done, dict) else None),
            # Copied, not referenced: the caller's dict is the LIVE foreground state's
            # channel, and it keeps changing after this call returns. A reference
            # would quietly turn "snapshot" into "whatever it is now", which is the
            # one thing §1.4 of the design says this must never be.
            seed_artifacts=(dict(seed_artifacts) if seed_artifacts else None),
            snapshot_at=(time.time() if seed_artifacts else 0.0),
            park_id=str(park_id or ""))
        with self._cond:
            self._runs[run_id] = rec
            self._prune_locked()
            self._threads[run_id] = threading.Thread(
                target=self._worker, args=(run_id,),
                name=f"bg-run-{run_id}", daemon=True)
            th = self._threads[run_id]
        th.start()
        logger.info("background run spawned: %s agents=%s prio=%s conv=%s",
                    run_id, agents_t, prio, conversation_id)
        return rec.public()

    def list_runs(self, conversation_id: str | None = None,
                  *, active_only: bool = False) -> list[dict]:
        with self._cond:
            runs = list(self._runs.values())
        out = []
        for r in runs:
            if conversation_id and r.conversation_id != conversation_id:
                continue
            if active_only and r.status in _TERMINAL:
                continue
            out.append(r.public())
        out.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return out

    def get(self, run_id: str) -> dict | None:
        with self._cond:
            rec = self._runs.get(run_id)
        return rec.public() if rec else None

    def abort(self, run_id: str) -> bool:
        """Signal a background run to stop (honoured between its super-steps, or
        immediately if still queued). False if unknown or already finished."""
        with self._cond:
            rec = self._runs.get(run_id)
            if rec is None or rec.status in _TERMINAL:
                return False
            rec._abort.set()
            self._cond.notify_all()  # wake it if it is waiting for admission
        logger.info("background run abort requested: %s", run_id)
        return True

    def abort_all(self, conversation_id: str | None = None) -> int:
        n = 0
        with self._cond:
            for r in self._runs.values():
                if conversation_id and r.conversation_id != conversation_id:
                    continue
                if r.status not in _TERMINAL:
                    r._abort.set()
                    n += 1
            self._cond.notify_all()
        return n

    def has_active(self, conversation_id: str | None = None) -> bool:
        with self._cond:
            for r in self._runs.values():
                if conversation_id and r.conversation_id != conversation_id:
                    continue
                if r.status not in _TERMINAL:
                    return True
        return False

    # ── admission control (priority + global cap + per-type quota) ───────────
    @staticmethod
    def _prio_rank(rec: BackgroundRun) -> int:
        return 1 if rec.priority == "high" else 0

    def _eligible_locked(self, rec: BackgroundRun) -> bool:
        """Would admitting rec respect the global cap AND its type quota? (Ignores
        priority ordering — that is _is_best_waiter_locked's job.)"""
        if len(self._running_ids) >= self._max_concurrent:
            return False
        if self._running_by_type.get(rec.agent_type, 0) >= self._per_type_quota:
            return False
        return True

    def _is_best_waiter_locked(self, rec: BackgroundRun) -> bool:
        """Is rec the highest-ranked ELIGIBLE waiter (priority desc, then oldest)?
        This is what makes a 'high' run jump ahead of 'normal' ones for a slot."""
        best_key = (self._prio_rank(rec), -rec.created_at)
        for r in self._runs.values():
            if r.run_id == rec.run_id or r.status != "queued":
                continue
            if not self._eligible_locked(r):
                continue
            if (self._prio_rank(r), -r.created_at) > best_key:
                return False
        return True

    def _can_admit_locked(self, rec: BackgroundRun) -> bool:
        return self._eligible_locked(rec) and self._is_best_waiter_locked(rec)

    # ── worker ──────────────────────────────────────────────────────────────
    def _worker(self, run_id: str) -> None:
        with self._cond:
            rec = self._runs.get(run_id)
        if rec is None:  # pragma: no cover — spawn always inserts first
            return
        # Wait for admission (NOT in spawn, so the caller stays non-blocking): the
        # run sits in `queued` until a slot + its type quota free AND it is the
        # best-ranked waiter. notify_all on every admit/finish re-evaluates waiters.
        with self._cond:
            while not (rec._abort.is_set() or self._can_admit_locked(rec)):
                self._cond.wait()
            if rec._abort.is_set():
                rec.status = "aborted"
                rec.finished_at = time.time()
                self._cond.notify_all()
                return
            self._running_ids.add(run_id)
            self._running_by_type[rec.agent_type] = (
                self._running_by_type.get(rec.agent_type, 0) + 1)
            rec.status = "running"
            rec.started_at = time.time()
            # let the NEXT-best waiter check too (multiple slots can fill at once)
            self._cond.notify_all()

        try:
            self._execute(rec)
        finally:
            rec.finished_at = time.time()
            # record duration for the smarter-decision learner (best-effort)
            if self._duration_sink is not None and rec.started_at is not None:
                try:
                    self._duration_sink(rec.agent_type, rec.finished_at - rec.started_at)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("duration_sink failed: %s", exc)
            # fire the cross-run completion hook ONLY on a clean done (not aborted/
            # failed) so a follow-up is never suggested off a broken run.
            if rec.status == "done" and self._completion_sink is not None:
                try:
                    self._completion_sink(rec)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("completion_sink failed: %s", exc)
            with self._cond:
                self._running_ids.discard(run_id)
                self._running_by_type[rec.agent_type] = max(
                    0, self._running_by_type.get(rec.agent_type, 0) - 1)
                self._cond.notify_all()

    def _execute(self, rec: BackgroundRun) -> None:
        """Run rec's run_fn, streaming its messages into the transcript + tracking
        progress. Runs OUTSIDE the admission lock (it is slow / blocking)."""
        self._emit_status(rec, f"{BG_TAG}后台任务启动:{rec.instruction}")

        def emit(agent_id: str, role: str, text: str) -> None:
            # Tag every background message so it is distinguishable in the merged
            # transcript; drop once aborted so a late chunk from a winding-down
            # stream doesn't append after the abort marker.
            if rec._abort.is_set():
                return
            rec.steps += 1
            rec.last_activity = str(text)[:120]
            # soft auto-progress from step count (monotonic, capped <100) so the
            # UI shows a moving bar even when run_fn reports no explicit percent.
            soft = min(90, 5 + rec.steps * 8)
            if rec.progress is None or rec.progress < soft:
                rec.progress = soft
            self._append(rec, "message", agent_id or "", role or "agent",
                         f"{BG_TAG}{text}")

        def _progress(pct: int, note: str = "") -> None:
            try:
                rec.progress = max(0, min(100, int(pct)))
            except (TypeError, ValueError):
                return
            if note:
                rec.last_activity = str(note)[:120]

        def _compaction(agent_id: str, text: str) -> None:
            """Record that the context middleware rewrote this run's history.

            a background run merges into the SAME group
            transcript the operator reads, so a compaction inside one is the same
            silent rewrite — it has to leave a marker there too. Written as a
            ``compaction`` row (not a ``message``) so the panel renders the same
            divider. The transcript sink has no ``meta`` channel, so the counts
            ride in the text and the row simply has no summary to expand — which
            the divider states outright rather than implying one exists.
            """
            if rec._abort.is_set():
                return
            self._append(rec, "compaction", agent_id or "", "", f"{BG_TAG}{text}")

        emit.progress = _progress  # type: ignore[attr-defined]
        emit.compaction = _compaction  # type: ignore[attr-defined]
        # Seeded product snapshot rides on ``emit`` rather than changing ``RunFn``'s
        # signature — the same OPTIONAL-attribute idiom the two hooks above use, and
        # for the same reason: every existing run_fn (and every test double) keeps
        # working untouched, and one that does not know about seeding simply does not
        # look. ``run_id``/``park_id`` travel too so the run can close its park and
        # file its products back under the right identity.
        emit.seed_artifacts = dict(rec.seed_artifacts or {})  # type: ignore[attr-defined]
        emit.snapshot_at = rec.snapshot_at  # type: ignore[attr-defined]
        emit.run_id = rec.run_id  # type: ignore[attr-defined]
        emit.park_id = rec.park_id  # type: ignore[attr-defined]

        try:
            final = self._run_fn(rec.instruction, rec.thread_id, rec.agents,
                                 emit, rec._abort)
            rec.final_text = (final or "").strip()
            if rec._abort.is_set():
                rec.status = "aborted"
                self._emit_status(rec, f"{BG_TAG}后台任务已中止")
            else:
                rec.status = "done"
                rec.progress = 100
                self._emit_status(
                    rec, f"{BG_TAG}后台任务完成:{rec.final_text[:200]}"
                    if rec.final_text else f"{BG_TAG}后台任务完成")
        except Exception as exc:  # noqa: BLE001 — a background failure must be
            # recorded, never crash the manager thread pool silently.
            rec.status = "failed"
            rec.error = f"{type(exc).__name__}: {exc}"
            logger.warning("background run %s failed: %s", rec.run_id, exc, exc_info=True)
            self._emit_status(rec, f"{BG_TAG}后台任务失败:{rec.error}")

    # ── transcript merge (best-effort; never breaks a run) ──────────────────
    def _append(self, rec: BackgroundRun, kind: str, agent_id: str,
                role: str, text: str) -> None:
        if self._sink is None or not rec.conversation_id:
            return
        try:
            self._sink(rec.conversation_id, kind, agent_id, role, text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("background transcript append failed: %s", exc)

    def _emit_status(self, rec: BackgroundRun, text: str) -> None:
        self._append(rec, "status", "", "", text)

    # ── housekeeping ────────────────────────────────────────────────────────
    def _prune_locked(self) -> None:
        """Drop the oldest FINISHED runs when over the history cap (called under
        the lock). Active runs are never pruned."""
        if len(self._runs) <= self._max_history:
            return
        finished = sorted(
            (r for r in self._runs.values() if r.status in _TERMINAL),
            key=lambda r: r.finished_at or r.created_at)
        drop = len(self._runs) - self._max_history
        for r in finished[:drop]:
            self._runs.pop(r.run_id, None)
            self._threads.pop(r.run_id, None)


def build_completion_hint(run: "BackgroundRun") -> str:
    """Format the FOREGROUND-supervisor note fired when a background run finishes
    cleanly (cross-run dependency, item ③).

    It SURFACES the result and — ONLY if the caller EXPLICITLY declared a follow-up
    via ``run.on_done`` ({"next_agent", "note"}) — names the suggested next step
    (never a guessed one). It is ADVISORY: it never triggers execution (the
    supervisor / HITL decides) and it is worded to discourage auto-running
    hardware. Pure + never raises, so it is unit-testable and safe on the worker
    thread's finally-block."""
    try:
        title = run.title or (run.instruction or "")[:40] or run.run_id
    except Exception:  # noqa: BLE001
        title = getattr(run, "run_id", "?")
    head = f"{BG_TAG}后台任务『{title}』已完成，其产出已并入本会话记录。"
    summary = (getattr(run, "final_text", "") or "").strip().replace("\n", " ")
    if summary:
        head += f"结论摘要：{summary[:200]}"
    od = getattr(run, "on_done", None)
    if isinstance(od, dict):
        nxt = str(od.get("next_agent") or "").strip()
        note = str(od.get("note") or "").strip()
        if nxt or note:
            tail = " 依先前显式声明，后续可考虑"
            if nxt:
                tail += f"交由 {nxt} 处理"
            if note:
                tail += f"（{note}）" if nxt else note
            head += (tail + "。是否继续由你/用户决定——不要自动执行，"
                     "尤其不要自动触发仪器（instrument_control）操作。")
    return head


__all__ = ["BackgroundRunManager", "BackgroundRun", "BG_TAG", "build_completion_hint"]
