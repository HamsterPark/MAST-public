"""CompositeSkillGraph executor — unified framework for graph-shaped composites.

Every composite skill in MAST is *named* a single skill but *executes* as a
sequence of sub-skills. Pre-Phase 7 these ran as opaque Python loops inside
``CompositeSkill.run_composite()`` — a single LangGraph tool call wrapped N
nested ``context.run()`` invocations. That meant:

  • A 100-point GridSTS that crashed at point 99 had to re-run from point 0.
  • LangGraph checkpointer never saw intermediate progress.
  • Live progress was invisible to the GUI.

This module is the unified framework that fixes those. A graph-shaped
composite declares its work as a *plan* (a list of :class:`CompositeStep`)
or as a *dynamic plan* (a generator that may inspect previous results).
The :class:`GraphExecutor` walks the plan, runs each step via
``context.run(...)``, and after every step emits a :class:`CompositeProgress`
snapshot through ``context.emit_progress(...)``. The wrap_skill adapter
catches that emit and writes it into ``MASTState.composite_progress`` so
the SqliteSaver checkpointer can flush it.

Resume semantics:
  • On re-invocation, the executor reads ``progress.completed_steps`` from
    the context (if present) and skips any step whose ``step_id`` is already
    in that set.
  • Optional steps (``CompositeStep.optional=True``) record their failure
    and continue.
  • Mandatory steps abort the composite on failure with a snapshot.

Authoring rules (see docs/v2/architecture/composite_skills.md when
generated):
  1. New composites SHALL subclass :class:`CompositeSkillGraph`, not
     :class:`CompositeSkill` directly.
  2. ``plan()`` returns a static list when N is known at parse time
     (params fully determine the step sequence). Use ``plan_dynamic()``
     when later step params depend on earlier results.
  3. Each step's ``step_id`` MUST be unique within the composite and
     stable across runs (so resume can match).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Step + Progress dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompositeStep:
    """One atomic step inside a composite.

    Attributes:
      step_id:    Unique ID within the composite, stable across runs.
                  Used for resume matching. Conventionally
                  "<phase>:<index>" e.g. "configure", "point_3_5",
                  "attempt_2:scan".
      skill_name: Name of the sub-skill to invoke (e.g. "MoveToXY").
      params:     Dict of sub-skill params.
      optional:   If True, a failure in this step does not abort the
                  composite; the executor records it in
                  ``failed_steps`` and proceeds.
      checkpoint_after: If True, the executor calls
                  ``context.checkpoint_flush()`` after this step
                  succeeds. Default True for long composites; set False
                  for tight inner loops where flush overhead matters.
      tags:       Free-form annotation (for debugging / UI).
    """
    step_id: str
    skill_name: str
    params: dict[str, Any]
    optional: bool = False
    checkpoint_after: bool = True
    tags: tuple[str, ...] = ()
    # P4: 钉住子技能版本（None = registry 最新 semver，既有行为）。
    skill_version: str | None = None


# Longest gap between a sidecar's last step and a resume that can still be the
# SAME interrupted run (LangGraph interrupt → operator decision → resume is
# minutes; anything older is a leftover from a previous session).
_SIDECAR_RESUME_WINDOW_S = 30 * 60.0

#: What the ABORT LATCH writes into ``aborted_reason``. That Event has exactly
#: two setters — the operator's 中止 button and E_STOP — so this is the ONE
#: abort that may honestly be reported to the caller as a user abort.
#:
#: It is stored ALREADY OPERATOR-FACING on purpose (2026-07-28). It used to be
#: the sentinel below, and ``abort_error_text`` translated it — but 25 of the 27
#: readers of ``aborted_reason`` do not go through that function, they render
#: ``progress.aborted_reason or "<skill> aborted"`` directly. So pressing 中止
#: reported "external abort flag before step" out of almost every composite:
#: executor jargon naming a flag, handed to the agent, which is precisely how
#: an agent once spent a run asking the operator to "release the abort
#: latch" that does not exist. A value that is only safe to show through one
#: particular accessor is a trap, and it caught 25 of 27 call sites. The two
#: sibling latch paths in this file already wrote "aborted by operator"; this
#: one was the outlier.
_USER_ABORT_TEXT = "aborted by user"

#: LEGACY sentinel — what the latch wrote before the above. Still recognised by
#: :func:`abort_error_text` so a sidecar written by an older build (a run
#: resumed across an upgrade) still translates instead of leaking jargon.
_ABORT_LATCH_REASON = "external abort flag before step"


def _drop_stale_abort(progress: "CompositeProgress", composite_name: str,
                      source: str) -> None:
    """恢复既往进度时清除上次中止标记；当前中止由 context.check_abort 在每步独立判断。"""
    if not getattr(progress, "aborted", False):
        return
    logger.warning(
        "[%s] %s 带着上一次的中止状态(%s)—— 清掉再恢复,否则这次「重试」会变成「重放」",
        composite_name, source, progress.aborted_reason or "(无原因)")
    progress.aborted = False
    progress.aborted_reason = ""


def _is_terminal(progress) -> bool:
    """这份既往进度描述的是一次**跑完了**的运行,不是一次被打断的。

    恢复的语义是「中断 → 用户决定 → 接着跑」。一份每一步都完成的快照不是
    中断,是一次**清理没做干净**的完成 —— 拿它当起点会把整个计划跳掉,而外面
    看到的是一次数字完全正常的成功(本仓已出货四次,见 ``__init__`` 与
    ``_load_or_init_progress`` 两处的事故清单)。

    ``total_steps > 0`` 是分母:流式 composite(步数事先不知道)在跑完前可能是 0,
    那时**不判 terminal** —— 宁可多恢复一次,也不要把一次真正的中断续跑判死。

    提成一个有名字的函数是因为它有**两个调用点**(sidecar 一个、上下文一个),
    而这两处以前只有一处有它 —— 「同一个坑的两个入口只堵了一个」正是这么来的。
    """
    try:
        total = int(getattr(progress, "total_steps", 0) or 0)
        done = len(getattr(progress, "completed_steps", None) or ())
    except (TypeError, ValueError):
        return False
    return total > 0 and done >= total


def _sidecar_dir():
    from mast._runtime_paths import project_root
    d = project_root() / "experiments" / "composite_progress"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(s: str, n: int = 80) -> str:
    import re
    return re.sub(r"[^\w一-鿿-]", "_", str(s))[:n]


def _sidecar_path(composite_name: str, run_id: str = ""):
    """Out-of-band step-progress file for one composite IN ONE RUN (P2-G).

    Keyed by (composite, run) — NOT by composite alone. The old name-only key
    made a sidecar a shared, cross-run mailbox: a finished AutoApproach left its
    "all 4 steps done" file behind and EVERY later approach resumed it, skipping
    the whole plan and reporting instant success without touching the instrument
    (the 2026-07-10 fake-进针). Terminal/stale guards patched that, but the key
    was the real defect: resume is only ever meaningful WITHIN one run
    (interrupt → operator decision → resume), so the run id belongs in the key
    and cross-run reuse becomes structurally impossible rather than merely
    guarded against.

    ``run_id`` empty (manual executor path, tests) → the legacy name-only file,
    still protected by the terminal/stale guards.
    """
    safe = _slug(composite_name) or "composite"
    d = _sidecar_dir()
    if run_id:
        return d / f"{safe}__{_slug(run_id, 40)}.json"
    return d / f"{safe}.json"


def _sweep_stale_sidecars(max_age_s: float = 24 * 3600.0) -> None:
    """Delete sidecars older than a day. Run-scoped keys mean a crashed run's
    file is never reused, but it would otherwise linger on disk forever."""
    import time as _t
    try:
        now = _t.time()
        for p in _sidecar_dir().glob("*.json"):
            try:
                if now - p.stat().st_mtime > max_age_s:
                    p.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:  # pragma: no cover — housekeeping must never break a run
        pass


def _is_graph_interrupt(exc: BaseException) -> bool:
    """True iff *exc* is LangGraph's control-flow interrupt — it must always
    bubble (pausing the graph), never be treated as a step failure."""
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:  # pragma: no cover — langgraph always pinned in v2
        return False
    return isinstance(exc, GraphInterrupt)


def _diag(kind: str, subject: str, reason: str, **fields) -> None:
    """Write one line to the refusal ledger. Never raises — a lost diagnostics
    line must never cost a composite its run."""
    try:
        from mast.core.diagnostics import record

        record(kind, subject, reason, **fields)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass


def _narrate(kind: str, /, **data) -> None:
    """向用户发一条**旁白**（"我们要打一发 10 V / 500 ms 的脉冲"）。

    ⚠️ ``kind`` 是**位置限定**（那个 ``/``）。不加的话，任何一条数据字段也叫
    ``kind`` 的旁白都会抛 ``TypeError: got multiple values for argument 'kind'``
    —— 2026-08-18 ``poke_decision``（决策码字段就叫 ``kind``）在另外三个发射器上
    当场把 63 条测试打红。四个发射器现在统一是位置限定的，由
    ``tests/v2/unit/chat/test_forge_outcomes_are_all_narratable.py`` 钉住。

    发点在这里而不是在每个技能里，是因为这是**每一个 composite 子步骤都必经**的
    地方，而且这里同时握着 ``step.skill_name`` / ``step.params`` / 结果 —— 一处接线
    覆盖全部 composite，零 per-skill 改动。

    「句子里的电压等于真正下发的电压」这件事成立的原因也在这里：交出去的
    ``params`` 就是下面那行 ``self._context.run(step.skill_name, step.params)``
    送进 ``skill.execute`` 的**同一份**，中间没有任何人有机会改写它。

    整体 try/except：旁白链路（import 失败、队列、DB）出任何事都只能丢一条旁白，
    绝不许影响正在跑的实验。这与 ``_diag`` 同形，理由也同一条。
    """
    try:
        from mast.chat import narration

        narration.narrate(kind, **data)
    except Exception:  # noqa: BLE001
        pass


def _is_abort_requested(exc: BaseException) -> bool:
    """True iff *exc* is the composite layer's AbortRequested — control flow
    (the operator stopped us), NOT a failure to be rolled back.

    Imported lazily: _base imports this module, so a top-level import here
    would close the cycle."""
    try:
        from mast.skills.composite._base import AbortRequested
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, AbortRequested)


@dataclass
class CompositeProgress:
    """Serializable snapshot of a composite's progress.

    All fields are plain JSON-serializable types — this dict is written
    into ``MASTState.composite_progress[composite_name]`` and flushed by
    SqliteSaver. Tensors, ndarrays, file handles are FORBIDDEN here
    (the ``block_scan_tensors_in_checkpointer`` hook enforces).
    """
    composite_name: str
    total_steps: int = 0
    completed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    # 按 step_id 保存失败原因，使 optional 步骤的调用方能决定后续行为。
    failed_reasons: dict[str, str] = field(default_factory=dict)
    current_step: str | None = None
    partial_data: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    last_update_at: float = field(default_factory=time.time)
    aborted: bool = False
    aborted_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_name": self.composite_name,
            "total_steps": self.total_steps,
            "completed_steps": list(self.completed_steps),
            "failed_steps": list(self.failed_steps),
            "current_step": self.current_step,
            "partial_data": dict(self.partial_data),
            "started_at": self.started_at,
            "last_update_at": self.last_update_at,
            "aborted": self.aborted,
            "aborted_reason": self.aborted_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompositeProgress:
        return cls(
            composite_name=d.get("composite_name", ""),
            total_steps=int(d.get("total_steps", 0)),
            completed_steps=list(d.get("completed_steps", [])),
            failed_steps=list(d.get("failed_steps", [])),
            current_step=d.get("current_step"),
            partial_data=dict(d.get("partial_data", {})),
            started_at=float(d.get("started_at", time.time())),
            last_update_at=float(d.get("last_update_at", time.time())),
            aborted=bool(d.get("aborted", False)),
            aborted_reason=str(d.get("aborted_reason", "")),
        )


# ─────────────────────────────────────────────────────────────────────────
# Context protocol — what GraphExecutor expects from the ExecutionContext
# ─────────────────────────────────────────────────────────────────────────


@runtime_checkable
class _ProgressCtx(Protocol):
    """Subset of ExecutionContext that the executor needs."""

    def run(self, skill_name: str, params: dict) -> Any: ...

    # Optional — the executor checks via hasattr/getattr and degrades
    # gracefully when missing.
    # def emit_progress(self, progress: CompositeProgress) -> None: ...
    # def get_progress(self, composite_name: str) -> CompositeProgress | None: ...
    # def checkpoint_flush(self) -> None: ...
    # def check_abort(self) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────
# The executor
# ─────────────────────────────────────────────────────────────────────────

# Type of a plan iterator. Static plan = list[CompositeStep]; dynamic plan =
# Iterator[CompositeStep] (can inspect prior results via the executor).
PlanSource = Iterator[CompositeStep]


class GraphExecutor:
    """Walks a CompositeStep plan, emitting progress + supporting resume."""

    def __init__(
        self,
        composite_name: str,
        context: _ProgressCtx,
        *,
        on_step_result: Callable[[CompositeStep, Any], None] | None = None,
        on_step_failed: Callable[[CompositeStep, str], bool] | None = None,
    ) -> None:
        """
        Args:
          composite_name:  Name from skill metadata; used as state key.
          context:         ExecutionContext-like object (see protocol).
          on_step_result:  Optional callback invoked after each successful
                           step with (step, sub_skill_result). Lets the
                           composite stash data into partial_data.
          on_step_failed:  Optional callback for failures. Returns True to
                           continue (treated as if optional=True), False
                           to abort. Default: abort iff step.optional is
                           False.
        """
        self._composite_name = composite_name
        self._context = context
        self._on_step_result = on_step_result
        self._on_step_failed = on_step_failed
        self._progress = self._load_or_init_progress()
        # 同一个坑的**两个入口**:既往进度可以从上下文来,也可以从 sidecar 来,
        # 两条路都会 ``from_dict`` 恢复 ``aborted``。只堵一个等于没堵。
        _drop_stale_abort(self._progress, composite_name, "state")
        # P2-G step-level OUT-OF-BAND persistence: the in-state progress only
        # lands when the whole tool call returns (emit/flush are no-ops on the
        # agent path), so a mid-run interrupt()/crash lost every completed
        # step — and the LangGraph resume REPLAYS the tool from the top,
        # re-executing instrument actions (spike-proven,
        # test_interrupt_spike.py). The sidecar JSON is written after every
        # step; on construction it wins over a staler in-state snapshot.
        # Identity = composite name (single-instrument serial reality);
        # cleared on successful completion by _base._graph_execute.
        self._sidecar_path = None
        try:
            # Scope the sidecar to THIS run (see _sidecar_path): a resume is only
            # ever meaningful inside the run that was interrupted.
            run_id = str(getattr(context, "run_id", "") or "")
            self._sidecar_path = _sidecar_path(composite_name, run_id)
            side = self._load_sidecar()
            if side is not None and (len(side.completed_steps)
                                     > len(self._progress.completed_steps)):
                # Resume is ONLY for the crash/interrupt window of the SAME
                # logical run. Two classes of sidecar must NOT resume
                # (2026-07-10 — a 9-day-old fully-completed
                # AutoApproach.json short-circuited every later approach into
                # an instant fake success, so "进针成功" was declared at
                # 0.17 pA vs a 500 pA setpoint):
                #   1. TERMINAL sidecars — every step completed. That is a
                #      finished run whose cleanup was missed, not an
                #      interrupted one; resuming it skips the entire plan.
                #   2. STALE sidecars — last touched longer ago than any
                #      plausible interrupt→resume window.
                # Rule 1 depends on total_steps being on disk. It is written by
                # the flush at the end of run_plan() — see the long note there
                # for why a streaming composite used to persist total_steps == 0
                # and so could never be recognised as finished (2026-07-27:
                # BatchRegionsScan regions 1..4 "completing" in 0.4 s each).
                # 判据走 :func:`_is_terminal` —— 上下文那条恢复路上用的是同一个
                # (2026-08-24 起)。两处各写一遍表达式,就是两处各漂各的。
                terminal = _is_terminal(side)
                age_s = max(0.0, time.time() - float(side.last_update_at or 0))
                stale = age_s > _SIDECAR_RESUME_WINDOW_S
                if terminal or stale:
                    logger.warning(
                        "[%s] discarding %s sidecar (age %.0fs, %d/%d steps) — "
                        "starting fresh", composite_name,
                        "terminal" if terminal else "stale", age_s,
                        len(side.completed_steps), side.total_steps)
                    self.clear_sidecar()
                else:
                    logger.info(
                        "[%s] resuming from step-level sidecar (%d completed > "
                        "%d in state)", composite_name,
                        len(side.completed_steps),
                        len(self._progress.completed_steps))
                    # 恢复时清除上次调用的中止状态，当前中止仍由 context.check_abort 独立检查。
                    _drop_stale_abort(side, composite_name, "sidecar")
                    self._progress = side
        except Exception:  # pragma: no cover — sidecar must never block a run
            logger.exception("[%s] sidecar init failed", composite_name)
        self._sub_results: dict[str, Any] = {}  # step_id → sub-skill result

    # -- public API --

    @property
    def progress(self) -> CompositeProgress:
        return self._progress

    @property
    def context(self):
        """The ExecutionContext this plan ran against.

        Exposed for ``aggregate()``, which gets ``sub_results`` + ``progress``
        but no context — so a composite that wants to state WHAT THE INSTRUMENT
        IS at the end had nowhere to read from and quoted its own plan instead
        (``ForgeAuTip`` printed "仪器留在结条件 0.05 V / 1 nA" while the hardware
        sat at 1.0 V / 100 pA, 2026-08-10). Reads are legal here even after an
        abort: ``ExecutionContext.safe_call`` refuses writes, never reads."""
        return self._context

    @property
    def sub_results(self) -> dict[str, Any]:
        """All sub-skill results from this execution, keyed by step_id.
        Note: on resume, only steps run *in this invocation* appear here
        (resumed-skipped steps are not re-executed, so no result is
        available)."""
        return dict(self._sub_results)

    def is_completed(self, step_id: str) -> bool:
        """True if a prior run already finished this step (resume case)."""
        return step_id in self._progress.completed_steps

    def run_plan(self, plan: PlanSource) -> bool:
        """Execute every step in the plan. Returns True iff none aborted.

        Skipped steps (already in progress.completed_steps) are NOT
        re-executed. Optional steps that fail are logged but do not abort.
        Mandatory failures abort and the executor marks
        ``progress.aborted = True``.
        """
        steps_seen = 0
        _plan = iter(plan)
        while True:
            try:
                step = next(_plan)
            except StopIteration:
                break
            except BaseException as exc:  # noqa: BLE001
                # A DYNAMIC plan runs its body inside the generator, so it can
                # abort from in there — ConditionTip's wait-for-scan helper raises
                # AbortRequested while polling. That is CONTROL FLOW, not a step
                # failure: the bare `for step in plan:` let it escape run_plan
                # entirely, up through skill.execute into skill_adapter's generic
                # `except Exception`, which fired the skill's ROLLBACK. An abort
                # would then undo work it had no business touching.
                if _is_graph_interrupt(exc):
                    raise                      # HITL pause — must bubble
                if _is_abort_requested(exc):
                    logger.info("[%s] plan aborted from inside the generator",
                                self._composite_name)
                    self._abort(reason="aborted by operator")
                    return False
                raise
            steps_seen += 1
            # Resume: skip steps that completed in a prior invocation
            if self.is_completed(step.step_id):
                logger.debug(
                    "[%s] skip resumed step %s",
                    self._composite_name, step.step_id,
                )
                # A SKIPPED step and a step that ran-and-succeeded are
                # indistinguishable from outside — which is exactly why an early
                # stop like this could not be diagnosed. If a
                # stale sidecar makes the executor resume-skip the leading points,
                # the composite silently starts in the middle and the operator sees
                # it "stop early". A skip is a DECISION; write it down.
                _diag("step_skip", f"{self._composite_name}.{step.step_id}",
                      "已完成（从 sidecar 断点恢复）——本次未执行",
                      skill=step.skill_name, ordinal=steps_seen,
                      completed=len(self._progress.completed_steps))
                continue
            # Pre-step abort check (watchdog / user abort)
            if self._check_abort():
                self._abort(reason=_USER_ABORT_TEXT)
                return False
            # Pre-step HALT check — a CRITICAL **physical** finding from the
            # current monitor (preamp railed / measurement chain dead / a giant
            # current step outside deliberate tip work).
            #
            # Built for five tip_quality_drop events
            # fired that day, every one only AFTER the hardware action had
            # finished, because the sole consumer ran between LLM calls and one
            # BatchRegionsScan tool call scans 4 regions in 107 s. Vision had
            # said "其后行不可信，建议中止扫描" and two more batches ran anyway.
            #
            # ⑰-C1 (2026-08-09): **the VISION source no longer arms this halt at
            # all** — in any mode, any scenario. That verdict is a guess about
            # morphology, it scored zero true positives over two field nights,
            # and it never once caught something the refusal-type protections
            # missed. So the 2026-07-27 event itself would no longer stop a run;
            # what still does is a physical excursion. The mechanism here is
            # unchanged and still load-bearing — only its vision trigger is gone.
            # See runtime.tip_halt_source / make_tip_halt_hook.
            #
            # Distinct from the abort above ON PURPOSE. Abort is a latch that
            # makes skill_adapter refuse every new instrument action, so
            # reusing it for a bad tip would block 修针 / 退针 / 停扫 — the very
            # remedies — and wedge the instrument against its own recovery. The
            # halt is one-shot and scoped to this run: it stops THIS plan here
            # and changes nothing else.
            halt = self._check_halt()
            if halt:
                logger.warning("[%s] halted at %s: %s",
                               self._composite_name, step.step_id, halt)
                self._abort(reason=halt)
                return False

            self._progress.current_step = step.step_id
            self._progress.last_update_at = time.time()

            # 动作旁白在执行前发出，文案使用未来时，避免把计划说成已完成。
            self._narrate_step_begin(step)

            try:
                # P4: 钉版本时显式 3 参调用（老的 2 参 fake/legacy context 在
                # 不钉版本的路径上完全不受影响）。
                if step.skill_version:
                    result = self._context.run(step.skill_name, step.params,
                                               version=step.skill_version)
                else:
                    result = self._context.run(step.skill_name, step.params)
            except Exception as exc:
                # P2-G: GraphInterrupt is CONTROL FLOW (a nested composite's
                # human node pausing the run), not a failure — let it bubble
                # so LangGraph pauses; progress is already in the sidecar.
                if _is_graph_interrupt(exc):
                    raise
                # An operator abort raised from inside a sub-skill is control
                # flow too: mark the composite aborted rather than routing it
                # through _handle_failure (which would count it as a failed step
                # and, one level up, trigger a rollback of work the abort never
                # touched).
                if _is_abort_requested(exc):
                    self._abort(reason="aborted by operator")
                    return False
                msg = f"{step.skill_name} raised {type(exc).__name__}: {exc}"
                if not self._handle_failure(step, msg):
                    return False
                continue

            success = getattr(result, "success", True)
            if not success:
                err = getattr(result, "error", "") or "(no error message)"
                msg = f"{step.skill_name} failed: {err}"
                # ⚠️ ``result.error`` 是**短码**（"rolled_back: diverged"），而技能
                # 写给人读的那句带数字的话在 ``data["detail"]`` 里
                # （"第 1 轮后残余 Z 占用 6.4 nm，未降到上一轮的 70% 以下"）。
                #
                # 一步失败之后 ``sub_results`` 不收它（下面那行只在成功分支），
                # 于是 ``_narrate_step_result`` 不发、``_data()`` 读不到 ——
                # **技能说得最清楚的那句话，正好在出事的时候被丢掉**。
                # 调平时的操作和效果也应该记入旁白——
                # 出事那次的读数比顺利那次更该留下来。
                #
                # 走**显式形参**，不走一个临时挂在 self 上的字段：后者等于给
                # ``_handle_failure`` 加了一个看不见的入参，而另一个调用方
                # （异常那一支）读到的会是上一次留下来的值。
                detail = str((getattr(result, "data", None) or {}).get("detail") or "")
                if not self._handle_failure(step, msg, detail=detail):
                    return False
                continue

            self._sub_results[step.step_id] = result
            self._narrate_step_result(step, result)
            self._progress.completed_steps.append(step.step_id)
            if self._on_step_result is not None:
                try:
                    self._on_step_result(step, result)
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "[%s] on_step_result callback raised",
                        self._composite_name,
                    )
            self._emit_progress()
            if step.checkpoint_after:
                self._checkpoint_flush()

        # Plan fully drained without abort
        self._progress.total_steps = max(self._progress.total_steps, steps_seen)
        self._progress.current_step = None
        self._emit_progress()
        # Persist the now-known step count. Until 2026-07-27 this assignment
        # happened AFTER the last _checkpoint_flush() (which only runs on steps
        # marked checkpoint_after), so a completed run's sidecar was left on disk
        # with total_steps == 0 — and the terminal-sidecar guard in __init__ needs
        # total_steps > 0 as its denominator. A STREAMING composite (step count
        # unknown up front, e.g. WaitScanComplete's one _phase_poll_<i> per poll)
        # therefore left behind a sidecar that could never be RECOGNISED as
        # finished, only as interrupted.
        #
        # Field consequence (2026-07-27 forensics): within one BatchRegionsScan
        # all regions share a run_id and therefore one sidecar file. Region 0
        # scanned for 105.6 s and left a 211-step, total_steps == 0 sidecar;
        # regions 1..4 "resumed" from it, skipped WaitScanComplete entirely, and
        # returned in 0.37/0.38/0.38/0.22 s — while the composite reported
        # success_count=5, fail_count=0, five distinct .sxm paths, and
        # recommended one of the fake regions as the best. That is the second
        # shipment of this failure mode; the first (AutoApproach, 2026-07-10
        # #42/#75) is recorded in the guard's own comment.
        #
        # Normal completion clears the sidecar right after this; this write is
        # what makes the guard correct when that cleanup does NOT happen.
        #
        # flush_sidecar(), NOT _checkpoint_flush(): the only thing that needs to
        # reach disk here is the step count. A full checkpoint would also drive
        # context.checkpoint_flush() (a DB write), turning every composite's
        # completion into an extra checkpoint — the "flush only on critical
        # phases" contract that test_auto_approach_graph / test_pattern_graph /
        # test_set_bias_ramp_graph pin.
        if self._sidecar_path is not None and not self._progress.aborted:
            self.flush_sidecar()
            # 正常完成后清理自身 sidecar；中断流程保留续跑记录，不能仅凭 total_steps 为零丢弃。
            self._clear_completed_sidecar()
        return not self._progress.aborted

    def set_total_steps(self, n: int) -> None:
        """Hint for UI / progress bars. Optional; the executor also
        updates total_steps as steps stream in."""
        self._progress.total_steps = max(self._progress.total_steps, int(n))

    def set_partial(self, key: str, value: Any) -> None:
        """Stash a scalar/dict/list into progress.partial_data (so it
        survives checkpoint + appears in the final SkillResult)."""
        self._progress.partial_data[key] = value
        self._progress.last_update_at = time.time()

    def set_partial_default(self, key: str, value: Any) -> None:
        """Set partial only if key not already present (resume-friendly).

        Use this for *accumulators* that should preserve their value
        across a resume (e.g. ``succeeded``, ``failed``, ``quality_history``).
        Use :meth:`set_partial` for *parameters* that should always reflect
        the current invocation (e.g. ``nx``, ``ny``, ``spacing_m``).
        """
        if key not in self._progress.partial_data:
            self._progress.partial_data[key] = value
            self._progress.last_update_at = time.time()

    def abort(self, reason: str) -> None:
        """Public abort entry point for composite skills (forwards to _abort)."""
        self._abort(reason=reason)

    # -- internals --

    def _load_sidecar(self) -> CompositeProgress | None:
        if self._sidecar_path is None or not self._sidecar_path.exists():
            return None
        try:
            import json
            d = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
            return CompositeProgress.from_dict(d)
        except Exception:  # pragma: no cover — corrupt sidecar ⇒ fresh start
            logger.warning("[%s] sidecar unreadable — ignoring",
                           self._composite_name)
            return None

    def flush_sidecar(self) -> None:
        """Atomically persist the current progress out-of-band. Called after
        every step and (critically) right BEFORE a human-node interrupt —
        the interrupt aborts the tool call, so anything not in the sidecar
        is re-executed on resume."""
        if self._sidecar_path is None:
            return
        try:
            import json
            import os
            import tempfile
            p = self._sidecar_path
            fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp",
                                       prefix=p.name + ".")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._progress.to_dict(), f, ensure_ascii=False)
            os.replace(tmp, str(p))
        except Exception:  # pragma: no cover — never block the run
            logger.warning("[%s] sidecar flush failed", self._composite_name)

    def clear_sidecar(self) -> None:
        """Remove the sidecar — the run finished; the next invocation of this
        composite must start fresh, not resume-skip stale steps."""
        if self._sidecar_path is None:
            return
        try:
            self._sidecar_path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover
            logger.warning("[%s] sidecar clear failed", self._composite_name)


    def _clear_completed_sidecar(self) -> None:
        """正常完成后删除本次运行的 sidecar。
        
        同一次会话的下一次调用不能恢复已经完成的计划。
        中止运行保留恢复依据；命名方法便于对完成分支做独立验证。
        """
        self.clear_sidecar()

    def _load_or_init_progress(self) -> CompositeProgress:
        get_prog = getattr(self._context, "get_progress", None)
        if callable(get_prog):
            try:
                prior = get_prog(self._composite_name)
            except Exception:
                prior = None
            if isinstance(prior, dict) and prior:
                prior = CompositeProgress.from_dict(prior)
            if isinstance(prior, CompositeProgress):
                if _is_terminal(prior):
                    # 内存进度与磁盘 sidecar 都必须拒绝恢复已完成快照；非终态的合法续跑仍然保留。
                    logger.warning(
                        "[%s] discarding TERMINAL prior progress from context "
                        "(%d/%d steps) — that is a FINISHED run, not an "
                        "interrupted one; starting fresh",
                        self._composite_name,
                        len(prior.completed_steps), prior.total_steps)
                    _diag("progress_discard", self._composite_name,
                          "上下文里的既往进度是**跑完了的**——本次从头跑,"
                          "不跳步(跳步 = 零仪器调用的假成功)",
                          completed=len(prior.completed_steps),
                          total=prior.total_steps)
                else:
                    logger.info(
                        "[%s] resuming from progress (%d/%d completed)",
                        self._composite_name,
                        len(prior.completed_steps), prior.total_steps,
                    )
                    return prior
        return CompositeProgress(composite_name=self._composite_name)

    def _diag_step(self, kind: str, step_id: str, reason: str, **f) -> None:
        _diag(kind, f"{self._composite_name}.{step_id}", reason, **f)

    # ── 旁白（向用户解说；agent 看不见，见 mast/chat/narration.py） ──────

    def _narrate_step_begin(self, step: CompositeStep) -> None:
        """这一步要动手之前说一句 —— 只对**查得到模板**的技能说。

        查不到就**不发**，不是发一句通用的「正在执行某个步骤」：后者既没有信息量，
        又会把真正有信息量的那几条淹掉。一次 ForgeAuTip 有几百个子步骤。
        """
        try:
            from mast.chat.narration_templates import BEGIN_KIND_FOR_SKILL

            kind = BEGIN_KIND_FOR_SKILL.get(step.skill_name)
            if not kind:
                return
            _narrate(kind, skill=step.skill_name, step_id=step.step_id,
                     composite=self._composite_name, params=step.params)
        except Exception:  # noqa: BLE001 — 见 _narrate 的 docstring
            pass

    def _narrate_step_result(self, step: CompositeStep, result: Any) -> None:
        """这一步**跑完之后**说一句结论 —— 只对查得到模板的技能说。

        2026-08-16 加。要求:旁白要更具体 —— 例如正在扎针尖时,团簇的
        分析图与分析结果也要各自带一条旁白。

        ``RESULT_KIND_FOR_SKILL`` 从 2026-08-11 起一直是空的,理由写在那儿:
        结论类旁白要读子技能的 ``result.data``,而那些字段当时正在被另一条线改,
        照着一份会变的词汇表写模板 = 一句永远走 fallback、看起来却完全正常的话。
        08-15/16 把 ``AssessClusterRoundness`` 的返回字段钉死之后,前提才成立。
        **其余技能仍然查不到模板,于是仍然不发** —— 这不是遗漏,是同一条纪律。

        ## 为什么图在这里取

        ``step.params["scan_path"]`` 是**刚刚送进 skill.execute 的那一份**,
        与判读用的是同一张图 —— 中间没有任何人有机会换掉它。
        这和 ``_narrate`` 那句「句子里的电压等于真正下发的电压」是同一条理由。
        取到的是 ``scan_path``,但**挂出去的不是它** —— 见下面那段注释:
        ``.sxm`` 走不通,所以在这里把分析图画好落盘,挂那张 PNG。
        猜错 origin 的后果不是没有图,是**借了别的图**(见 narration._normalise_image)。
        """
        try:
            from mast.chat.narration_templates import RESULT_KIND_FOR_SKILL

            kind = RESULT_KIND_FOR_SKILL.get(step.skill_name)
            if not kind:
                return
            data = getattr(result, "data", None)
            if not isinstance(data, dict):
                return
            # 附带判据分解图，使用户能核对判断依据。
            image = None
            src = str((step.params or {}).get("scan_path") or "").strip()
            if src:
                from mast.vision.cluster_panel import render_cluster_panel

                png = render_cluster_panel(src, data)
                if png:
                    image = {"src": png, "origin": "milestone_png"}
            _narrate(kind, skill=step.skill_name, step_id=step.step_id,
                     composite=self._composite_name, params=step.params,
                     result=data, image=image)
        except Exception:  # noqa: BLE001 — 见 _narrate 的 docstring
            pass

    def _narrate_failure(self, step: CompositeStep, msg: str,
                         continued: bool, detail: str = "") -> None:
        """这一步没成。``continued`` 是用户看到「没成」之后要知道的第一件事：
        **这次跑还在不在跑**。它由这里的三条出路各自决定，不由 ``step.optional``
        推断 —— ``on_step_failed`` 回调也能放行一个非 optional 的步骤。"""
        _narrate("step_failed", skill=step.skill_name, step_id=step.step_id,
                 composite=self._composite_name, reason=msg,
                 continued=continued, detail=detail)

    def _handle_failure(self, step: CompositeStep, msg: str, *,
                        detail: str = "") -> bool:
        """Returns True to continue, False to abort.

        ``detail`` 是失败的技能写给**人**读的那句话（``result.data["detail"]``），
        与 ``msg``（短码）互补。默认空串：异常那一支根本没有 ``result``，
        而「读不到」不许被填成一句话。
        """
        logger.warning("[%s] step %s failed: %s",
                       self._composite_name, step.step_id, msg)
        if self._on_step_failed is not None:
            try:
                cont = bool(self._on_step_failed(step, msg))
            except Exception:  # pragma: no cover
                cont = False
            if cont:
                self._progress.failed_steps.append(step.step_id)
                self._progress.failed_reasons[step.step_id] = str(msg or "")
                self._diag_step("step_fail", step.step_id, msg,
                                skill=step.skill_name, continued=True)
                self._narrate_failure(step, msg, continued=True, detail=detail)
                self._emit_progress()
                return True
        if step.optional:
            self._progress.failed_steps.append(step.step_id)
            self._progress.failed_reasons[step.step_id] = str(msg or "")
            self._diag_step("step_fail", step.step_id, msg,
                            skill=step.skill_name, continued=True, optional=True)
            self._narrate_failure(step, msg, continued=True, detail=detail)
            self._emit_progress()
            return True
        self._diag_step("step_fail", step.step_id, msg,
                        skill=step.skill_name, continued=False)
        self._narrate_failure(step, msg, continued=False, detail=detail)
        self._abort(reason=msg)
        return False

    # NB: module-level twin lives at the bottom of this file —
    # :func:`abort_error_text`. Hand-rolled ``run_composite`` overrides call it
    # instead of hard-coding "aborted by user".

    def _abort(self, *, reason: str) -> None:
        self._progress.aborted = True
        self._progress.aborted_reason = reason
        self._progress.last_update_at = time.time()
        # WHICH step it stopped on, and why. "The composite stopped" plus a bare
        # reason string leaves a run undiagnosable — one can see it
        # halt at point 5 and nobody could say whether point 5 ran, was skipped,
        # or was never reached.
        _diag("step_abort",
              f"{self._composite_name}.{self._progress.current_step or '?'}",
              reason,
              completed=len(self._progress.completed_steps),
              failed=list(self._progress.failed_steps),
              total=self._progress.total_steps)
        self._emit_progress()

    def _check_abort(self) -> bool:
        ca = getattr(self._context, "check_abort", None)
        if callable(ca):
            try:
                return bool(ca())
            except Exception:  # pragma: no cover
                return False
        return False

    def _check_halt(self) -> str:
        """A pending run-scoped stop reason, or ``""``.

        Duck-typed exactly like ``check_abort`` above: contexts that don't
        provide ``check_halt`` (legacy / test fakes) simply never halt. The
        call CONSUMES the signal — it is one-shot by design, so a resolved tip
        event cannot keep stopping every later plan in the same run."""
        ch = getattr(self._context, "check_halt", None)
        if not callable(ch):
            return ""
        try:
            reason = ch()
        except Exception:  # pragma: no cover — a broken check can't stop work
            logger.debug("check_halt raised; treating as no halt", exc_info=True)
            return ""
        if isinstance(reason, str):
            return reason
        # A halt must be a REASON, and only a real str is one. Anything else is
        # a broken (or auto-generated) check, and the safe reading of "I don't
        # understand this answer" is "no halt" — a MagicMock context, which is
        # how most composite tests build their ExecutionContext, otherwise
        # returns a truthy sentinel and stops every plan at its first step.
        if reason not in (None, False):
            logger.debug("check_halt returned %s, not a reason string — ignoring",
                         type(reason).__name__)
        return ""

    def _emit_progress(self) -> None:
        # P2-G: the sidecar IS the real step-level persistence (the ctx emit
        # below is a no-op on the agent path) — flush on every progress beat.
        self.flush_sidecar()
        emit = getattr(self._context, "emit_progress", None)
        if callable(emit):
            try:
                emit(self._progress)
            except Exception:  # pragma: no cover
                logger.exception(
                    "[%s] emit_progress raised", self._composite_name,
                )

    def _checkpoint_flush(self) -> None:
        flush = getattr(self._context, "checkpoint_flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # pragma: no cover
                logger.exception(
                    "[%s] checkpoint_flush raised", self._composite_name,
                )


def abort_error_text(progress: CompositeProgress) -> str:
    """The error text for a composite that stopped early — WHY it stopped.

    the run was reported as user-aborted
    when no one had touched it.

    ``progress.aborted`` is set by more than one thing. The operator's 中止 and
    E_STOP set it, but so does a CRITICAL tip-quality halt, and so does a
    mandatory step failing. Every hand-rolled ``run_composite`` translated all
    of them into the same string, ``"aborted by user"`` — discarding
    ``aborted_reason``, which already held the real cause.

    The consequence was not cosmetic. The operator was told three times running
    that they had aborted a scan they had not touched, and the AGENT was told
    the same thing, so it kept retrying a scan that a bad tip had stopped —
    while reporting "疑似用户在按中止" to the very person who wasn't. A stop
    signal that lies about its own origin makes the run undiagnosable from
    either end.

    Only the ABORT LATCH is rendered as "aborted by user": that Event has
    exactly two setters, the operator's 中止 button and E_STOP. Every other
    reason is passed through verbatim.

    Since 2026-07-28 the latch already STORES that text (see ``_USER_ABORT_TEXT``),
    so this function is no longer the only thing standing between executor jargon
    and the operator — the 25 call sites that read ``aborted_reason`` directly are
    safe too. The legacy sentinel is still translated for sidecars written by an
    older build.
    """
    reason = (getattr(progress, "aborted_reason", "") or "").strip()
    if not reason or reason == _ABORT_LATCH_REASON:
        return _USER_ABORT_TEXT
    return reason


def abort_facts(progress: CompositeProgress) -> "dict":
    """停止这件事的**机器可读**版本 —— 给下游用,不是给人读的那句话。

    ``abort_error_text`` 给的是一句话;下游要判「这是用户喊停,还是它自己失败了」
    就得去猜那句话的措辞 —— 而那正是 #46 的翻版(判据落在措辞上,措辞一改就失效)。

    ``aborted_by_operator`` 只在**中止闩**被扳时为 True(那个 Event 恰好两个设置者:
    用户的中止按钮和 E_STOP)。CRITICAL 针尖停机、必要步骤失败一律 False —— 它们
    也让 ``aborted`` 为 True,但**没有人喊过停**。

    「用户喊停」和「跑失败了」是两句话:前者不该触发重试、不该记成故障,
    后者该。缺陷⑬ 的验收哲学是「用户的停止意图必须有保底路径穿透到硬件」,
    而穿透之后它得**以本来面目**到达下游。
    """
    aborted = bool(getattr(progress, "aborted", False))
    reason = (getattr(progress, "aborted_reason", "") or "").strip()
    by_operator = aborted and (
        not reason or reason in (_ABORT_LATCH_REASON, _USER_ABORT_TEXT))
    return {
        "aborted": aborted,
        "aborted_by_operator": bool(by_operator),
        "abort_reason": abort_error_text(progress) if aborted else "",
    }


__all__ = [
    "CompositeStep",
    "CompositeProgress",
    "GraphExecutor",
    "PlanSource",
    "abort_error_text",
    "abort_facts",
]
