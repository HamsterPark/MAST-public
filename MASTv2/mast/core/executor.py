"""SkillExecutor: orchestrates skill execution with safety, logging, and rollback."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

from mast.core.connection import ConnectionPool
from mast.core.registry import SkillRegistry
from mast.core.safety import (
    SafetyGuard,
    is_coarse_drive_change,
    is_coarse_sample_approach,
    is_unguarded_lateral_coarse_move,
)
from mast.core.state import InstrumentState
from mast.core.types import (
    HardwareState,
    NanonisCallRecord,
    SkillResult,
)

logger = logging.getLogger(__name__)

# Default threshold for post-execution current anomaly (100 nA)
_CRASH_CURRENT_THRESHOLD_A = 100e-9


class SkillExecutor:
    """Orchestrates skill execution: safety check -> execute -> log -> rollback."""

    def __init__(
        self,
        pool: ConnectionPool,
        registry: SkillRegistry,
        safety: SafetyGuard,
        state: InstrumentState,
    ):
        self._pool = pool
        self._registry = registry
        self._safety = safety
        self._state = state
        self._execution_stack: list[tuple[str, dict, HardwareState]] = []
        self._abort_event = threading.Event()
        self._pause_event = threading.Event()
        self._watchdog = None
        # Reentrancy guard for failure-driven rollback. A failed skill whose
        # rollback_skill itself fails (and whose rollback chain loops back —
        # e.g. A.rollback==B, B.rollback==A, or a skill whose rollback is
        # itself) would otherwise recurse forever through run(). We bound the
        # rollback chain and never roll back a skill already on the chain.
        self._rollback_depth = 0
        self._rollback_chain: list[str] = []

    # Hard cap on how deep an automatic failure->rollback chain may recurse.
    _MAX_ROLLBACK_DEPTH = 8

    @property
    def _scope_admitted(self) -> bool:
        """True while we are inside a rollback chain — never gate a rollback.

        A rollback runs specifically because something already went wrong; a
        record-keeping rule must not be what stops the instrument from being put
        back into a safe state.
        """
        return self._rollback_depth > 0

    def start_watchdog(
        self,
        current_threshold_a: float | None = None,
        interval_s: float = 0.5,
        window_size: int = 8,
    ) -> None:
        """启动后台电流看门狗。
        
        通过监控端口读取隧道电流；滑动窗口内持续越限时，经 emergency 端口请求退针。
        current_threshold_a=None 时实时读取 cm_sat_current_a，与其他饱和判据共用配置，
        不另设可能高于前放量程的独立阈值。该配置必须与目标仪器的实际量程一致。
        
        suppress_getter 使用 core.tip_intent.active_tip_work，从加锁的仪器令牌读取
        有意修针状态。脉冲和扎针的计划瞬态与持续异常需要区分，避免看门狗与正在执行
        的物理动作冲突；不在此处另造一套修针判据。"""
        from mast.core.watchdog import SafetyWatchdog

        def _threshold_a() -> float:
            """当前的前放满量程。真源只有 ``cm_sat_current_a`` 一处。"""
            from mast.monitoring.thresholds import get_monitor_thresholds

            return float(get_monitor_thresholds().cm_sat_current_a)

        def _tip_work_now() -> str:
            """正在进行的蓄意针尖动作名,没有就是 ``""``。永不抛。"""
            from mast.core.tip_intent import active_tip_work

            return active_tip_work()

        def _token_idle_s():
            """仪器令牌空闲了多久;``None`` = 正被持有(MAST 在驱动)。

            用来回答「现在动仪器的是我们还是人」。**不是**「最近发过写命令」——
            一次长扫描是**持续操作、零写命令**(2026-08-08 那 13 分钟就是),
            用写命令做判据会把它判成「人在操作」。见 ``instrument_lock.idle_s``。
            """
            from mast.core.instrument_lock import instrument_lock

            return instrument_lock().idle_s()

        def on_anomaly():
            logger.critical("Watchdog anomaly — executing SafeRetract via emergency port")
            # Stop any powered coarse-approach FIRST: the fine-Z withdraw below
            # only retracts the piezo, but if the Nanonis AutoApproach / motor is
            # still stepping toward the sample it eats that retract in a few steps.
            # Best-effort on the emergency port; failures must not block withdraw.
            # urgent_call: bounded role-lock wait + force-unstick. The watchdog
            # runs on the monitor port but retracts on emergency/main, and a
            # wedged main socket used to swallow the fallback retract entirely
            # (审计 致命三).
            _urgent = getattr(self._pool, "urgent_call", None) or self._pool.safe_call
            for stop_call in (("AutoApproach_OnOffSet", 0), ("Motor_StopMove",)):
                try:
                    _urgent(*stop_call, role="emergency")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Watchdog pre-retract %s failed: %s",
                                   stop_call[0], exc)
            # ZCtrl_Withdraw(Wait_until_finished, Timeout_ms) — BOTH args are
            # required. The old single-arg call raised TypeError on real hardware
            # (swallowed into record.error), so the emergency retract NEVER ran —
            # the last-resort tip protection was dead (2026-06-10 review C1).
            # -1 = wait indefinitely until the tip is fully withdrawn.
            #
            # Retract must be CONFIRMED or the watchdog re-arms and retries: a
            # single failed withdraw used to latch the anomaly flag and disarm the
            # safety net for the whole session. If the
            # dedicated emergency port is down (TCP jitter / never connected) we
            # fall back to the main port — a live main port still retracts.
            retract_ok = False
            for role in ("emergency", "main"):
                rec = _urgent("ZCtrl_Withdraw", 1, -1, role=role)
                if not getattr(rec, "error", ""):
                    retract_ok = True
                    if role != "emergency":
                        logger.critical(
                            "Watchdog SafeRetract succeeded via fallback role=%s "
                            "(emergency port failed)", role)
                    break
                logger.critical("Watchdog SafeRetract via %s FAILED: %s",
                                role, rec.error)
            self._abort_event.set()
            # 修复项 review fix: bridge the anomaly into the buffer as an E_STOP
            # event so the AGENT path aborts too — the GUI registers a
            # synchronous critical hook that sets the orchestrator abort Event
            # (a running composite stops at its next check_abort()). Without
            # this the watchdog only stopped the manual/executor path; this
            # was the only production E_STOP emitter gap. Best-effort: a dead
            # buffer must never break the retract that already happened.
            try:
                from mast.buffer.active import get_active_buffer
                buf = get_active_buffer()
                if buf is not None:
                    from mast.buffer.schemas import make_e_stop
                    buf.emit_event(make_e_stop(
                        "watchdog",
                        "tunneling-current anomaly — SafeRetract executed",
                        seqno=buf.next_seq(),
                    ))
            except Exception as exc:  # noqa: BLE001 — never mask the retract
                logger.warning("watchdog E_STOP event emit failed: %s", exc)
            # Tell the watchdog whether the retract was confirmed. False → it
            # keeps the net armed and retries after its cooldown instead of
            # latching disarmed forever.
            return retract_ok

        self._watchdog = SafetyWatchdog(
            pool=self._pool,
            on_anomaly=on_anomaly,
            # None(默认)⇒ 不钉死,交给下面的 getter 每 tick 派生。
            current_threshold_a=current_threshold_a,
            interval_s=interval_s,
            window_size=window_size,
            threshold_getter=_threshold_a,
            suppress_getter=_tip_work_now,
            idle_getter=_token_idle_s,
        )
        self._watchdog.start()

    def stop_watchdog(self) -> None:
        """Stop the safety watchdog."""
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

    def run(
        self,
        skill_name: str,
        params: dict,
        approval_source: str = "auto",
    ) -> SkillResult:
        """Execute a skill with full safety pipeline:
        1. Look up skill in registry
        2. Refresh instrument state
        3. Validate params + preconditions via SafetyGuard
        4. Check approval level
        5. Snapshot state_before
        6. Execute skill
        7. Snapshot state_after
        8. Record in execution stack
        9. Return SkillResult
        On failure, attempt rollback if skill defines rollback_skill."""
        t0 = time.perf_counter()

        # 1. Look up skill
        try:
            skill_cls = self._registry.get(skill_name)
        except KeyError as e:
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=str(e),
                elapsed_s=time.perf_counter() - t0,
            )

        meta = self._registry._get_metadata(skill_cls)

        # 2. Refresh state
        try:
            current_state = self._state.refresh()
        except Exception as e:
            current_state = self._state.snapshot()
            logger.warning("State refresh failed, using cached: %s", e)

        # 3. Validate (retry once after 100ms if precondition fails —
        #    Nanonis hardware state propagation can lag behind TCP replies)
        is_safe, issues = self._safety.validate_execution(meta, params, current_state)
        if not is_safe:
            precond_fail = any("Precondition" in i for i in issues)
            if precond_fail:
                time.sleep(0.1)
                try:
                    current_state = self._state.refresh()
                except Exception:
                    pass
                is_safe, issues = self._safety.validate_execution(meta, params, current_state)
            if not is_safe:
                logger.info(
                    "Skill '%s' rejected by safety validation: %s",
                    skill_name, "; ".join(issues),
                )
                return SkillResult(
                    skill_name=skill_name,
                    success=False,
                    error=f"Safety validation failed: {'; '.join(issues)}",
                    elapsed_s=time.perf_counter() - t0,
                    state_before=current_state,
                )

        # 3b. Sample gate (2026-07-28). A data-producing skill needs a sample to
        # belong to — an unattributable .sxm is a measurement nobody can use
        # later. This covers the manual/GUI path (the agent path is gated in
        # skill_adapter). Reads, stops, retracts and E-STOP are never gated, and
        # an unclassifiable skill is ALLOWED — see mast.core.sample_gate for why
        # this gate fails open while the instrument token fails closed.
        #
        # ExecutionContext.run sets _scope_admitted on the context so a composite
        # that already passed the gate isn't re-gated mid-way: a composite halted
        # halfway is worse than one that never started.
        try:
            from mast.core.sample_gate import check_sample_scope
            from mast.logging.experiment_log import get_active_log
            gate_msg = check_sample_scope(meta, skill_name, get_active_log())
        except Exception:  # noqa: BLE001 — a broken gate must not block work
            gate_msg = None
        if gate_msg and not self._scope_admitted:
            logger.info("Skill '%s' rejected: no active sample", skill_name)
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=gate_msg,
                elapsed_s=time.perf_counter() - t0,
                state_before=current_state,
            )

        # 4. Approval. 这里把两种「要人」拆开 —— 见
        # ``core/execution_context.py`` 里同款的那一段,两条路必须同口径。
        #   * **参数条件硬闸**(下面三条):拒绝型防护,原样保留;
        #   * **基线 DANGEROUS**(``safety_level`` 派生):转成「执行 + 留痕 + 通知」。
        #     原来这里的拒绝语让人「去审批面板批准」,而那个面板已经不会再为
        #     DANGEROUS 技能产生任何东西 —— 指向死路的提示比没有提示更坏。
        required_approval = self._safety.requires_approval(meta)
        hard_gate: str | None = None
        # The one physically-dangerous action — an open-loop coarse Z step toward
        # the sample — regardless of the skill's baseline safety_level (MotorMove
        # is CONFIRM for x/y/retract). 2026-06-11.
        if is_coarse_sample_approach(skill_name, params):
            hard_gate = "开环粗动 Z 向样品进针（没有反馈能让它停下来）"
        # The coarse stepper's DRIVE voltage: unrecoverable if wrong (some stacks
        # fail at 300 V on a controller that offers 400), and nothing reads back
        # which rig this is. The per-rig ceiling is the operator's to declare, so
        # the operator is who sets it. 2026-07-31.
        elif is_coarse_drive_change(skill_name, params):
            hard_gate = "设置粗动马达驱动电压/频率"
        # A RAW lateral coarse step. The guarded composite (RelocateCoarseXY)
        # stays autonomous; the bare motor command — no coarse-Z clearance, no
        # vacuum check, no drive readback, no odometer — needs a human.
        elif is_unguarded_lateral_coarse_move(skill_name, params):
            hard_gate = "裸横向粗动（未走 RelocateCoarseXY 的防护路径）"
        if hard_gate is not None and approval_source != "human":
            logger.info(
                "Skill '%s' refused: %s (source=%s)",
                skill_name, hard_gate, approval_source,
            )
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=(
                    f"硬闸拒绝：'{skill_name}' 属于「{hard_gate}」，"
                    "自主路径上一律不执行——这不是等待审批，没有任何批准会到来。"
                    "STOP retrying。确实需要时请用户在 GUI 手动执行。"
                ),
                elapsed_s=time.perf_counter() - t0,
                state_before=current_state,
            )
        if required_approval == "human" and approval_source != "human":
            try:
                from mast.core.auto_approval import notify, would_have_asked

                _reason = would_have_asked(meta, tool_name=skill_name, args=params)
                notify(skill_name,
                       f"{_reason or 'DANGEROUS 技能'} —— 已直接执行并通知，"
                       "不再等待人工批准",
                       args=params, approval_source=approval_source,
                       path="executor")
            except Exception:  # noqa: BLE001 — 通知坏了绝不能让技能失败
                logger.debug("auto_approval 通知失败(已忽略)", exc_info=True)
        if required_approval == "llm" and approval_source == "auto":
            logger.info(
                "Skill '%s' rejected: LLM/human approval required (source=auto)",
                skill_name,
            )
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error="LLM or human approval required for CONFIRM skill",
                elapsed_s=time.perf_counter() - t0,
                state_before=current_state,
            )

        # 5. Snapshot state_before
        state_before = replace(current_state)

        # 6. Execute
        ctx = self.create_context(approval_source=approval_source)
        from mast.core.instrument_lock import InstrumentBusy, hold_for_skill
        try:
            skill_instance = skill_cls()
            # The manual/GUI path is one of the drivers of the one instrument
            # (inventory: instrument_lock docstring). Same re-entrant token; reads and
            # stop/retract remedies skip it (审计 致命一).
            with hold_for_skill(meta, skill_name, "手动/执行器"):
                result: SkillResult = skill_instance.execute(ctx, params)
            result.state_before = state_before
        except InstrumentBusy as busy:
            return SkillResult(
                skill_name=skill_name, success=False, error=busy.message(),
                elapsed_s=time.perf_counter() - t0, state_before=state_before,
            )
        except (ValueError, TypeError) as e:
            # Most common cause: a Get* skill assumed Nanonis would return
            # numeric values, but the corresponding hardware module isn't
            # loaded / configured (e.g. CurrentBEEM, Motor Control,
            # SafeTip), so Nanonis returned empty strings/lists which crash
            # int()/float() conversion. Surface this to the LLM as
            # "module not available, move on" rather than a cryptic
            # ValueError that triggers debugging loops.
            result = SkillResult(
                skill_name=skill_name,
                success=False,
                error=(
                    f"Skill returned no usable data — likely the corresponding "
                    f"Nanonis module is not loaded or configured on this "
                    f"instrument. Skip this state field and continue. "
                    f"(internal: {type(e).__name__}: {e})"
                ),
                state_before=state_before,
            )
        except Exception as e:
            result = SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"Execution error: {type(e).__name__}: {e}",
                state_before=state_before,
            )

        # 7. Snapshot state_after + post-execution anomaly detection 
        try:
            state_after = self._state.refresh()
        except Exception:
            state_after = self._state.snapshot()
        result.state_after = state_after
        result.elapsed_s = time.perf_counter() - t0

        # Post-execution current anomaly check (crash detection)
        if state_after.current_a is not None:
            if abs(state_after.current_a) > _CRASH_CURRENT_THRESHOLD_A:
                if result.data is None:
                    result.data = {}
                result.data["anomaly_warning"] = "current_spike"
                logger.warning(
                    "Post-execution anomaly: current=%.2e A exceeds threshold",
                    state_after.current_a,
                )

        # 8. Record in execution stack
        self._execution_stack.append((skill_name, params, state_before))

        # Attempt rollback on failure.
        #
        # Guards (review): the rollback is dispatched back through run(), so a
        # rollback chain that loops (A.rollback==B & B.rollback==A, or a skill
        # whose rollback is itself) or simply runs too deep would recurse
        # without bound and blow the stack mid-experiment. We:
        #   * skip a rollback target already on the active chain (cycle), and
        #   * cap total rollback depth (_MAX_ROLLBACK_DEPTH), and
        #   * inspect the rollback's own result instead of discarding it, so a
        #     failed rollback is surfaced on the original result + logged loudly
        #     (a silently-failed rollback can leave the instrument unsafe).
        if not result.success and meta.rollback_skill:
            rb = meta.rollback_skill
            if self._rollback_depth >= self._MAX_ROLLBACK_DEPTH:
                logger.error(
                    "Skill '%s' failed; rollback via '%s' SKIPPED — max rollback "
                    "depth %d reached (chain=%s)",
                    skill_name, rb, self._MAX_ROLLBACK_DEPTH,
                    " -> ".join(self._rollback_chain),
                )
                self._note_rollback(result, f"rollback aborted (max depth {self._MAX_ROLLBACK_DEPTH})")
            elif rb == skill_name or rb in self._rollback_chain:
                logger.error(
                    "Skill '%s' failed; rollback via '%s' SKIPPED — would form a "
                    "rollback cycle (chain=%s)",
                    skill_name, rb, " -> ".join(self._rollback_chain + [skill_name]),
                )
                self._note_rollback(result, f"rollback aborted (cycle via {rb})")
            else:
                logger.warning(
                    "Skill '%s' failed, attempting rollback via '%s'",
                    skill_name, rb,
                )
                self._rollback_depth += 1
                self._rollback_chain.append(skill_name)
                try:
                    rb_result = self.run(rb, {}, approval_source="auto")
                finally:
                    self._rollback_chain.pop()
                    self._rollback_depth -= 1
                if rb_result is not None and not rb_result.success:
                    logger.error(
                        "Rollback '%s' for failed skill '%s' ALSO FAILED: %s",
                        rb, skill_name, rb_result.error,
                    )
                    self._note_rollback(
                        result, f"rollback '{rb}' failed: {rb_result.error}"
                    )
                else:
                    logger.info(
                        "Rollback '%s' for failed skill '%s' succeeded", rb, skill_name
                    )

        logger.info(
            "Skill '%s' %s in %.3fs",
            skill_name,
            "succeeded" if result.success else "failed",
            result.elapsed_s,
        )

        # Push skill completion event to UI via WebSocket
        try:
            from mast.core.events import EventBus
            EventBus.get().publish_skill_step(
                skill=skill_name,
                step=len(self._execution_stack),
                success=result.success,
            )
        except Exception:
            pass

        return result

    @staticmethod
    def _note_rollback(result: SkillResult, note: str) -> None:
        """Record the outcome of a failure-driven rollback on the original
        result's ``data`` so a failed/aborted rollback is never silently lost
        (a botched rollback can leave the instrument in an unsafe state)."""
        if result.data is None:
            result.data = {}
        result.data["rollback_status"] = note

    def rollback_last(self) -> SkillResult | None:
        """Rollback the most recent skill execution."""
        if not self._execution_stack:
            return None

        skill_name, params, state_before = self._execution_stack.pop()

        try:
            skill_cls = self._registry.get(skill_name)
            meta = self._registry._get_metadata(skill_cls)
        except KeyError:
            return None

        if not meta.rollback_skill:
            logger.warning("Skill '%s' has no rollback_skill defined", skill_name)
            return None

        return self.run(meta.rollback_skill, {}, approval_source="auto")

    def create_context(
        self,
        approval_source: str = "auto",
        abort_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> "_LegacyExecutionContext":
        """Create an execution context for composite skills to call sub-skills.

        NOTE: this returns the legacy ``_LegacyExecutionContext`` (executor-
        backed, recursive ``executor.run`` dispatch), NOT the active v2
        ``mast.core.execution_context.ExecutionContext`` (registry-backed,
        constructed as ``ExecutionContext(pool, state, registry)``). The two
        used to share the name ``ExecutionContext``, which made the
        ``(self, pool, state, ...)`` constructor here silently incompatible
        with the v2 one — . They are now distinct types so the
        collision can't recur.
        """
        return _LegacyExecutionContext(
            self, self._pool, self._state,
            approval_source=approval_source,
            abort_event=abort_event or self._abort_event,
            pause_event=pause_event or self._pause_event,
        )


class _LegacyExecutionContext:
    """Legacy executor-backed context (v1-style recursive dispatch).

    The active v2 context is ``mast.core.execution_context.ExecutionContext``
    (registry-backed). This class is kept only for ``SkillExecutor`` — the v1
    pipeline still imported by gui/app.py, llm/planner.py and llm/quickask.py —
    and is intentionally private so nothing re-introduces the ``ExecutionContext``
    name collision (). Provides access to executor, pool, state.
    """

    def __init__(
        self,
        executor: SkillExecutor,
        pool: ConnectionPool,
        state: InstrumentState,
        approval_source: str = "auto",
        abort_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ):
        self.executor = executor
        self.pool = pool
        self.state = state
        self._approval_source = approval_source
        self._abort = abort_event or threading.Event()
        self._pause = pause_event or threading.Event()

    def run(self, skill_name: str, params: dict) -> SkillResult:
        """Convenience: execute a sub-skill through the executor.
        Inherits approval_source from parent skill context."""
        return self.executor.run(skill_name, params, approval_source=self._approval_source)

    def safe_call(self, method_name: str, *args, role: str = "main") -> NanonisCallRecord:
        """Direct Nanonis call through the connection pool."""
        return self.pool.safe_call(method_name, *args, role=role)

    def check_abort(self) -> bool:
        """Check abort/pause flags. Composite skills should call between steps.

        If paused: pauses Nanonis scan, blocks until resumed, then resumes scan.
        Returns True if aborted (caller should return immediately).
        """
        if self._pause.is_set():
            logger.info("Execution paused")
            self.safe_call("Scan_Action", 2, 0)  # Pause scan
            while self._pause.is_set() and not self._abort.is_set():
                time.sleep(0.2)
            if not self._abort.is_set():
                self.safe_call("Scan_Action", 3, 0)  # Resume scan (3=Resume; 1=Stop)
                logger.info("Execution resumed")
        return self._abort.is_set()
