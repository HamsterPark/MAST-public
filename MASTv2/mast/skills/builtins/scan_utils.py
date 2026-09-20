"""Scan utility skills: frame readback and end-of-scan wait.

P (Ported) from v1 mast/skills/builtins/scan_utils.py.
v0.3.14 polling implementation replaces the original Scan_WaitEndOfScan call
which blocked the TCP socket for the full timeout (Stop button unresponsive,
LLM unaware when scan finished).

Phase 7 migration (2026-05-19): WaitScanComplete now exposes its polling loop
as a graph-shaped composite. The polling iterations are dispatched as
``_phase_poll_<i>`` synthetic steps via the same ``_PhaseCtx`` wrapper used
by AssessImageQuality, plus a final ``_phase_finalize`` step. Polls are
``optional=True`` (a single TCP hiccup must not abort the wait), the finalize
step is mandatory.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_error_text,
    abort_facts,
)

logger = logging.getLogger(__name__)


# Synthetic phase identifiers — NOT registered in SkillRegistry; intercepted
# by :class:`_WaitScanCompletePhaseCtx`.
_PHASE_POLL_PREFIX = "_phase_poll_"
_PHASE_FINALIZE = "_phase_finalize"


class GetScanFrame(BaseSkill):
    """Read back the current scan frame parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScanFrame",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前扫描框（中心、尺寸、角度）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "frame", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Scan_FrameGet")
        if record.error:
            return SkillResult(
                skill_name="GetScanFrame",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 5:
                data = {
                    "center_x_m": float(vals[0]),
                    "center_y_m": float(vals[1]),
                    "width_m": float(vals[2]),
                    "height_m": float(vals[3]),
                    "angle_deg": float(vals[4]),
                }
        return SkillResult(
            skill_name="GetScanFrame",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class _WaitScanCompletePhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names."""

    def __init__(self, real_ctx, skill: "WaitScanComplete") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class WaitScanComplete(CompositeSkillGraph):
    """Wait for scan completion with nonblocking, abort-aware polling.
    Each polling iteration is a synthetic composite step so progress and checkpointing
    remain visible. A stopped status alone does not prove a complete frame; the buffer
    line count distinguishes completion from early stopping.
    The outcome is completed, stopped_early, timed_out, or restarted. Abort stops the
    scan and returns failure. If line verification is unavailable, the existing
    completed outcome is retained with lines_verified=False; unknown line count must
    not silently become verified completion or an assertion of truncation.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WaitScanComplete",
            version="1.2.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="等待当前扫描结束，或等到超时。",
            parameters=[
                ParameterSpec(
                    name="timeout_ms",
                    type="int",
                    description="超时时长，单位毫秒（-1 = 无限等待）",
                    unit="ms",
                    required=False,
                    default=-1,
                    min_value=-1,
                ),
            ],
            estimated_duration_s=60.0,
            composition_level=1,
            tags=["scan", "wait", "read"],
        )

    # ------------------------------------------------------------------
    # Dynamic plan — yield one poll step per iteration, then finalize.
    # ------------------------------------------------------------------

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        # Configuration captured into partial_data so resumed runs see the
        # same timeout window (we don't extend the wait across restarts).
        timeout_ms = int(params.get("timeout_ms", -1))
        timeout_s = float("inf") if timeout_ms < 0 else timeout_ms / 1000.0
        self._timeout_s = timeout_s
        # On timeout, STOP the scan by default: the old code
        # returned success with the scan STILL RUNNING, so batch/full-scan
        # continued to save + reconfigure + restart on a live scan. A scan that
        # overran its timeout is stuck or mis-estimated; stopping it is safe.
        self._stop_on_timeout = bool(params.get("stop_on_timeout", True))
        # Instance/class override wins, so a test can shrink the cadence BEFORE
        # ``max_polls`` is derived from it. Assigning 0.5 unconditionally here
        # meant a test could only patch it afterwards — by which point the poll
        # budget was already computed from the production value.
        self._poll_interval_s = float(
            params.get("poll_interval_s")
            or getattr(self, "_poll_interval_s", None) or 0.5)
        # The budget is what the caller asked for; the DEADLINE is what we are
        # currently waiting until. They differ only after an extension, and both
        # are reported — "we waited past your budget, here is why" has to be
        # visible or the next reader is back to guessing.
        self._deadline_s = timeout_s
        self._extensions = 0
        self._last_lines: "int | None" = None
        self._frame_restarted = False
        #: 见过 ``Scan_StatusGet != 0`` 了吗。**「还没起来」和「已经停了」是
        #: 两件事**,而在此之前它们走的是同一支代码。见 ``_START_GRACE_S``。
        self._seen_running = False

        # Hard cap on poll-step count so the UI progress bar has a number.
        # Negative timeout (wait indefinitely) → use a large soft cap.
        if timeout_s == float("inf"):
            max_polls = 100000  # essentially unbounded
        else:
            # Derived from the LONGEST deadline this wait could legitimately
            # reach, not from the base budget: the poll budget and the time
            # budget are two limits on the same loop, and sizing one for a
            # shorter horizon than the other makes the extension unreachable —
            # i.e. dead code that reads like a working guard. (Caught by its own
            # test: the first version ran out of polls at 2 and never extended.)
            max_deadline = timeout_s * (
                1.0 + self._MAX_EXTENSIONS * self._EXTENSION_FRACTION)
            max_polls = max(1, int(max_deadline / self._poll_interval_s) + 2)
        executor.set_total_steps(max_polls + 1)  # +1 finalize

        i = 0
        while True:
            # Stop polling once the previous iteration recorded a terminal
            # status (scan_done flag in partial_data).
            if executor.progress.partial_data.get("scan_done"):
                break
            if executor.progress.aborted:
                return
            yield CompositeStep(
                step_id=f"poll_{i}",
                skill_name=f"{_PHASE_POLL_PREFIX}{i}",
                params={"index": i},
                # A single failed poll must not abort the wait — TCP hiccups
                # are recoverable. The finalize step is mandatory.
                optional=True,
                checkpoint_after=False,
                tags=("poll", f"i={i}"),
            )
            if executor.progress.aborted:
                return
            i += 1
            if i >= max_polls:
                break

        # Always emit a finalize step so SkillResult.data is well-formed.
        yield CompositeStep(
            step_id="finalize",
            skill_name=_PHASE_FINALIZE,
            params={},
            optional=False,
            checkpoint_after=True,
            tags=("finalize",),
        )

    # ------------------------------------------------------------------
    # Phase dispatch — invoked by _WaitScanCompletePhaseCtx.run()
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name.startswith(_PHASE_POLL_PREFIX):
            return self._phase_poll(params, real_ctx)
        if skill_name == _PHASE_FINALIZE:
            return self._phase_finalize(real_ctx)
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    # Bound the number and size of progress-based deadline extensions.
    # Each extension uses a fraction of the original budget; the limit prevents
    # a misestimated or unreadable scan from becoming an unbounded wait.
    _MAX_EXTENSIONS = 2
    _EXTENSION_FRACTION = 0.25

    # 开扫后的握手宽限期，与完整扫描的等待预算分开。
    # StartScan 返回后，第一次状态读取可能早于扫描模块真正进入运行状态。
    # 从未观察到运行且仍处于宽限期内时，不立即把 status=0 判为中途停止。
    # 一旦观察到运行，此宽限不再参与停止判定。默认时间是工程预算，需要适配实际设备。
    _START_GRACE_S = 5.0

    # Class-level fallbacks: an abort can fire before ``plan_dynamic`` runs, and
    # the finalize data assembly reads all four unconditionally.
    _timeout_s: float = float("inf")
    _deadline_s: float = float("inf")
    _extensions: int = 0
    _last_lines: "int | None" = None
    _frame_restarted: bool = False

    def _grant_extension(self, real_ctx, elapsed: float) -> bool:
        """Extend the deadline only while verified line count strictly increases.
        Equal or unreadable counts do not justify an extension. Extensions are bounded.
        A decreasing count indicates that the counted frame was replaced, so report
        restarted rather than treating continuous scanning as a stalled frame.
        """
        if self._extensions >= self._MAX_EXTENSIONS:
            return False
        try:
            lines = self._measure_lines(real_ctx)
            done = lines.get("lines_done")
        except Exception:  # noqa: BLE001 — a diagnostic must not break the wait
            return False
        if not isinstance(done, int):
            return False
        prev, self._last_lines = self._last_lines, done
        if prev is not None and done < prev:
            self._frame_restarted = True
            logger.warning(
                "WaitScanComplete: 行数从 %s 掉到 %s —— 不是卡住,是仪器已经扫完一帧"
                "又开了新的一帧(Nanonis 的 Continuous scan 开着)。等「扫描结束」"
                "等不到,现在停下。", prev, done,
            )
            return False
        if prev is None or done <= prev:
            return False
        self._extensions += 1
        self._deadline_s = elapsed + self._timeout_s * self._EXTENSION_FRACTION
        logger.warning(
            "WaitScanComplete: 预算 %.0fs 到点但扫描仍在推进(%s → %s 行),"
            "延长 %.0fs(第 %d/%d 次)。估计帧时偏小,不是扫描卡住。",
            self._timeout_s, prev, done,
            self._timeout_s * self._EXTENSION_FRACTION,
            self._extensions, self._MAX_EXTENSIONS)
        return True

    def _phase_poll(self, params: dict, real_ctx) -> SkillResult:
        """One polling iteration: abort check → status get → sleep."""
        # Abort check FIRST so a freshly-set abort_event aborts even before
        # we issue the first Scan_StatusGet.
        check_abort = getattr(real_ctx, "check_abort", None)
        if callable(check_abort) and check_abort():
            try:
                stop_rec = real_ctx.safe_call("Scan_Action", 1, 0)
                self._call_log.append(stop_rec)
            except Exception as exc:
                logger.warning(
                    "WaitScanComplete abort: Scan_Action stop failed: %s", exc,
                )
            self._executor.set_partial("scan_done", True)
            self._executor.set_partial("aborted", True)
            self._executor.set_partial("elapsed_s",
                                       time.monotonic() - self._start)
            return SkillResult(
                skill_name=_PHASE_POLL_PREFIX + "abort",
                success=False,
                error="aborted by user",
            )

        elapsed = time.monotonic() - self._start
        # Seed the progress reference ONCE, at the halfway mark. Without it the
        # first expiry has nothing to compare against and could never extend —
        # the extension would be dead code that reads as a working guard.
        # (Line counting is a buffer grab, so it stays off the per-poll path:
        # one extra read per wait, not one per 0.5 s.)
        if (self._last_lines is None
                and self._timeout_s != float("inf")
                and elapsed >= 0.5 * self._timeout_s):
            try:
                seed = self._measure_lines(real_ctx).get("lines_done")
                if isinstance(seed, int):
                    self._last_lines = seed
            except Exception:  # noqa: BLE001
                pass
        if elapsed >= self._deadline_s:
            # On budget expiry, inspect verified line progress. Continuing acquisition may
            # earn a bounded extension; absent progress does not. Report which case occurred
            # so the caller can distinguish a small budget from a stalled scan.
            if self._grant_extension(real_ctx, elapsed):
                self._executor.set_partial("deadline_s", self._deadline_s)
                self._executor.set_partial("extensions", self._extensions)
            else:
                if getattr(self, "_stop_on_timeout", True):
                    try:
                        stop_rec = real_ctx.safe_call("Scan_Action", 1, 0)  # stop scan
                        self._call_log.append(stop_rec)
                    except Exception as exc:
                        logger.warning(
                            "WaitScanComplete timeout stop-scan failed: %s", exc)
                self._executor.set_partial("scan_done", True)
                self._executor.set_partial("timed_out", True)
                self._executor.set_partial(
                    "outcome",
                    "restarted" if self._frame_restarted else "timed_out")
                self._executor.set_partial(
                    "frame_restarted", self._frame_restarted)
                self._executor.set_partial("elapsed_s", elapsed)
                # Always report estimate, actual budget, elapsed time and line progress.
                # These quantities answer different questions and must not stand in for each other.
                self._executor.set_partial("budget_s", self._timeout_s)
                self._executor.set_partial("last_lines_done", self._last_lines)
                return SkillResult(
                    skill_name=_PHASE_POLL_PREFIX + "timeout",
                    success=True,
                    data={"timed_out": True,
                          "budget_s": self._timeout_s,
                          "deadline_s": self._deadline_s,
                          "elapsed_s": elapsed,
                          "extensions": self._extensions,
                          "lines_done": self._last_lines,
                          "frame_restarted": self._frame_restarted,
                          "scan_stopped": getattr(self, "_stop_on_timeout", True)},
                )

        rec = real_ctx.safe_call("Scan_StatusGet")
        self._call_log.append(rec)
        self._polls += 1
        if not rec.error:
            parsed = rec.return_value
            # Scan_StatusGet returns (err, raw, [status]) where status 0
            # means "not scanning" (== finished).
            status = (
                parsed[2][0]
                if isinstance(parsed, (list, tuple)) and len(parsed) > 2
                else parsed
            )
            if status != 0:
                # 看见它在跑了。从这一刻起,再读到 0 就**确实**是「停了」。
                self._seen_running = True
            elif not self._seen_running and elapsed < self._START_GRACE_S:
                # 开始命令返回不等于扫描已进入运行；宽限期还需结合缓冲区判断。
                # 有有效行时说明扫描已运行，按正常完成逻辑处理；确认零行才继续握手等待。
                # 缓冲区不可读时沿用既定未知处理，不能直接断言扫描被截断。
                probe = self._measure_lines(real_ctx)
                probed = probe.get("lines_done")
                if isinstance(probed, int) and probed == 0:
                    logger.debug(
                        "WaitScanComplete: 尚未见到扫描运行且缓冲区为空"
                        "(%.2fs/%.1fs 宽限)", elapsed, self._START_GRACE_S)
                    time.sleep(self._poll_interval_s)
                    return SkillResult(
                        skill_name=_PHASE_POLL_PREFIX + "starting",
                        success=True,
                        data={"timed_out": False, "waiting_for_start": True},
                    )
            if status == 0:
                # The scan STOPPED. Did it finish? Ask the buffer, once.
                lines = self._measure_lines(real_ctx)
                stopped_early = bool(lines.get("stopped_early"))
                # 宽限期内一次都没见它跑、而且缓冲区**确证**一行都没有
                # ⇒ 它**从没开始**。这与「跑了一半被停下」是两个事实,指向的
                # 下一步也不同:前者要查我们自己的发起时序,后者才要去查用户
                # Stop / Nanonis 自停。混成一句话,读的人会去查一件没发生的事。
                #
                # ⚠️ `lines_done is None` 是**读不到**,不是**零行**。
                # 第一版写的是 `not lines.get("lines_done")` —— 那把 None 和 0
                # 判成同一件事,于是一个缓冲区读不出来的装机会被告知「扫描从没
                # 开始」。而本类自述里那条纪律正好相反:**读不到就 fail open**,
                # 保持 `completed` + `lines_verified=False`,「不可测」不是「被截断」。
                # 闸门当场红了 6 条。这是我今天第三次栽在「不知道 ≠ 零」上。
                done_lines = lines.get("lines_done")
                never_started = bool(
                    not self._seen_running
                    and isinstance(done_lines, int)
                    and done_lines == 0)
                self._executor.set_partial("scan_done", True)
                self._executor.set_partial("timed_out", False)
                self._executor.set_partial("never_started", never_started)
                self._executor.set_partial(
                    "stopped_early", stopped_early and not never_started)
                self._executor.set_partial(
                    "outcome",
                    "never_started" if never_started
                    else ("stopped_early" if stopped_early else "completed"))
                for key in ("lines_done", "lines_total", "lines_verified"):
                    if lines.get(key) is not None:
                        self._executor.set_partial(key, lines[key])
                self._executor.set_partial("elapsed_s",
                                           time.monotonic() - self._start)
                return SkillResult(
                    skill_name=_PHASE_POLL_PREFIX + "done",
                    success=True,
                    data={"timed_out": False, **lines},
                )

        time.sleep(self._poll_interval_s)
        return SkillResult(
            skill_name=_PHASE_POLL_PREFIX + "continue",
            success=True,
            data={},
        )

    def _measure_lines(self, real_ctx) -> dict:
        """How many lines of the frame actually carry data. One-shot, fail-open.

        Returns ``{"lines_done", "lines_total", "lines_verified", "stopped_early"}``.
        ``lines_verified=False`` means the instrument could not tell us — the
        caller must then treat the frame as it did before this check existed
        (``stopped_early=False``), because "unmeasurable" is not "truncated".

        Two calls, issued exactly once when the status first reads 0:

        * ``Scan_BufferGet`` → the configured line count. Parsed by
          :func:`mast.io.nanonis_files.parse_buffer_get`, the ONE place that
          knows the real rig hands back channel ids as 1-tuples ``[(0,), (30,)]``.
        * ``Scan_FrameDataGrab`` → the frame, whose all-NaN rows are the ones
          never acquired (:func:`~mast.io.nanonis_files.frame_acquired_lines`).

        The grab moves a full frame (≈1 MB at 512²) and is why this is not done
        per poll: at 0.5 s polling that would be 2 MB/s of TCP against the same
        socket the scan is using.
        """
        out: dict = {"lines_done": None, "lines_total": None,
                     "lines_verified": False, "stopped_early": False}
        try:
            from mast.io.nanonis_files import (
                decode_reply,
                frame_acquired_lines,
                parse_buffer_get,
            )

            channel = 0
            rec_buf = real_ctx.safe_call("Scan_BufferGet")
            self._call_log.append(rec_buf)
            buf = (None if getattr(rec_buf, "error", "")
                   else parse_buffer_get(getattr(rec_buf, "return_value", None)))
            if buf:
                configured = buf.get("lines")
                if configured and configured > 0:
                    out["lines_total"] = int(configured)
                ids = buf.get("channel_indexes") or []
                if ids:
                    channel = int(ids[0])

            rec_frame = real_ctx.safe_call("Scan_FrameDataGrab", channel, 1)
            self._call_log.append(rec_frame)
            measured = (None if getattr(rec_frame, "error", "")
                        else frame_acquired_lines(
                            getattr(rec_frame, "return_value", None)))
            if measured is None:
                return out
            done, rows = measured
            configured = out["lines_total"]
            if configured and rows != configured:
                # The grab did not give us the frame we asked about — a reply we
                # mis-shaped, a buffer resized under us, an instrument that does
                # not allocate the whole frame. Whatever it is, the two numbers
                # describe different objects and dividing one by the other would
                # manufacture a truncation. Refuse to judge; do not invent.
                logger.warning(
                    "WaitScanComplete: frame has %d rows but the buffer is "
                    "configured for %d — not verifying the line count",
                    rows, configured,
                )
                return out
            total = configured or rows
            out["lines_done"] = int(done)
            out["lines_total"] = int(total)
            out["lines_verified"] = True
            out["stopped_early"] = bool(total > 0 and done < total)
        except Exception as exc:  # noqa: BLE001 — a wait must never die measuring
            logger.warning(
                "WaitScanComplete: could not verify the line count (%s); "
                "reporting the scan as complete but unverified", exc,
            )
        return out

    def _phase_finalize(self, real_ctx) -> SkillResult:
        """Build final aggregate. Always succeeds; aborted state surfaced via aggregate()."""
        elapsed = self._executor.progress.partial_data.get(
            "elapsed_s", time.monotonic() - self._start,
        )
        self._executor.set_partial("elapsed_s", float(elapsed))
        self._executor.set_partial("polls", self._polls)
        return SkillResult(
            skill_name=_PHASE_FINALIZE,
            success=True,
            data={},
        )

    # ------------------------------------------------------------------
    # Driver: wrap context so the executor dispatches ``_phase_*``.
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []
        self._polls = 0
        self._start = time.monotonic()

        wrapped = _WaitScanCompletePhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # Resume-friendly defaults
        executor.set_partial_default("scan_done", False)
        executor.set_partial_default("timed_out", False)
        executor.set_partial_default("stopped_early", False)
        executor.set_partial_default("frame_restarted", False)
        executor.set_partial_default("lines_verified", False)
        executor.set_partial_default("aborted", False)
        executor.set_partial_default("polls", 0)
        self._executor = executor

        # Roll forward the poll counter on resume so newly-yielded poll ids
        # don't collide with already-completed ones.
        if executor.progress.completed_steps:
            for step_id in executor.progress.completed_steps:
                if step_id.startswith("poll_"):
                    try:
                        idx = int(step_id.split("_", 1)[1])
                        self._polls = max(self._polls, idx + 1)
                    except (ValueError, IndexError):
                        pass

        plan_iter = self.plan_dynamic(params, executor)
        executor.run_plan(plan_iter)

        # The executor calls check_abort() BEFORE each step and marks the
        # whole composite aborted (progress.aborted=True) before any of our
        # phase code runs. In that case we still want to honour the v1
        # contract — issue Scan_Action(1, 0) to stop the scan — and return
        # success=False with the right error message.
        externally_aborted = bool(executor.progress.aborted)
        aborted_partial = bool(executor.progress.partial_data.get("aborted",
                                                                  False))
        aborted = externally_aborted or aborted_partial

        if externally_aborted and not aborted_partial:
            # External abort fired before any poll ran — issue the stop call
            # ourselves so the scan doesn't keep running.
            try:
                stop_rec = context.safe_call("Scan_Action", 1, 0)
                self._call_log.append(stop_rec)
            except Exception as exc:
                logger.warning(
                    "WaitScanComplete external abort: Scan_Action stop "
                    "failed: %s", exc,
                )

        partial = executor.progress.partial_data
        timed_out = bool(partial.get("timed_out", False))
        stopped_early = bool(partial.get("stopped_early", False))
        elapsed_s = float(partial.get(
            "elapsed_s", time.monotonic() - self._start))
        polls = int(partial.get("polls", self._polls))

        # One field the caller can switch on, instead of a pile of booleans it
        # has to combine correctly. `aborted` wins because it is the only one
        # that also flips success.
        outcome = str(partial.get("outcome") or "")
        if aborted:
            outcome = "aborted"
        elif not outcome:
            # No poll reached a terminal state (e.g. the plan ran out of poll
            # steps). Not "completed" — nothing said it completed.
            outcome = "unknown"

        data: dict[str, Any] = {
            "timed_out": timed_out,
            "stopped_early": stopped_early,
            # 从未开始与中途停止是不同结果：前者检查启动时序，后者检查停止原因。
            # 报告相应状态，避免用同一句话掩盖差异。
            "never_started": bool(partial.get("never_started", False)),
            "outcome": outcome,
            # 缺陷⑬:「用户喊停」与「超时」「它自己失败了」是三句话。
            # timed_out 已经把超时分了出去;这一位把「有人喊停」分出去。
            **abort_facts(executor.progress),
            "lines_done": partial.get("lines_done"),
            "lines_total": partial.get("lines_total"),
            "lines_verified": bool(partial.get("lines_verified", False)),
            # 「换帧了」和「卡住了」是两句话,所以这里是两个字段而不是一个
            # ——outcome 给要分支的调用方,这一位给要写话的那一层。
            "frame_restarted": bool(partial.get("frame_restarted", False)),
            "polls": polls,
            "elapsed_s": elapsed_s,
            # What we were waiting FOR and how long we ended up waiting. Present
            # on every outcome, not just timeouts: a number that exists only on
            # the path that needs it is missing exactly when somebody asks.
            "budget_s": (None if self._timeout_s == float("inf")
                         else self._timeout_s),
            "deadline_s": (None if self._deadline_s == float("inf")
                           else self._deadline_s),
            "extensions": int(partial.get("extensions", self._extensions)),
            "_progress": executor.progress.to_dict(),
        }

        # Outcomes:
        #   * abort        → success=False, error=<the reason it actually stopped>
        #   * timeout      → success=True,  data.timed_out=True
        #   * stopped early→ success=True,  data.stopped_early=True
        #   * completed    → success=True,  both False
        #
        # ``stopped_early`` keeps success=True for the same reason ``timed_out``
        # does: this skill's job is to WAIT, and it waited correctly — the scan
        # really did stop. Whether a truncated frame is acceptable is the
        # caller's judgement, and FullScan makes it (a hard failure there). The
        # alternative, failing here, would change control flow inside seven
        # declarative composites that have never seen this field, without any
        # of them gaining a better error message from it.
        #
        # The error text used to be the literal "aborted by user" for EVERY
        # early stop — including a CRITICAL tip-quality halt that no one had
        # actually triggered. Blaming the wrong cause sent people chasing an
        # "abort latch" that did not exist. abort_error_text() reports the
        # cause the executor already recorded.
        if aborted:
            return SkillResult(
                skill_name=self._skill_name(),
                success=False,
                error=abort_error_text(executor.progress),
                data=data,
                nanonis_calls=list(self._call_log),
            )
        return SkillResult(
            skill_name=self._skill_name(),
            success=True,
            data=data,
            nanonis_calls=list(self._call_log),
        )
