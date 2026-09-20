"""ConditionTip — rule-based tip conditioning loop (Phase 7 graph migration).

Dynamic-plan composite: the number of attempts and the per-attempt step
sequence depend on the previous attempt's FFT quality. Each attempt:

  pulse -> configure -> set_speed -> start_scan -> wait -> assess

After ``assess``, the plan checks ``executor.sub_results`` for the FFT
quality. If quality >= ``target_quality``, the generator returns early.
Otherwise the next attempt's steps are yielded. ``wait_scan_complete`` is
not itself a sub-skill — it is invoked synchronously inside the generator
between ``start_scan`` and ``assess`` and registers its NanonisCallRecords
onto ``self._all_calls`` for the final SkillResult.

Inspired by Scanbot rule-based conditioning and DeepSPM.
"""

from __future__ import annotations

import logging
from typing import Iterator

from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite._helpers import wait_scan_complete
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.agents._shared.skill_adapter import wrap_skill

logger = logging.getLogger(__name__)


class ConditionTip(CompositeSkillGraph):
    """Automatic tip conditioning: pulse -> scan -> evaluate -> repeat.

    Iterates up to *max_attempts* times:
      1. Apply a bias pulse via TipPulse.
      2. Configure and run a test scan.
      3. Wait for the scan to finish.
      4. Evaluate the resulting image with AssessImageQuality.
      5. If quality >= *target_quality*, stop.

    Inspired by Scanbot rule-based conditioning and DeepSPM.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConditionTip",
            version="2.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"bias_pulse"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "修针自动闭环：脉冲 → 扫描 → FFT 评质，重复到达标为止。"
            ),
            parameters=[
                ParameterSpec(
                    name="pulse_v",
                    type="float",
                    description=(
                        "修针用的偏压脉冲电压。**没有具体理由就别填** —— 留空时，"
                        "它按已登记针尖（材料 × 制法 × 形态）从针尖策略表里取。"
                        "你**填了**的值会被采用，但超出该针尖的安全包络时是"
                        "**拒绝**（不是夹紧）。"
                    ),
                    unit="V",
                    required=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="max_attempts",
                    type="int",
                    description=(
                        "脉冲+扫描的最大轮次（留空 → 针尖策略表）"
                    ),
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=50,
                ),
                ParameterSpec(
                    name="target_quality",
                    type="float",
                    description="FFT 质量分阈值（0-1）",
                    required=False,
                    default=0.3,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="测试扫描中心 X（默认：当前位置）",
                    unit="m",
                    required=False,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="测试扫描中心 Y（默认：当前位置）",
                    unit="m",
                    required=False,
                ),
                ParameterSpec(
                    name="scan_width_m",
                    type="float",
                    description="测试扫描宽度",
                    unit="m",
                    required=False,
                    default=10e-9,
                    min_value=1e-10,
                    max_value=1e-6,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=120.0,
            composition_level=3,
            tags=["tip", "conditioning", "composite", "quality"],
        )

    # ------------------------------------------------------------------
    # Dynamic plan — yields steps per attempt; runs wait_scan_complete
    # synchronously between StartScan and AssessImageQuality; checks
    # the assess result and stops early once target_quality is met.
    # ------------------------------------------------------------------

    def _emit_scan_and_assess(
        self, executor: GraphExecutor, label: str,
        cx: float, cy: float, scan_w: float, scan_speed: float,
        *, abort_on_timeout: bool = True,
    ) -> Iterator[CompositeStep]:
        """Yield configure→set_speed→start_scan, wait for completion inline, then
        yield assess. On scan timeout, STOP the still-running scan (the wait
        helper only stops on abort, not timeout — 审查). When
        ``abort_on_timeout`` is False (the best-effort initial assessment), a
        timeout just stops the scan and returns WITHOUT the assess step, so the
        caller falls through to conditioning instead of aborting the whole run."""
        yield CompositeStep(
            step_id=f"{label}:configure", skill_name="ConfigureScan",
            params={"center_x_m": cx, "center_y_m": cy,
                    "width_m": scan_w, "height_m": scan_w},
            optional=False, checkpoint_after=False,
            tags=("attempt", label, "configure"),
        )
        if executor.progress.aborted:
            return
        yield CompositeStep(
            step_id=f"{label}:set_speed", skill_name="SetScanSpeed",
            params={"fwd_speed": scan_speed, "bwd_speed": scan_speed,
                    "fwd_line_time": 0.1, "bwd_line_time": 0.1, "keep_const": 0},
            optional=True, checkpoint_after=False,
            tags=("attempt", label, "set_speed"),
        )
        if executor.progress.aborted:
            return
        yield CompositeStep(
            step_id=f"{label}:start_scan", skill_name="StartScan", params={},
            optional=False, checkpoint_after=False,
            tags=("attempt", label, "start_scan"),
        )
        if executor.progress.aborted:
            return
        # Wait for scan completion (synchronous helper — NOT a step).
        scan_finished = wait_scan_complete(
            self._context, timeout_s=60.0, call_accumulator=self._all_calls,
        )
        if not scan_finished:
            # The wait helper leaves the scan RUNNING on timeout — stop it so the
            # instrument isn't left scanning unmonitored after we bail.
            try:
                self._all_calls.append(self._context.safe_call("Scan_Action", 1, 0))
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("ConditionTip timeout stop-scan failed: %s", exc)
            if abort_on_timeout:
                executor.abort(reason=f"Scan timed out ({label}) — scan stopped")
            else:
                logger.info("ConditionTip initial scan timed out — skipping "
                            "initial assessment, proceeding to condition")
            return
        yield CompositeStep(
            step_id=f"{label}:assess", skill_name="AssessImageQuality", params={},
            optional=True, checkpoint_after=True,
            tags=("attempt", label, "assess"),
        )

    @staticmethod
    def _capture_frame(context) -> dict | None:
        """Snapshot the current scan frame (cx, cy, w, h, angle) for restore.
        Best-effort — returns None if it can't be read."""
        try:
            rec = context.safe_call("Scan_FrameGet")
            if getattr(rec, "error", ""):
                return None
            parsed = getattr(rec, "return_value", None)
            vals = parsed[2] if (isinstance(parsed, (list, tuple)) and len(parsed) > 2) else None
            if isinstance(vals, (list, tuple)) and len(vals) >= 5:
                return {"cx": float(vals[0]), "cy": float(vals[1]),
                        "w": float(vals[2]), "h": float(vals[3]),
                        "angle": float(vals[4])}
        except Exception:  # pragma: no cover - best-effort
            return None
        return None

    def _restore_frame(self, context, frame: dict | None) -> None:
        """Restore a previously-captured scan frame. Best-effort, never raises."""
        if not frame:
            return
        try:
            rec = context.safe_call(
                "Scan_FrameSet", frame["cx"], frame["cy"],
                frame["w"], frame["h"], frame["angle"])
            if getattr(self, "_all_calls", None) is not None:
                self._all_calls.append(rec)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("ConditionTip frame restore failed: %s", exc)

    @staticmethod
    def _quality_of(executor: GraphExecutor, assess_step_id: str) -> float | None:
        """Fresh FFT quality from a just-run assess step, or None if resumed
        (result not in this invocation's sub_results)."""
        if assess_step_id not in executor.sub_results:
            return None
        r = executor.sub_results.get(assess_step_id)
        if (r is not None and getattr(r, "success", False)
                and "fft_quality" in (getattr(r, "data", {}) or {})):
            return float(r.data["fft_quality"])
        return 0.0

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 针尖安全包络(超上限拒绝,不夹紧)。"""
        errors = super().validate_params(params)
        from mast.skills.builtins._tip_policy import apply_tip_policy
        _, plan = apply_tip_policy(params, ("pulse_v",))
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        return errors

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        # 没给的参数按当前针尖查方案表。钨腐蚀针和铂铱剪切针不是一个打法,
        # 而在此之前这里只有一个模型现编的数。
        from mast.skills.builtins._tip_policy import (
            apply_tip_policy,
            policy_fields_for_result,
        )
        params, tip_plan = apply_tip_policy(
            params, ("pulse_v", "pulse_duration_s", "pulse_count",
                     "target_quality", "max_attempts"))
        if tip_plan is not None and not tip_plan.ok:
            executor.progress.aborted = True
            executor.progress.aborted_reason = "；".join(tip_plan.refusals)
            return
        for _k, _v in policy_fields_for_result(tip_plan).items():
            executor.set_partial(_k, _v)

        pulse_v = params.get("pulse_v")
        if pulse_v is None:
            pulse_v = 3.0        # 解析层不可用时的保守兜底,不是 KeyError
        pulse_duration_s = float(params.get("pulse_duration_s", 0.1) or 0.1)
        pulse_count = int(params.get("pulse_count", 1) or 1)
        max_attempts = int(params.get("max_attempts", 5))
        target_quality = float(params.get("target_quality", 0.3))
        scan_w = float(params.get("scan_width_m", 10e-9))
        # cx / cy already resolved into partial_data by run_composite()
        cx = executor.progress.partial_data.get("center_x_m", 0.0) or 0.0
        cy = executor.progress.partial_data.get("center_y_m", 0.0) or 0.0
        scan_speed = scan_w / 0.1  # 0.1s per line

        # Hint upper-bound step count for UI progress bars (initial assess + loop)
        executor.set_total_steps(5 * max_attempts + 4)

        # 0. Initial assessment BEFORE any pulse: if the tip
        #    is already good, don't fire a pulse into a perfectly good imaging
        #    region. Only on a FRESH run — a resumed run (completed_steps already
        #    populated) continues straight into the loop, so a0 never perturbs
        #    resume step-matching or the quality_history indices.
        is_resume = bool(executor.progress.completed_steps)
        if not is_resume:
            yield from self._emit_scan_and_assess(
                executor, "a0", cx, cy, scan_w, scan_speed,
                abort_on_timeout=False)
            if executor.progress.aborted:
                return
            q0 = self._quality_of(executor, "a0:assess")
            if q0 is not None:
                executor.set_partial("initial_quality", q0)
                executor.set_partial("final_quality", q0)
                if q0 >= target_quality:
                    logger.info("ConditionTip: tip already good (q=%.3f ≥ %.3f) "
                                "— no pulse fired", q0, target_quality)
                    executor.set_partial("target_reached", True)
                    executor.set_partial("attempts", 0)
                    return

        for attempt in range(1, max_attempts + 1):
            logger.info("ConditionTip attempt %d/%d", attempt, max_attempts)

            # 1. Apply tip pulse
            yield CompositeStep(
                step_id=f"a{attempt}:pulse",
                skill_name="TipPulse",
                params={"pulse_v": pulse_v, "duration_s": pulse_duration_s,
                        "count": pulse_count},
                optional=False,
                checkpoint_after=True,
                tags=("attempt", f"attempt={attempt}", "pulse"),
            )
            if executor.progress.aborted:
                return

            # 2-4. Configure + fast test scan + wait (stops scan on timeout).
            yield from self._emit_scan_and_assess(
                executor, f"a{attempt}", cx, cy, scan_w, scan_speed)
            if executor.progress.aborted:
                return

            # Inspect the assess result. On resume, the executor skips
            # already-completed steps, so the resumed-step result is NOT
            # in ``executor.sub_results`` (which only carries this
            # invocation's results). For resumed attempts the quality
            # is already in partial_data["quality_history"][attempt-1]
            # — we should not re-append it.
            assess_step_id = f"a{attempt}:assess"
            ran_fresh = assess_step_id in executor.sub_results
            if ran_fresh:
                assess_result = executor.sub_results.get(assess_step_id)
                if (
                    assess_result is not None
                    and getattr(assess_result, "success", False)
                    and "fft_quality" in (
                        getattr(assess_result, "data", {}) or {})
                ):
                    quality = float(assess_result.data["fft_quality"])
                else:
                    quality = 0.0

                # Append fresh quality to history.
                history = list(executor.progress.partial_data.get(
                    "quality_history", []))
                history.append(quality)
                executor.set_partial("quality_history", history)
                executor.set_partial("attempts", attempt)
                executor.set_partial("final_quality", quality)

                logger.info(
                    "ConditionTip attempt %d: quality=%.3f (target=%.3f)",
                    attempt, quality, target_quality,
                )
            else:
                # Resumed attempt — read the recorded quality from
                # the persisted history (which run_composite seeded).
                history = list(executor.progress.partial_data.get(
                    "quality_history", []))
                if attempt - 1 < len(history):
                    quality = float(history[attempt - 1])
                else:
                    quality = 0.0
                logger.info(
                    "ConditionTip attempt %d: resumed quality=%.3f "
                    "(target=%.3f)", attempt, quality, target_quality,
                )

            if quality >= target_quality:
                # Mark success on the progress so aggregate() can read it
                executor.set_partial("target_reached", True)
                return

        # Exhausted all attempts without hitting target — generator ends
        # naturally; run_composite()'s aggregate path emits the failure.

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def aggregate(
        self, sub_results: dict, progress: CompositeProgress,
    ) -> dict:
        data = {
            "attempts": int(progress.partial_data.get("attempts", 0)),
            "final_quality": float(
                progress.partial_data.get("final_quality", 0.0)),
            "quality_history": list(
                progress.partial_data.get("quality_history", [])),
            "target_quality": float(
                progress.partial_data.get("target_quality", 0.0)),
        }
        return data

    # ------------------------------------------------------------------
    # run_composite override — seeds cx/cy resolution + accumulators,
    # then drives the GraphExecutor like the default impl does, then
    # synthesizes the final SkillResult with v1-compatible success
    # semantics (success iff target_quality reached).
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        target_quality = float(params.get("target_quality", 0.3))
        max_attempts = int(params.get("max_attempts", 5))

        # Stash context for helpers that run between yields (not sub-skills).
        self._context = context

        # Capture the operator's ORIGINAL scan frame so we can (a) fall back to
        # its centre when the tip position is unknown — instead of yanking the
        # test scan to absolute (0,0) — and (b) restore it after conditioning
        # (the test scan reconfigures the frame to a small 10 nm box; leaving
        # that behind loses the operator's field of view).
        self._orig_frame = self._capture_frame(context)

        # Resolve scan center: params → live tip position → original frame centre.
        cx = params.get("center_x_m")
        cy = params.get("center_y_m")
        if cx is None or cy is None:
            fx = fy = None
            try:
                state = context.state.snapshot()
                fx = state.x_pos_m
                fy = state.y_pos_m
            except Exception:
                pass
            frame = self._orig_frame or {}
            cx = (cx if cx is not None else
                  fx if fx is not None else frame.get("cx", 0.0))
            cy = (cy if cy is not None else
                  fy if fy is not None else frame.get("cy", 0.0))

        # ⑫ tip-crash guard: conditioning a spot that keeps crashing the tip is
        # exactly the ~5-min in-place spin this state machine exists to stop —
        # refuse and tell the agent to withdraw + coarse-move to a fresh region.
        from mast.core.tip_crash_tracker import crash_guard
        escape = crash_guard(context, cx, cy)
        if escape:
            return self.fail(escape, repeated_crash=True)

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # Resolved scan center (always reflects current invocation).
        executor.set_partial("center_x_m", cx)
        executor.set_partial("center_y_m", cy)
        executor.set_partial("target_quality", target_quality)
        executor.set_partial("max_attempts", max_attempts)
        # Accumulators — preserve across resume.
        executor.set_partial_default("attempts", 0)
        executor.set_partial_default("final_quality", 0.0)
        executor.set_partial_default("quality_history", [])
        executor.set_partial_default("target_reached", False)
        self._executor = executor

        try:
            all_good = executor.run_plan(self.plan_dynamic(params, executor))
        finally:
            # Restore the operator's original scan frame (the loop reconfigured
            # it to a small test box). Best-effort; never masks the result.
            self._restore_frame(context, getattr(self, "_orig_frame", None))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        target_reached = bool(
            executor.progress.partial_data.get("target_reached", False))

        # Success iff target_reached (v1 contract: any attempt's quality
        # >= target). If executor aborted, fail with the abort reason.
        if target_reached and all_good:
            return self.ok(**data)

        if executor.progress.aborted:
            return self.fail(
                executor.progress.aborted_reason or "composite aborted",
                **data,
            )

        # All attempts ran but none met target
        attempts = data["attempts"] or max_attempts
        history = data["quality_history"]
        best = max(history) if history else 0.0
        return self.fail(
            f"Target quality {target_quality} not reached after "
            f"{attempts} attempts (best: {best:.3f})",
            **data,
        )


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(ConditionTip, context_provider)
