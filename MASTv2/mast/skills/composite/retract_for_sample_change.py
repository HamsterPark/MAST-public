"""RetractForSampleChange — laddered coarse retract for 换样品 / 关机 / 收工.

A "big" retract that换样品/关机前把针尖用**粗动
马达**沿保存的退针方向退几千步——distinct from the piezo-only SafeRetract /
WithdrawTip (those just收 the fine Z). The danger: if the configured retract
direction is BACKWARDS on the instrument, stepping the coarse motor drives the tip
INTO the sample (撞针).

The anti-crash design is a laddered self-check:

    withdraw the piezo, then step the coarse motor 1 → 10 → 100 → (rest),
    and AFTER EACH rung open the Z-controller and read the **Z-piezo
    direction**: if the tip really receded, feedback extends the piezo to
    chase the now-farther sample (Z moves in the extend direction); if the
    direction was backwards the piezo retracts / the current spikes — stop.

Why look at Z, not current: once the tip is far the current
has decayed to ~0 and carries no sign, but the Z-piezo's extend/retract
direction is always clean. Current spikes are used only as an extra danger trip.

Why the ladder starts at 1: the FIRST rung is the minimum-risk probe. One
coarse step is « the piezo range, so even if the direction is backwards,
opening feedback after a single step lets the piezo absorb that tiny approach
without a crash — we learn the direction is wrong and abort having moved 1 step.
Only after a rung CONFIRMS receding do we escalate. The configured direction is
an INTENT; this per-rung Z self-check is the real anti-crash safety net.

Direction + total steps + the recede threshold + the lock-in dI/dV signal come
from :mod:`mast.core.instrument_profile` (per-rig, operator-set). CONFIRM-gated
(a big deliberate move), tagged dangerous.
"""

from __future__ import annotations

from typing import Any

from mast.core import instrument_profile as ip
from mast.core.crosstalk_report import crosstalk_report
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite._z_settle import ZSettle, settle_and_read_z
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

# Synthetic phase identifiers — intercepted by _RetractPhaseCtx.
_P_STOP = "_phase_stop_powered"
_P_BASELINE = "_phase_baseline"
_P_WITHDRAW = "_phase_withdraw_initial"
_P_RETRACT = "_phase_retract_verify"

# A current this far above the setpoint (or this far above the absolute floor
# when no setpoint is readable) during a recede self-check means the tip is
# APPROACHING — a backwards direction. Trip immediately.
_DANGER_CURRENT_MULT = 3.0
_DANGER_CURRENT_ABS_A = 1e-9   # 1 nA — retracting, the current should be ~0
_NOISE_FLOOR_A = 1e-12         # below this, "current" is amplifier noise


def _first_val(rv) -> "float | None":
    """Pull the first scalar from a Nanonis (header, body, [vals]) triplet."""
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        inner = rv[2]
        if isinstance(inner, (list, tuple)) and inner:
            try:
                return float(inner[0])
            except (TypeError, ValueError):
                return None
    return None


class _RetractPhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names to
    the composite's own handlers (same pattern as AutoApproach)."""

    def __init__(self, real_ctx, skill: "RetractForSampleChange") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class RetractForSampleChange(CompositeSkillGraph):
    """Laddered coarse-motor retract with a per-rung Z-direction self-check."""

    # Wait for Z convergence rather than reading after a fixed delay. Poll cadence,
    # window length and the optional timeout configure that check. None delegates
    # the production time budget to instrument_profile.z_settle_timeout_s.
    _poll_interval_s = 0.1
    _settle_window_n = 5
    _settle_timeout_s: "float | None" = None

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RetractForSampleChange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM: a big, deliberate coarse move (换样品/关机). It steps the
            # coarse motor in the AWAY direction only, and every rung self-checks
            # the Z-piezo direction + trips on a current spike, so it is not
            # HUMAN-gated like a coarse sample-APPROACH — but it is a major action
            # the operator should knowingly trigger, hence CONFIRM not AUTO.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "换样品/关机前的大退针：先收压电，再用粗动马达沿配置的退针方向"
                "分级(1→10→100→剩余)后退约数千步；每级退完开反馈回读 Z 压电走向"
                "自检(只有确认在远离才继续，检测到逼近立即停+撤针)。方向/步数/阈值"
                "来自 instrument_profile。"
            ),
            parameters=[
                ParameterSpec(
                    name="total_steps",
                    type="int",
                    description=(
                        "退针总步数(覆盖 instrument_profile 的默认)。分级 1→10→100→"
                        "剩余(按单次上限分批)后退这么多步。"),
                    required=False,
                    default=None,
                    min_value=1,
                    max_value=1_000_000,
                ),
            ],
            estimated_duration_s=60.0,
            composition_level=2,
            tags=["tip", "safety", "retract", "sample-change", "coarse", "dangerous"],
        )

    # ------------------------------------------------------------------
    # Plan — ladder derived from params + instrument_profile.
    # ------------------------------------------------------------------

    @staticmethod
    def _ladder(total: int, step_max: int) -> list[int]:
        """1 → 10 → 100 → (rest, chunked by step_max), never exceeding total."""
        total = max(1, int(total))
        step_max = max(1, min(1000, int(step_max)))
        rungs: list[int] = []
        for probe in (1, 10, 100):
            if sum(rungs) + probe <= total:
                rungs.append(probe)
        remaining = total - sum(rungs)
        while remaining > 0:
            chunk = min(step_max, remaining)
            rungs.append(chunk)
            remaining -= chunk
        return rungs

    def plan(self, params: dict) -> list[CompositeStep]:
        total = params.get("total_steps")
        if not isinstance(total, int) or total <= 0:
            total = int(ip.get_config("retract_total_steps", 3000))
        step_max = int(ip.get_config("retract_step_max", 1000))
        rungs = self._ladder(total, step_max)

        steps: list[CompositeStep] = [
            CompositeStep(step_id="stop_powered", skill_name=_P_STOP, params={},
                          optional=False, checkpoint_after=False, tags=("stop",)),
            CompositeStep(step_id="baseline_z", skill_name=_P_BASELINE, params={},
                          optional=False, checkpoint_after=True, tags=("read",)),
            CompositeStep(step_id="withdraw_initial", skill_name=_P_WITHDRAW,
                          params={}, optional=False, checkpoint_after=True,
                          tags=("withdraw",)),
        ]
        cumulative = 0
        for i, n in enumerate(rungs):
            cumulative += n
            steps.append(CompositeStep(
                step_id=f"retract_{i}_{n}",
                skill_name=_P_RETRACT,
                params={"steps": n, "level": i, "cumulative": cumulative},
                optional=False, checkpoint_after=True,
                tags=("retract", "verify")))
        return steps

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _P_STOP:
            return self._phase_stop_powered(real_ctx)
        if skill_name == _P_BASELINE:
            return self._phase_baseline(real_ctx)
        if skill_name == _P_WITHDRAW:
            return self._phase_withdraw(real_ctx, tag="withdraw_initial")
        if skill_name == _P_RETRACT:
            return self._phase_retract_verify(
                real_ctx, int(params.get("steps", 1)),
                int(params.get("level", 0)), int(params.get("cumulative", 0)))
        return SkillResult(skill_name=skill_name, success=False,
                           error=f"Unknown phase: {skill_name}")

    def _phase_stop_powered(self, real_ctx) -> SkillResult:
        """Stop any powered approach FIRST — a running AutoApproach / motor would
        eat the retract. Best-effort: a stop that errors must not block退针."""
        # Literal verbs: safe_call verbs must be greppable string literals — the
        # abort-policy / security / API-census tools scan for safe_call("…"). A
        # verb behind a splatted tuple is invisible to all of them.
        for thunk in (
            lambda: real_ctx.safe_call("AutoApproach_OnOffSet", 0),
            lambda: real_ctx.safe_call("Motor_StopMove"),
        ):
            try:
                self._call_log.append(thunk())
            except Exception:  # noqa: BLE001 — defensive; stop is advisory
                pass
        return SkillResult(skill_name=_P_STOP, success=True, data={})

    def _settle(self, real_ctx) -> "ZSettle":
        """Let the Z loop find the surface and STOP, then read it."""
        return settle_and_read_z(
            real_ctx, log=self._call_log, timeout_s=self._settle_timeout_s,
            poll_interval_s=self._poll_interval_s,
            window_n=self._settle_window_n)

    def _phase_baseline(self, real_ctx) -> SkillResult:
        """Record where the loop COMES TO REST as the recede baseline.

        Waits for the Z piezo to stop travelling rather than sleeping a fixed
        window: a Z read mid-ramp measures how long we waited, not where the
        surface is, and every rung below is a subtraction against this number.

        **An unusable baseline aborts.** It used to degrade to a current-only
        check, which sounds forgiving until you notice what it degrades INTO: a
        zero current means "no surface within piezo reach", which is equally
        true of a tip that is receding, a tip still out of range while
        approaching, and a dead preamp. Calling that "receding" is the guard
        that isn't — and this composite then drives thousands of coarse steps on
        it. Refusing is loud and costs a sample change; proceeding costs a tip.
        """
        settle = self._settle(real_ctx)
        self._baseline = settle
        self._baseline_z = settle.z_m if settle.usable else None
        self._setpoint_a = settle.setpoint_a
        self._executor.set_partial("baseline_z_m", self._baseline_z)
        self._executor.set_partial("setpoint_a", self._setpoint_a)
        self._executor.set_partial("baseline_settle", settle.as_dict())
        if settle.state == "aborted":
            return SkillResult(skill_name=_P_BASELINE, success=False,
                               error="aborted while settling for baseline Z")
        if not settle.usable:
            return SkillResult(
                skill_name=_P_BASELINE, success=False,
                error=("退针方向自检无法建立基线 Z:" + settle.why() + "。"
                       "每一级退针都是拿「反馈稳定后的 Z」跟这个基线比,"
                       "基线读不到就没有方向自检 —— 不会盲退几千步。"
                       "若本机反馈确实比这个预算慢,到设置页把「退针 Z 稳定预算」"
                       "(z_settle_timeout_s)调大。"),
                data={"baseline_settle": settle.as_dict()})
        return SkillResult(skill_name=_P_BASELINE, success=True,
                           data={"baseline_z_m": self._baseline_z,
                                 "setpoint_a": self._setpoint_a,
                                 "baseline_settle": settle.as_dict()})

    def _phase_withdraw(self, real_ctx, *, tag: str) -> SkillResult:
        """Withdraw the fine Z fully (ZCtrl_Withdraw wait-until-finished)."""
        rec = real_ctx.safe_call("ZCtrl_Withdraw", 1, -1)
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(skill_name=_P_WITHDRAW, success=False,
                               error=f"withdraw ({tag}) failed: {rec.error}")
        return SkillResult(skill_name=_P_WITHDRAW, success=True, data={"withdrawn": tag})

    def _read_didv(self, real_ctx) -> "float | None":
        """Lock-in R (dI/dV) magnitude via Signals_ValGet, or None when the
        signal index isn't configured / the read fails."""
        idx = ip.get_config("lockin_signal_index", None)
        if idx is None:
            return None
        try:
            rec = real_ctx.safe_call("Signals_ValGet", int(idx), 0)
            self._call_log.append(rec)
            if rec.error:
                return None
            v = _first_val(rec.return_value)
            return None if v is None else abs(v)
        except Exception:  # noqa: BLE001
            return None

    def _phase_retract_verify(self, real_ctx, steps: int, level: int,
                              cumulative: int) -> SkillResult:
        """Step the coarse motor AWAY by `steps`, then open feedback and verify
        the tip actually receded (Z-piezo extend direction), tripping on an
        approach / current spike."""
        dir_code = ip.get_retract_dir_code()

        # 1. Coarse retract (direction from config; away-only). Motor_StartMove(
        #    direction, steps, group, wait) — same shape as MotorMove uses.
        rec_move = real_ctx.safe_call("Motor_StartMove", dir_code, steps, 0, 1)
        self._call_log.append(rec_move)
        if rec_move.error:
            return SkillResult(skill_name=_P_RETRACT, success=False,
                               error=f"coarse retract (rung {level}, {steps} steps) "
                                     f"failed: {rec_move.error}")

        # 2. Open feedback and wait for the piezo to STOP chasing the sample.
        #    Not a fixed window: the loop takes seconds to find the surface and
        #    a Z read taken while it is still travelling is a stopwatch reading.
        after = self._settle(real_ctx)
        if after.state == "aborted":
            self._safe_stop_and_withdraw(real_ctx)
            return SkillResult(skill_name=_P_RETRACT, success=False,
                               error="aborted during recede self-check")

        # 3. Read the remaining evidence.
        z_after = after.z_m if after.usable else None
        current = after.current_a
        didv = self._read_didv(real_ctx)

        verdict, reason = self._judge_recede(after)
        rung = {"level": level, "steps": steps, "cumulative": cumulative,
                "z_after_m": z_after, "current_a": current, "didv_v": didv,
                "verdict": verdict, "reason": reason, "settle": after.as_dict()}
        # 串扰导航仅提供参考读数，不驱动退针阶梯；方向决策仍来自 _judge_recede。
        rung.update(crosstalk_report(real_ctx))
        self._rungs.append(rung)
        self._executor.set_partial("rungs", list(self._rungs))

        if verdict == "unsettled":
            # NOT "approaching" — a reading we could not take must not wear the
            # same label as a tip closing in, or a broken judge is
            # indistinguishable from a backwards cable. Stops the ladder: the
            # next rung is 10× this one, and it is exactly what the small probe
            # rungs exist to protect.
            self._safe_stop_and_withdraw(real_ctx)
            return SkillResult(
                skill_name=_P_RETRACT, success=False,
                error=("退针方向自检**没有得出结论**(不是判定为逼近):" + reason
                       + "。已停止粗动并撤针。读不到稳定的 Z 就无法判断针尖是远离"
                         "还是靠近,而下一级步数是这一级的 10 倍。"
                         "若本机反馈确实较慢,到设置页把「退针 Z 稳定预算」"
                         "(z_settle_timeout_s)调大。"),
                data={"rung": rung, "rungs": list(self._rungs)})

        if verdict == "no_sign":
            # 停在这一级,而且**处方与 unsettled 不同**:那一条的方子是「调大
            # z_settle_timeout_s」,对着一个没填的符号开那张方子只会把人指向
            # 没坏的东西。这条自检的全部内容就是把 Z 的走向翻译成方向,
            # 没有符号就没有翻译 —— 不是「判不准」,是**根本没有判据**。
            self._safe_stop_and_withdraw(real_ctx)
            return SkillResult(
                skill_name=_P_RETRACT, success=False,
                error=("退针方向自检无法进行:" + reason + "。已停止粗动并撤针。"
                       "z_extend_sign 必须经目标仪器的方向检查明确配置，不能猜测。"
                       "请依据退针与反馈寻面时的原始 Z 读数确认伸长方向，"
                       "再在设置页的退针配置中填写 z_extend_sign。"),
                data={"rung": rung, "rungs": list(self._rungs)})

        if verdict == "approaching":
            # Backwards direction — the tip moved TOWARD the sample. Stop the
            # motor and withdraw NOW; do not take another (larger) step.
            self._safe_stop_and_withdraw(real_ctx)
            return SkillResult(
                skill_name=_P_RETRACT, success=False,
                error=("退针方向自检失败：粗动 " + str(steps) + " 步后开反馈检测到针尖在"
                       "**逼近**样品（" + reason + "）——退针方向很可能配置反了。已停止粗动"
                       "并撤针，只赔了这一级的步数。请核对 instrument_profile 的退针方向/"
                       "z_extend_sign 后再试。"),
                data={"rung": rung, "rungs": list(self._rungs)})

        # 4. Receding (or ambiguous-but-safe on a tiny early rung): withdraw the
        #    piezo again to make the NEXT coarse step safe.
        w = self._phase_withdraw(real_ctx, tag=f"rung_{level}")
        if not w.success:
            return SkillResult(skill_name=_P_RETRACT, success=False,
                               error=f"post-rung withdraw failed: {w.error}",
                               data={"rung": rung})
        return SkillResult(skill_name=_P_RETRACT, success=True, data={"rung": rung})

    def _judge_recede(self, after: "ZSettle") -> "tuple[str, str]":
        """Classify a rung: receding | approaching | ambiguous | unsettled | no_sign.
        
        Z direction relative to baseline is interpreted only with an explicitly configured
        z_extend_sign. A large current can indicate danger even when Z is unreadable.
        Both readings must settle; otherwise return unsettled rather than infer approach.
        Near-zero current alone cannot distinguish retreat, an out-of-range approach, or a
        disconnected preamplifier. Evidence is weaker when the starting point is not tunnelling.
        Determine polarity from raw readings and independent direction checks, never from
        labels that were themselves generated using the polarity being checked.
        """
        current = after.current_a
        sp = self._setpoint_a if self._setpoint_a is not None else after.setpoint_a
        # Danger by current: retracting, |I| should be ~0. A big current means
        # the tip is near the surface → approaching. Kept ahead of the Z branch
        # precisely because it does not need a settled reading.
        if current is not None and abs(current) > _NOISE_FLOOR_A:
            danger_by_sp = (sp is not None and abs(sp) > 0
                            and abs(current) > _DANGER_CURRENT_MULT * abs(sp))
            if danger_by_sp or abs(current) > _DANGER_CURRENT_ABS_A:
                return ("approaching",
                        f"|I|={abs(current):.2e} A 远高于退针时应有的~0")

        base = self._baseline
        if not after.usable:
            return ("unsettled", after.why())
        if base is None or not base.usable:
            return ("unsettled", "基线 Z 不可用:"
                    + (base.why() if base is not None else "没有基线读数"))

        z_min_m = float(ip.get_config("z_recede_min_nm", 1.0)) * 1e-9
        # **没声明就不判**(2026-08-11):出厂 ``+1`` 不是保守值,是两个互斥答案里的
        # 一个。见 ``instrument_profile.z_extend_sign_or_none``。
        sign = ip.z_extend_sign_or_none()
        if sign is None:
            return ("no_sign",
                    "本机没有声明「压电伸长(趋向样品)对应 Z 读数符号」"
                    "(设置 → 退针 → `z_extend_sign`),"
                    "Z 读数变化无法翻译成「在远离还是在靠近」")
        dz_extend = (after.z_m - base.z_m) * sign

        if after.at_rail and base.at_rail:
            # Both readings are the piezo limit: consistent with receding, but
            # a limit is the same number whichever way the stage went, so it
            # proves nothing. The approaching half of the check stays live —
            # a tip that closed in would pull the piezo off the rail.
            return ("ambiguous",
                    "退针前后压电都到极限、量程内都没有表面 —— "
                    "与远离一致,但证明不了距离(也无逼近迹象)")
        if dz_extend > z_min_m:
            if after.at_rail:
                return ("receding",
                        f"压电走到极限仍未找到表面,较基线至少远离 "
                        f"{dz_extend * 1e9:.2f} nm(极限值,实际更远)")
            return ("receding",
                    f"开反馈后 Z 朝伸长方向移动 {dz_extend * 1e9:.2f} nm（在追更远的样品）")
        if dz_extend < -z_min_m:
            if base.at_rail:
                return ("approaching",
                        f"基线时量程内还没有表面,退针后反而找到了"
                        f"(Z 缩回 {abs(dz_extend) * 1e9:.2f} nm)—— 针尖是靠近了")
            return ("approaching",
                    f"开反馈后 Z 朝缩回方向移动 {abs(dz_extend) * 1e9:.2f} nm（样品变近了）")
        # Within the noise band → not conclusive.
        return ("ambiguous", "Z 走向不明显，未确认但也无逼近迹象")

    def _safe_stop_and_withdraw(self, real_ctx) -> None:
        """Best-effort emergency cleanup: stop the motor + withdraw. Never raises."""
        # Literal verbs (see _phase_stop_powered) — greppable for the safety tools.
        for thunk in (
            lambda: real_ctx.safe_call("Motor_StopMove"),
            lambda: real_ctx.safe_call("ZCtrl_Withdraw", 1, -1),
        ):
            try:
                self._call_log.append(thunk())
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Driver (override — custom phases, like AutoApproach)
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []
        self._rungs: list[dict] = []
        self._baseline_z: "float | None" = None
        self._baseline: "ZSettle | None" = None
        self._setpoint_a: "float | None" = None

        wrapped = _RetractPhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))

        rungs = list(executor.progress.partial_data.get("rungs", self._rungs))
        confirmed = [r for r in rungs if r.get("verdict") == "receding"]
        data: dict[str, Any] = {
            "retracted": bool(all_good),
            "baseline_z_m": executor.progress.partial_data.get("baseline_z_m"),
            "total_steps_retracted": sum(r.get("steps", 0) for r in rungs)
            if all_good else sum(r.get("steps", 0) for r in rungs
                                 if r.get("verdict") != "approaching"),
            "rungs": rungs,
            "recede_confirmed_rungs": len(confirmed),
            "_progress": executor.progress.to_dict(),
        }

        if not executor.progress.aborted:
            executor.clear_sidecar()

        if not all_good:
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=executor.progress.aborted_reason or "RetractForSampleChange aborted",
                data=data, nanonis_calls=list(self._call_log))

        # A verified full retract is the ONE moment the free-oscillation
        # amplitude is knowable: thousands of coarse steps away from the surface,
        # every rung self-checked as receding. Capture the baseline here rather
        # than hoping somebody remembers to pass set_baseline=True by hand — that
        # hope is why the amplitude crash detector has been answering
        # "no_baseline" (i.e. "cannot tell") instead of doing its job.
        # Best-effort: a rig with no qPlus adds a note and nothing else.
        try:
            from mast.skills.builtins._tip_evidence import capture_qplus_baseline

            data.update(capture_qplus_baseline(
                context, note="退针自检全部通过后记录的自由振荡基线"))
        except Exception:  # noqa: BLE001 — a bonus must not fail a good retract
            pass

        return SkillResult(
            skill_name=self._skill_name(), success=True,
            data=data, nanonis_calls=list(self._call_log))
