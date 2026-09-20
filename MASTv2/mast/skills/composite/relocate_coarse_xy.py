"""RelocateCoarseXY — move the sample stage sideways to a fresh patch, safely.

WHY THIS EXISTS
===============
``map_analysis.coarse_move_advice`` has been able to say *whether* to relocate for
a while. Actually doing it was a bare ``MotorMove(direction='x+', steps=N)``:

* its only precondition is ``state.withdrawn`` — the FINE-Z PIEZO at its high
  limit, one or two microns — and that check passes when the state is unknown;
* it never steps the coarse Z motor back, which is where the tens of microns of
  real clearance come from;
* it does not look at the chamber pressure, and driving a coarse piezo through
  the Paschen band arcs across the stack;
* it does not check what the drive amplitude is currently set to;
* it fires one blocking ``Motor_StartMove`` and looks at nothing until it
  returns;
* and it leaves no record of where on the sample the stage has already been.

The gap is exact: moving in XY without first retracting the tip by a safe margin
is exactly the kind of precaution this workflow did not know to take.

THE PHASES
==========
``preflight`` → ``clear`` → ``move`` → ``verify`` → ``reapproach``

``clear`` is the interesting one. It reuses the retract ladder's idea rather than
its code path: step, then open feedback and watch which way the Z piezo goes. A
configured direction is an INTENT; the runtime self-check is the actual anti-crash
guard, and the first rung is one step so a backwards configuration costs exactly
one step to discover.

Clearance is then CONFIRMED, not assumed: the current must be at the noise floor
and — where a qPlus sensor exists — the oscillation amplitude must have recovered
to near its free value. That second witness answers the question the current
cannot: the current is zero both when the tip is far away and when the preamp is
dead, while the amplitude answers "is the tip mechanically free to move".

``move`` goes in chunks with a look between each one, because a blocking move of
several hundred steps is several hundred steps during which nothing is watching.

TWO THINGS THAT WILL BITE A FUTURE EDITOR
=========================================
1. **This composite calls ``Motor_StartMove`` through ``safe_call`` directly**,
   like ``RetractForSampleChange`` does, to avoid inheriting ``MotorMove``'s
   hardcoded semantics. That means the recorder never sees a ``MotorMove``
   payload for it — so ``runtime._LATERAL_COARSE_SKILLS`` has to include
   ``relocatecoarsexy``, or the stage moves, every recorded coordinate becomes
   meaningless, and ``coord_epoch`` never advances to say so.
2. **The result must report ``direction`` and ``steps`` in ``data``.** That is
   what ``lateral_coarse_move_info`` reads to write the ``coarse_move`` marker,
   which is in turn the only source the coarse map's odometer has.
3. **…and ``lateral_steps_taken`` as well, because a failed composite still
   moved the stage.** Until 2026-08-05 the whole result was discarded on
   failure, so a run that slid 301 steps and then failed to re-approach left the
   odometer, the coarse map and ``coord_epoch`` all describing the position the
   sample had LEFT — 301 steps of drift between the physical stage and every
   later plan, silently. ``steps`` cannot carry that on its own: on the failure
   path a reader would need to know which failures happen before the move and
   which after. ``lateral_steps_taken`` is the explicit promise ("the stage
   physically took this many, whatever the verdict"), it is the only key the
   recorder will accept from a failed result, and it is 0 in ``dry_run``.
"""

from __future__ import annotations

import logging
from typing import Any

from mast.core import instrument_profile as ip
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

logger = logging.getLogger(__name__)

# Synthetic phase identifiers — intercepted by _RelocatePhaseCtx.
_P_PREFLIGHT = "_phase_preflight"
_P_CLEAR = "_phase_clear"
_P_MOVE = "_phase_move"
_P_VERIFY = "_phase_verify"
_P_REAPPROACH = "_phase_reapproach"

#: Below this the "current" is amplifier noise, not tunnelling. Same value the
#: approach path uses as its engagement floor — one instrument, one noise floor.
_NOISE_FLOOR_A = 1e-12

#: During a lateral move the tip should see NOTHING. A current above this while
#: the stage is sliding means the clearance was not real; stop immediately.
_MOVE_DANGER_CURRENT_A = 1e-11

# 横移前降低偏压，随后仍保留独立的电流危险检查。
# 低偏压默认值不保证目标仪器上不会场发射；动作工作点需由使用者验证。
_MOVE_BIAS_V = 0.5

_DIRECTIONS = {("x", "+"): "x+", ("x", "-"): "x-",
               ("y", "+"): "y+", ("y", "-"): "y-"}
#: Nanonis Motor_StartMove direction codes.
_DIR_CODE = {"x+": 0, "x-": 1, "y+": 2, "y-": 3}


def _first_val(rv) -> "float | None":
    """First scalar out of a Nanonis (header, body, [vals]) triplet."""
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        inner = rv[2]
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return float(inner)
        if isinstance(inner, (list, tuple)) and inner:
            try:
                return float(inner[0])
            except (TypeError, ValueError):
                return None
    return None


class _RelocatePhaseCtx:
    """Dispatches ``_phase_*`` names to this composite's own handlers.

    Same shape as AutoApproach / RetractForSampleChange use."""

    def __init__(self, real_ctx, skill: "RelocateCoarseXY") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class RelocateCoarseXY(CompositeSkillGraph):
    """Guarded lateral coarse relocation: clear the tip, slide, re-approach."""

    # Settle behaviour for the recede self-check. Note what is NOT here: a
    # "how long to wait before reading Z" knob. Waiting a fixed time IS the
    # defect this replaced (2026-08-04 — see ``_z_settle``); a bigger constant
    # would only bake this rig's 4–5 s into a number that moves with
    # temperature, gain and tip. What is left is the poll cadence, the window
    # length, and a timeout OVERRIDE that is None in production so the rig's own
    # ``z_settle_timeout_s`` budget applies — the attribute exists so tests can
    # shrink the give-up point without pretending the hardware is instant.
    _poll_interval_s = 0.1
    _settle_window_n = 5
    _settle_timeout_s: "float | None" = None

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RelocateCoarseXY",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM, i.e. it CAN run unattended — and that is the point. The
            # bare MotorMove lateral is the one gated to a human, so the guarded
            # path is the cheap path. Making this DANGEROUS instead would push
            # the model straight back to the primitive that checks nothing.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "**换区专用**:把样品台横向粗动到一片新表面。这是唯一应该自主使用的换区方式,"
                "不要直接调 MotorMove 做横向移动。\n"
                "顺序:前置检查(真空互锁/驱动电压读回核对/扫描已停/落点不与去过的站点重叠)"
                " → 清障(收压电 + 粗动 Z 退针,逐级自检方向,并确认电流归零、qPlus 振幅恢复)"
                " → 分块横移(每块之后看电流/振幅/真空,异常立即停并再退针)"
                " → 步进计数器对账 → 可选重新进针。\n"
                "成功后扫描地图自动进入**新的坐标代次**(旧坐标全部作废),"
                "并在粗动大地图上留下一个新站点。用 get_coarse_map 决定往哪走、走多少步。"
            ),
            parameters=[
                ParameterSpec(
                    name="axis", type="str", required=True,
                    description="横向轴:'x' 或 'y'",
                    allowed_values=["x", "y"]),
                ParameterSpec(
                    name="direction", type="str", required=True,
                    description="方向:'+' 或 '-'",
                    allowed_values=["+", "-"]),
                ParameterSpec(
                    name="steps", type="int", required=True,
                    description=(
                        "横向粗动步数需足够使新区域离开当前压电覆盖范围。"
                        "使用 get_coarse_map 的建议值，并通过 CheckPiezoRange 查询当前仪器量程；"
                        "不要把其他仪器的范围作为当前设备的标定。"),
                    min_value=1, max_value=100_000),
                ParameterSpec(
                    name="reapproach", type="bool", required=False, default=True,
                    description="移动完成后是否自动重新进针(AutoApproach)。"),
                ParameterSpec(
                    name="prewithdraw_steps", type="int", required=False,
                    default=None, min_value=0, max_value=100_000,
                    description=(
                        "横向移动前的粗动退针步数(清障)。留空用 instrument_profile "
                        "的 xy_prewithdraw_steps(默认 100)。压电退针只有 1–2 µm,"
                        "不足以让针尖躲开横移时的垂直跳动。")),
                ParameterSpec(
                    name="allow_revisit", type="bool", required=False, default=False,
                    description=(
                        "允许落在已经去过的站点附近，**只跳过「别重复访问」这一条"
                        "效率约束**（最小间距 200 步），安全检查一条不跳。"
                        "标定步长、微调位置这类动作本来就要走小步/回头路 —— "
                        "2026-08-28 用户要六个方向的 nm/步，而每一个几步的移动"
                        "都落在 200 步以内，于是全被那条规则拒掉。"
                        "常规换区**不要**开它：重复用同一片表面是真的浪费。")),
                ParameterSpec(
                    name="dry_run", type="bool", required=False, default=False,
                    description=(
                        "跑完整条相位但**不发出任何移动命令** —— 用于无硬件验收:"
                        "前置检查、看护逻辑、记录都会真的跑一遍。")),
            ],
            preconditions=["vacuum_ok_for_coarse", "scan_not_running"],
            estimated_duration_s=180.0,
            composition_level=3,
            tags=["motor", "coarse", "relocate", "safety", "dangerous"],
        )

    # ── plan ────────────────────────────────────────────────────────────

    def plan(self, params: dict) -> list[CompositeStep]:
        pre = params.get("prewithdraw_steps")
        if not isinstance(pre, int) or pre < 0:
            pre = int(ip.get_config("xy_prewithdraw_steps", 100))
        chunk = max(1, int(ip.get_config("xy_move_chunk_steps", 50)))
        total = int(params.get("steps", 0))

        steps: list[CompositeStep] = [
            CompositeStep(step_id="preflight", skill_name=_P_PREFLIGHT,
                          params=dict(params), optional=False,
                          checkpoint_after=True, tags=("check",)),
            CompositeStep(step_id="clear", skill_name=_P_CLEAR,
                          params={"prewithdraw_steps": pre}, optional=False,
                          checkpoint_after=True, tags=("withdraw", "verify")),
        ]
        # Chunked so something looks between bursts. A single blocking
        # Motor_StartMove of `total` steps is `total` steps during which nothing
        # can notice the clearance was wrong.
        done = 0
        idx = 0
        while done < total:
            n = min(chunk, total - done)
            done += n
            steps.append(CompositeStep(
                step_id=f"move_{idx}_{n}", skill_name=_P_MOVE,
                params={"steps": n, "index": idx, "cumulative": done,
                        "total": total, "axis": params.get("axis"),
                        "direction": params.get("direction"),
                        "dry_run": bool(params.get("dry_run", False))},
                optional=False, checkpoint_after=True, tags=("move", "watch")))
            idx += 1

        steps.append(CompositeStep(step_id="verify", skill_name=_P_VERIFY,
                                   params={}, optional=True,
                                   checkpoint_after=True, tags=("read",)))
        if params.get("reapproach", True):
            steps.append(CompositeStep(
                step_id="reapproach", skill_name=_P_REAPPROACH,
                params={"dry_run": bool(params.get("dry_run", False))},
                optional=False, checkpoint_after=True, tags=("approach",)))
        return steps

    # ── phase dispatch ──────────────────────────────────────────────────

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _P_PREFLIGHT:
            return self._phase_preflight(real_ctx, params)
        if skill_name == _P_CLEAR:
            return self._phase_clear(real_ctx, int(params.get("prewithdraw_steps", 100)))
        if skill_name == _P_MOVE:
            return self._phase_move(real_ctx, params)
        if skill_name == _P_VERIFY:
            return self._phase_verify(real_ctx)
        if skill_name == _P_REAPPROACH:
            return self._phase_reapproach(real_ctx, params)
        return SkillResult(skill_name=skill_name, success=False,
                           error=f"Unknown phase: {skill_name}")

    # ── phases ──────────────────────────────────────────────────────────

    def _phase_preflight(self, real_ctx, params: dict) -> SkillResult:
        """Everything that must be true BEFORE anything moves.

        Ordered cheapest-and-most-decisive first, so a refusal costs one call and
        names the real reason rather than the first symptom."""
        checks: dict[str, Any] = {}

        # 1. Vacuum. Fail-closed: an unknown pressure is exactly the case this
        #    refuses, because a gauge that cannot see is a gauge in the discharge
        #    band or at atmosphere with nothing to distinguish them.
        from mast.core import vacuum_interlock as vac

        verdict = vac.check()
        checks["vacuum"] = verdict.as_dict()
        if not verdict.allow:
            return SkillResult(
                skill_name=_P_PREFLIGHT, success=False,
                error=f"真空互锁拒绝粗动:{verdict.reason}", data={"checks": checks})

        # 2. Drive amplitude AS IT IS NOW. The declared ceiling constrains what
        #    MAST writes; it says nothing about what somebody set in the Nanonis
        #    UI. Unreadable is a refusal — a check that passes when it cannot see
        #    is not a check.
        from mast.core import coarse_drive
        from mast.skills.builtins.motor import _parse_freq_amp

        rec = real_ctx.safe_call("Motor_FreqAmpGet", 0)
        self._call_log.append(rec)
        freq, amp = (None, None) if rec.error else _parse_freq_amp(rec.return_value)
        ok, why = coarse_drive.readback_matches(amp, freq)
        checks["drive"] = {"frequency_hz": freq, "amplitude_v": amp,
                           "ok": ok, "note": why}
        if not ok:
            return SkillResult(skill_name=_P_PREFLIGHT, success=False,
                               error=f"粗动驱动核对失败:{why}",
                               data={"checks": checks})

        # 3. Destination sanity against the coarse map. The LLM picked `steps`;
        #    this is where "don't go back to where we've been" is enforced — as
        #    geometry on a map, not as a rule someone has to remember.
        checks["destination"] = self._check_destination(real_ctx, params)
        if checks["destination"].get("blocked"):
            return SkillResult(
                skill_name=_P_PREFLIGHT, success=False,
                error=("目标落点被拒:" + str(checks["destination"].get("reason"))),
                data={"checks": checks})

        # 4. Conditions worth RECORDING rather than gating on. Coarse step size
        #    is strongly temperature-dependent — the same 100 steps travel much
        #    further at 300 K than at 4 K — so the odometer entry is only
        #    interpretable with the temperature beside it.
        checks["temperature_k"] = self._read_temperature()
        self._temperature_k = checks["temperature_k"]

        # 5. Is the amplitude witness usable at all? Recorded now so the clear
        #    phase can tell "the tip is not free" from "we have no way to know".
        try:
            from mast.skills.builtins._tip_evidence import qplus_fields
            checks["qplus"] = qplus_fields(real_ctx)
        except Exception:  # noqa: BLE001
            checks["qplus"] = {}

        self._checks = checks
        return SkillResult(skill_name=_P_PREFLIGHT, success=True,
                           data={"checks": checks})

    def _check_destination(self, real_ctx, params: dict) -> dict:
        """Would this move land on a patch of sample we have already used?

        ``allow_revisit=True`` skips ONLY this check. Specifying the coarse step
        size in nm needs moves of a few steps
        — and every such move lands inside ``site_spacing_steps`` (200) of the
        current site, so this check refused all of them. That rule is a
        **surface-budget** rule ("don't re-use ground"), stated as such in this
        method's own docstring; it is not a safety rule. Everything that IS a
        safety rule — vacuum interlock, drive-voltage readback, scan stopped,
        coarse-Z clearance ladder, bias lowered, current proven at the noise
        floor — still runs. The marker is still recorded, so the map keeps
        knowing where the stage went.

        Degrades to "cannot tell, allow" when the map is unavailable: this is a
        surface-budget question, not a safety one, and refusing to relocate
        because the database is unreachable would strand a run over bookkeeping."""
        if bool(params.get("allow_revisit")):
            return {"blocked": False, "reason": (
                "allow_revisit=True —— 跳过「别重复访问」这条效率约束"
                "（标定步长、微调位置这类动作本来就要走回头路）。"
                "安全检查一条没跳。")}

        try:
            from mast.io.coarse_map import (
                AXIS_OF, SIGN_OF, CoarseMapConfig, _blocked_by, derive_sites,
            )

            rows, cfg = self._map_inputs(real_ctx)
            if rows is None:
                return {"blocked": False, "reason": "无法读取粗动记录,跳过落点复核"}
            cfg = cfg or CoarseMapConfig()
            sites = derive_sites(rows, cfg)
            cur = sites[-1]
            direction = _DIRECTIONS.get(
                (str(params.get("axis")), str(params.get("direction"))))
            steps = int(params.get("steps", 0))
            if direction is None or steps <= 0:
                return {"blocked": True, "reason": "轴/方向/步数无效"}
            if not cur.position_known:
                # Relative motion along one axis is still monotone with a broken
                # odometer; a return trip is not, because it needs the absolute
                # position we no longer have.
                last = self._last_direction(sites)
                if last is not None and last != direction:
                    return {"blocked": True, "reason": (
                        f"粗动里程表已失效(某次粗动缺方向/步数),此时只允许沿上一次的方向"
                        f"{last} 继续前进,不允许改向或回头 —— 回到某个位置需要绝对坐标,"
                        f"而那个数字现在是假的。")}
                return {"blocked": False, "reason": "里程表不确定,但方向与上次一致,放行",
                        "position_known": False}
            nx = cur.x_steps + (SIGN_OF[direction] * steps if AXIS_OF[direction] == "x" else 0)
            ny = cur.y_steps + (SIGN_OF[direction] * steps if AXIS_OF[direction] == "y" else 0)
            travelled = abs(nx) if AXIS_OF[direction] == "x" else abs(ny)
            if travelled > cfg.axis_step_budget:
                return {"blocked": True, "reason": (
                    f"落点 {travelled} 步超出单轴行程预算 {cfg.axis_step_budget} 步 —— "
                    "粗动台走到头只会空滑,但位置认知会全部丢失。")}
            blocked, clearance = _blocked_by((nx, ny), sites, cfg)
            if blocked:
                return {"blocked": True, "lands_at": [nx, ny], "reason": (
                    f"落点 ({nx}, {ny}) 会压在已经去过的站点上"
                    f"(最小间距 {cfg.site_spacing_steps} 步,还要加上那个站点的"
                    f"不确定半径)。用 get_coarse_map 取一个可用的方向/步数。")}
            return {"blocked": False, "lands_at": [nx, ny],
                    "clearance_steps": round(clearance, 1)}
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not strand a run
            logger.debug("destination check unavailable: %s", exc)
            return {"blocked": False, "reason": f"落点复核不可用({exc}),放行"}

    @staticmethod
    def _last_direction(sites) -> str | None:
        from mast.io.coarse_map import _last_known_direction
        return _last_known_direction(sites)

    @staticmethod
    def _map_inputs(real_ctx):
        """(marker rows, CoarseMapConfig) for the current scope, or (None, None).

        Through an injected provider rather than off the context: ExecutionContext
        carries the pool, the state, the registry and the abort events — nothing
        about experiment records — and reaching around it into the live app object
        would invert the layering and make this skill untestable."""
        from mast.core import coarse_map_provider
        return coarse_map_provider.markers_and_config()

    @staticmethod
    def _read_temperature() -> float | None:
        """Sample temperature, if a thermometer is wired. Recorded, never gated on.

        A relocation at 4 K and one at 300 K travel very different distances for
        the same step count, so an odometer entry without a temperature beside it
        cannot be compared with the next one."""
        from mast.core import coarse_map_provider
        return coarse_map_provider.temperature_k()

    def _phase_clear(self, real_ctx, prewithdraw_steps: int) -> SkillResult:
        """Get the tip genuinely out of the way, and PROVE it before moving.

        The piezo withdraw alone buys ~1 µm. What a sliding stage can present is
        sample tilt, vertical runout and the tip's own length — tens of microns.
        So the coarse Z motor steps back too, in a ladder whose first rung is a
        single step: if the configured retract direction is backwards, one step
        is « the piezo range and opening feedback reveals it having cost nothing.
        The CONFIG is an intent; this self-check is the guard."""
        log = self._call_log
        # ⭐ 降压必须排在**清障与自检之前** —— 清障阶梯本身也在读电流。
        self._lower_bias_for_move(real_ctx)

        # Stop anything powered first — a running AutoApproach would fight every
        # retract step, and a motor already moving must not be re-commanded.
        for thunk in (lambda: real_ctx.safe_call("AutoApproach_OnOffSet", 0),
                      lambda: real_ctx.safe_call("Motor_StopMove"),
                      lambda: real_ctx.safe_call("Scan_Action", 1, 0)):
            try:
                log.append(thunk())
            except Exception:  # noqa: BLE001 — best-effort quiescing
                pass

        # On self, not just local: the per-rung verdicts ARE the output of the
        # anti-crash self-check, and until 2026-08-04 they reached the composite
        # result only by being quoted in an error string — so a run that PASSED
        # left no record of what it had proved, and the acceptance question
        # ("did reversing the direction reverse the verdict?") could not be
        # answered from the result at all.
        rungs = self._rungs
        baseline = None
        if prewithdraw_steps > 0:
            # Baseline: where does the loop COME TO REST when it goes looking
            # for the surface? Every rung asks the same question afterwards, and
            # the answer is only a measure of the tip–sample gap once the piezo
            # has stopped travelling — so this waits for convergence instead of
            # sleeping a fixed 1.5 s and reading whatever the ramp was passing
            # through. See ``_z_settle`` for the real-machine evidence.
            baseline = self._settle(real_ctx, log)
            self._baseline_settle = baseline.as_dict()
            if baseline.state == "aborted":
                return SkillResult(skill_name=_P_CLEAR, success=False,
                                   error="aborted before clearance retract")
            if not baseline.usable:
                # FAIL CLOSED. Every rung's verdict is a comparison against this
                # reading; without it the ladder would drive the full
                # prewithdraw with NO direction check at all — which is the bare
                # MotorMove this composite exists to replace. Refusing here
                # costs a relocation; proceeding costs a tip.
                return SkillResult(
                    skill_name=_P_CLEAR, success=False,
                    error=("清障退针自检无法建立基线:" + baseline.why() + "。"
                           "方向自检要拿「反馈稳定后的 Z」和退针后的同一个量比,"
                           "基线读不到就整条自检都不成立 —— 不会盲退。"
                           "若本机反馈确实比这个预算慢,到设置页把「退针 Z 稳定预算」"
                           "(z_settle_timeout_s)调大。"),
                    data={"baseline": baseline.as_dict()})

            dir_code = ip.get_retract_dir_code()
            for n in self._clearance_ladder(prewithdraw_steps):
                # Piezo out of the way BEFORE the coarse motor steps — a coarse
                # step with the piezo extended is the crash this ladder is for.
                # (The settle below withdraws too, but that one is about the
                # MEASUREMENT starting from a repeatable place; this one is the
                # safety invariant, and it has to hold at this line.)
                log.append(real_ctx.safe_call("ZCtrl_Withdraw", 1, -1))
                rec = real_ctx.safe_call("Motor_StartMove", dir_code, n, 0, 1)
                log.append(rec)
                if rec.error:
                    return SkillResult(
                        skill_name=_P_CLEAR, success=False,
                        error=f"清障退针失败({n} 步):{rec.error}",
                        data={"rungs": rungs})

                after = self._settle(real_ctx, log)
                if after.state == "aborted":
                    self._panic(real_ctx)
                    return SkillResult(skill_name=_P_CLEAR, success=False,
                                       error="aborted during clearance retract",
                                       data={"rungs": rungs})
                verdict, why = self._judge_recede(baseline, after)
                # Two extras the no-displacement guard below needs, recorded per
                # rung so the refusal is auditable afterwards rather than being a
                # conclusion nobody can re-derive.
                #
                # dz_m is measured against the SHARED baseline, not against the
                # previous rung — so the last rung's dz_m is already the
                # CUMULATIVE displacement over the whole ladder. That is what
                # makes the guard cumulative without summing anything.
                #
                # 符号没声明时 ``dz_m`` 是 ``None`` 而不是「用出厂值算出来的数」:
                # 这一列是台账,一个方向可能反了的位移写进台账,事后没人分得出
                # 它是哪个方向 —— 而 ``_no_displacement`` 只用 ``abs(dz)``,不受影响。
                _s = ip.z_extend_sign_or_none()
                has_ruler = baseline.measures_gap and after.measures_gap
                dz_m = (None if (baseline.z_m is None or after.z_m is None
                                 or _s is None)
                        else (after.z_m - baseline.z_m) * _s)
                rungs.append({"steps": n, "z_after_m": after.z_m,
                              "current_a": after.current_a, "verdict": verdict,
                              "reason": why, "settle": after.as_dict(),
                              "has_ruler": has_ruler, "dz_m": dz_m})
                if verdict == "approaching":
                    self._panic(real_ctx)
                    return SkillResult(
                        skill_name=_P_CLEAR, success=False,
                        # 按拒绝依据给出处置建议，不能用 z_extend_sign 翻译出的结论反过来证明该符号正确。
                        error=("清障退针自检失败:粗动 " + str(n) + " 步后判定针尖在"
                               "**逼近**样品(" + why + ")。已停止粗动并撤针。"
                               + self._approach_prescription(why)),
                        data={"rungs": rungs})
                if verdict == "unsettled":
                    # NOT "approaching". The old code could only ever reach the
                    # approaching branch, which is exactly what made a broken
                    # judge indistinguishable from a backwards cable. A reading
                    # we could not take gets its own answer and its own message.
                    #
                    # And it stops the ladder: the value we would otherwise
                    # judge is the mid-ramp number this whole fix exists to
                    # reject, and the rungs after this one are precisely the
                    # steps the one-step probe was built to protect.
                    self._panic(real_ctx)
                    return SkillResult(
                        skill_name=_P_CLEAR, success=False,
                        error=("清障退针自检**没有得出结论**(不是判定为逼近):"
                               + why + "。已停止粗动并撤针。"
                               "这一级读不到稳定的 Z,就无法判断针尖是远离还是靠近;"
                               "继续加大步数正是这条自检要挡住的事。"
                               "若本机反馈确实较慢,到设置页把「退针 Z 稳定预算」"
                               "(z_settle_timeout_s)调大。"),
                        data={"rungs": rungs})
                if verdict == "no_sign":
                    # 与 ``unsettled`` 同样停梯子,但**处方完全不同**:那一条的方子是
                    # 「调大 z_settle_timeout_s」,对着一个没填的符号开那张方子,
                    # 又是一次把人指向没坏的东西。
                    self._panic(real_ctx)
                    return SkillResult(
                        skill_name=_P_CLEAR, success=False,
                        error=("清障退针自检**无法进行**:" + why + "。已停止粗动并撤针。"
                               "这一项没有可用的默认值,且因机型而异 —— "
                               "「猜一个」和「猜对了」长得一模一样。"
                               "判法:开反馈让压电去找表面,看 Z 读数往哪边走 —— "
                               "那一边就是**伸长**;再到设置页 → 退针 → "
                               "`z_extend_sign` 如实填。"),
                        data={"rungs": rungs})

            # The ladder finished. Did the stage actually GO anywhere?
            stuck, why_stuck = self._no_displacement(rungs, prewithdraw_steps)
            if stuck:
                self._panic(real_ctx)
                return SkillResult(skill_name=_P_CLEAR, success=False,
                                   error=why_stuck, data={"rungs": rungs})

        # Put the piezo away before the lateral move. The last settle left it
        # tracking the surface (that is what makes the reading meaningful), so
        # this is not optional — sliding the stage with an extended piezo is the
        # crash the whole phase is about.
        log.append(real_ctx.safe_call("ZCtrl_Withdraw", 1, -1))
        # Feedback OFF for the lateral move: with it on, the piezo would chase
        # the surface sliding underneath and drive the tip straight back down.
        log.append(real_ctx.safe_call("ZCtrl_OnOffSet", 0))

        proof = self._prove_clear(real_ctx)
        if proof.get("clear") is False:
            self._panic(real_ctx)
            return SkillResult(skill_name=_P_CLEAR, success=False,
                               error=("脱离确认失败:" + str(proof.get("reason"))),
                               data={"rungs": rungs, "clearance": proof})

        # The tip is verified clear — the one moment the free-oscillation
        # amplitude is knowable. Capture it now; this is what stops the crash
        # detector from spending its life reporting "no_baseline".
        try:
            from mast.skills.builtins._tip_evidence import capture_qplus_baseline
            proof.update(capture_qplus_baseline(
                real_ctx, note="横向粗动前确认脱离后记录"))
        except Exception:  # noqa: BLE001
            pass

        self._clearance = proof
        return SkillResult(skill_name=_P_CLEAR, success=True,
                           data={"rungs": rungs, "clearance": proof,
                                 "prewithdraw_steps": prewithdraw_steps,
                                 # The reference every rung was judged against.
                                 # Without it the rung verdicts in the record
                                 # are unfalsifiable after the fact.
                                 "baseline": baseline.as_dict() if baseline else None})

    @staticmethod
    def _clearance_ladder(total: int) -> list[int]:
        """1 → 10 → rest, chunked at 100.

        The single-step first rung is the minimum-risk probe: one coarse step is
        much smaller than the piezo range, so even a backwards direction is
        absorbed and revealed for the price of one step."""
        total = max(0, int(total))
        out: list[int] = []
        for probe in (1, 10):
            if sum(out) + probe <= total:
                out.append(probe)
        remaining = total - sum(out)
        while remaining > 0:
            n = min(100, remaining)
            out.append(n)
            remaining -= n
        return out

    def _settle(self, real_ctx, log) -> ZSettle:
        """Let the Z loop find the surface and STOP, then read it."""
        return settle_and_read_z(
            real_ctx, log=log, timeout_s=self._settle_timeout_s,
            poll_interval_s=self._poll_interval_s,
            window_n=self._settle_window_n)

    @staticmethod
    def _no_displacement(rungs: list[dict], commanded_steps: int) -> tuple[bool, str]:
        """Did the coarse motor take our commands and move NOTHING? (KNOWN_ISSUES 2.28)

        A motor can accept ``Motor_StartMove``, return success, and not move the
        stage at all — the coarse-approach HV switched off, a drive amplitude
        below the slip threshold, the axis at the end of its travel, a mechanical
        jam. Nothing else in this composite notices: ``Motor_StepCounterGet`` is
        unsupported on most controllers, and the mid-move current watch only
        answers "is anything touching the tip", which is equally true of a stage
        that never moved. So the run would report SUCCESS and the odometer would
        record a relocation that did not happen — every later "have we scanned
        here?" then rests on a coordinate frame that was never entered.

        THE DISTINCTION THIS RESTS ON
        =============================
        "We could not measure it" and "we measured zero" are different facts, and
        writing them as one word is what makes a broken check look like a normal
        result. (Same shape as ``unsettled`` vs ``approaching`` above.)

        So this does NOT trigger on ``ambiguous``. It triggers on having had a
        working RULER and reading zero with it: both the baseline and the rung
        settled while TRACKING a real junction, so the piezo was measuring an
        actual tip–sample distance at each end, and that distance did not change.

        Refusing on "every rung was ambiguous" instead would be wrong, and the
        counter-example is common: a tip that starts BEYOND piezo range gives a
        railed baseline and railed rungs — legitimately ambiguous every time —
        while the motor works perfectly. That happens after a sample-change
        retract, after a failed approach, after a manual withdraw.

        WHY IT NEEDS NO MODE SWITCH
        ===========================
        The guard is only applicable in exactly the situation it detects:

          * motor working  → the tip leaves piezo range within a rung or two →
            rail readings → no ruler → this abstains (cannot false-positive);
          * motor dead     → the tip stays in range for the whole ladder →
            every rung tracking → this fires.

        CUMULATIVE, NOT PER-RUNG
        ========================
        The first rung is a single step, and stick-slip travel shrinks sharply at
        low temperature, so one step can legitimately fall under the threshold. A
        per-rung veto would false-positive there. ``dz_m`` is measured against the
        SHARED baseline, so the last rung's value is already the total travel over
        the whole ladder — rungs 2 and 3 are 10 and 89 steps, 10–100× the reach.
        """
        if not rungs or not all(r.get("has_ruler") for r in rungs):
            return False, ""          # no ruler ⇒ no opinion (see docstring)
        dz = rungs[-1].get("dz_m")
        if dz is None:
            return False, ""
        thresh = float(ip.get_config("z_recede_min_nm", 1.0)) * 1e-9
        if abs(dz) > thresh:
            return False, ""
        return True, (
            f"粗动马达没有产生位移:下了 {commanded_steps} 步退针命令,"
            f"而压电全程都握着隧道结、量到的总位移只有 {abs(dz) * 1e9:.2f} nm"
            f"(阈值 {thresh * 1e9:.2f} nm)。"
            "**这不是「测不出来」,是「测出来是零」** —— 每一级的基线与退针后读数都"
            "稳定在真实隧道结上,尺子是好的,它说台子没动。\n"
            "常见原因:**粗逼近高压没开**(压电马达靠高压黏滑步进,高压没了"
            "`Motor_StartMove` 照样返回成功)、驱动幅度低于起动阈、"
            "该轴走到行程尽头空滑、机械卡死。\n"
            "已停止粗动并撤针,**本次不计入粗动里程表**(坐标代次不推进,"
            "粗动大地图不加站点)—— 记下一次没发生的位移,比不换区危险得多。")

    @staticmethod
    def _approach_prescription(why: str) -> str:
        """按拒绝的证据给出处置建议：Z 方向与电流阈值分别解释，不能从下游判决反证极性配置正确。"""
        from_current = ("阈值" in why) or ("满量程" in why)
        if not from_current:
            return ("—— 压电缩回就是针尖靠近了。请核对 instrument_profile 的"
                    "退针方向 / z_extend_sign(**方向很可能配反**),再重试换区。")
        return ("\n**下一步按上面那句话里的依据分**:\n"
                "• 若写着「未收敛」——反馈可能还在追,先把设置页的"
                "「退针 Z 稳定预算」(z_settle_timeout_s)调大再重试;\n"
                "• 若写着「由**绝对地板**决定」而你的工作点本来就跑在接近 1 nA ——"
                "那个地板是照成像条件(~100 pA)定的,对这条流程不适用,"
                "先把工作电流降到成像量级再换区;\n"
                "• 若写着「由**相对项**决定」且读数已收敛 ——这才是真的在逼近,"
                "去核对 instrument_profile 的退针方向 / z_extend_sign。")

    def _judge_recede(self, baseline: ZSettle, after: ZSettle) -> tuple[str, str]:
        """'receding' | 'approaching' | 'ambiguous' | 'unsettled' | 'no_sign'.

        Primary evidence is the Z-PIEZO DIRECTION, not the current: once the tip
        is far the current has decayed to zero and carries no sign at all, while
        the piezo extending to chase a now-farther sample is unambiguous. The
        current is used only as a danger trip in the other direction.

        **Both readings must be converged.** Comparing anything else measures
        the wait, not the gap — that is the 2026-08-04 defect, where reversing
        ``retract_motor_dir`` left the verdict unchanged because both numbers
        came off the same ramp. A reading that could not be taken now returns
        ``unsettled``, which is deliberately NOT ``approaching``: collapsing
        "the tip is closing in" and "the check could not run" into one answer is
        what made a broken judge look exactly like a backwards cable, and left
        the operator no way to tell which they had.
        """
        # 1. Danger trip by current. Independent of Z, so it still works when
        #    the Z reading is unusable — this is the one verdict that must
        #    survive a failed settle.
        current = after.current_a
        setpoint = (after.setpoint_a if after.setpoint_a is not None
                    else baseline.setpoint_a)
        if current is not None and abs(current) > _NOISE_FLOOR_A:
            # 报告阈值由哪一项决定、setpoint 是否可读及读数是否稳定；未知与零不能混为一谈。
            floor = 1e-9
            rel = None if setpoint is None else 3.0 * abs(float(setpoint))
            if setpoint is None:
                # 设定点不可读时改用独立的放大器饱和检查；满量程也不可读时不伪造电流结论。
                # Z 方向仍是主证据，缺失 setpoint 不应错误地被报告成 Z 未稳定。
                full_scale = None
                try:
                    fs = ip.get_config("preamp_full_scale_a", None)
                    full_scale = float(fs) if fs else None
                except Exception:  # noqa: BLE001 — 读不到就当没有这个界
                    full_scale = None
                if full_scale and abs(current) >= full_scale:
                    return "approaching", (
                        f"电流 {abs(current):.3g} A 已达前置放大器满量程 "
                        f"{full_scale:.3g} A —— 放大器饱和,与工作点无关。"
                        "(读不到 setpoint,所以用的是这个与工作点无关的界,"
                        "不是相对判据。)")
                why_no_sp = (after.setpoint_why or baseline.setpoint_why
                             or "两次读数都没有留下原因")
                skipped = (
                    f"(电流 {abs(current):.3g} A 这一条**没判**:读不到 setpoint"
                    f"[{why_no_sp}],相对判据 3×setpoint 无从算起;"
                    + (f"满量程界 {full_scale:.3g} A 也没超。"
                       if full_scale else
                       "而 instrument_profile 也没声明 preamp_full_scale_a,"
                       "没有与工作点无关的界可用。")
                    + "已改用 Z 压电方向这条**主证据**。)")
                self._current_trip_note = skipped
            else:
                bar = max(rel, floor)
                if abs(current) > bar:
                    which = ("相对项 3×setpoint" if rel >= floor else "绝对地板")
                    settled = ("读数已收敛" if after.usable
                               else f"⚠️ **这次读数未收敛**({after.why()})")
                    return "approaching", (
                        f"电流 {abs(current):.3g} A 高于阈值 {bar:.3g} A"
                        f"(由**{which}**决定:setpoint 读到 {float(setpoint):.3g} A,"
                        f"相对项 {rel:.3g} A,地板 {floor:.3g} A);{settled}")

        # 2. The Z comparison — only between two readings that actually stopped.
        note = getattr(self, "_current_trip_note", "")
        self._current_trip_note = ""

        def _say(verdict: str, why: str) -> tuple[str, str]:
            """Z 那一支的结论 + 「电流这条这次判没判」。

            跳过一条判据必须**出现在结论里**:否则「电流没超」与「电流这条没判」
            长得一模一样,而那正是本仓反复栽的形状。
            """
            return verdict, (why + note if note else why)

        if not after.usable:
            return _say("unsettled", after.why())
        if not baseline.usable:
            return _say("unsettled", "基线 Z 不可用:" + baseline.why())

        # **没声明就不判**(2026-08-11)。出厂 ``+1`` 在这里不是保守值,是两个互斥
        # 答案里的一个 —— 见 ``instrument_profile.z_extend_sign_or_none`` 的表:
        # 同一台机器在 08-05 与 08-08 的原始读数隐含的符号是**相反**的。
        # 猜错的两个方向不对称:猜成 approaching 只是白撤一次针(烦,安全);
        # 猜成 receding 是**针尖在靠近却说在远离**,然后梯子照爬到 89 步。
        sign = ip.z_extend_sign_or_none()
        if sign is None:
            return _say("no_sign", (
                "本机没有声明「压电伸长(趋向样品)对应 Z 读数符号」"
                "(设置 → 退针 → `z_extend_sign`),Z 读数变化无法翻译成"
                "「在远离还是在靠近」"))
        dz = (after.z_m - baseline.z_m) * sign
        thresh = float(ip.get_config("z_recede_min_nm", 1.0)) * 1e-9

        if after.at_rail and baseline.at_rail:
            # Nothing within piezo reach either side of the step. Consistent
            # with receding but it does not PROVE it — two rail readings are the
            # same number whichever way the stage went. Says so, and stays safe:
            # a tip that had closed in would have pulled the piezo off the rail,
            # so the approaching half of the check is still live.
            return _say("ambiguous", (
                "退针前后压电都到极限、量程内都没有表面 —— "
                "与远离一致,但两次都是极限值,证明不了距离(没有逼近迹象)"))
        if dz > thresh:
            if after.at_rail:
                # The piezo ran out of range looking for a surface that used to
                # be within reach: receding, and the magnitude is a lower bound.
                return _say("receding", (
                    f"压电走到极限仍未找到表面,较基线至少远离 {dz * 1e9:.1f} nm"
                    "(极限值,实际更远)"))
            return _say("receding", f"Z 压电伸长 {dz * 1e9:.1f} nm(在追远离的样品)")
        if dz < -thresh:
            if baseline.at_rail:
                # Was beyond reach, now the loop can hold a junction: the tip
                # came back INTO range, i.e. closer. The safety-critical verdict
                # survives a railed baseline, which is why a rail is reported
                # rather than treated as a failed reading.
                return _say("approaching", (
                    f"基线时量程内还没有表面,退针后反而找到了(Z 缩回 "
                    f"{abs(dz) * 1e9:.1f} nm)—— 针尖是靠近了"))
            return _say("approaching", f"Z 压电缩回 {abs(dz) * 1e9:.1f} nm")
        return _say("ambiguous", "Z 压电几乎没动")

    def _prove_clear(self, real_ctx) -> dict:
        """Confirm the tip is off the surface. Two witnesses, neither sufficient alone.

        The current going to zero is necessary but weak — a dead preamp reads
        zero too. The qPlus amplitude recovering to its free value is the direct
        answer to "is the tip mechanically free", but many rigs have no such
        sensor. So: the current must be quiet, and the amplitude must either
        agree or abstain. An amplitude that positively says "still damped" blocks
        the move even when the current is silent."""
        out: dict[str, Any] = {}
        rec = real_ctx.safe_call("Current_Get")
        self._call_log.append(rec)
        current = None if rec.error else _first_val(rec.return_value)
        out["current_a"] = current
        if current is None:
            out["clear"] = None
            out["reason"] = ("读不到电流,无法确认针尖已脱离。"
                             "读不到 ≠ 已脱离,但也不足以判定失败 —— "
                             "继续依赖退针自检的结论。")
        elif abs(current) > _NOISE_FLOOR_A * 10:
            out["clear"] = False
            out["reason"] = (f"退针后电流仍有 {abs(current):.3g} A(应到噪声底)"
                             " —— 针尖可能还在隧穿距离内,**不要横向移动**。")
            return out
        else:
            out["clear"] = True
            out["reason"] = f"电流已到噪声底({abs(current):.3g} A)"

        try:
            from mast.skills.builtins._tip_evidence import qplus_recovered
            verdict, why = qplus_recovered(real_ctx)
            out["qplus_recovered"] = verdict
            out["qplus_note"] = why
            if verdict is False:
                out["clear"] = False
                out["reason"] = why
            elif verdict is True and out.get("clear") is None:
                # The amplitude answered even though the current could not.
                out["clear"] = True
                out["reason"] = why
        except Exception as exc:  # noqa: BLE001
            # 「取证没做成」和「没人试过取证」必须是两句话。
            #
            # 原来这里是 `pass`:qplus 取证一旦抛异常,``out`` 里就**没有**
            # ``qplus_recovered`` 这个键 —— 而「本机没有 qPlus 所以没取证」出来的
            # 也是同一个「没有这个键」。下游读到的两种情形完全一致,于是一次失败的
            # 取证被当成「这台机器不适用」,悄悄降级成只看电流那一路。
            #
            # 三态,和 `clear` 同款:True/False = 取到证了;None + note = 试过但没成。
            out["qplus_recovered"] = None
            out["qplus_note"] = f"qPlus 取证未完成({type(exc).__name__}: {exc})"
            out["qplus_probe_failed"] = True
            logger.warning("relocate: qPlus 复原取证抛异常(按未取证处理): %r", exc)
        return out

    def _phase_move(self, real_ctx, params: dict) -> SkillResult:
        """One chunk of lateral travel, then look.

        Nothing should happen electrically while the stage slides: the tip is
        tens of microns away and the feedback is off. So any current at all is
        evidence the clearance was not what we proved it was, and the right
        response is to stop mid-move rather than finish and find out."""
        n = int(params.get("steps", 0))
        direction = _DIRECTIONS.get(
            (str(params.get("axis")), str(params.get("direction"))))
        if direction is None or n <= 0:
            return SkillResult(skill_name=_P_MOVE, success=False,
                               error=f"无效的移动参数:{params}")
        if params.get("dry_run"):
            self._moved_steps += n
            return SkillResult(skill_name=_P_MOVE, success=True,
                               data={"dry_run": True, "steps": n,
                                     "direction": direction,
                                     "cumulative": params.get("cumulative")})

        rec = real_ctx.safe_call("Motor_StartMove", _DIR_CODE[direction], n, 0, 1)
        self._call_log.append(rec)
        if rec.error:
            self._panic(real_ctx)
            return SkillResult(
                skill_name=_P_MOVE, success=False,
                error=f"横向粗动失败(第 {params.get('index')} 块,{n} 步):{rec.error}",
                data={"moved_steps": self._moved_steps})
        self._moved_steps += n

        watch: dict[str, Any] = {"cumulative": params.get("cumulative")}
        rc = real_ctx.safe_call("Current_Get")
        self._call_log.append(rc)
        current = None if rc.error else _first_val(rc.return_value)
        watch["current_a"] = current
        if current is not None and abs(current) > _MOVE_DANGER_CURRENT_A:
            self._panic(real_ctx)
            return SkillResult(
                skill_name=_P_MOVE, success=False,
                error=(f"横向移动中检测到电流 {abs(current):.3g} A —— "
                       "针尖离表面太近(清障不足或样品倾斜)。已立即停止粗动并撤针。"
                       f"已移动 {self._moved_steps} 步。"),
                data={"moved_steps": self._moved_steps, "watch": watch})

        try:
            from mast.skills.builtins._tip_evidence import qplus_fields, qplus_says_crashed
            q = qplus_fields(real_ctx)
            watch.update(q)
            if qplus_says_crashed(q):
                self._panic(real_ctx)
                return SkillResult(
                    skill_name=_P_MOVE, success=False,
                    error=("横向移动中 qPlus 振幅塌了 —— 针尖已经接触表面。"
                           "已立即停止粗动并撤针。"
                           f"已移动 {self._moved_steps} 步。"),
                    data={"moved_steps": self._moved_steps, "watch": watch})
        except Exception:  # noqa: BLE001
            pass

        # Pressure can change DURING a move (a pump trips, a valve opens), and
        # the remaining chunks are still hundreds of volts of drive.
        try:
            from mast.core import vacuum_interlock as vac
            v = vac.check()
            if not v.allow:
                self._panic(real_ctx)
                return SkillResult(
                    skill_name=_P_MOVE, success=False,
                    error=f"移动过程中真空互锁转为拒绝:{v.reason} 已停止粗动。",
                    data={"moved_steps": self._moved_steps, "watch": watch})
        except Exception:  # noqa: BLE001
            pass

        self._watches.append(watch)
        return SkillResult(skill_name=_P_MOVE, success=True,
                           data={"steps": n, "direction": direction, "watch": watch})

    def _phase_verify(self, real_ctx) -> SkillResult:
        """Reconcile against the step counter — where one exists.

        Only Attocube ANC150 controllers expose ``Motor_StepCounterGet``. On
        everything else this reports ``unavailable`` and says so, rather than
        printing a reassuring tick for a check that never ran."""
        rec = real_ctx.safe_call("Motor_StepCounterGet", 0, 0, 0)
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(skill_name=_P_VERIFY, success=True,
                               data={"counter": "unavailable",
                                     "note": ("本控制器不支持步进计数器读回"
                                              "(仅 Attocube ANC150 支持)—— "
                                              "移动步数无法对账,这是如实记录,不是通过。")})
        parsed = rec.return_value
        data: dict[str, Any] = {"counter": "read", "commanded_steps": self._moved_steps}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                data.update({"step_counter_x": int(vals[0]),
                             "step_counter_y": int(vals[1]),
                             "step_counter_z": int(vals[2])})
        return SkillResult(skill_name=_P_VERIFY, success=True, data=data)

    def _phase_reapproach(self, real_ctx, params: dict) -> SkillResult:
        """Bring the tip back down — through AutoApproach, never by stepping Z.

        AutoApproach is the controller's own current-feedback approach: it stops
        on contact. An open-loop coarse Z step toward the sample has no such stop
        and is the one action in MAST gated to a human."""
        if params.get("dry_run"):
            return SkillResult(skill_name=_P_REAPPROACH, success=True,
                               data={"dry_run": True, "skipped": "AutoApproach"})
        # 谁改谁恢复：横移前是这条流程把偏压降下去的，进针**之前**放回去。
        # 放在进针之前而不是之后 —— 进针要在调用方原本的工作点上完成，
        # 否则「进好的针」是在一个没人要求过的偏压下建立的。
        self._restore_bias(real_ctx)
        res = real_ctx.run("ApproachTip", {})
        ok = bool(getattr(res, "success", False))
        return SkillResult(
            skill_name=_P_REAPPROACH, success=ok,
            # WHAT THIS SENTENCE MAY CLAIM (2026-08-05). It used to assert
            # 「坐标代次已经推进」—— a hardcoded statement of what the author
            # believed the recorder did, written at a time when a failed
            # composite's data was discarded and the epoch therefore did NOT
            # advance. On the rig the stage had moved 301 steps, the ledger had
            # not, and the message said the opposite. So: state the PHYSICAL
            # fact (which this phase watched happen) and point at the thing that
            # can be checked, instead of narrating a bookkeeping step that runs
            # elsewhere, after this returns, and can fail on its own.
            error=None if ok else (
                f"换区后重新进针失败:{getattr(res, 'error', 'unknown')}。"
                f"**横向移动本身已经完成**(实际走了 {self._moved_steps} 步),"
                "针尖现在是退开的 —— 重新进针即可,**不要再移动一次**。"
                "旧的压电坐标已经不指向原来那片表面了;"
                "账面位置用 get_coarse_map 核对,不要凭这句话推断。"),
            data={"approach": getattr(res, "data", None) or {}})

    def _lower_bias_for_move(self, real_ctx) -> None:
        """横移前降低偏压并记住原值，收尾只恢复本流程实际改动的状态。"""
        try:
            rec = real_ctx.safe_call("Bias_Get")
            self._call_log.append(rec)
            now = None if getattr(rec, "error", "") else _first_val(rec.return_value)
            if now is not None and abs(float(now)) > _MOVE_BIAS_V:
                self._bias_restore = float(now)
                self._call_log.append(
                    real_ctx.safe_call("Bias_Set", float(_MOVE_BIAS_V)))
        except Exception:  # noqa: BLE001 — 降压失败不该让整条流程炸
            logger.warning("横移前降偏压失败，继续（拦截可能因此误触发）", exc_info=True)

    def _restore_bias(self, real_ctx) -> None:
        """把横移前降下去的偏压放回原值。**幂等**：放回一次就清掉记号。

        三条路径都要调它：正常收尾（reapproach 之前）、panic、以及 reapproach=False
        时的结束。漏掉任何一条，调用方就会在一个自己没要求过的偏压上继续工作 ——
        而那正是 08-26 追了半夜的「工作点被上一个 skill 改掉且不改回来」。
        """
        want = getattr(self, "_bias_restore", None)
        if want is None:
            return
        self._bias_restore = None
        try:
            self._call_log.append(real_ctx.safe_call("Bias_Set", float(want)))
        except Exception:  # noqa: BLE001
            logger.warning("恢复偏压到 %.3f V 失败 —— 请手动确认", want, exc_info=True)

    def _panic(self, real_ctx) -> None:
        """Stop the motor and withdraw. Best-effort, never raises.

        **但「没能停下来」必须留下痕迹。** 原来两个动作都包在 ``except: pass`` 里:
        急停发不出去、退针发不出去,调用方拿到的东西与两条都成功时**一模一样**
        —— 而这是本技能里唯一一条「针尖可能正贴着表面而马达还在走」的路径。
        调用方接下来只会报「移动失败」,没有一个字提到针尖没退开。

        失败改成往 ``_call_log`` 里塞一条**带 error 的合成记录**:``nanonis_calls``
        本来就随每个 SkillResult 一起回去,所以九个调用点一行都不用改,而
        「急停没发出去」在调用日志里是看得见的。另外记进 ``_panic_failures``,
        由 ``run_composite`` 的结果把它顶到 error 文本里(见 ``_panic_note``)。
        """
        # 动词写成**字面量**并装进 thunk,不要 `safe_call(verb, *args)` ——
        # 那样一来本仓库的每一个安全工具都看不见它们(中止策略检查 / 安全审计 /
        # API 覆盖率普查全靠 grep `safe_call("…")`)。
        #
        # 这是同一个错误的**第二次**:`core/runtime.py` 的 EMERGENCY STOP 原来也是
        # 一张 tuple 表 splat 进去,于是「急停自己发出的那三个动词」对中止策略检查器
        # 隐形 —— 它们碰巧在允许名单里,而没有任何东西验证过这一点。那次改成了
        # 持有字面量的 thunk;这次(2026-08-10,由 `test_safe_call_verbs_are_literal`
        # 逮到)在这里又写回了旧形状。**急停路径是最不该对审计工具隐形的地方。**
        for verb, args, fire in (
            ("Motor_StopMove", (),
             lambda: real_ctx.safe_call("Motor_StopMove")),
            ("ZCtrl_Withdraw", (1, -1),
             lambda: real_ctx.safe_call("ZCtrl_Withdraw", 1, -1)),
        ):
            try:
                self._call_log.append(fire())
            except Exception as exc:  # noqa: BLE001
                self._panic_failures.append(f"{verb}: {exc}")
                self._call_log.append(NanonisCallRecord(
                    method=verb, args=args,
                    error=f"panic {verb} 未能下发: {exc}"))
                logger.error("relocate 急停动作 %s 未能下发: %r", verb, exc)
        # 急停也要把偏压放回去 —— 否则一次失败的粗动会让调用方停在 0.5 V 上，
        # 而它自己以为还在原来的成像偏压。
        self._restore_bias(real_ctx)

    def _panic_note(self) -> str:
        """急停没做成时,要顶到用户读的那句话里去的一行。空串 = 都发出去了。"""
        if not self._panic_failures:
            return ""
        return ("\n⚠️ **紧急停止未能完整下发**(" + "；".join(self._panic_failures)
                + ")。请立即到 Nanonis 界面确认马达已停、针尖已退开 —— "
                "不要假设本技能已经把针尖收回去了。")

    # ── driver ──────────────────────────────────────────────────────────

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []
        #: 急停动作里**没能下发**的那些(见 :meth:`_panic`)。空 = 都发出去了。
        self._panic_failures: list[str] = []
        self._watches: list[dict] = []
        self._moved_steps = 0
        self._rungs: list[dict] = []
        self._baseline_settle: dict[str, Any] | None = None
        self._checks: dict[str, Any] = {}
        self._clearance: dict[str, Any] = {}
        self._temperature_k: float | None = None
        #: 横移前被本流程降下去的偏压原值；None = 没降过 / 已放回。
        self._bias_restore: float | None = None

        direction = _DIRECTIONS.get(
            (str(params.get("axis")), str(params.get("direction"))))
        if direction is None:
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=(f"无效的轴/方向:axis={params.get('axis')!r} "
                       f"direction={params.get('direction')!r};"
                       "轴取 'x'/'y',方向取 '+'/'-'。"))

        wrapped = _RelocatePhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor
        all_good = executor.run_plan(iter(self.plan(params)))

        data: dict[str, Any] = {
            # direction + steps are the CONTRACT with the recorder: they are what
            # runtime.lateral_coarse_move_info reads to write the coarse_move
            # marker that advances coord_epoch and feeds the coarse map's
            # odometer. Reporting the steps ACTUALLY moved, not the steps asked
            # for — a move that stopped halfway still moved the stage, and an
            # odometer fed the request instead of the outcome is fiction.
            "direction": direction,
            "steps": int(self._moved_steps),
            # The odometer's admission ticket on the FAILURE path. Separate key,
            # separate promise: "the stage physically took this many lateral
            # steps, whatever the verdict of this composite turns out to be".
            # ``steps`` above cannot carry that meaning on its own — a reader
            # would have to know which failures happened before the move and
            # which after, and that knowledge does not survive being a number in
            # a dict. Zero in dry_run: a rehearsal commands nothing, and a
            # rehearsal that ages every coordinate on the sample is a phantom
            # relocation like any other. See runtime.lateral_coarse_move_info.
            "lateral_steps_taken": (0 if params.get("dry_run")
                                    else int(self._moved_steps)),
            "requested_steps": int(params.get("steps", 0)),
            "reapproached": bool(params.get("reapproach", True)) and all_good,
            "dry_run": bool(params.get("dry_run", False)),
            "checks": self._checks,
            "clearance": self._clearance,
            # The direction self-check's own evidence, on success as well as on
            # failure. "Which way did the tip go, and how did we know" is the
            # only thing that makes the clearance falsifiable after the fact.
            "clearance_rungs": list(self._rungs),
            "clearance_baseline": self._baseline_settle,
            "watches": self._watches,
            "_progress": executor.progress.to_dict(),
        }
        if self._temperature_k is not None:
            data["temperature_k"] = self._temperature_k

        if not executor.progress.aborted:
            executor.clear_sidecar()

        # 急停有没有真的发出去,是**两条路都要说**的事。
        # 只在失败分支说不够:一次「成功」的移动里如果中途 panic 过而且没发下去,
        # 针尖状态同样不确定,而 success=True 会让人完全不去看。
        data["panic_failures"] = list(self._panic_failures)
        panic_note = self._panic_note()

        if not all_good:
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=((executor.progress.aborted_reason
                        or "RelocateCoarseXY aborted") + panic_note),
                data=data, nanonis_calls=list(self._call_log))
        if panic_note:
            # 走到这里说明:动作整体判成功,但中途某次急停没能下发。
            # 那不是「成功」—— 针尖是不是退开的,本技能答不上来。
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=("横向移动的各项判据都过了,但" + panic_note.lstrip("\n")),
                data=data, nanonis_calls=list(self._call_log))
        return SkillResult(skill_name=self._skill_name(), success=True,
                           data=data, nanonis_calls=list(self._call_log))


__all__ = ["RelocateCoarseXY"]
