"""One-shot tunnelling-current noise characterisation → a stored baseline.

## What it does

Walks a matrix of working points, waits for the current monitor to record real
segments at each one, then turns the whole sweep into a
:mod:`~mast.monitoring.baseline` the live monitor judges against.

Two sweeps, and both are needed::

    setpoint sweep, bias fixed   →  exponent a_I
    bias sweep, setpoint fixed   →  exponent a_V

A mechanical vibration and pickup on the bias line produce **the same** a_I of
+1 — the setpoint sweep alone cannot separate them. Holding the current with
feedback and moving the bias splits them: a gap modulation's dz is set by the
mechanics (a_V = 0), while voltage noise arrives as dI = dV/R and R is
proportional to V at fixed current (a_V = -1). Running only the first sweep
gives a baseline that judges, but cannot say what to go fix.

## What it does NOT do

It does not acquire anything itself. The current monitor is already sampling at
2 kHz and writing segments; this skill sets a working point, waits, and reads
what the monitor recorded. That is deliberate — a second acquisition path would
be a second set of numbers to reconcile, and the whole value of the baseline is
that it is measured with the same sampling the live judgement uses (segments
are 1 s at 2 kHz, per-segment linear detrend, and
``baseline.width_stats`` is bit-for-bit ``features.detrended_rms``).

## Safety

``CONFIRM``: it moves the setpoint and the bias, and the tip follows. After
every change it verifies the junction actually followed — the register agreeing
is not the same as the current agreeing — and it restores the as-found working
point in a ``finally``, so an abort mid-sweep does not leave the instrument
parked somewhere the operator did not choose.

Bias is bounded to +-2 V by the parameter spec rather than SetBias's +-10 V.
These are workflow defaults, not instrument calibration. Validate working points
for the current junction before a sweep.
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Optional

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

logger = logging.getLogger(__name__)

_PHASE_COLLECT = "_phase_collect_"
_PHASE_FINISH = "_phase_finish"

#: Default example setpoint sweep, in amps; configure for the current junction.
#: The low end helps distinguish additive preamplifier noise from multiplicative noise.
_DEFAULT_SETPOINTS = (20e-12, 50e-12, 100e-12, 200e-12, 500e-12, 1000e-12)

# Default bias sweep in volts, covering both polarities.
# Symmetric conditions let the analysis check polarity dependence instead of assuming it.
# Rectification or density-of-states asymmetry may invalidate a sign-independent model.
_DEFAULT_BIASES = (0.04, 0.20, 0.50, -0.04, -0.20, -0.50)

#: Additional workflow Z limit; verify it against the current instrument's range.
_Z_LIMIT_M = 1.60e-7

#: The junction must be following, not just the register. 25 % is loose on
#: purpose: a fixed amplifier offset alone can already be a sizeable fraction
#: of a low (tens of pA) setpoint.
_FOLLOW_TOL = 0.25

# Uncommissioned example integration bands: (name, lo_hz, hi_hz).
# These bands do not describe a particular instrument or site and require configuration
# before using line attribution as an instrument diagnosis.
_LINES: tuple[tuple[str, float, float], ...] = (
    ("vib_6hz", 4.0, 8.0), ("l_29hz", 28.0, 31.0), ("l_50hz", 48.5, 51.5),
    ("l_87hz", 85.0, 90.0), ("l_100hz", 98.5, 101.5),
    ("l_200hz", 198.0, 202.0), ("l_450hz", 445.0, 455.0),
    ("l_example_800hz", 795.0, 805.0),
)


class _PhaseCtx:
    """Wraps the real ExecutionContext so ``_phase_*`` names dispatch locally."""

    def __init__(self, real_ctx, skill: "CharacteriseCurrentNoise") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


def _num(rec, field: str) -> Optional[float]:
    """A number out of a SkillResult's data, or None. Never a substituted 0."""
    d = getattr(rec, "data", None)
    if not isinstance(d, dict):
        return None
    v = d.get(field)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class CharacteriseCurrentNoise(CompositeSkillGraph):
    """Sweep working points, measure the noise at each, store a baseline."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CharacteriseCurrentNoise",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "一键测出这台机器的电流噪声基线：扫一遍设定电流、再扫一遍偏压，"
                "在每个工况上读电流监控已经录下的 2 kHz 段，算出涨落幅度、宽度分布、"
                "噪声谱与每条谱线的机制归因，存成基线供实时判据当分母用。"
                "同时采 Z 位移通道并算出 Z-电流相干性与衰减常数 kappa。"
                "会改设定点与偏压（含负偏压；结束后恢复原工作点）。"
                "典型耗时 = 点数 ×（settle + 采集）≈ 20-30 分钟。"
            ),
            parameters=[
                ParameterSpec(
                    name="setpoints_a",
                    type="str",
                    description=(
                        "设定电流列表，JSON 数组，单位安培。"
                        "留空用默认的 [20p, 50p, 100p, 200p, 500p, 1n]。"
                        "**要跨至少一个数量级** —— 加性项(前放)与乘性项(振动)"
                        "只有在电流跨度够大时才分得开，点太密而范围太窄会拟合出"
                        "一条没有分辨力的曲线。"
                    ),
                    required=False, default="",
                ),
                ParameterSpec(
                    name="biases_v",
                    type="str",
                    description=(
                        "偏压列表，JSON 数组，单位伏特。留空用默认 [0.04, 0.2, 0.5]。"
                        "传 '[]' 则跳过偏压组 —— 那样仍然得到可用的基线曲线，"
                        "但**没有机制归因**（分不开机械振动与偏压线拾取）。"
                    ),
                    required=False, default="",
                ),
                ParameterSpec(
                    name="seconds_per_point", type="float",
                    description="每个工况采多久（秒）。75 s ≈ 70 段，PSD 平均已经很稳。",
                    unit="s", required=False, default=75.0,
                    min_value=20.0, max_value=600.0,
                ),
                ParameterSpec(
                    name="settle_s", type="float",
                    description="改完工况等多久再开始采（秒）。要盖过 Z 反馈的整定。",
                    unit="s", required=False, default=25.0,
                    min_value=5.0, max_value=300.0,
                ),
                ParameterSpec(
                    name="z_burst_s", type="float",
                    description=(
                        "每个工况额外采多久的 Z + 电流同步 burst（秒）。0 = 不采 Z。"
                        "默认 30 s。它把示波器的 B 路改成 Z，而电流监控"
                        "**只认 A 路**（pump.channel_is_ours 位置 1 刻意不判，"
                        "因为 zburst 每 30 min 就会合法地改它），所以电流采集不受影响；"
                        "代价只是 burst 期间与泵抢 data 角色锁，可能丢几帧。"
                    ),
                    unit="s", required=False, default=30.0,
                    min_value=0.0, max_value=300.0,
                ),
                ParameterSpec(
                    name="label", type="str",
                    description="给这份基线起个名字，例如「干净针尖 Au(111) 无磁场位」。",
                    required=False, default="",
                ),
                ParameterSpec(
                    name="note", type="str",
                    description="条件备注：针尖、样品、扫描位、磁场、两个腔的泵开关。",
                    required=False, default="",
                ),
                ParameterSpec(
                    name="allow_while_scanning", type="bool",
                    description=("允许在扫描进行中采集。**默认 False** —— 扫描段"
                                 "量到的是样品形貌，不是仪器噪声。"),
                    required=False, default=False,
                ),
                ParameterSpec(
                    name="activate", type="bool",
                    description=(
                        "跑完是否立刻启用这份基线做实时判据。默认 true。"
                        "设 false 则只记录，需要人工在设置里选用。"
                    ),
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="max_bias_v", type="float",
                    description=(
                        "偏压绝对值上限，默认 2 V；默认偏压组只到 0.5 V。这些是工作流设置，使用前应确认适合当前结状态。"
                    ),
                    unit="V", required=False, default=2.0,
                    min_value=0.01, max_value=10.0,
                ),
            ],
            estimated_duration_s=1800.0,
            composition_level=1,
            tags=["noise", "baseline", "characterisation", "current", "write"],
        )

    # ── plan ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_list(raw: str, default: tuple) -> list[float]:
        s = (raw or "").strip()
        if not s:
            return list(default)
        try:
            items = json.loads(s)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("CharacteriseCurrentNoise: 解析不了 %r，用默认值", raw)
            return list(default)
        if not isinstance(items, (list, tuple)):
            return list(default)
        out = []
        for x in items:
            try:
                v = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                out.append(v)
        return out

    def plan(self, params: dict) -> list[CompositeStep]:
        sps = self._parse_list(params.get("setpoints_a", ""), _DEFAULT_SETPOINTS)
        bss = self._parse_list(params.get("biases_v", ""), _DEFAULT_BIASES)
        max_b = abs(float(params.get("max_bias_v", 2.0) or 2.0))
        bss = [b for b in bss if abs(b) <= max_b]
        # 按 |V| 排序再交错正负，让扫描从小偏压开始 ——
        # 结不稳定最先出现在大偏压上，晚一点撞到它意味着前面的点已经拿到手了。
        bss.sort(key=lambda v: (abs(v), -v))
        steps: list[CompositeStep] = []
        n = 0

        # Sweep A — setpoint, bias left wherever the operator had it.
        for sp in sps:
            steps.append(CompositeStep(
                step_id=f"set_sp_{n}", skill_name="SetSetpoint",
                params={"setpoint_a": float(sp)},
                optional=False, checkpoint_after=False,
                tags=("sweep=setpoint", f"setpoint_a={sp:g}")))
            steps.append(CompositeStep(
                step_id=f"collect_{n}", skill_name=f"{_PHASE_COLLECT}{n}",
                params={"ordinal": n, "sweep": "setpoint", "setpoint_a": float(sp)},
                optional=False, checkpoint_after=True,
                tags=("collect", f"point={n}")))
            n += 1

        ref_sp = float(sps[len(sps) // 2]) if sps else 200e-12

        # 重复点：回到参照设定点再采一次。它**不参与拟合**（sweep 标 repeat，
        # build_models 只取 setpoint），存在的唯一理由是给出「同一工作点隔了
        # 二十分钟重测，能重现到什么程度」——而那正是判断极性差异、判断比值阈值
        # 是否合理的唯一尺度。没有它的话，任何「差了 3%」都无从判断是不是差异。
        if sps:
            steps.append(CompositeStep(
                step_id=f"set_sp_rep_{n}", skill_name="SetSetpoint",
                params={"setpoint_a": ref_sp},
                optional=False, checkpoint_after=False, tags=("repeat",)))
            steps.append(CompositeStep(
                step_id=f"collect_{n}", skill_name=f"{_PHASE_COLLECT}{n}",
                params={"ordinal": n, "sweep": "repeat", "setpoint_a": ref_sp},
                optional=False, checkpoint_after=True,
                tags=("collect", "repeat", f"point={n}")))
            n += 1

        # Sweep B — bias, current pinned at the reference setpoint. The
        # setpoint is re-sent for every bias point rather than once: sweep A
        # left it at its own last value, and a bias sweep is only interpretable
        # with the current actually held.
        for b in bss:
            steps.append(CompositeStep(
                step_id=f"set_bias_{n}", skill_name="SetBias",
                params={"bias_v": float(b), "slew_rate_v_per_s": 0.5},
                optional=False, checkpoint_after=False,
                tags=("sweep=bias", f"bias_v={b:g}")))
            steps.append(CompositeStep(
                step_id=f"set_sp_b_{n}", skill_name="SetSetpoint",
                params={"setpoint_a": ref_sp},
                optional=False, checkpoint_after=False,
                tags=("sweep=bias",)))
            steps.append(CompositeStep(
                step_id=f"collect_{n}", skill_name=f"{_PHASE_COLLECT}{n}",
                params={"ordinal": n, "sweep": "bias", "setpoint_a": ref_sp,
                        "bias_v": float(b)},
                optional=False, checkpoint_after=True,
                tags=("collect", f"point={n}")))
            n += 1

        steps.append(CompositeStep(
            step_id="finish", skill_name=_PHASE_FINISH,
            params={"n_points": n}, optional=False, checkpoint_after=True,
            tags=("fit", "store")))
        return steps

    # ── phases ──────────────────────────────────────────────────────────

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name.startswith(_PHASE_COLLECT):
            return self._phase_collect(params, real_ctx)
        if skill_name == _PHASE_FINISH:
            return self._phase_finish(params, real_ctx)
        return SkillResult(skill_name=skill_name, success=False,
                           error=f"Unknown phase: {skill_name}")

    def _read_state(self, ctx) -> dict:
        out: dict = {}
        for skill, key in (("GetBias", "bias_v"), ("GetCurrent", "current_a"),
                           ("GetSetpoint", "setpoint_a"), ("GetZPosition", "z_pos_m")):
            try:
                out[key] = _num(ctx.run(skill, {}), key)
            except Exception:  # noqa: BLE001 — a read that fails is a None, not a crash
                logger.debug("%s failed during characterisation", skill, exc_info=True)
                out[key] = None
        return out

    def _phase_collect(self, params: dict, ctx) -> SkillResult:
        """Wait out the settle, verify the junction followed, read the segments."""
        import numpy as np

        from mast.monitoring import baseline as B
        from mast.monitoring import features as F
        from mast.monitoring.store import get_store

        name = f"{_PHASE_COLLECT}{params.get('ordinal')}"
        p = self._params or {}
        settle = float(p.get("settle_s", 25.0) or 25.0)
        secs = float(p.get("seconds_per_point", 75.0) or 75.0)
        target_sp = params.get("setpoint_a")

        if not self._sleep(settle, ctx):
            return SkillResult(skill_name=name, success=False, error="已中止")

        st = self._read_state(ctx)
        z = st.get("z_pos_m")
        i_now = st.get("current_a")
        if z is None or abs(z) > _Z_LIMIT_M:
            return SkillResult(skill_name=name, success=False,
                               error=f"Z={z} 超出安全行程 ±{_Z_LIMIT_M} m，中止表征")
        if i_now is None:
            return SkillResult(skill_name=name, success=False,
                               error="电流读不到 —— 拒绝把一个读不到的工况记进基线")
        # 比较电流幅度与正值 setpoint，避免把负偏压下的电流符号误当成跟踪误差。
        if target_sp and abs(abs(i_now) - abs(target_sp)) / abs(target_sp) > _FOLLOW_TOL:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"电流 |{i_now:.4g}| A 与设定点 |{target_sp:.4g}| A 相差超过 "
                       f"{_FOLLOW_TOL:.0%} —— 结没有跟上，这个点测出来的不是基线"))

        t0 = time.time()
        if not self._sleep(secs, ctx):
            return SkillResult(skill_name=name, success=False, error="已中止")
        t1 = time.time()
        st_after = self._read_state(ctx)

        store = get_store()
        if store is None:
            return SkillResult(skill_name=name, success=False,
                               error="拿不到电流监控存储 —— 监控没在跑？")
        try:
            segs = (store.segments_query(since=t0, until=t1, limit=2000)
                    or {}).get("segments") or []
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f"段落查询失败: {exc}")
        runs: list = []
        seg_ids: list[int] = []
        fs = 0.0
        for s in segs:
            sid = int(s.get("seg_id") or s.get("id") or 0)
            if not sid:
                continue
            d = store.read_segment_decimated(sid, max_points=200000)
            if not d or d.get("decimated"):
                # Decimated data aliases everything above the new Nyquist onto
                # the bands the baseline is about to describe. Skip, don't use.
                continue
            y = np.asarray(d.get("i_a") or [], dtype=np.float64)
            if y.size < 256:
                continue
            runs.append(y)
            seg_ids.append(sid)
            fs = float(d.get("fs_hz") or fs)
        if len(runs) < 5:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"这个工况只取到 {len(runs)} 段全速率数据（需要 ≥5）。"
                       "监控可能没在采，或者原始波形已被清理。"))

        det = np.concatenate([B._detrend_linear(r)[0] for r in runs])
        w = B.width_stats(np.concatenate(runs))
        w_det = B.width_stats(det)
        freqs, psd = F.psd_of_runs(runs, fs)
        white = B.robust_white_floor(freqs, psd)
        i_mean = float(np.mean([r.mean() for r in runs]))

        # sigma is RMS-combined across segments, matching how the live column is
        # produced (one value per segment) rather than one std of everything.
        sig = float(np.sqrt(np.mean([B._detrend_linear(r)[0].std() ** 2 for r in runs])))
        _fw, hist = B.fwhm(det)

        lines = {}
        for nm, lo, hi in _LINES:
            m = B.peak_metrics(freqs, psd, lo, hi, i_mean)
            if m:
                lines[nm] = m

        # Z burst 安排在电流采集之后。它切换 B 通道，电流泵检查的是 A 通道归属。
        # 两者仍共享 data 角色锁；先完成当前工况的电流窗口，避免锁竞争造成缺帧。
        z_summary, z_freqs, z_psd = {"available": False}, None, None
        coupling: dict = {"available": False}
        coh_curve = None
        z_secs = float(p.get("z_burst_s", 30.0) or 0.0)
        if z_secs > 0:
            burst = self._dual_burst(ctx, z_secs)
            if burst:
                z_summary, z_freqs, z_psd = B.z_stats(burst["z"], burst["fs_hz"])
                coupling = B.z_current_coupling(burst["z"], burst["i"],
                                                burst["fs_hz"], i_mean)
                if coupling.get("available"):
                    coh_curve = (np.asarray(coupling["freqs_hz"]),
                                 np.asarray(coupling["coherence"]))
            else:
                z_summary = {"available": False,
                             "detail": "双通道 burst 没采到 —— 通道被占或被改动"}

        metrics = {
            "bias_v": st.get("bias_v"), "setpoint_a": target_sp,
            "i_measured_a": i_mean, "z_m": st.get("z_pos_m"),
            "seg_id_lo": min(seg_ids), "seg_id_hi": max(seg_ids),
            "n_segments": len(runs),
            "sigma_a": sig,
            "sigma_iqr_a": w_det.get("sigma_iqr_a"),
            "sigma_mad_a": w_det.get("sigma_mad_a"),
            "fwhm_a": w_det.get("fwhm_a"),
            "fwhm_over_sigma": w_det.get("fwhm_over_sigma"),
            "iqr_a": w_det.get("iqr_a"), "p99_p1_a": w_det.get("p99_p1_a"),
            "ptp_a": w_det.get("ptp_a"),
            "kurtosis": w_det.get("kurtosis"), "skewness": w_det.get("skewness"),
            "mean_a": w.get("mean_a"),
            "white_a2hz": white,
            # Z 与联合量：读不到就留 None。0 m 的 Z 噪声不是一个可能的读数，
            # 所以「没采到」绝不能塌成 0。
            "z_sigma_m": z_summary.get("sigma_m"),
            "z_step_rms_m": z_summary.get("step_rms_m"),
            "z_white_m2hz": z_summary.get("white_floor_m2hz"),
            "z_ptp_m": z_summary.get("ptp_m"),
            "z_fwhm_m": z_summary.get("fwhm_m"),
            "z_mean_m": z_summary.get("z_mean_m"),
            "z_n_runs": z_summary.get("n_runs"),
            "coh_fraction": coupling.get("coherent_fraction"),
            "coh_n_pairs": coupling.get("n_pairs"),
            "kappa_per_m": coupling.get("kappa_per_m"),
            "apparent_barrier_ev": coupling.get("apparent_barrier_ev"),
        }
        bid = self._executor.progress.partial_data.get("baseline_id")
        if not bid:
            # 缺少 baseline_id 时本点无法保存，必须报告失败。
            # 否则后续拟合无法区分未测到的工况与已测到但丢失的工况。
            return SkillResult(
                skill_name=name, success=False,
                error="拿不到 baseline_id —— 这个工况点测出来了却无处可存")
        pid = None
        if bid:
            pid = store.add_baseline_point(
                int(bid), tag=str(params.get("sweep") or ""),
                ordinal=int(params.get("ordinal") or 0), metrics=metrics,
                psd=(freqs, psd), hist=hist,
                z_psd=((z_freqs, z_psd) if z_freqs is not None else None),
                coh=coh_curve,
                extra={"lines": lines, "fs_hz": fs,
                       "state_before": st, "state_after": st_after,
                       "t0": t0, "t1": t1,
                       "z": {k: v for k, v in z_summary.items()
                             if k not in ("freqs", "psd")},
                       "coupling": {k: v for k, v in coupling.items()
                                    if k not in ("freqs_hz", "coherence",
                                                 "transfer_a_per_m")}})
        if pid is None:
            # 存不下就让这一步失败。这个点的数据只活在内存里，扫描继续下去的话
            # 它会在最后的拟合里悄悄缺席 —— 而「少了两个点」和「本来就没测那两个点」
            # 在结果里长得一模一样。存储层按设计吞异常，所以只有这里能把它变成一句话。
            return SkillResult(
                skill_name=name, success=False,
                error=(f"工况点测到了（{len(runs)} 段，I={i_mean:.4g} A）但写不进基线 "
                       f"#{bid} —— 看服务日志里 add_baseline_point 那条 debug"))
        pts = list(self._executor.progress.partial_data.get("points", []))
        pts.append({**metrics, "sweep": params.get("sweep"), "lines": lines})
        self._executor.set_partial("points", pts)
        return SkillResult(skill_name=name, success=True,
                           data={"ordinal": params.get("ordinal"),
                                 "i_measured_a": i_mean, "sigma_a": sig,
                                 "n_segments": len(runs)})

    def _dual_burst(self, ctx, seconds: float):
        """一次 Z + 电流同步 burst。复用 envhistory 的通道纪律，不另写一套。

        那套纪律（采前采后都校验通道、对不上整轮丢弃、只在 Z 不在画面上时才改、
        结束改回去）对这条路径同样必要，理由一模一样：Osci2T 是用户屏幕上的
        共享模块。再写一份的话，两份里迟早只有一份是对的。
        """
        try:
            from mast.envhistory.zburst import run_dual_burst
            pool = getattr(ctx, "pool", None)
            if pool is None:
                return None
            return run_dual_burst(lambda: pool,
                                  stop=getattr(ctx, "abort_event", None),
                                  burst_s=float(seconds))
        except Exception:  # noqa: BLE001 — Z 是补充信息，拿不到不该毁掉这个工况
            logger.debug("dual burst failed inside characterisation", exc_info=True)
            return None

    def _sleep(self, seconds: float, ctx) -> bool:
        """Sleep in slices so an abort is honoured. False = aborted.

        A composite that runs for half an hour must be interruptible at a
        finer grain than its own steps; the abort event is checked once a
        second rather than once per working point.
        """
        ev = getattr(ctx, "abort_event", None)
        end = time.time() + max(0.0, float(seconds))
        while time.time() < end:
            if ev is not None and ev.is_set():
                return False
            time.sleep(min(1.0, end - time.time()))
        return not (ev is not None and ev.is_set())

    def _phase_finish(self, params: dict, ctx) -> SkillResult:
        """Fit across the points, attribute the lines, close the baseline."""
        from mast.monitoring import baseline as B
        from mast.monitoring.store import get_store

        pts = list(self._executor.progress.partial_data.get("points", []))
        bid = self._executor.progress.partial_data.get("baseline_id")
        store = get_store()
        if not pts:
            return SkillResult(skill_name=_PHASE_FINISH, success=False,
                               error="一个工况点都没测到")

        # 跨点拟合与归因走 baseline.build_models —— 与导入外部测量的那条路
        # 用的是同一份实现。两份的话就是同一批数据出两组系数，而差异要等到有人
        # 把存下来的基线和新测的比较时才会现形。
        models = B.build_models(pts)
        sigma_model = models["sigma_model"]
        white_model = models["white_model"]
        lines_out = models["lines"]
        rep = models["repeatability"]
        polarity = models.get("polarity")
        bias_mag = models.get("bias_magnitude")

        status = "complete" if sigma_model is not None else "aborted"
        if store is not None and bid:
            store.finish_baseline(
                int(bid), status=status,
                sigma_model=sigma_model.to_dict() if sigma_model else None,
                white_model=white_model.to_dict() if white_model else None,
                lines=lines_out, repeatability=rep, polarity=polarity,
                bias_magnitude=bias_mag)
            if status == "complete" and bool((self._params or {}).get("activate", True)):
                store.activate_baseline(int(bid))
        if sigma_model is None:
            return SkillResult(
                skill_name=_PHASE_FINISH, success=False,
                error=("设定点扫描的有效点不足 3 个，拟合不出 sigma 曲线。"
                       "基线已记录为 aborted —— 它测到的点仍然保留，但不会被"
                       "判据使用（一份没有曲线的基线等于没有基线）。"))
        return SkillResult(skill_name=_PHASE_FINISH, success=True,
                           data={"baseline_id": bid, "status": status,
                                 "sigma_model": sigma_model.to_dict(),
                                 "white_model": white_model.to_dict() if white_model else None,
                                 "lines": lines_out, "polarity": polarity,
                                 "bias_magnitude": bias_mag,
                                 "n_setpoint": models["n_setpoint"],
                                 "n_bias": models["n_bias"]})

    @staticmethod
    def _polarity_check(pts: list) -> "dict | None":
        """委托给 :func:`baseline.polarity_check` —— 导入外部测量的那条路也要用它,
        两份实现会让同一批数据按入口不同给出不同的结论。"""
        from mast.monitoring.baseline import polarity_check
        return polarity_check(pts)

    # ── driver ──────────────────────────────────────────────────────────

    @staticmethod
    def _scanning_state(context) -> "bool | None":
        """仪器在扫吗。``True`` / ``False`` / ``None``（读不到）。

        ``Scan_StatusGet`` 的历史包袱在 ``scan_utils`` 里记着（连续扫描开着时
        它从不返回 0）。这里只关心「有没有在动」，而且**读不到就是读不到** ——
        返回 None，由调用方决定要不要冒险，不折叠成 False。
        """
        try:
            rec = context.safe_call("Scan_StatusGet")
        except Exception:  # noqa: BLE001
            return None
        if getattr(rec, "error", ""):
            return None
        v = getattr(rec, "return_value", None)
        if isinstance(v, (list, tuple)) and len(v) > 2:
            body = v[2]
            if isinstance(body, (list, tuple)) and body:
                first = body[0]
                if isinstance(first, (list, tuple)) and first:
                    first = first[0]
                if isinstance(first, (bool, int)):
                    return bool(first)
        return None

    def run_composite(self, context, params: dict) -> SkillResult:
        from mast.monitoring.store import get_store

        # ═══════════════════════════════════════════════════════════════
        # 扫描中量到的不是仪器噪声，是**这块样品的形貌**
        # ═══════════════════════════════════════════════════════════════
        # ``monitoring.alerts`` 早就写明了这件事：扫描段的判据一律 suppressed，
        # 因为「从扫描段导出的阈值是关于这个样品的陈述，不是关于仪器噪声的」。
        # 而基线正是要拿去当阈值用的 —— 在扫描中建的基线会把形貌起伏写进
        # 「正常噪声」，此后真正的噪声异常就再也报不出来了。
        #
        # 读不到状态时**不阻止**：一个读不到就罢工的前置检查，比它防的问题
        # 更常见。但要记进 conditions，让这份基线的可信度可查。
        scanning = self._scanning_state(context)
        if scanning is True and not bool(params.get("allow_while_scanning", False)):
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=("仪器正在扫描 —— 现在量到的是这块样品的形貌起伏，不是"
                       "仪器噪声，拿它当基线会把形貌写进「正常」。先停扫描"
                       "（StopScan），或显式传 allow_while_scanning=true。"),
                data={"scanning": True})
        self._scanning_at_start = scanning
        self._params = dict(params or {})
        wrapped = _PhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(), context=wrapped,
            on_step_result=self.on_step_result, on_step_failed=self.on_step_failed)
        executor.set_partial_default("points", [])
        self._executor = executor

        as_found = self._read_state(context)
        logger.info("CharacteriseCurrentNoise: 原工作点 %s", as_found)
        store = get_store()
        bid = None
        if store is not None:
            bid = store.create_baseline(
                label=str(params.get("label") or ""),
                note=str(params.get("note") or ""),
                conditions=self._conditions(as_found))
        executor.set_partial("baseline_id", bid)

        ok = False
        try:
            ok = executor.run_plan(iter(self.plan(params)))
        finally:
            # Restore in a finally, not as a last step: an abort or a failed
            # step must not leave the instrument parked at whatever working
            # point the sweep happened to reach. Best-effort and never raises —
            # a restore that fails is logged and reported, not swallowed into
            # an exception that hides the original failure.
            self._restore(context, as_found)
            if store is not None and bid and not ok:
                store.finish_baseline(int(bid), status="aborted")

        data = dict(executor.progress.partial_data)
        data["as_found"] = as_found
        data["_progress"] = executor.progress.to_dict()
        if not ok:
            return SkillResult(skill_name=self._skill_name(), success=False,
                               error=executor.progress.aborted_reason
                               or "表征未跑完（已恢复原工作点）",
                               data=data)
        return SkillResult(skill_name=self._skill_name(), success=True, data=data)

    def _conditions(self, state: dict) -> dict:
        """Condition snapshot, keyed the same as :data:`baseline.CONDITION_KEYS`.

        Anything unreadable stays absent rather than being guessed — the diff
        function reports an unknown on either side as unknown, not as a match.
        """
        out: dict = {"as_found_bias_v": state.get("bias_v"),
                     "as_found_setpoint_a": state.get("setpoint_a")}
        try:
            from mast.core import instrument_profile as ip
            out["tip_id"] = ip.get_config("active_tip_id", None)
        except Exception:  # noqa: BLE001
            pass
        try:
            from mast.monitoring.store import get_store
            st = get_store()
            if st is not None:
                row = st.stats() if hasattr(st, "stats") else None
                if isinstance(row, dict) and row.get("fs_hz"):
                    out["fs_hz"] = float(row["fs_hz"])
        except Exception:  # noqa: BLE001
            pass
        return out

    def _restore(self, context, as_found: dict) -> None:
        if not (self._params or {}).get("restore", True):
            return
        b, s = as_found.get("bias_v"), as_found.get("setpoint_a")
        for skill, key, val in (("SetBias", "bias_v", b),
                                ("SetSetpoint", "setpoint_a", s)):
            if val is None:
                logger.warning("CharacteriseCurrentNoise: 原始 %s 没读到，"
                               "无法恢复这一项", key)
                continue
            try:
                p = {key: float(val)}
                if skill == "SetBias":
                    p["slew_rate_v_per_s"] = 0.5
                r = context.run(skill, p)
                if not getattr(r, "success", False):
                    logger.warning("CharacteriseCurrentNoise: 恢复 %s 失败: %s",
                                   key, getattr(r, "error", ""))
            except Exception:  # noqa: BLE001
                logger.warning("CharacteriseCurrentNoise: 恢复 %s 抛异常",
                               key, exc_info=True)


__all__ = ["CharacteriseCurrentNoise"]
