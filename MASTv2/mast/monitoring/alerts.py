"""Turning current features into verdicts — and, rarely, into an interruption.

The monitor is ADVISORY: it never touches the tip, the bias, Z or the motors.
(It does configure the oscilloscope's display — see pump.py — but that moves
nothing physical.) The strongest thing it can do is emit a CRITICAL
``tip_quality_drop`` onto the vision buffer, which
two existing consumers already act on — the composite halt hook stops the
running skill at its next step boundary, and the HITL middleware interrupts the
planner with a suggested remedy. Deciding whether to actually reshape the tip
stays with a human or an agent.

That reach is exactly why the CRITICAL rules are narrow. Only three qualify:

* **saturation** — the current is pinned at the preamp rail,
* **freeze** — the readout is returning one exact value, i.e. nobody is measuring,
* **giant_spike** — an excursion tens of sigma out with a step to match.

Each is a statement about the *instrument*, verifiable from the segment alone,
and each still needs N consecutive segments plus a cool-down before it fires.
Everything about tip *quality* — noise level, telegraph switching, mains pickup,
jump bursts — stays WARN. Those are the interesting numbers, but their operating
points have not been calibrated against real tips yet, and an alert that halts a
ten-minute scan on an uncalibrated threshold is an alert nobody will leave
enabled.

Fail-closed: a feature that could not be computed (``None``) never triggers a
rule. Missing evidence is not evidence.

**While a scan runs, two of the WARN rules are not measuring the instrument.**
See :data:`SCAN_SUPPRESSED_RULES`. They are still computed and still stored —
the verdict just reads ``suppressed`` instead of ``warn``, the same distinction
the aux channels already draw between "we chose not to judge this" and "we
judged it fine".
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Advisory rules — surface in the UI, never stop anything.
WARN_RULES = ("rms_high", "rtn_bistable", "line_hum", "jump_burst", "spike_warn",
              # instrument-level findings demoted because the context does not
              # support a hard call — see AlertEngine.evaluate
              "saturation_no_junction", "freeze_no_junction")

#: Rules allowed to escalate. See module docstring for why the list is this short.
CRIT_RULES = ("saturation", "freeze", "giant_spike")

#: Buffer payload ``signal`` value per critical rule.
_CRIT_SIGNAL = {
    "saturation": "current_saturation",
    "freeze": "current_freeze",
    "giant_spike": "current_giant_spike",
}

def critical_signals() -> frozenset[str]:
    """The ``signal`` values a CURRENT-MONITOR critical can carry.

    Public because a second consumer now has to tell "the preamp is railed /
    the chain is dead / the tip just took a giant step" apart from every OTHER
    thing published under the same ``tip_quality_drop`` event kind — see
    ``agents/_shared/buffer_hitl.py``. Exposed as a derived accessor rather
    than copied over there: a fourth entry in :data:`CRIT_RULES` must not
    require anybody to remember a list in another package."""
    return frozenset(_CRIT_SIGNAL.values())


#: A giant spike must also be a big STEP, not just a big z-score. On a very
#: quiet trace the robust sigma can be tiny, and then a perfectly ordinary
#: fluctuation scores hundreds of sigma.
_GIANT_SPIKE_STEP_OVER_RMS = 10.0

# Scanning can make topographic changes dominate jump and spike features. Record these advisory rules as suppressed during scans. Critical saturation, freeze and giant-spike rules remain active; other rules retain their declared policy. Commission noise thresholds from quiet segments, not sample topography.
SCAN_SUPPRESSED_RULES: frozenset[str] = frozenset({"jump_burst", "spike_warn"})

# Lock-in modulation can produce periodic excursions in current. Suppress the corresponding advisory rules while modulation is confirmed active, but preserve records and critical protection. Keep the modulation and scanning rule sets independently configurable.
MODULATION_SUPPRESSED_RULES: frozenset[str] = frozenset({"jump_burst", "spike_warn"})


@dataclass
class Verdict:
    """One segment's adjudication before consecutive-confirmation is applied."""

    level: str = "ok"                    # ok | warn | suppressed | critical_candidate
    rules: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    #: Rules that DID trigger but were not judged in this context. Kept apart
    #: from ``rules`` on purpose: everything downstream that alerts iterates
    #: ``rules``, so a rule parked here cannot reach an alert by accident, and a
    #: reader still sees it fired.
    suppressed_rules: list[str] = field(default_factory=list)


def _num(v) -> Optional[float]:
    """Coerce to float, treating None/NaN as 'not measured'."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


class AlertEngine:
    """Per-segment adjudication plus the consecutive/cool-down state machine."""

    def __init__(self, thresholds_getter=None):
        if thresholds_getter is None:
            from mast.monitoring.thresholds import get_monitor_thresholds
            thresholds_getter = get_monitor_thresholds
        self._th = thresholds_getter
        self._streak_rule: str | None = None
        self._streak_n = 0
        self._last_fired: dict[str, float] = {}

    # ── per-segment rules ───────────────────────────────────────────────────

    def evaluate(self, feats: dict, ctx: dict | None = None,
                 baseline: "Any | None" = None) -> Verdict:
        """Judge one segment. Pure — no state, no I/O, safe to test directly.

        ``baseline`` is a :class:`~mast.monitoring.baseline.SigmaModel` or None.
        It arrives as an ARGUMENT rather than being looked up here on purpose:
        this method's whole contract is that it does no I/O, and a database read
        per segment would also put a lock on the acquisition thread's path. The
        caller (:class:`~mast.monitoring.service.CurrentMonitorService`) holds
        the cached model and passes it down.

        ``ctx`` gates the instrument-level rules. Saturation and a frozen readout
        are only meaningful claims while a tunnelling junction actually exists:
        with the Z controller off the tip may be retracted, parked, or mid-
        spectroscopy, and "the current is pinned" then describes the experiment
        rather than a fault. Since these two rules can halt a running composite
        skill, they stay silent when the context does not support them — an
        unknown context (no snapshot at all) is treated as permissive, because
        refusing to judge whenever state is missing would disable the monitor on
        every install without a live InstrumentState.

        ``ctx_scanning`` gates :data:`SCAN_SUPPRESSED_RULES` the other way round:
        it must be EXPLICITLY true to suppress. Unknown means "do not suppress",
        so an install with no InstrumentState keeps the rules it has today
        instead of quietly losing two of them — the same polarity argument as
        above, applied to a gate that removes judgement rather than adds it.
        """
        th = self._th()
        ctx = ctx or {}
        crit: list[str] = []
        warn: list[str] = []
        detail: dict = {}

        # None = unknown (no snapshot); False = explicitly off.
        junction = ctx.get("ctx_zctrl_on")
        junction_ok = junction is None or bool(junction)

        sat_frac = _num(feats.get("sat_frac"))
        if sat_frac is not None and sat_frac > th.cm_sat_frac_crit:
            if junction_ok:
                crit.append("saturation")
            else:
                warn.append("saturation_no_junction")
            detail["sat_frac"] = sat_frac

        frozen = _num(feats.get("frozen"))
        if frozen is not None and frozen >= 1:
            if junction_ok:
                crit.append("freeze")
            else:
                warn.append("freeze_no_junction")

        spike_sigma = _num(feats.get("spike_max_sigma"))
        max_step = _num(feats.get("max_step_a"))
        rms = _num(feats.get("rms_detrended_a"))
        if spike_sigma is not None and spike_sigma > th.cm_spike_sigma_crit:
            if (max_step is not None and rms is not None and rms > 0
                    and max_step > _GIANT_SPIKE_STEP_OVER_RMS * rms):
                crit.append("giant_spike")
                detail["spike_max_sigma"] = spike_sigma
                detail["max_step_a"] = max_step
            else:
                warn.append("spike_warn")
                detail["spike_max_sigma"] = spike_sigma
        elif spike_sigma is not None and spike_sigma > th.cm_spike_sigma_warn:
            warn.append("spike_warn")
            detail["spike_max_sigma"] = spike_sigma

        # ``rms_high`` —— 两条路，选哪条由「有没有活跃基线」决定。
        #
        # 噪声可能随电流工作点改变。单一绝对阈值在整个范围上可能过松或过紧，
        # 因而有基线时需对照相同电流下的预期噪声。
        #
        # 所以有基线时，判的是**相对基线在同一电流下预期值的倍数**。
        #
        # 三条边界，每条都是刻意的：
        # * **没有基线 ⇒ 行为与今天逐字节相同**。一台还没做过表征的机器不能因为
        #   这次改动而悄悄换了判据——那会变成在别的机器上的静默行为漂移。
        # * **扫描中 ⇒ 走固定阈值那条路**。rms_detrended 在扫描时携带形貌的交流
        #   成分(见本文件 SCAN_SUPPRESSED_RULES 上方那段论证)，比值会必然超标。
        #   `is True` 不能用：SQLite 把布尔存成 0/1，`1 is True` 在 Python 里是 False。
        # * **基线判不了 ⇒ 回落固定阈值，并把原因记进 detail**。「判不了」不等于
        #   「没问题」，但也不该让这一段完全失去判据。
        if rms is not None:
            scanning = ctx.get("ctx_scanning")
            bl = None
            if baseline is not None and not (scanning is not None and bool(scanning)):
                from mast.monitoring.baseline import compare as _bl_compare
                bl = _bl_compare(rms, _num(feats.get("mean_a")), baseline)
            if bl is not None and bl.judged:
                detail["rms_ratio"] = bl.ratio
                detail["rms_expected_a"] = bl.expected_a
                detail["rms_detrended_a"] = rms
                if bl.ratio is not None and bl.ratio > th.cm_rms_ratio_warn:
                    warn.append("rms_high")
            else:
                if bl is not None and bl.reason:
                    detail["rms_baseline_unjudged"] = bl.reason
                if rms > th.cm_rms_warn_a:
                    warn.append("rms_high")
                    detail["rms_detrended_a"] = rms

        rtn = _num(feats.get("rtn_score"))
        if rtn is not None and rtn > th.cm_rtn_score_warn:
            warn.append("rtn_bistable")
            detail["rtn_score"] = rtn
            detail["rtn_rate_hz"] = _num(feats.get("rtn_rate_hz"))

        line = _num(feats.get("line_ratio"))
        if line is not None and line > th.cm_line_ratio_warn:
            warn.append("line_hum")
            detail["line_ratio"] = line

        jump = _num(feats.get("jump_rate_hz"))
        if jump is not None and jump > th.cm_jump_rate_warn_hz:
            warn.append("jump_burst")
            detail["jump_rate_hz"] = jump

        # ── context suppression: scanning ──────────────────────────────────
        # `is not None and bool(...)` rather than `is True`: SQLite hands these
        # back as 0/1 ints on every path that round-trips through the store, and
        # `1 is True` is False in Python. That exact trap emptied a calibration
        # group once already (commission._tri_bool). None (unknown) → no
        # suppression, deliberately.
        scanning = ctx.get("ctx_scanning")
        suppressed: list[str] = []
        if scanning is not None and bool(scanning) and warn:
            suppressed = [r for r in warn if r in SCAN_SUPPRESSED_RULES]
            warn = [r for r in warn if r not in SCAN_SUPPRESSED_RULES]

        # ── context suppression: lock-in modulation ────────────────────────
        # Suppress modulation-sensitive warnings only when modulation is explicitly active.
        # Unknown state retains the warnings; critical protection remains active.
        modulating = ctx.get("ctx_lockin_on")
        if modulating is not None and bool(modulating) and warn:
            more = [r for r in warn if r in MODULATION_SUPPRESSED_RULES]
            if more:
                suppressed = list(suppressed) + more
                warn = [r for r in warn if r not in MODULATION_SUPPRESSED_RULES]

        if crit:
            return Verdict("critical_candidate", crit, detail, suppressed)
        if warn:
            return Verdict("warn", warn, detail, suppressed)
        if suppressed:
            # Not "ok". "ok" would claim we looked at these numbers and found
            # them fine; we declined to look, and a later reader has to be able
            # to tell those apart (module docstring, and the aux precedent).
            return Verdict("suppressed", [], detail, suppressed)
        return Verdict("ok", [], detail)

    # ── confirmation ────────────────────────────────────────────────────────

    def confirm(self, verdict: Verdict, now: float | None = None) -> Optional[str]:
        """Escalate to CRITICAL only after N consecutive segments of the same rule.

        Returns the confirmed rule, or None. A single clean segment resets the
        streak: an intermittent rail touch is a WARN-worthy oddity, a sustained
        one means the tip is in the surface.
        """
        th = self._th()
        now = time.time() if now is None else now
        if verdict.level != "critical_candidate" or not verdict.rules:
            self._streak_rule, self._streak_n = None, 0
            return None

        rule = verdict.rules[0]
        if rule == self._streak_rule:
            self._streak_n += 1
        else:
            self._streak_rule, self._streak_n = rule, 1
        if self._streak_n < th.crit_consecutive:
            return None

        # A rule that has never fired is not "in cool-down". Defaulting the
        # last-fired time to 0 would make that depend on how large the clock
        # happens to be, which is fine for epoch seconds and wrong for any
        # other time base.
        last = self._last_fired.get(rule)
        if last is not None and now - last < th.cm_alert_cooldown_s:
            return None
        self._last_fired[rule] = now
        self._streak_n = 0
        return rule

    def should_emit_warn(self, rule: str, now: float | None = None) -> bool:
        """De-bounce WARNs per rule so the UI does not get a wall of duplicates."""
        th = self._th()
        now = time.time() if now is None else now
        key = f"warn:{rule}"
        last = self._last_fired.get(key)
        if last is not None and now - last < th.cm_alert_cooldown_s:
            return False
        self._last_fired[key] = now
        return True

    def reset(self) -> None:
        self._streak_rule, self._streak_n = None, 0
        self._last_fired.clear()


# ── human-readable summaries ────────────────────────────────────────────────

def summarize_zh(rule: str, feats: dict, detail: dict | None = None) -> str:
    """One Chinese sentence: what was measured, and what it suggests."""
    d = detail or {}
    if rule == "saturation":
        frac = (d.get("sat_frac") or _num(feats.get("sat_frac")) or 0.0) * 100
        # ⚠️ 2026-08-25 改：先分诊，再给处置。
        #
        # 原文是「疑似撞针或前置放大器过载,建议停止扫描并检查针尖」——
        # 它点了两个病因，而**最常见的那个不在里面**：Z 顶在退针上限 + 电流远超
        # setpoint = **热漂移**（样品漂近、压电吃满行程），不是针尖问题。
        # 按原处方去「检查针尖」，等于去修一根根本没坏的针尖。
        #
        # 三者用 **Z 的位置** 分得开，所以把判别式写进告警正文 ——
        # 读它的是 agent，而它此刻手上就有 GetZControllerState。
        return (f"隧道电流持续贴轨饱和(段内 {frac:.0f}% 的样本达到满量程)。"
                f"**先读 Z 再下结论**："
                f"① Z 顶在**退针上限**(接近 +压电半程) ⇒ **热漂移**，样品漂近、"
                f"压电吃满行程 —— 处置是 SafeRetract + 粗动 z-retract **退几步**"
                f"再 AutoApproach，**不是**修针；"
                f"② Z 在中段而电流贴轨 ⇒ 疑似撞针或针尖脏 —— 停扫、查针尖；"
                f"③ 换个偏压电流不变 ⇒ 那是**前放满量程这条轨**，不是真实电流，"
                f"读数本身不可信。")
    if rule == "freeze":
        return ("隧道电流读数完全不变(整段只有一个数值)——测量链路可能已中断"
                "(ADC/前放/通信),读到的数不能当作真实电流。")
    if rule == "giant_spike":
        sig = d.get("spike_max_sigma") or _num(feats.get("spike_max_sigma")) or 0.0
        step = d.get("max_step_a") or _num(feats.get("max_step_a")) or 0.0
        return (f"隧道电流出现巨幅瞬变(偏离背景噪声 {sig:.0f}σ,最大跳变 "
                f"{step * 1e12:.0f} pA)——疑似放电或针尖接触表面。")
    if rule == "rms_high":
        rms = d.get("rms_detrended_a") or _num(feats.get("rms_detrended_a")) or 0.0
        return f"电流噪声偏高(去趋势 RMS {rms * 1e12:.1f} pA)——针尖或环境可能不稳。"
    if rule == "rtn_bistable":
        score = d.get("rtn_score") or _num(feats.get("rtn_score")) or 0.0
        rate = d.get("rtn_rate_hz") or _num(feats.get("rtn_rate_hz")) or 0.0
        gap = _num(feats.get("rtn_gap_a")) or 0.0
        return (f"电流在两个电平之间来回跳变(RTN 判分 {score:.2f},速率 "
                f"{rate:.1f} Hz,间距 {gap * 1e12:.1f} pA)——典型的针尖顶端不稳定。")
    if rule == "line_hum":
        ratio = d.get("line_ratio") or _num(feats.get("line_ratio")) or 0.0
        return f"工频干扰明显(50 Hz 峰高出邻频 {ratio:.0f} 倍)——检查接地与屏蔽。"
    if rule == "jump_burst":
        rate = d.get("jump_rate_hz") or _num(feats.get("jump_rate_hz")) or 0.0
        return f"电流频繁突跳({rate:.1f} 次/秒)——针尖状态可能正在变化。"
    if rule == "saturation_no_junction":
        frac = (d.get("sat_frac") or _num(feats.get("sat_frac")) or 0.0) * 100
        return (f"电流贴轨饱和(段内 {frac:.0f}%),但 Z 反馈未开——可能是退针、"
                f"停泊或正在做谱,未按故障处理,仅记录。")
    if rule == "freeze_no_junction":
        return "电流读数完全不变,但 Z 反馈未开——未按故障处理,仅记录。"
    if rule == "spike_warn":
        sig = d.get("spike_max_sigma") or _num(feats.get("spike_max_sigma")) or 0.0
        return f"电流出现瞬时尖峰(偏离背景噪声 {sig:.0f}σ)——留意放电或接触。"
    return f"电流监控规则 {rule} 触发。"


# ── evidence ────────────────────────────────────────────────────────────────

def render_evidence_png(segment, feats: dict, out_dir: Path,
                        rule: str = "alert") -> Optional[str]:
    """Trace + spectrum PNG for one alert. Returns the path, or None.

    An alert that can halt a scan has to come with something an operator can
    look at — the vision monitor holds the same contract with its frame PNGs.
    Best effort: a plotting failure must never suppress the alert itself.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np

        y = segment.samples
        fs = float(getattr(segment, "fs_hz", 0.0) or 0.0)
        if y.size == 0 or fs <= 0:
            return None

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cm_{int(time.time() * 1000):013d}_{rule}.png"
        # Written via a temp file and renamed: this runs on a daemon thread, and
        # a shutdown mid-render would otherwise leave a truncated PNG at the
        # path the alert row already points at — the UI would show a broken
        # image instead of falling back to "no evidence".
        tmp = path.with_suffix(".png.part")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.2), dpi=110)
        t = np.arange(y.size) / fs
        ax1.plot(t, y * 1e12, lw=0.5, color="#2563eb")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("current (pA)")
        ax1.set_title(f"{rule} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        ax1.grid(alpha=0.25)

        from mast.io.signal_fft import compute_fft
        spec = compute_fft({"samples": y.tolist(), "fs_hz": fs, "unit": "A"},
                           window="hann", detrend=True, output="power")
        if spec:
            f = np.asarray(spec["freqs_hz"])
            p = np.asarray(spec["spectrum"])
            m = (f > 0) & (p > 0)
            if m.any():
                ax2.loglog(f[m], p[m], lw=0.6, color="#7c3aed")
        ax2.set_xlabel("frequency (Hz)")
        ax2.set_ylabel("PSD (A²/Hz)")
        ax2.grid(alpha=0.25, which="both")

        note = "  ".join(
            f"{k}={v:.3g}" for k, v in sorted(feats.items())
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k in ("rms_detrended_a", "sat_frac", "spike_max_sigma",
                      "rtn_score", "line_ratio", "jump_rate_hz")
        )
        if note:
            fig.text(0.01, 0.01, note, fontsize=7, color="#555")
        fig.tight_layout()
        fig.savefig(tmp, format="png")   # explicit: the .part suffix is not a format
        plt.close(fig)
        os.replace(tmp, path)
        return str(path)
    except Exception:  # noqa: BLE001 — evidence is a nicety, the alert is not
        logger.debug("evidence PNG failed (swallowed)", exc_info=True)
        return None


# ── escalation to the vision buffer ─────────────────────────────────────────

def _safe_mode_quiet() -> bool:
    """SAFE-mode predicate for alert wording. Never raises, never blocks."""
    try:
        from mast.core.operating_mode import safe_mode_active
        return safe_mode_active()
    except Exception:  # noqa: BLE001
        return False


def emit_critical(rule: str, summary_zh: str, feats: dict,
                  evidence_png: str | None, segment_id: int | None,
                  scan_id: str = "") -> bool:
    """Publish a CRITICAL tip-quality drop. Returns True if the buffer took it.

    The payload shape matches what the vision scan monitor already emits, so the
    existing consumers need no changes: the halt hook reads ``summary_zh``, the
    HITL middleware gates on kind+severity, and the GUI thumbnails
    ``frame_path``. ``recommend`` is a hint for the agent that receives the
    interrupt — it does not execute anything.
    """
    try:
        from mast.buffer.active import get_active_buffer
        from mast.buffer.schemas import Severity, VisionEvent, VisionEventType

        buf = get_active_buffer()
        if buf is None:
            return False                     # standalone / tests: DB + bus only
        payload: dict[str, Any] = {
            "signal": _CRIT_SIGNAL.get(rule, f"current_{rule}"),
            "scan_id": scan_id or "",
            "summary_zh": summary_zh,
            "network_free": True,
            "frame_path": evidence_png or "",
            "features": {k: float(v) for k, v in feats.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)
                         and k in ("sat_frac", "railed_frac", "spike_max_sigma",
                                   "max_step_a", "rms_detrended_a", "rtn_score",
                                   "frozen")},
            # In SAFE the gate refuses ConditionTip, so recommending it would be
            # the system asking for an action it will then reject — the shape of
            # . The CRITICAL itself still fires and still halts:
            # this is physical safety, and SAFE only means "do not repair the tip".
            "recommend": (["StopScan"] if _safe_mode_quiet()
                          else ["StopScan", "ConditionTip"]),
            "source": "current_monitor",
        }
        buf.emit_event(VisionEvent(
            seqno=buf.next_seq(),
            kind=VisionEventType.TIP_QUALITY_DROP,
            severity=Severity.CRITICAL,
            payload=payload,
            cause_ref=f"current_monitor#{segment_id}" if segment_id else "current_monitor",
        ))
        return True
    except Exception:  # noqa: BLE001 — a broken buffer must not lose the alert
        logger.warning("current monitor: could not publish CRITICAL to the "
                       "vision buffer (alert still recorded)", exc_info=True)
        return False


__all__ = ["AlertEngine", "Verdict", "WARN_RULES", "CRIT_RULES",
           "SCAN_SUPPRESSED_RULES", "MODULATION_SUPPRESSED_RULES",
           "critical_signals",
           "summarize_zh", "render_evidence_png", "emit_critical"]
