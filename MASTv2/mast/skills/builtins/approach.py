"""Auto approach and withdraw skills.

vendored from v1 mast/skills/builtins/approach.py 2026-04-23. Zero behavioural changes.
3 skills: AutoApproach, WithdrawTip, GetAutoApproachStatus.

Phase 7 migration (2026-05-19): AutoApproach is now a CompositeSkillGraph.
The original two-call sequence (Open → OnOffSet(1)) is expanded into 4
mandatory phases — open_module / start_approach / wait_complete /
verify_status — so per-phase progress is visible to the GUI and the
LangGraph checkpointer can flush after each phase. ``wait_complete`` and
``verify_status`` are NEW phases that issue an AutoApproach_OnOffGet check
to confirm the approach has actually started; if the OnOffGet TCP call
fails the whole composite aborts (caller probably wants to know).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.core import instrument_profile as ip
from mast.core import safety_escalation as esc
from mast.core.crosstalk_report import crosstalk_report
from mast.skills.base import BaseSkill
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite._preflight import (
    apply_approach_preset,
    close_modulation,
    restore_zctrl,
)
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_facts,
)


# Synthetic phase identifiers — intercepted by _AutoApproachPhaseCtx.
logger = logging.getLogger(__name__)

_PHASE_OPEN = "_phase_open_module"
_PHASE_START = "_phase_start_approach"
_PHASE_WAIT = "_phase_wait_complete"
_PHASE_VERIFY = "_phase_verify_status"

# Engagement requires both a fraction of the setpoint and an absolute floor.
# Otherwise an arbitrarily small setpoint can make noise look like engagement.
# This software criterion must be checked against the target
# instrument noise; it is not a shipped noise-floor measurement.
_MIN_ENGAGED_CURRENT_A = 1e-12


def _engagement_bar(setpoint_a: "float | None") -> "float | None":
    """The |I| a real engagement must clear, or None when it cannot be computed.

    Two bars, whichever is higher: half the setpoint (feedback is actually
    regulating) and the absolute noise floor (so lowering the setpoint cannot
    make amplifier noise pass as engagement — )."""
    if setpoint_a is None or setpoint_a == 0:
        return None
    return max(0.5 * abs(setpoint_a), _MIN_ENGAGED_CURRENT_A)


@dataclass
class EngageVerdict:
    """Did the tunnelling current SETTLE on one side of the engagement bar?

    ``engaged`` has three values and they are three different facts:

    * ``True``  — settled above the bar. The tip is in tunnelling.
    * ``False`` — settled below it. The tip is not engaged; this is a MEASURED
      zero, not a missing measurement.
    * ``None``  — the readings never agreed with themselves inside the budget
      (or could not be read at all). **We do not know.** Callers must fail on
      this, but they must not report it as "not engaged" — "没测出来" and
      "测出来是零" are two different sentences, and printing one for the other
      is what once sent an operator looking at the motor range for the wrong reason.
    """

    engaged: "bool | None" = None
    current_a: "float | None" = None
    setpoint_a: "float | None" = None
    bar_a: "float | None" = None
    #: The most recent |I| reads, oldest first — the evidence. Capped at
    #: :data:`_KEEP_READS`: with the production 1 s spacing a full window is
    #: ~20 numbers, but nothing stops an operator shrinking the interval, and a
    #: failure message is useless if it is three screens of floats.
    reads_a: list[float] = field(default_factory=list)
    #: How many reads were actually taken (``reads_a`` may be a tail of these).
    total_reads_n: int = 0
    #: How many consecutive reads landed on the same side before we stopped.
    agreed_n: int = 0
    elapsed_s: float = 0.0
    budget_s: float = 0.0
    interval_s: float = 0.0
    aborted: bool = False
    #: Reads that produced no parseable current / setpoint at all.
    unreadable_n: int = 0

    def evidence(self) -> str:
        """The measured numbers, and nothing else.

        What used to stand in the failure message was a list of GUESSES —
        「量程耗尽 / Z 在极限 / 状态过期」—— none of which this code had looked
        at, and they were not always true. A verdict may print what it measured;
        the causes are for whoever can actually observe them."""
        reads = ", ".join(_fmt_a(v) for v in self.reads_a) or "(无)"
        bar = _fmt_a(self.bar_a) if self.bar_a is not None else "无法计算(设定点读不到或为 0)"
        scope = ("窗口内实测 |I| 依次为"
                 if self.total_reads_n <= len(self.reads_a)
                 else f"窗口内共读 {self.total_reads_n} 次,最近 {len(self.reads_a)} 次 |I| 依次为")
        # 明确标注电流判定窗口，避免把窗口耗时误读为进针模块运行时长。
        return (f"判据 |I| ≥ {bar}(= max(50%×设定点 {_fmt_a(self.setpoint_a)}, "
                f"噪声底 {_fmt_a(_MIN_ENGAGED_CURRENT_A)}));"
                f"{scope} [{reads}],"
                f"电流判定窗口 {self.elapsed_s:.1f}s / 判定预算 {self.budget_s:.1f}s,"
                f"取样间隔 {self.interval_s:.1f}s")

    def as_dict(self) -> dict[str, Any]:
        return {"engaged": self.engaged, "measured_current_a": self.current_a,
                "setpoint_a": self.setpoint_a, "engagement_bar_a": self.bar_a,
                "reads_a": list(self.reads_a), "total_reads_n": self.total_reads_n,
                "agreed_n": self.agreed_n,
                "elapsed_s": round(self.elapsed_s, 3), "budget_s": self.budget_s,
                "settle_interval_s": self.interval_s, "aborted": self.aborted,
                "unreadable_reads": self.unreadable_n}


@dataclass
class WaitProgress:
    """报告等待阶段的模块运行时间与 Z 总行程。
    模块运行时间与模块停止后的电流验证窗口是两个不同的量，必须分开报告。
    否则调用方可能把一次很短的后验验证误解为整个进针过程持续时间。
    未出现隧穿电流时，Z 行程仍可用于判断机构是否运动。进针循环的 Z 往复
    不等同于已经接触表面，但可以区分持续推进与完全未动。
    协议没有直接提供粗动完成步数，因此这里使用可获得的运动读数。
    z_travel_m 为零也是有效信息；未知读数不能当作零行程。
    """

    #: 轮询了几次模块状态,其中几次读到 running=1。
    polls_n: int = 0
    running_polls_n: int = 0
    #: 从**第一次**读到 running=1 到**最后一次**读到 running=1 的秒数 —— 模块实际在跑
    #: 的时长(下界:第一次读到之前它可能已经跑了不到一个轮询周期)。
    module_ran_s: float = 0.0
    #: 整个等待相花了多久(含判定前的轮询)。
    waited_s: float = 0.0
    #: Z 采样:读了几次、区间、**总行程**(逐次 |Δz| 累加)。
    z_reads_n: int = 0
    z_min_m: "float | None" = None
    z_max_m: "float | None" = None
    z_travel_m: float = 0.0
    #: 读到 running=0、复读一次却又变回 1 的次数(见 ``_confirm_stopped``)。
    status_flap_n: int = 0
    #: **状态位读不懂**的次数(``_parse_running`` 回 None)。与「读到 0」分开记:
    #: 循环对两者的处置一样(都往「可能停了」走),但事后查账时它们要做的事完全
    #: 不同 —— 一个是查仪器,一个是查 TCP / parser。合成一个数就再也分不开了。
    status_unreadable_n: int = 0

    def note_running(self, running: bool, elapsed_s: float) -> None:
        self.polls_n += 1
        self.waited_s = float(elapsed_s)
        if running:
            self.running_polls_n += 1
            if self._first_running_s is None:
                self._first_running_s = float(elapsed_s)
            self.module_ran_s = float(elapsed_s) - self._first_running_s

    def note_z(self, z_m: "float | None") -> None:
        if z_m is None:
            return
        z = float(z_m)
        if self._last_z is not None:
            self.z_travel_m += abs(z - self._last_z)
        self._last_z = z
        self.z_reads_n += 1
        self.z_min_m = z if self.z_min_m is None else min(self.z_min_m, z)
        self.z_max_m = z if self.z_max_m is None else max(self.z_max_m, z)

    #: 内部游标。dataclass 字段而不是普通属性,免得一个未初始化的实例在第一次
    #: ``note_*`` 上炸掉一趟真进针。
    _first_running_s: "float | None" = None
    _last_z: "float | None" = None

    @property
    def z_span_m(self) -> "float | None":
        if self.z_min_m is None or self.z_max_m is None:
            return None
        return self.z_max_m - self.z_min_m

    @property
    def cycles_approx(self) -> "float | None":
        """≈ 几个进退循环。一个循环 = 走一个来回 = 2 × 行程区间。

        故意只给一个 ``≈``:1 Hz 采样看一个 ~5 s 的循环够用来数,但不够精确到整数,
        而一个没有 ≈ 的循环数,读的人会拿它当刻度用(同 ``crosstalk_report`` 的误差带
        那条理由)。
        """
        span = self.z_span_m
        if not span or span <= 0:
            return None
        return self.z_travel_m / (2.0 * span)

    def motion_text(self) -> str:
        """一句话说清「模块跑了多久 + 台子动没动」。永远说得出是哪一种。"""
        head = (f"模块实际运行 **{self.module_ran_s:.1f}s**"
                f"(共轮询 {self.polls_n} 次,其中 {self.running_polls_n} 次读到 running=1"
                + (f",另有 {self.status_flap_n} 次读到停止但复读又在跑"
                   if self.status_flap_n else "")
                + (f",另有 {self.status_unreadable_n} 次**状态位读不懂**"
                   f"(那不是「没在跑」,是没问出来)"
                   if self.status_unreadable_n else "")
                + ")")
        if self.z_reads_n == 0:
            return head + ";这段时间**没读到 Z 压电位置**,粗动有没有推进无法判断。"
        span = self.z_span_m or 0.0
        if self.z_travel_m <= 0:
            return (head + f";期间 Z 压电**纹丝不动**(采样 {self.z_reads_n} 次,"
                    f"始终 {_fmt_m(self.z_min_m)})—— 模块报在跑,压电却没动。")
        cyc = self.cycles_approx
        cyc_txt = f",≈{cyc:.0f} 个进退循环" if cyc and cyc >= 1 else ""
        return (head + f";期间 Z 压电总行程 **{_fmt_m(self.z_travel_m)}**"
                f"(区间 {_fmt_m(self.z_min_m)} … {_fmt_m(self.z_max_m)},"
                f"摆幅 {_fmt_m(span)}{cyc_txt},采样 {self.z_reads_n} 次)"
                f" —— **粗动确实在推进**。")

    def as_dict(self) -> dict[str, Any]:
        return {"polls_n": self.polls_n, "running_polls_n": self.running_polls_n,
                "module_ran_s": round(self.module_ran_s, 3),
                "waited_s": round(self.waited_s, 3),
                "z_reads_n": self.z_reads_n, "z_min_m": self.z_min_m,
                "z_max_m": self.z_max_m, "z_travel_m": self.z_travel_m,
                "z_cycles_approx": self.cycles_approx,
                "status_flap_n": self.status_flap_n,
                "status_unreadable_n": self.status_unreadable_n}


#: How many CONSECUTIVE reads must land on the same side of the bar before the
#: verdict is given. Two is the smallest number that can tell a settled junction
#: from a single sample of a transient — which is the whole defect this replaces.
_ENGAGE_AGREE_N = 2

#: Spacing between those reads. It must be LONGER than the feedback handover
#: transient, or two samples of the same transient "agree" and we are back where
#: we started. 1 s is orders of magnitude above the preamp bandwidth and above
#: the loop's own response, while being small against the settle budget below.
_ENGAGE_INTERVAL_S = 1.0

#: How many of the most recent reads to keep as evidence. See EngageVerdict.
_KEEP_READS = 12


def _engage_budget_s() -> float:
    """How long to keep asking before giving up — DERIVED, not a new constant.

    ``z_settle_timeout_s`` is already the rig's declared answer to "how long may
    this feedback loop take to converge", and it is already on the settings page
    (「退针 Z 稳定预算」). The current settling after the approach module hands
    over is the same loop converging, so a second knob beside it would be a
    second number to keep equal — and the way those two drift apart is that
    nobody remembers the second one exists."""
    try:
        from mast.skills.composite._z_settle import settle_timeout_s

        return float(settle_timeout_s())
    except Exception:  # noqa: BLE001 — defensive; never let a config read decide
        return 20.0


def settle_engagement(
    read_pair: "Callable[[], tuple[float | None, float | None]]",
    *,
    interval_s: float = _ENGAGE_INTERVAL_S,
    budget_s: "float | None" = None,
    agree_n: int = _ENGAGE_AGREE_N,
    check_abort: "Callable[[], bool] | None" = None,
) -> EngageVerdict:
    """Judge engagement from consecutive agreeing reads.

    The handover from approach to Z feedback may include a transient. Poll until
    ``agree_n`` consecutive reads lie on the same side of the criterion; a
    disagreement means the readings have not settled, up to the time budget.

    ``read_pair`` returns ``(current_a, setpoint_a)``. An unreadable value or a
    reader exception is treated as an unreadable pair rather than raised.
    """
    import time as _t

    budget = float(_engage_budget_s() if budget_s is None else budget_s)
    out = EngageVerdict(budget_s=budget, interval_s=float(interval_s))
    t0 = _t.monotonic()
    run_side: "bool | None" = None
    run = 0

    while True:
        if callable(check_abort) and check_abort():
            out.aborted = True
            out.elapsed_s = _t.monotonic() - t0
            return out

        try:
            cur, sp = read_pair()
        except Exception:  # noqa: BLE001 — a failed read is an unreadable read
            cur = sp = None
        out.elapsed_s = _t.monotonic() - t0

        bar = _engagement_bar(sp)
        if cur is None or bar is None:
            # Unreadable. It breaks any run in progress — a verdict may only
            # rest on reads we actually took, and an unread sample is not
            # evidence that the previous one still holds.
            out.unreadable_n += 1
            run_side, run = None, 0
        else:
            out.current_a = float(cur)
            out.setpoint_a = float(sp)
            out.bar_a = bar
            out.total_reads_n += 1
            out.reads_a.append(abs(float(cur)))
            if len(out.reads_a) > _KEEP_READS:
                out.reads_a.pop(0)
            side = abs(float(cur)) >= bar
            if side is run_side:
                run += 1
            else:
                run_side, run = side, 1
            out.agreed_n = run
            if run >= max(2, int(agree_n)):
                out.engaged = side
                return out

        if out.elapsed_s >= budget:
            # Never agreed with itself. engaged stays None — see EngageVerdict.
            return out
        if out.total_reads_n == 0 and out.unreadable_n >= max(2, int(agree_n)):
            # Several tries and not ONE parseable pair. Waiting out the rest of
            # the budget cannot turn a dead readback into a reading, and the
            # caller needs the diagnosis now — "cannot read the current" is a
            # different repair from "this junction is still settling". Same
            # early-out, and the same reason, as ``_z_settle``.
            return out
        _t.sleep(min(interval_s, max(0.0, budget - out.elapsed_s)))


def _approach_evidence(context) -> dict:
    """Independent witnesses to attach to a successful approach.

    Two channels the current cannot supply on its own:

    * **qPlus amplitude** — mechanically independent of the current preamp, and
      it answers a question the current cannot: "is the tip still free to move?"
      A tip that has ploughed in is damped to a standstill, which is exactly the
      state a current-only check reads as a healthy engagement at setpoint.
    * **lock-in dI/dV** — reported as a ratio against the learned at-contact
      value. Trend only; see ``_tip_evidence.didv_trend_fields`` for why it does
      not gate.

    Never raises and never subtracts: an STM with no qPlus contributes nothing
    here, which is a normal configuration and not a fault."""
    out: dict = {}
    try:
        from mast.skills.builtins._tip_evidence import didv_trend_fields, qplus_fields

        out.update(qplus_fields(context))
        out.update(didv_trend_fields(context))
    except Exception:  # noqa: BLE001 — a bonus witness must not break the verdict
        return out
    return out


def _evidence_warning(evidence: dict) -> str:
    """A sentence to append when a second witness contradicts the current.

    Deliberately does NOT flip ``success``. The current-based verification is the
    contract every caller already relies on, and a rig whose amplitude channel is
    mis-scaled would otherwise start failing perfectly good approaches. But the
    contradiction has to be visible — an engaged-at-setpoint reading with a
    collapsed oscillation is what a tip buried in the surface looks like."""
    try:
        from mast.skills.builtins._tip_evidence import qplus_says_crashed

        if qplus_says_crashed(evidence):
            return (" ⚠️ 但 qPlus 振幅已塌到自由振荡基线的 "
                    f"{evidence.get('qplus_fraction', '?')} —— 这是**针尖已扎进表面**"
                    "的典型特征。电流判据说进针成功,振幅判据说针尖不自由;"
                    "两者矛盾时以更悲观的为准:先退针复查,不要直接扫图。")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _chain_scope(context) -> str:
    """Identify the driver chain for scoping the Layer-0d approach refusal.

    A refusal is process-global for the shared instrument, but clearing it is
    scoped to its originating chain. Engagement in another chain must not erase
    that refusal. See ``safety_escalation.clear_approach_refusal``.
    """
    owner = str(getattr(context, "owner", "") or "")
    run_id = str(getattr(context, "run_id", "") or "")
    return f"{owner}#{run_id}" if (owner or run_id) else ""


def _fmt_a(x: "float | None") -> str:
    """Human-readable ampere value (for example, '0.25 pA' or '0.75 nA')."""
    if x is None:
        return "unreadable"
    ax = abs(x)
    if ax >= 1e-9:
        return f"{x / 1e-9:.2f} nA"
    return f"{x / 1e-12:.2f} pA"


def _fmt_m(x: "float | None") -> str:
    """Human-readable metre value (for example, '125.00 nm' or '2.50 µm')."""
    if x is None:
        return "unreadable"
    ax = abs(x)
    if ax >= 1e-3:
        return f"{x / 1e-3:.2f} mm"
    if ax >= 1e-6:
        return f"{x / 1e-6:.2f} µm"
    if ax >= 1e-9:
        return f"{x / 1e-9:.2f} nm"
    return f"{x / 1e-12:.2f} pm"


def _parse_running(rv) -> "bool | None":
    """Extract Status from AutoApproach.OnOffGet's (error, raw, parsed) response.
    The scalar status is parsed[0]. Unknown or malformed responses return None,
    never False: decoding failure is not evidence that the module has stopped.
    Legacy flat shapes and singleton tuples are also accepted defensively.
    The protocol declares a scalar here; tuple support does not imply that a
    particular device has returned an array wrapper. Unwrap before bool conversion
    because a tuple containing zero is itself truthy.
    """
    inner = rv
    if isinstance(rv, (list, tuple)):
        if not rv:
            return None
        inner = rv[2] if len(rv) > 2 else rv
    # parsed 列表 → 状态位；再解一层 1-元组。空 = 上游没解出来 ⇒ 读不懂。
    for _ in range(2):
        if not isinstance(inner, (list, tuple)):
            break
        if not inner:
            return None
        inner = inner[0]
    if isinstance(inner, bool):
        return inner
    if isinstance(inner, (int, float)):
        return bool(inner)
    try:
        return bool(int(str(inner).strip()))
    except (TypeError, ValueError):
        return None


class _AutoApproachPhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names."""

    def __init__(self, real_ctx, skill: "AutoApproach") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class AutoApproach(CompositeSkillGraph):
    """Start auto approach procedure (moves tip toward surface)."""

    # Default ceiling on how long the coarse approach may run before we stop the
    # module and report "did not complete" (rather than silently claiming done).
    # This is only a BACKSTOP against a stuck module — the wait loop exits the
    # instant the module reaches the setpoint and stops, so a generous ceiling
    # costs nothing on a normal approach but stops truncating a legitimately long
    # coarse approach from far away (300–900 s was too
    # short and cut off approaches that had not yet reached the surface). 30 min.
    _DEFAULT_WAIT_TIMEOUT_S = 1800.0
    # Wait-phase poll cadence + startup grace window. Class attributes so tests
    # (which drive the wrap_skill path and can't touch the instance) can override
    # them at the class level to keep the poll loop fast.
    _poll_interval_s = 0.5
    _grace_s = 3.0

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AutoApproach",
            version="1.2.0",
            category=SkillCategory.WRITE,
            # AUTO: Nanonis 内置进针的硬件级安全保护（电流限幅 + Z 上限），
            # 软件层再加 confirm 反而卡死——曾经确认框弹不出来，
            # 导致进针完全走不动。把确认要求去掉，进针直接执行。
            # bias_nonzero precondition (2026-07-03 review): the hardware
            # current-feedback stop ONLY works if there is a tunnelling current to
            # detect. With bias=0 V the current stays 0 even on contact, so the
            # module never stops and the coarse stepper grinds the tip into the
            # surface. Refuse to start a coarse approach at zero bias.
            safety_level=SafetyLevel.AUTO,
            description="启动 auto approach 流程。它会把针尖朝表面移动。",
            estimated_duration_s=60.0,
            rollback_skill="WithdrawTip",
            composition_level=1,
            preconditions=["bias_nonzero"],
            parameters=[
                ParameterSpec(
                    name="wait_timeout_s", type="float",
                    description=(
                        "等待粗动 approach 达到 setpoint 的最长秒数；到点则停掉"
                        "该模块并报告「未完成」。它**只是一道兜底（BACKSTOP）** —— 模块一旦"
                        "达到 setpoint，等待就立刻结束，所以保留那个宽松的"
                        "默认值（1800 s = 30 分钟）是安全的。从远处开始的粗动 "
                        "approach 可能要花好几分钟；**不要**把它降到几百秒，"
                        "否则你可能会在一次有效的 approach 触到表面之前"
                        "就把它截断。"),
                    required=False, default=self._DEFAULT_WAIT_TIMEOUT_S,
                    min_value=5.0, max_value=3600.0),
            ],
            tags=["approach", "tip"],
        )

    # ------------------------------------------------------------------
    # Plan — 4 mandatory phases. All failures abort the composite.
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        return [
            CompositeStep(
                step_id="open_module",
                skill_name=_PHASE_OPEN,
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("setup", "open"),
            ),
            CompositeStep(
                step_id="start_approach",
                skill_name=_PHASE_START,
                params={},
                optional=False,
                checkpoint_after=True,    # critical state change — flush
                tags=("write", "start"),
            ),
            CompositeStep(
                step_id="wait_complete",
                skill_name=_PHASE_WAIT,
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("read", "wait"),
            ),
            CompositeStep(
                step_id="verify_status",
                skill_name=_PHASE_VERIFY,
                params={},
                optional=False,
                checkpoint_after=True,    # final state — flush
                tags=("read", "verify"),
            ),
        ]

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _PHASE_OPEN:
            return self._phase_open_module(real_ctx)
        if skill_name == _PHASE_START:
            return self._phase_start_approach(real_ctx)
        if skill_name == _PHASE_WAIT:
            return self._phase_wait_complete(real_ctx)
        if skill_name == _PHASE_VERIFY:
            return self._phase_verify_status(real_ctx)
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    def _phase_open_module(self, real_ctx) -> SkillResult:
        rec = real_ctx.safe_call("AutoApproach_Open")
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_OPEN,
                success=False,
                error=rec.error,
            )
        return SkillResult(skill_name=_PHASE_OPEN, success=True, data={})

    def _phase_start_approach(self, real_ctx) -> SkillResult:
        rec = real_ctx.safe_call("AutoApproach_OnOffSet", 1)
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_START,
                success=False,
                error=rec.error,
            )
        self._executor.set_partial("approach_started", True)
        return SkillResult(skill_name=_PHASE_START, success=True, data={})

    # 报告级串扰导航的节奏。0 或负数 = 关掉。类属性,好让测试改而不必碰实例。
    _crosstalk_every_s = 15.0

    def _maybe_report_crosstalk(self, real_ctx) -> None:
        """隔一阵读一次 lock-in X,报「≈还剩多少步」。**不驱动任何决策。**

        两个开销上的选择:

        * **调制关着就一次性熄火**(`_crosstalk_off` 闩)。进针流程开跑前会自动关调制
          (用完即关),所以常态下这是关的 —— 那就不该每 15 秒再去问一次,一次
          `GetLockInConfig` 是 6 个来回,而一次进针可以跑半小时。
        * 取样很短(3 点),因为这段时间里我们没在轮询模块;而模块自己会在设定点停,
          我们只是在旁边看着。

        永不抛异常,永不改变等待逻辑 —— 它只是往 partial 里放一句话。
        """
        import time as _t     # 与本文件其它计时处一致:_t 是函数内局部导入

        every = float(getattr(self, "_crosstalk_every_s", 0.0) or 0.0)
        if every <= 0 or getattr(self, "_crosstalk_off", False):
            return
        now = _t.monotonic()
        last = getattr(self, "_crosstalk_last_t", None)
        if last is not None and (now - last) < every:
            return
        self._crosstalk_last_t = now
        try:
            rep = crosstalk_report(real_ctx, n=3, interval_s=0.05)
            if rep.get("crosstalk_modulation_off"):
                self._crosstalk_off = True      # 熄火:这一趟不再问
            executor = getattr(self, "_executor", None)
            if executor is not None:
                executor.set_partial("crosstalk", rep)
        except Exception:  # noqa: BLE001 — 报告不该动进针
            logger.debug("串扰报告失败(跳过)", exc_info=True)

    def _phase_wait_complete(self, real_ctx) -> SkillResult:
        """Actually WAIT for the coarse approach to finish.

        The old implementation did a single OnOffGet and returned success
        regardless of whether the module was still approaching, had finished,
        OR had refused to start — so ApproachTip declared "engaged, stopped at
        the setpoint" while the motor was possibly still stepping (or never
        started). This polls AutoApproach_OnOffGet until the module stops
        (running 1→0 = reached setpoint), honouring the shared abort Event and a
        timeout; a module that never runs within a short grace window (start
        rejected) fails, unless a tunnelling current confirms an ultra-fast
        completion.
        """
        import time as _t
        check_abort = getattr(real_ctx, "check_abort", None)
        poll_interval = float(getattr(self, "_poll_interval_s", 0.5))
        grace_s = float(getattr(self, "_grace_s", 3.0))
        timeout_s = float(getattr(self, "_wait_timeout_s", self._DEFAULT_WAIT_TIMEOUT_S))
        max_consecutive_errors = 5  # bail fast on a dead link (don't wait full timeout)
        start = _t.monotonic()
        observed_running = False
        consecutive_errors = 0
        # 逐趟局部变量,**不挂到 self 上**:技能实例会被复用,而一个挂在实例上的进展
        # 记录在「等待相这趟根本没跑到」的时候会留着上一趟的数字 —— 那正是 sidecar
        # 那个缺陷的形状(一个 9 天前的 _progress 被当成本次结果)。
        progress = WaitProgress()
        z_every = float(getattr(self, "_z_sample_every_s", 1.0) or 0.0)
        z_last_t: "float | None" = None

        while True:
            if callable(check_abort) and check_abort():
                self._stop_module(real_ctx)
                self._executor.set_partial("aborted", True)
                self._executor.set_partial("wait_progress", progress.as_dict())
                return SkillResult(
                    skill_name=_PHASE_WAIT, success=False,
                    error=("aborted by user — AutoApproach module stopped"
                           + self.stop_note()),
                    data={"wait_progress": progress.as_dict(),
                          "stop_failures": list(self._stop_failures)})

            elapsed = _t.monotonic() - start
            rec = real_ctx.safe_call("AutoApproach_OnOffGet")
            self._call_log.append(rec)
            if rec.error:
                # Transient poll failure — retry a few times, but bail rather than
                # spin until the (long) timeout if the link is genuinely dead.
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors or elapsed >= timeout_s:
                    self._executor.set_partial("wait_progress", progress.as_dict())
                    return SkillResult(
                        skill_name=_PHASE_WAIT, success=False,
                        error=f"OnOffGet failed persistently: {rec.error}",
                        data={"wait_progress": progress.as_dict()})
                _t.sleep(poll_interval)
                continue
            consecutive_errors = 0
            self._maybe_report_crosstalk(real_ctx)

            running = self._parse_running(rec.return_value)
            if running is None:
                # 循环的处置不变(读不懂与读到 0 一样往「可能停了」走 —— 那条
                # 保守语义有它自己的理由,见 `_confirm_stopped`),但**记下来**:
                # 事后要能分出「模块报停了」和「我们没问出来」。
                progress.status_unreadable_n += 1
            progress.note_running(bool(running), elapsed)
            # Z 采样比状态轮询慢一档 —— 见 ``_z_sample_every_s``。
            now_t = _t.monotonic()
            if z_every > 0 and (z_last_t is None or (now_t - z_last_t) >= z_every):
                z_last_t = now_t
                progress.note_z(self._read_z(real_ctx))
            if running:
                observed_running = True
            else:
                if observed_running and not self._confirm_stopped(real_ctx, progress):
                    # 复读说它还在跑 —— 那个 0 是一次读数,不是一个事实。
                    _t.sleep(poll_interval)
                    continue
                if observed_running:
                    # A stopped module does not prove engagement: stops can have
                    # multiple causes. Confirm using consecutive agreeing current
                    # readings so a handover transient does not decide the result.
                    v = self._judge_engagement(real_ctx)
                    self._executor.set_partial("wait_progress", progress.as_dict())
                    if v.engaged is True:
                        self._executor.set_partial("running_after_start", False)
                        return SkillResult(
                            skill_name=_PHASE_WAIT, success=True,
                            data={"running": False, "completed": True,
                                  "engagement": v.as_dict(),
                                  "wait_progress": progress.as_dict()})
                    self._stop_module(real_ctx)
                    # Built BEFORE the SkillResult call, not inside it: a
                    # keyword argument nested in a SkillResult(...) call reads
                    # as a SkillResult field to the source-level guard in
                    # test_all_skills_execute (which exists to catch
                    # `message=` typos and cannot parse this apart).
                    why = self._engage_failure_text(v, stopped=True, progress=progress)
                    return SkillResult(
                        skill_name=_PHASE_WAIT, success=False,
                        error=why + self.stop_note(),
                        data={"engagement": v.as_dict(),
                              "wait_progress": progress.as_dict(),
                              "stop_failures": list(self._stop_failures)})
                # Never observed running.
                if elapsed >= grace_s:
                    # Either the start was rejected, or it completed faster than
                    # the first poll. Confirm via tunnelling current before deciding.
                    v = self._judge_engagement(real_ctx)
                    self._executor.set_partial("wait_progress", progress.as_dict())
                    if v.engaged is True:
                        self._executor.set_partial("running_after_start", False)
                        return SkillResult(
                            skill_name=_PHASE_WAIT, success=True,
                            data={"running": False, "completed": True,
                                  "note": "completed before first poll (current confirms)",
                                  "engagement": v.as_dict(),
                                  "wait_progress": progress.as_dict()})
                    why = self._engage_failure_text(v, stopped=False, progress=progress)
                    return SkillResult(
                        skill_name=_PHASE_WAIT, success=False,
                        error=why + self.stop_note(),
                        data={"engagement": v.as_dict(),
                              "wait_progress": progress.as_dict(),
                              "stop_failures": list(self._stop_failures)})

            if elapsed >= timeout_s:
                # 总预算需允许正常但缓慢的推进完成，同时为无法完成的过程提供有限上界。
                # 超时与模块停止后的进针判定分开报告，不把两者混为同一个阶段。
                self._stop_module(real_ctx)
                self._executor.set_partial("running_after_start", True)
                self._executor.set_partial("timed_out", True)
                self._executor.set_partial("wait_progress", progress.as_dict())
                why = (f"AutoApproach did not reach the setpoint within "
                       f"{timeout_s:.0f}s — stopped the module。"
                       f"{progress.motion_text()}"
                       + (" 仍在推进却被预算砍断 —— 这不是「卡住」,"
                          "**再调一次 AutoApproach 会从当前粗动位置继续**"
                          "(必要时把 wait_timeout_s 调大)。"
                          if progress.z_travel_m > 0 else ""))
                return SkillResult(
                    skill_name=_PHASE_WAIT, success=False,
                    error=why + self.stop_note(),
                    data={"wait_progress": progress.as_dict(),
                          "stop_failures": list(self._stop_failures)})
            _t.sleep(poll_interval)

    # Z 采样与状态轮询使用各自的频率，以平衡链路开销与运动进度可见性。
    # 设为零表示关闭额外的 Z 进度采样。
    _z_sample_every_s = 1.0

    def _read_z(self, real_ctx) -> "float | None":
        """当前 Z 压电位置;读不到返回 ``None``。永不抛异常。

        **不从监控库拿**,虽然 aux 采集器正好也在 1 Hz 读同一个量:那个采集器可以是
        关着的(出厂就关),而一个只在监控开着时才有的进展判据,会在最需要它的那台机器上
        恰好不存在。这里多一个来回,换的是「这条判据自己站得住」。
        """
        try:
            rec = real_ctx.safe_call("ZCtrl_ZPosGet")
        except Exception:  # noqa: BLE001 — 进展观测失败不该带走进针
            return None
        if getattr(rec, "error", ""):
            return None
        return self._first_value(getattr(rec, "return_value", None))

    def _confirm_stopped(self, real_ctx, progress: "WaitProgress") -> bool:
        """复读 OnOffGet 确认模块停止，True 表示保留停止判断。
        单次零读数可能是瞬态；后续会判进针失败并发送停止命令，因此先确认。
        复读失败时保留先前停止读数，避免坏链路使等待无限持续。
        status_flap_n 记录复读发现的状态翻转，供调用方检查这一防线是否触发。
        """
        try:
            rec = real_ctx.safe_call("AutoApproach_OnOffGet")
        except Exception:  # noqa: BLE001
            return True
        self._call_log.append(rec)
        if getattr(rec, "error", ""):
            return True
        again = self._parse_running(getattr(rec, "return_value", None))
        if again is True:
            progress.status_flap_n += 1
            return False
        if again is None:
            # 复读回来一个读不懂的包。**结论不变**(同 TCP 报错那条:不推翻已经
            # 读到的那个 0),但它和「复读也说停了」不是一回事 —— 前者要查
            # TCP/parser,后者才是仪器真的停了。分开记,别混进 status_flap_n。
            progress.status_unreadable_n += 1
        return True

    def _stop_module(self, real_ctx) -> None:
        """Best-effort AutoApproach_OnOffSet(0) — never raises.

        ``allow_on_abort``: this is the cleanup that runs BECAUSE of an abort, so
        it must pass the post-abort hardware gate. (It would pass anyway — the
        gate recognises OnOffSet(0) as a stop — but saying so explicitly keeps
        the intent obvious and survives any future tightening of that policy.)

        **「没停下来」必须留下痕迹**(2026-08-10)。原来整段包在 ``except: pass``
        里:停机命令根本没下发,而调用方拿到的东西和停成功时**一模一样** ——
        接下来它只会报「进针失败」,一个字都不会提「而且模块可能还在走」。
        进针模块还在跑意味着针尖还在往表面压,这是本文件里代价最高的一种不知情。

        失败塞一条**带 error 的合成记录**进 ``_call_log``:``nanonis_calls`` 随
        SkillResult 一起回去,所以三个调用点一行都不用改;再记进
        ``_stop_failures``,由调用方顶进用户读的那句话(见 ``stop_note``)。
        """
        # 懒初始化:这个方法也可能在 `_run_approach_graph` 之外被调到(旧测试、
        # 直接驱动某个 phase)。一个「记录失败」的机制自己抛 AttributeError,
        # 就把它要报告的那次失败一起吞了。
        if not hasattr(self, "_stop_failures"):
            self._stop_failures = []
        try:
            try:
                stop = real_ctx.safe_call("AutoApproach_OnOffSet", 0,
                                          allow_on_abort=True)
            except TypeError:
                # Contexts that predate the kwarg (older fakes/tests).
                stop = real_ctx.safe_call("AutoApproach_OnOffSet", 0)
            self._call_log.append(stop)
            err = str(getattr(stop, "error", "") or "").strip()
            if err:
                # TCP 层没抛,但仪器拒了 —— 同样是「没停下来」。
                self._stop_failures.append(f"AutoApproach_OnOffSet(0): {err}")
                logger.error("进针模块停机命令被拒: %s", err)
        except Exception as exc:  # pragma: no cover - defensive
            self._stop_failures.append(f"AutoApproach_OnOffSet(0): {exc}")
            self._call_log.append(NanonisCallRecord(
                method="AutoApproach_OnOffSet", args=(0,),
                error=f"停机命令未能下发: {exc}"))
            logger.error("进针模块停机命令未能下发: %r", exc)

    def stop_note(self) -> str:
        """停机没做成时顶进 error 文本的一行。空串 = 命令确实发出去了。"""
        if not getattr(self, "_stop_failures", None):
            return ""
        return ("\n⚠️ **进针模块的停机命令没有成功下发**("
                + "；".join(self._stop_failures)
                + ")。模块可能仍在推进针尖 —— 请立即到 Nanonis 的 Auto Approach "
                  "面板确认它已停止,不要假设本技能已经把它停下了。")

    # Settle knobs, on the class so tests can shrink the window without
    # pretending the hardware is instant (same discipline as _poll_interval_s).
    _engage_interval_s = _ENGAGE_INTERVAL_S
    _engage_budget_s: "float | None" = None

    def _judge_engagement(self, real_ctx) -> EngageVerdict:
        """Is there tunnelling, judged from readings that agree with each other?

        Used to disambiguate an ultra-fast completed approach (running went 1→0
        before our first poll) from a rejected start, and to verify that a
        stopped module actually reached the setpoint rather than exhausting its
        range. Only the LAST read pair is added to the call log — a 20 s window
        is 20 round-trips, and burying the result under its own polling is the
        mistake ``_z_settle`` already declined to make."""
        last: list = []

        def _read() -> "tuple[float | None, float | None]":
            cur_rec = real_ctx.safe_call("Current_Get")
            sp_rec = real_ctx.safe_call("ZCtrl_SetpntGet")
            last[:] = [cur_rec, sp_rec]
            return (self._first_value(getattr(cur_rec, "return_value", None)),
                    self._first_value(getattr(sp_rec, "return_value", None)))

        v = settle_engagement(
            _read,
            interval_s=float(getattr(self, "_engage_interval_s", _ENGAGE_INTERVAL_S)),
            budget_s=getattr(self, "_engage_budget_s", None),
            check_abort=getattr(real_ctx, "check_abort", None),
        )
        self._call_log.extend(last)
        return v

    @staticmethod
    def _engage_failure_text(v: EngageVerdict, *, stopped: bool,
                             progress: "WaitProgress | None" = None) -> str:
        """Report measured evidence without assigning unobserved failure causes.

        Present the engagement verdict first. Report approach-module duration
        and current-verification-window duration separately, followed by the
        current evidence. See :class:`WaitProgress`.
        """
        head = ("AutoApproach 模块已停止" if stopped
                else "AutoApproach 在宽限窗口内始终报告未运行(启动可能被拒绝)")
        motion = f"{progress.motion_text()}" if progress is not None else ""
        if v.aborted:
            return (f"{head},进针判定被中止 —— **进针状态未知**,"
                    f"不要按「已进针」继续。{motion}{v.evidence()}")
        if v.engaged is None:
            # NOT the same sentence as engaged=False. See EngageVerdict.
            return (f"{head},但**判不出**有没有隧穿电流:预算耗尽前始终没有出现"
                    f"{_ENGAGE_AGREE_N} 次一致的读数"
                    + (f"(其中 {v.unreadable_n} 次根本读不到电流/设定点)"
                       if v.unreadable_n else "")
                    + f"。{motion}{v.evidence()}。**这不等于没进针**,也不等于进针了 —— "
                      "读数一直在动。不要扫图;先看反馈是否合上、电流量程是否合适,"
                      "若本机反馈确实较慢,到设置页把「退针 Z 稳定预算」"
                      "(z_settle_timeout_s)调大。")
        return (f"{head},且电流**稳定地**没有达到进针判据 —— 针尖未进入隧穿,"
                f"不要扫图。{motion}{v.evidence()}。")

    @staticmethod
    def _first_value(rv):
        if isinstance(rv, (list, tuple)) and len(rv) > 2:
            inner = rv[2]
            if isinstance(inner, (list, tuple)) and inner:
                try:
                    return float(inner[0])
                except (TypeError, ValueError):
                    return None
        return None

    def _phase_verify_status(self, real_ctx) -> SkillResult:
        """Final readback so the composite's data includes the latest status."""
        rec = real_ctx.safe_call("AutoApproach_OnOffGet")
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_VERIFY,
                success=False,
                error=f"verify OnOffGet failed: {rec.error}",
            )
        running = self._parse_running(rec.return_value)
        # ``None`` 原样往下传,**不折成 False**。这一相是进针跑完之后的记账式
        # 回读,所以读不懂**不该**让整次(可能已经成功的)进针失败 —— 那是拿一次
        # 读故障去否定一件已经做成的事。但它也绝不能写成「模块已停」:
        # `run_composite` 顶层那个 `running` 会照样把 None 带出去。
        self._executor.set_partial("final_running", running)
        return SkillResult(
            skill_name=_PHASE_VERIFY,
            success=True,
            data={"running": running},
        )

    @staticmethod
    def _parse_running(rv) -> bool:
        return _parse_running(rv)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        """切到进针参数组 → 跑图 → **放回调用前的值**(缺陷⑫)。

        为什么落在这儿而不是加两个 step:`GraphExecutor` 某一步失败就停下,「最后
        那个 step」根本不会跑 —— 一个不保证执行的收尾不是 finally,是一句看起来像
        finally 的话。而增益是我们改的,失败和中止时**更**需要放回去。

        `finally` 里放回:中止时异常穿过去(由 `CompositeSkill.execute` 收成
        `fail(...)`),放回照样发生 —— 与⑪的「中止不动手」**刻意相反**,判据是
        「这个状态是不是我改的」,见 `_preflight` 里那段。
        """
        # 每次调用重置逐趟状态。**放在这里而不是图体里**:这是「一次调用」的入口,
        # 而技能实例可能被复用 —— 「上一趟调制是关的」不是这一趟的事实。一个不重置的
        # 闩会把串扰报告永久关掉,而且没有任何症状(它本来就是一句安静的报告)。
        self._crosstalk_off = False
        self._crosstalk_last_t = None

        snapshot, note = apply_approach_preset(context, skill_name="AutoApproach")
        try:
            res = self._run_approach_graph(context, params)
        finally:
            note.update(restore_zctrl(context, snapshot, skill_name="AutoApproach"))
        if res.data is None:
            res.data = {}
        res.data.update(note)
        return res

    def _run_approach_graph(self, context, params: dict) -> SkillResult:
        """图本体。**这个类不走 `_base._graph_execute`** —— 见下面 sidecar 那段。"""
        self._call_log: list[NanonisCallRecord] = []
        #: 停机命令里**没能下发或被仪器拒掉**的那些(见 :meth:`_stop_module`)。
        self._stop_failures: list[str] = []
        try:
            self._wait_timeout_s = float(
                (params or {}).get("wait_timeout_s", self._DEFAULT_WAIT_TIMEOUT_S))
        except (TypeError, ValueError):
            self._wait_timeout_s = self._DEFAULT_WAIT_TIMEOUT_S
        wrapped = _AutoApproachPhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial_default("approach_started", False)
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))

        data: dict[str, Any] = {
            "approach_started": bool(
                executor.progress.partial_data.get("approach_started", False)),
            # ``None`` = 没问出来(回读没跑,或回包读不懂)。**别 bool() 它** ——
            # 那正是 A4 那条缺陷的出口:一次没做成的观测会变成「模块已停」。
            "running": (None if executor.progress.partial_data.get(
                "final_running") is None else bool(
                    executor.progress.partial_data["final_running"])),
            # 等待相的进展观测抬到顶层:埋在 `_progress.partial_data` 里的数字,
            # 读结果的人(和 agent)不会去翻。
            "wait_progress": executor.progress.partial_data.get("wait_progress"),
            "_progress": executor.progress.to_dict(),
            # 缺陷⑬:用户喊停要以本来面目到达下游,而不是变成一次「失败」。
            **abort_facts(executor.progress),
        }

        # This override bypasses _base._graph_execute, so it must clear its own
        # completed step sidecar. Only an aborted/interrupted run retains the
        # sidecar for crash-resume; completed runs must not resume-skip phases.
        if not executor.progress.aborted:
            executor.clear_sidecar()

        if not all_good:
            return SkillResult(
                skill_name=self._skill_name(),
                success=False,
                error=executor.progress.aborted_reason or "AutoApproach aborted",
                data=data,
                nanonis_calls=list(self._call_log),
            )
        return SkillResult(
            skill_name=self._skill_name(),
            success=True,
            data=data,
            nanonis_calls=list(self._call_log),
        )


class WithdrawTip(BaseSkill):
    """Withdraw tip fully."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WithdrawTip",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把针尖从表面完全退开。",
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["tip", "withdraw", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # ZCtrl_Withdraw(Wait_until_finished, Timeout_ms). The old (0, 1) meant
        # "don't wait, 1 ms timeout" — the call returned withdrawn:True while the
        # tip was STILL climbing, so a coarse XY move could start before the tip
        # was clear. Wait until the withdraw actually finishes
        # (indefinite timeout) before reporting it done.
        record = context.safe_call("ZCtrl_Withdraw", 1, -1)
        if record.error:
            return SkillResult(
                skill_name="WithdrawTip",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="WithdrawTip",
            success=True,
            data={"withdrawn": True},
            nanonis_calls=[record],
        )


class StopAutoApproach(BaseSkill):
    """Emergency-stop the Nanonis auto-approach module (AutoApproach_OnOffSet 0).

    The counterpart to StopMotor for the current-feedback approach. Before this
    there was NO software path to halt a running approach — a runaway coarse
    approach could only be stopped by physically reaching the Nanonis GUI. AUTO
    (like StopMotor): stopping motion is always safe and must never be gated."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopAutoApproach",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=("立即停止 auto-approach 流程"
                         "（AutoApproach_OnOffSet 0）。用于紧急叫停正在跑的粗动 "
                         "approach —— 永远安全，从不设闸。"),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["approach", "stop", "safety", "emergency"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("AutoApproach_OnOffSet", 0)
        if record.error:
            return SkillResult(
                skill_name="StopAutoApproach",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopAutoApproach",
            success=True,
            data={"stopped": True},
            nanonis_calls=[record],
        )


class GetAutoApproachStatus(BaseSkill):
    """Get the on/off status of the auto-approach procedure."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetAutoApproachStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 auto-approach 流程当前是否在运行。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["approach", "status", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("AutoApproach_OnOffGet")
        if record.error:
            return SkillResult(
                skill_name="GetAutoApproachStatus",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # AutoApproach.OnOffGet ResponseTypes=["H"] -> parsed[2][0] is the
        # Status (0=Off, 1=running). Reading parsed[0] (the empty error
        # string) made this silently report "not running" on real hardware.
        running = _parse_running(record.return_value)
        if running is None:
            # 解码失败必须返回未知，不能用 False 代替；否则调用方会误以为模块已停止。
            return SkillResult(
                skill_name="GetAutoApproachStatus",
                success=False,
                error=("AutoApproach_OnOffGet 回来了,但状态位读不懂"
                       f"(return_value={record.return_value!r})—— "
                       "这**不是**「没在进针」,是没问出来。"),
                data={"running": None},
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetAutoApproachStatus",
            success=True,
            data={"running": running},
            nanonis_calls=[record],
        )


class ApproachTip(BaseSkill):
    """Smart, safe '进针': establish tunnelling by the LEAST-risky means available.

    进针 is inherently ambiguous — the tip may already be within tunnelling range
    (just turn feedback ON) or far away (needs the coarse current-feedback approach).
    The choice carries tip-crash risk, so it must NOT hinge on the LLM re-reading a
    flag and escalating by hand. This skill decides deterministically + safely:

      1. TryEngageController — turn the Z-controller ON and watch for tunnelling
         current (NO motor motion). If |current| reaches ~setpoint → engaged, done.
         This also transparently handles the "already tunnelling" case.
      2. ONLY if feedback can't reach tunnelling (needs_auto_approach) → AutoApproach,
         the Nanonis current-feedback approach module. It hardware-STOPS at the
         setpoint, so there is NO tip-crash risk — this is deliberately NOT the
         open-loop coarse Z stepper (MotorMove z-approach), which stays behind the
         human-approval gate and is never touched here.

    Route a plain '进针 / engage / approach the tip' here. For an explicit coarse
    '自动粗逼近' call AutoApproach directly; to only try feedback (no approach) call
    TryEngageController.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ApproachTip",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # Both phases are current-feedback protected (engage = no motor;
            # AutoApproach stops at setpoint). The one crash-risky action —
            # open-loop coarse MotorMove z-approach — is NEVER invoked here and
            # stays human-gated, so this smart dispatcher is AUTO.
            safety_level=SafetyLevel.AUTO,
            description=(
                "面对一句朴素的「进针」，用**安全**的方式建立隧道：先试着让 "
                "Z-controller 进上（feedback 开，不动马达）；**只有**当这条路"
                "够不到隧道时，才退回到带电流反馈的 "
                "AutoApproach（Nanonis 会在 setpoint 处停住 —— 不会撞针）。"
                "含糊的「进针 / engage / 把针进上」这类请求，**默认**就用它。"
            ),
            parameters=[
                ParameterSpec(
                    name="settle_s", type="float",
                    description=("engage 阶段里，一边轮询电流、一边让 feedback "
                                 "稳定下来的秒数。"),
                    required=False, default=1.5, min_value=0.1, max_value=30.0),
            ],
            estimated_duration_s=60.0,
            rollback_skill="WithdrawTip",
            composition_level=2,
            tags=["approach", "engage", "tip", "进针", "smart"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        """外壳:切进针参数组 → 跑进针 → 放回去(缺陷⑫)。主体在 :meth:`_approach`。

        套在最外层而不是只套在 AutoApproach 上,是因为**第一阶段就可能进成**
        (TryEngageController 直接建立隧穿,根本不走粗动)。那一段同样受增益快慢
        影响,而它整段都在 AutoApproach 之外。

        嵌套是免费的:里层看到增益已经是进针组的值,就不写也不需要放回。
        """
        snapshot, note = apply_approach_preset(context, skill_name="ApproachTip")
        try:
            res = self._approach(context, params)
        finally:
            note.update(restore_zctrl(context, snapshot, skill_name="ApproachTip"))
        if res.data is None:
            res.data = {}
        res.data.update(note)
        return res

    def _approach(self, context, params: dict) -> SkillResult:
        settle_s = float(params.get("settle_s", 1.5))
        steps: list = []

        # ── Phase 1: safe engage (feedback ON, no motor). Also detects the
        #    "already tunnelling" case (peak current ≥ threshold → engaged). ──
        r_eng = context.run("TryEngageController", {"settle_s": settle_s})
        eng_data = getattr(r_eng, "data", None) or {}
        steps.append({"skill": "TryEngageController",
                      "success": bool(getattr(r_eng, "success", False)),
                      "engaged": eng_data.get("engaged"),
                      "needs_auto_approach": eng_data.get("needs_auto_approach"),
                      "error": getattr(r_eng, "error", None)})

        if not getattr(r_eng, "success", False):
            # This is a REFUSAL to escalate, not just a failed step: engage
            # could not prove the Z-feedback state, and running the coarse
            # approach without that proof is the crash case. Make it stick, or
            # the very next tool call re-enters the same coarse approach
            # through AutoApproach with no gate at all (
            # 13:50:40 refused → 13:52:53 AutoApproach, 133 s later).
            err = f"engage phase failed: {getattr(r_eng, 'error', 'unknown')}"
            esc.record_approach_refusal(err, owner=_chain_scope(context))
            return SkillResult(
                skill_name="ApproachTip", success=False,
                error=err,
                data={"phase": "engage", "steps": steps})

        if eng_data.get("engaged"):
            esc.clear_approach_refusal("ApproachTip engaged via Z-controller",
                                      owner=_chain_scope(context))
            didv_cal = self._didv_calibration_window(
                context, setpoint_a=eng_data.get("setpoint_a"))
            # Second opinion + distance trend. ADDITIVE: a crash verdict from the
            # amplitude channel downgrades success to a warning in the message
            # but does not overrule the current-based verdict, and an absent
            # channel adds nothing at all.
            evidence = _approach_evidence(context)
            return SkillResult(
                skill_name="ApproachTip", success=True,
                data={"engaged": True, "method": "engage_controller",
                      "auto_approach_used": False,
                      "peak_current_a": eng_data.get("peak_current_a"),
                      "setpoint_a": eng_data.get("setpoint_a"),
                      "didv_at_contact_v": didv_cal,
                      "message": ("Tip already within tunnelling range — engaged via "
                                  "the Z-controller; no coarse approach needed."
                                  + _evidence_warning(evidence)),
                      "steps": steps, **evidence})

        # ── Phase 2: feedback couldn't reach tunnelling → tip is far. Use the
        #    current-feedback AutoApproach (safe: hardware-stops at setpoint). ──
        if not eng_data.get("needs_auto_approach"):
            # Ambiguous outcome (not engaged, not clearly asking for approach) →
            # do NOT drive the approach on a maybe. Stop and report. Same as the
            # engage-failure branch: this is a refusal to escalate and it has to
            # outlive this tool call ().
            err = ("engage did not establish tunnelling and did not flag "
                   "needs_auto_approach — stopping rather than approaching on a "
                   "maybe; check the tip/Z state.")
            esc.record_approach_refusal(err, owner=_chain_scope(context))
            return SkillResult(
                skill_name="ApproachTip", success=False,
                error=err,
                data={"engaged": False, "auto_approach_used": False, "steps": steps})

        # Escalation DECIDED, on fresh evidence — this supersedes any earlier
        # refusal (and is what makes "just run ApproachTip again" the escape
        # from the gate rather than "wait out the TTL").
        esc.clear_approach_refusal("ApproachTip escalating to AutoApproach",
                                  owner=_chain_scope(context))
        r_app = context.run("AutoApproach", {})
        app_data = getattr(r_app, "data", None) or {}
        steps.append({"skill": "AutoApproach",
                      "success": bool(getattr(r_app, "success", False)),
                      "data": app_data, "error": getattr(r_app, "error", None)})
        if not getattr(r_app, "success", False):
            return SkillResult(
                skill_name="ApproachTip", success=False,
                error=f"auto-approach phase failed: {getattr(r_app, 'error', 'unknown')}",
                data={"engaged": False, "method": "auto_approach",
                      "auto_approach_used": True, "steps": steps})

        # Independently verify engagement using consecutive actual-current reads,
        # regardless of the composite result. Return the measured evidence so
        # callers can assess the verdict and an unsettled transient stays visible.
        v = self._verify_engagement(context)
        cur, sp = v.current_a, v.setpoint_a
        if v.engaged is not True:
            return SkillResult(
                skill_name="ApproachTip", success=False,
                error=self._verify_failure_text(v),
                data={"engaged": False, "method": "auto_approach",
                      "auto_approach_used": True,
                      "measured_current_a": cur, "setpoint_a": sp,
                      "engagement": v.as_dict(),
                      "steps": steps})
        didv_cal = self._didv_calibration_window(context, setpoint_a=sp)
        evidence = _approach_evidence(context)
        return SkillResult(
            skill_name="ApproachTip", success=True,
            data={"engaged": True, "method": "auto_approach", "auto_approach_used": True,
                  "measured_current_a": cur, "setpoint_a": sp,
                  "didv_at_contact_v": didv_cal,
                  "engagement": v.as_dict(),
                  "message": ("Tip was too far to engage by feedback alone — ran the "
                              "current-feedback AutoApproach; tunnelling verified at "
                              f"|I|={_fmt_a(cur)} vs setpoint {_fmt_a(sp)} "
                              f"({v.agreed_n} 次一致读数)."
                              + _evidence_warning(evidence)),
                  "steps": steps, **evidence})

    # Same knobs as AutoApproach, same reason (tests shrink the window).
    _engage_interval_s = _ENGAGE_INTERVAL_S
    _engage_budget_s: "float | None" = None

    def _verify_engagement(self, context) -> EngageVerdict:
        """Post-approach 进针 verdict from agreeing reads, via the READ skills."""
        return settle_engagement(
            lambda: self._read_current_and_setpoint(context),
            interval_s=float(getattr(self, "_engage_interval_s", _ENGAGE_INTERVAL_S)),
            budget_s=getattr(self, "_engage_budget_s", None),
            check_abort=getattr(context, "check_abort", None),
        )

    @staticmethod
    def _verify_failure_text(v: EngageVerdict) -> str:
        """Report verification evidence without guessing unmeasured causes."""
        if v.aborted:
            return ("post-approach verification 被中止 —— **进针状态未知**,"
                    f"不要按「已进针」继续。{v.evidence()}")
        if v.engaged is None:
            return ("post-approach verification 判不出结果:预算耗尽前始终没有出现 "
                    f"{_ENGAGE_AGREE_N} 次一致的电流读数"
                    + (f"(其中 {v.unreadable_n} 次根本读不到电流/设定点)"
                       if v.unreadable_n else "")
                    + f"。{v.evidence()}。**这不等于没进针**,也不等于进针了。"
                      "不要扫图;先确认反馈已合上、电流量程合适。")
        return ("post-approach verification FAILED:电流**稳定地**没有达到进针判据,"
                f"针尖未进入隧穿,不要扫图。{v.evidence()}。")

    @staticmethod
    def _read_current_and_setpoint(context) -> tuple[float | None, float | None]:
        """Best-effort (current_a, setpoint_a) via the READ skills; None on failure."""
        cur = sp = None
        try:
            r = context.run("GetCurrent", {})
            if getattr(r, "success", False):
                cur = (getattr(r, "data", None) or {}).get("current_a")
        except Exception:  # noqa: BLE001
            pass
        try:
            r = context.run("GetSetpoint", {})
            if getattr(r, "success", False):
                sp = (getattr(r, "data", None) or {}).get("setpoint_a")
        except Exception:  # noqa: BLE001
            pass
        return (
            float(cur) if isinstance(cur, (int, float)) else None,
            float(sp) if isinstance(sp, (int, float)) else None,
        )

    #: 标定窗:开调制之后等多久再读。qPlus/lock-in 的解调滤波需要几个时间常数才稳,
    #: 而一个在瞬态里读到的 dI/dV 会被 EWMA 永久带进标定库。类属性,好让测试缩短它。
    _didv_settle_s = 2.5

    def _didv_calibration_window(self, context, *, setpoint_a):
        """在进针完成后的短窗口记录 dI/dV 标定，再恢复调制状态。
        进针主过程关闭调制以减少判据干扰，而 dI/dV 标定只接受调制已开启时的读数。
        因此此处临时开启调制、获取标定，然后关闭由此处开启的调制。
        原本已经开启的调制保持原状，遵守谁开启谁恢复的约束。
        标定是进针的副产物：异常不能改变主进针结论，也不能覆盖调用前状态。
        """
        import time as _t

        opened = False
        try:
            if ip.get_config("lockin_signal_index", None) is None:
                return None                     # 没有 dI/dV 通道 ⇒ 无窗可开
            if not self._modulation_confirmed_on(context):
                from mast.core.lockin_presets import PresetRejected, resolve

                try:
                    preset = resolve()
                except PresetRejected:
                    return None                 # 档案没配 ⇒ 不编数字,不开窗
                if not preset.usable:
                    return None
                res = context.run("ConfigureLockIn",
                                  preset.skill_params(mod_on=True))
                if not getattr(res, "success", False):
                    logger.info("engage 标定窗:开调制失败,跳过标定:%s",
                                getattr(res, "error", ""))
                    return None
                opened = True
                _t.sleep(max(0.0, float(self._didv_settle_s)))
            return self._record_didv_calibration(context, setpoint_a=setpoint_a)
        except Exception:  # noqa: BLE001 — 标定永远不该影响进针结论
            logger.debug("engage 标定窗失败(跳过)", exc_info=True)
            return None
        finally:
            if opened:
                try:
                    close_modulation(context, skill_name="ApproachTip(标定窗)")
                except Exception:  # noqa: BLE001
                    logger.warning("engage 标定窗:调制关不回去了 —— 请手动确认")

    def _record_didv_calibration(self, context, *, setpoint_a):
        """成功进针后记录 dI/dV-at-contact 标定（best-effort，供下轮判距离）。

        读当前 lock-in R (dI/dV) + bias，经 instrument_profile.set_calibration()
        EWMA 写回跨 run 存储（绑定 bias/setpoint/调制幅度）。lock-in 信号索引未
        配置 / 读不到 bias 时静默跳过。**绝不因标定失败影响进针成功**——这是
        进针的副产物，不是它的前置条件。返回记录到的 dI/dV（None=未记录）。
        """
        try:
            idx = ip.get_config("lockin_signal_index", None)
            if idx is None:
                return None
            # 调制没开的时候,这一路读到的**不是 dI/dV** —— 是一个没有被调制驱动
            # 的通道的读数。记进标定库就是把一个假值当基准存下来,而它之后每一次
            # 被引用都不会自己声明「我是在调制关着时量的」。
            #
            # 这条从 2026-08-05 起是必需的:那天起「用完即关」让进针类流程在开跑前
            # 自动关掉调制(_preflight.ensure_modulation_off),所以走到这里时调制
            # **通常是关的** —— 不加这道闸,这次改动会把一个坏标定写进持久存储。
            # 读不到调制状态同样不记:「没读到」不是「开着」。
            if not self._modulation_confirmed_on(context):
                return None
            rec = context.safe_call("Signals_ValGet", int(idx), 0)
            didv = self._first_scalar(getattr(rec, "return_value", None))
            if didv is None:
                return None
            didv = abs(float(didv))
            brec = context.safe_call("Bias_Get")
            bias_v = self._first_scalar(getattr(brec, "return_value", None))
            if bias_v is None:
                return didv  # 无法绑定 bias → 不写标定（标定值必须绑定 bias）
            sp = setpoint_a
            if sp is None:
                sprec = context.safe_call("ZCtrl_SetpntGet")
                sp = self._first_scalar(getattr(sprec, "return_value", None))
            if sp is None:
                return didv
            ip.set_calibration(didv, bias_v=float(bias_v), setpoint_a=float(sp),
                               mod_amp_v=ip.get_config("lockin_mod_amp_v", 0.02))
            return didv
        except Exception:  # noqa: BLE001 — calibration must never break 进针
            return None

    @staticmethod
    def _modulation_confirmed_on(context) -> bool:
        """True only when the rig SAYS the lock-in modulation is on.

        Three-valued underneath, collapsed deliberately toward "do not record":
        on / off / unreadable. Only the first one licenses writing a dI/dV
        calibration — an unreadable modulation state is not evidence that the
        signal we are about to read means anything."""
        try:
            rec = context.safe_call("LockIn_ModOnOffGet", 1)
            if getattr(rec, "error", ""):
                return False
            rv = getattr(rec, "return_value", None)
            vals = rv[2] if isinstance(rv, (list, tuple)) and len(rv) > 2 else None
            raw = vals[0] if isinstance(vals, (list, tuple)) and vals else vals
            if isinstance(raw, (list, tuple)) and raw:      # 1-元组形态(§2.21)
                raw = raw[0]
            return bool(int(raw)) if raw is not None else False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _first_scalar(rv):
        """First scalar from a Nanonis (header, body, [vals]) triplet, or None."""
        if isinstance(rv, (list, tuple)) and len(rv) > 2:
            inner = rv[2]
            if isinstance(inner, (list, tuple)) and inner:
                try:
                    return float(inner[0])
                except (TypeError, ValueError):
                    return None
        return None
