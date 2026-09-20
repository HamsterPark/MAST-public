"""Wait for stable Z feedback before interpreting a Z reading.

A fixed delay can sample the middle of a feedback ramp. Loop response depends on
gain, temperature and tip, so use convergence plus a configured time budget.

Stable Z with current supports a tunnel-junction measurement. Stable Z at the
current floor may instead indicate a piezo rail: report a directional bound,
not a measured displacement. Moving Z supports neither conclusion.

Each comparison begins from the withdraw position so baseline and ladder rungs
share the same starting condition. Stopped alone is insufficient: a loop that
has not started is also stopped. Require a live junction or evidence of actual
piezo travel before accepting convergence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from mast.core import instrument_profile as ip

logger = logging.getLogger(__name__)

#: Below this a "current" is preamp noise, not tunnelling. Same value both
#: composites already use as their engagement floor — one instrument, one floor.
_NOISE_FLOOR_A = 1e-12

#: The convergence band is DERIVED from the decision threshold
#: (``z_recede_min_nm``) rather than configured beside it, because the two are
#: not independent: the band has to be tighter than the threshold, or a reading
#: could be declared "settled" while still drifting far enough to flip the
#: verdict it is about to feed. Half is the loosest value for which residual
#: motion cannot account for a full threshold's worth of the answer. Deriving
#: also means an operator who raises the threshold for a noisy rig gets a
#: matching band for free — a second config key would silently stay at the old
#: value, which is how limits and the units they are measured in drift apart.
_TOL_FRACTION_OF_THRESHOLD = 0.5

#: Poll cadence and window length. 0.1 s / 5 samples = a 0.5 s window. For
#: scale: the measured ramp above is ~110 nm/s, i.e. ~55 nm of travel per
#: window against a default 0.5 nm band — two orders of magnitude of separation
#: between "travelling" and "arrived", which is what makes the criterion robust
#: to Z noise rather than sensitive to it.
_DEFAULT_POLL_S = 0.1
_DEFAULT_WINDOW_N = 5

_STATE_LABELS = {
    "tracking": "已稳定,反馈正握着隧道结(Z 是真实的间距读数)",
    "out_of_range": "已稳定在压电极限,量程内没有表面(Z 是下界,不是距离)",
    "moving": "超时:Z 还在走,没有可读的稳定值",
    "unreadable": "读不到 Z",
    "aborted": "被中止",
}


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


@dataclass
class ZSettle:
    """Outcome of one settle. ``usable`` is the only thing a judge may branch on.

    Everything else exists so that a run which could NOT produce a reading says
    what it saw instead of quietly handing over a number.
    """

    z_m: "float | None" = None
    current_a: "float | None" = None
    setpoint_a: "float | None" = None
    # Z 稳定等待需保留失败原因，读取不足不能静默折叠为不满足判据。
    setpoint_why: str = ""
    settled: bool = False
    #: tracking | out_of_range | moving | unreadable | aborted
    state: str = "unreadable"
    elapsed_s: float = 0.0
    samples: int = 0
    tol_m: float = 0.0
    timeout_s: float = 0.0
    #: Net drift across the final window — the quantity the criterion tests.
    drift_m: "float | None" = None
    #: Total travel seen since the settle began. Distinguishes "arrived" from
    #: "never started".
    excursion_m: "float | None" = None
    #: What ZCtrl_OnOffGet said at the start. None = could not read. Never gated
    #: on (the RT controller lags the write by design) — it is here so a timeout
    #: can say whether the loop was even closed.
    loop_confirmed_on: "bool | None" = None

    @property
    def usable(self) -> bool:
        """May the recede judgement compare this Z against another one?

        Only a converged reading qualifies. A mid-flight Z is not a worse
        measurement of the gap — it is a measurement of the elapsed time.
        """
        return (self.settled and self.z_m is not None
                and self.state in ("tracking", "out_of_range"))

    @property
    def at_rail(self) -> bool:
        """Settled with nothing within piezo reach (Z is a bound, not a gap)."""
        return self.settled and self.state == "out_of_range"

    @property
    def measures_gap(self) -> bool:
        """Is this reading a RULER — an actual tip–sample distance?

        Narrower than :attr:`usable` on purpose. A rail reading is usable (it can
        be compared, and it can still reveal an approach) but it is a bound, so a
        pair of them cannot tell "the stage moved out of reach" from "the stage
        never moved". Only two gap readings can, which is what lets a measured
        zero mean *zero displacement* rather than *no measurement* — the
        distinction the no-displacement guard is built on.
        """
        return self.settled and self.state == "tracking"

    def why(self) -> str:
        """One operator-facing line: what happened and, if it failed, what to do."""
        base = _STATE_LABELS.get(self.state, self.state)
        if self.state == "moving":
            rate = ""
            if self.drift_m is not None and self.samples > 1:
                per_window = abs(self.drift_m) * 1e9
                rate = f",最后一个窗口仍在以 ~{per_window:.1f} nm/窗口 移动"
            loop = ""
            if self.loop_confirmed_on is False:
                loop = ("。**实时控制器在开始时回报 Z 反馈是断开的** —— "
                        "先查 Z 反馈开关是否真的合上了,而不是加大预算")
            elif self.loop_confirmed_on is None:
                loop = "。(无法读回 Z 反馈开关状态,不能排除反馈根本没合上)"
            return (f"{base}(等了 {self.elapsed_s:.1f}s / 预算 {self.timeout_s:.1f}s,"
                    f"稳定判据 {self.tol_m * 1e9:.2f} nm{rate}){loop}")
        if self.state == "out_of_range":
            return (f"{base};已等 {self.elapsed_s:.1f}s,"
                    f"压电共走了 {abs(self.excursion_m or 0.0) * 1e9:.0f} nm")
        if self.state == "tracking":
            return (f"{base};已等 {self.elapsed_s:.1f}s,"
                    f"|I| = {abs(self.current_a or 0.0):.3g} A")
        if self.state == "unreadable":
            return (f"{base}(ZCtrl_ZPosGet 连试 {self.samples} 次都没有可解析的读数)"
                    " —— 这是读回链路的问题,不是等得不够久,加大预算没有用")
        return base

    def as_dict(self) -> dict[str, Any]:
        return {"z_m": self.z_m, "current_a": self.current_a,
                # 记录设定点读数及其缺失原因，让相对电流判据可追溯。
                "setpoint_a": self.setpoint_a,
                "setpoint_why": self.setpoint_why,
                "settled": self.settled, "state": self.state,
                "usable": self.usable, "elapsed_s": round(self.elapsed_s, 3),
                "samples": self.samples, "tol_m": self.tol_m,
                "timeout_s": self.timeout_s, "drift_m": self.drift_m,
                "excursion_m": self.excursion_m,
                "loop_confirmed_on": self.loop_confirmed_on,
                "why": self.why()}


def settle_timeout_s() -> float:
    """The rig's declared budget for the loop to find the surface.

    A BUDGET, not a physical constant — which is why exceeding it is reported
    with the measured drift rate beside it, the way ``ZControllerOnOff`` reports
    ``waited_s`` against the rig's own switch-off delay. Those two numbers are
    what separate "we gave up too early" from "the loop is not moving at all",
    and a timeout that prints neither leaves the operator guessing.
    """
    try:
        return float(ip.get_config("z_settle_timeout_s", 20.0))
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return 20.0


def settle_tolerance_m() -> float:
    """Convergence band, derived from the decision threshold. See the constant."""
    try:
        thresh = float(ip.get_config("z_recede_min_nm", 1.0)) * 1e-9
    except (TypeError, ValueError):  # pragma: no cover — defensive
        thresh = 1e-9
    return max(_TOL_FRACTION_OF_THRESHOLD * thresh, 0.0)


def settle_and_read_z(
    ctx,
    *,
    log: "list | None" = None,
    timeout_s: "float | None" = None,
    poll_interval_s: float = _DEFAULT_POLL_S,
    window_n: int = _DEFAULT_WINDOW_N,
    withdraw_first: bool = True,
) -> ZSettle:
    """Withdraw, close the loop, and wait for Z to stop travelling.

    Returns a :class:`ZSettle`. **It never raises and never guesses**: if the
    loop does not converge inside the budget the result says ``moving`` and
    carries no usable Z, rather than handing back the value it happened to be
    passing through.

    ``log`` collects the Nanonis records worth keeping. The poll reads are
    deliberately NOT logged — a 5 s settle at 10 Hz is 100 round-trips, and a
    hundred records per rung would bury the result (and the checkpoint) under
    its own diagnostics. The final reading is logged, and the summary numbers
    that matter (samples, elapsed, drift, excursion) ride on the ZSettle.
    """
    append = log.append if log is not None else (lambda _r: None)

    # A window of one has zero net drift by construction, so it would call the
    # very first reading converged — the defect, rebuilt out of an off-by-one.
    window_n = max(3, int(window_n))

    out = ZSettle(tol_m=settle_tolerance_m(),
                  timeout_s=float(settle_timeout_s() if timeout_s is None
                                  else timeout_s))

    # 0. Same initial condition every time — see docstring note 1. Without this
    #    the baseline and the rungs are not the same measurement, and two
    #    converged readings still are not comparable.
    if withdraw_first:
        append(ctx.safe_call("ZCtrl_Withdraw", 1, -1))

    # 1. Close the loop so the piezo goes looking for the surface.
    append(ctx.safe_call("ZCtrl_OnOffSet", 1))

    # 1b. Ask the REAL-TIME controller whether it is closed — for the record
    #     only. ``settle=False`` because we are about to poll for seconds
    #     anyway; a second waiting loop on top would buy nothing. And we do NOT
    #     gate on it: the RT controller lags the write by design (Nanonis'
    #     manual), so an OFF here at t=0 may be pure latency. If the loop really
    #     is open, Z will not move and the timeout below reports it — with this
    #     reading quoted, which is the sentence that tells an operator whether
    #     to check the wiring or raise the budget.
    try:
        from mast.skills.verify import verify_z_controller

        v = verify_z_controller(ctx, expect=True, settle=False)
        append(v["record"])
        out.loop_confirmed_on = v["on"] if v["verified"] else None
    except Exception:  # noqa: BLE001 — a diagnostic must not break the settle
        out.loop_confirmed_on = None

    # 相对危险阈值依赖实际 setpoint；读取缺失与数值为零须区分。
    rec_sp = ctx.safe_call("ZCtrl_SetpntGet")
    append(rec_sp)
    if rec_sp.error:
        out.setpoint_a = None
        out.setpoint_why = f"ZCtrl_SetpntGet 报错:{rec_sp.error}"
    else:
        out.setpoint_a = _first_val(rec_sp.return_value)
        if out.setpoint_a is None:
            out.setpoint_why = (
                f"ZCtrl_SetpntGet 回包读不懂(shape={type(rec_sp.return_value).__name__}"
                f",repr 前 120 字:{str(rec_sp.return_value)[:120]})")

    # 2. Poll Z until it stops travelling.
    t0 = time.monotonic()
    window: list[float] = []
    z_lo: "float | None" = None
    z_hi: "float | None" = None
    last_pair: tuple = ()
    reads_ok = 0
    check_abort = getattr(ctx, "check_abort", None)

    while True:
        if callable(check_abort) and check_abort():
            out.state = "aborted"
            out.elapsed_s = time.monotonic() - t0
            for rec in last_pair:
                append(rec)
            return out

        rec_z = ctx.safe_call("ZCtrl_ZPosGet")
        rec_c = ctx.safe_call("Current_Get")
        last_pair = (rec_z, rec_c)
        z = None if rec_z.error else _first_val(rec_z.return_value)
        cur = None if rec_c.error else _first_val(rec_c.return_value)
        out.elapsed_s = time.monotonic() - t0
        out.samples += 1

        if z is not None:
            reads_ok += 1
            out.z_m = z
            out.current_a = cur
            window.append(z)
            if len(window) > window_n:
                window.pop(0)
            z_lo = z if z_lo is None else min(z_lo, z)
            z_hi = z if z_hi is None else max(z_hi, z)
            out.excursion_m = z_hi - z_lo

            if len(window) >= window_n:
                # NET drift across the window, not peak-to-peak: random noise
                # does not accumulate into a net displacement but a ramp does,
                # so this detects MOTION rather than quietness. A noisy but
                # stationary Z passes; a smooth slow ramp does not.
                out.drift_m = abs(window[-1] - window[0])
                tracking = cur is not None and abs(cur) > _NOISE_FLOOR_A
                travelled = (out.excursion_m or 0.0) > out.tol_m
                # See docstring note 2: "not moving" is also true of a loop that
                # has not started. Require a live junction, or proof the piezo
                # actually went somewhere.
                if out.drift_m <= out.tol_m and (tracking or travelled):
                    out.settled = True
                    out.state = "tracking" if tracking else "out_of_range"
                    for rec in last_pair:
                        append(rec)
                    return out

        if out.elapsed_s >= out.timeout_s:
            break
        if out.samples >= window_n and reads_ok == 0:
            # A whole window's worth of tries and not one parseable Z. Waiting
            # out the rest of the budget cannot turn a dead readback into a
            # reading, and the caller needs the diagnosis now — "cannot read Z"
            # is a different repair from "this rig is slower than declared".
            break
        if poll_interval_s > 0:
            time.sleep(min(poll_interval_s,
                           max(0.0, out.timeout_s - out.elapsed_s)))

    out.state = "moving" if out.z_m is not None else "unreadable"
    for rec in last_pair:
        append(rec)
    logger.warning("Z settle timed out after %.2fs (state=%s, drift=%s)",
                   out.elapsed_s, out.state, out.drift_m)
    return out


__all__ = ["ZSettle", "settle_and_read_z", "settle_timeout_s",
           "settle_tolerance_m"]
