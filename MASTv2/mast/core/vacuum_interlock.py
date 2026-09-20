"""May the coarse piezo be driven right now? A pressure question, answered fail-closed.

THE PHYSICS
===========
A coarse stepper is driven by a few-hundred-volt sawtooth. In high vacuum that is
fine: the mean free path is enormous and there is nothing to ionise. At
atmosphere it is also mostly fine — that is why benches get tested in air. The
danger is **in between**. Around the Paschen minimum a few hundred volts is
enough to strike a discharge across the drive electrodes, and the arc tracks
across the piezo's insulation. The repo's own knowledge base has carried the
number as prose for a long time::

    corona_danger_zone_mbar: "10 to 1e-3"

i.e. **1e-3 … 10 mbar = 0.1 … 1000 Pa**. That band is not exotic: it is exactly
what a chamber passes through while pumping down and while venting — the moments
somebody is most likely to be moving things.

THE RULE, WHICH IS NOT ABOUT ANY PARTICULAR GAUGE
=================================================
    Permission requires POSITIVE EVIDENCE that the pressure is below the limit.
    A reading is evidence only while it lies INSIDE the gauge's usable band.
    Outside it — at either end — the gauge is not measuring, it is saturating,
    and a saturating gauge's number is not data.

Everything model-specific lives in two operator-set numbers: ``min_pa`` and
``full_scale_pa``. The logic above is the same for a DL-7, an ion gauge, a
Pirani, or something nobody has bought yet.

It is worth saying why, because the DL-7 makes it easy to believe otherwise: its
ceiling is 1e-1 Pa = 1e-3 mbar, which is *exactly* the bottom edge of the danger
band, so on the instrument "a valid in-range reading" and "below the danger band"
happen to coincide. That is a happy property of THESE DEFAULTS — it means a DL-7
rig is safe out of the box — and not a premise. Building on the coincidence
would have produced an interlock that quietly stops meaning anything the day
somebody fits a different gauge.

Where a different gauge really does change the answer is the BOTTOM of the band:

  * DL-7 / ion gauge, floor 5e-8 Pa — far below the permit limit. Bottoming out
    means "even lower than that", which is emphatically safe. Refusing a UHV
    chamber for being too clean would be absurd.
  * Pirani / capacitance manometer, floor ~0.5–1 Pa — ABOVE the permit limit and
    inside the discharge band. Bottoming out means "somewhere below 1 Pa", which
    includes 0.5 Pa, which is in the band. That reading proves nothing, and the
    number it emits (its floor, zero, or noise) looks exactly like an excellent
    vacuum — the same failure shape as the placeholder's 0.0, reached from the
    other direction.

So the under-range test is not "is it bottoming out" but "**is this gauge's floor
already below the limit**". A rough-vacuum-only gauge fails that permanently and
should; :func:`gauge_config_problem` says so once, plainly, instead of letting it
be discovered as a mysterious standing refusal.

SIX WAYS TO GET THIS WRONG, ALL OF WHICH ARE REAL
=================================================
1. ``PlaceholderSensor.read()`` returns ``value=0.0, status="unavailable"``. An
   interlock that reads ``.value`` and compares it against a threshold turns
   **no gauge installed** into **perfect vacuum** and waves the move through.
2. ``DL7VacuumSensor`` reports ``"Pa"``; the placeholder reports ``"mbar"``.
   Same number, 100× apart. Comparing without checking the unit lands you inside
   the danger band while believing you are two decades below it.
3. ``CoreRuntime._on_env_alarm`` deliberately ignores ``status="unavailable"`` —
   a gauge that drops off the bus must not abort an overnight run. Correct for
   an ALARM, backwards for an INTERLOCK. They are different questions and must
   not share a predicate.
4. The DL-7 frame parser does not validate the exponent field, so an over-range
   condition can decode to a plausible-looking small number. The band check
   below is what catches that; the driver will not raise.
5. A reading from five minutes ago says nothing about the pressure now. Age is
   part of the verdict, not a detail.
6. A gauge below its floor emits a small number that reads as an excellent
   vacuum. Harmless on a DL-7, fail-open on a rough-vacuum gauge (see above).

WHAT HAPPENS WITH NO GAUGE
==========================
Operator decision (2026-07-31): **默认拒绝，但用户可以签字**. Default mode is
``gauge_or_attest`` — a valid reading permits; otherwise the operator may sign a
time-limited statement ("已通大气" / "已抽至高真空但真空计不可用") and that
permits until it expires. The attestation is process-local and dies with the
process, like ``safety_escalation``'s refusal: a restart is a change of scene,
and re-signing costs ten seconds while a stale authorisation costs a stack.

``gauge_only`` refuses to accept a signature at all. ``off`` does not block —
but still computes and reports the verdict, so "we chose not to check" stays
visible instead of quietly becoming "there is nothing to check".

Dependency-light: stdlib only, no hardware import. The monitor is injected as a
callable, so the safety layer never depends on the environment layer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Constants: the ONE numeric source of truth for the discharge band ────────

#: Corona / Paschen danger band in Pa (= 1e-3 … 10 mbar). Referenced by
#: ``knowledge.safety_constraints`` instead of being restated there as prose.
CORONA_ZONE_PA: tuple[float, float] = (0.1, 1.0e3)

#: Default permit threshold: a decade below the band's lower edge, and
#: comfortably inside a cold-cathode gauge's range. Any real UHV STM sits many
#: decades below this.
DEFAULT_MAX_PRESSURE_PA = 1.0e-2

#: The installed gauge's USABLE BAND, in Pa. Defaults are the DL-7's
#: (5e-8 … 1e-1 Pa); any other gauge sets both from the settings page.
#:
#: The rule this band implements is gauge-independent and is the whole basis of
#: the interlock:
#:
#:     Permission requires POSITIVE EVIDENCE that the pressure is below the
#:     limit. A reading is evidence only while it lies inside the band. Outside
#:     it — either end — the gauge is not measuring, it is saturating, and a
#:     saturating gauge's number is not data.
#:
#: The DL-7's ceiling happening to coincide with the discharge band's lower edge
#: is a nice property of THESE DEFAULTS, not the design: it means a DL-7 rig is
#: safe out of the box. Nothing below depends on it.
DEFAULT_GAUGE_FULL_SCALE_PA = 1.0e-1
DEFAULT_GAUGE_MIN_PA = 5.0e-8
_OVERRANGE_FRACTION = 0.8
_UNDERRANGE_FACTOR = 1.25

#: A reading older than this cannot authorise anything.
DEFAULT_MAX_AGE_S = 60.0

#: Default lifetime of an operator's signature.
DEFAULT_ATTESTATION_TTL_S = 8 * 3600.0

MODES = ("gauge_or_attest", "gauge_only", "off")

ATTESTATION_REASONS: dict[str, str] = {
    "vented_to_atmosphere": "已通大气（腔体在常压）",
    "high_vacuum_gauge_unavailable": "已抽至高真空，但真空计不可用",
}

#: Sensor classes that are REAL pressure gauges. An allow-list, not a deny-list:
#: an unrecognised class is refused with its name in the message, which is a
#: one-line fix, whereas a deny-list silently accepts every future placeholder.
REAL_GAUGE_CLASSES: frozenset[str] = frozenset({"DL7VacuumSensor"})

#: Known stand-ins. Listed separately only so the refusal can say WHY.
PLACEHOLDER_CLASSES: frozenset[str] = frozenset({"PlaceholderSensor", "VacuumSensor"})

#: Unit → multiplier to Pa. Anything absent is refused rather than assumed.
_UNIT_TO_PA: dict[str, float] = {
    "pa": 1.0, "pascal": 1.0,
    "mbar": 100.0, "millibar": 100.0, "hpa": 100.0,
    "bar": 1.0e5,
    "torr": 133.322, "mmhg": 133.322,
    "mtorr": 0.133322, "micron": 0.133322,
}


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PressureSample:
    """One gauge reading, with everything needed to distrust it."""
    value: float
    unit: str
    status: str
    timestamp: str = ""          # ISO, as SensorReading carries it
    sensor_name: str = ""
    sensor_class: str = ""

    def age_s(self, now: float | None = None) -> float | None:
        if not self.timestamp:
            return None
        try:
            import datetime as _dt
            t = _dt.datetime.fromisoformat(str(self.timestamp)).timestamp()
        except (ValueError, TypeError):
            return None
        return max(0.0, (time.time() if now is None else now) - t)


@dataclass(frozen=True)
class Attestation:
    """An operator's signed, time-limited statement about the chamber."""
    reason: str
    signed_by: str = ""
    signed_at: float = 0.0
    ttl_s: float = DEFAULT_ATTESTATION_TTL_S
    note: str = ""

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (time.time() if now is None else now) - float(self.signed_at))

    def expired(self, now: float | None = None) -> bool:
        return self.age_s(now) > float(self.ttl_s)

    def remaining_s(self, now: float | None = None) -> float:
        return max(0.0, float(self.ttl_s) - self.age_s(now))

    def label(self) -> str:
        return ATTESTATION_REASONS.get(self.reason, self.reason or "未说明原因")


@dataclass(frozen=True)
class Verdict:
    """May we drive the coarse piezo? Plus everything the answer rests on."""
    allow: bool
    reason: str                       # Chinese, a complete sentence
    source: str = "none"              # gauge | attestation | disabled | none
    pressure_pa: float | None = None
    age_s: float | None = None
    mode: str = "gauge_or_attest"
    over_range: bool = False
    gauge_ok: bool = False            # a valid in-range reading exists
    attested: bool = False
    attestation_remaining_s: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow, "reason": self.reason, "source": self.source,
            "pressure_pa": self.pressure_pa, "age_s": self.age_s, "mode": self.mode,
            "over_range": self.over_range, "gauge_ok": self.gauge_ok,
            "attested": self.attested,
            "attestation_remaining_s": self.attestation_remaining_s,
            **({"detail": self.detail} if self.detail else {}),
        }


# ── Pure evaluation ──────────────────────────────────────────────────────────

def to_pascal(value: Any, unit: str) -> float | None:
    """Convert to Pa, or None when the unit is not one we recognise.

    Refusing an unknown unit is the point. ``"mbar"`` and ``"Pa"`` are both in
    live use in this codebase and are 100× apart; a default multiplier of 1.0
    would put a 1e-3 mbar reading (safe) and a 1e-3 Pa reading (also safe) into
    the same bucket as a 1 mbar reading (mid-band, dangerous) depending on which
    sensor happened to answer."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    mult = _UNIT_TO_PA.get(str(unit or "").strip().lower())
    if mult is None:
        return None
    return val * mult


def gauge_config_problem(*, min_pa: float, full_scale_pa: float,
                         max_pa: float) -> str:
    """Why this gauge can never authorise coarse motion — or "" if it can.

    A configuration check, not a reading check. It answers a question the
    operator should not have to discover empirically: *is the gauge I have
    capable of proving what this interlock needs proved?*

    A rough-vacuum gauge (Pirani, capacitance manometer — floor around
    0.5–1 Pa) is a perfectly good gauge that simply cannot see low enough. With
    only that fitted, every coarse move is refused forever and correctly, and the
    operator deserves to be told that once, plainly, instead of reading a
    puzzling refusal every time and wondering whether something is broken."""
    if min_pa <= 0 or full_scale_pa <= 0:
        return ""
    if min_pa >= full_scale_pa:
        return (f"真空计量程配置有误：下限 {min_pa:.3g} Pa ≥ 上限 "
                f"{full_scale_pa:.3g} Pa。在设置里把这两个数按规的实际量程填对。")
    if min_pa >= max_pa:
        return (f"这只真空计的量程下限是 {min_pa:.3g} Pa，高于粗动允许上限 "
                f"{max_pa:.3g} Pa —— **它永远无法证明压强足够低**，"
                f"因此在只有这只规的情况下粗动会一直被拒绝。这不是故障："
                f"粗糙真空规（Pirani / 电容薄膜规）本来就看不到那么低。"
                f"需要一只冷阴极规或电离规，或者由用户签署「当前气压安全」。")
    if full_scale_pa * _OVERRANGE_FRACTION <= max_pa:
        return (f"真空计满量程 {full_scale_pa:.3g} Pa 太低：还没到粗动允许上限 "
                f"{max_pa:.3g} Pa 就已经算超量程，判据永远给不出「通过」。"
                f"核对量程配置，或调低允许上限。")
    return ""


def gauge_verdict(
    sample: PressureSample | None,
    *,
    max_pa: float = DEFAULT_MAX_PRESSURE_PA,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    full_scale_pa: float = DEFAULT_GAUGE_FULL_SCALE_PA,
    min_pa: float = DEFAULT_GAUGE_MIN_PA,
    now: float | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Does the GAUGE alone permit coarse motion? ``(ok, reason, detail)``.

    Six refusals, in the order that produces the most useful message. None of
    them is specific to a gauge model — the model-specific part is entirely in
    ``min_pa`` / ``full_scale_pa``, which the operator sets."""
    if sample is None:
        return False, ("没有可用的真空计读数 —— 无法判断当前是否处在放电区间"
                       f"（{CORONA_ZONE_PA[0]:g}–{CORONA_ZONE_PA[1]:g} Pa）。"), {}

    cls = (sample.sensor_class or "").strip()
    detail: dict[str, Any] = {"sensor": sample.sensor_name, "sensor_class": cls}

    # 1. A stand-in is not a gauge. It reports value=0.0, which reads as a
    #    perfect vacuum to anything that only looks at the number.
    if cls in PLACEHOLDER_CLASSES:
        return False, ("真空计是**占位实现**（未接真实真空计）。它的读数恒为 0，"
                       "这不是「完美真空」而是「没有数据」—— 拒绝粗动。"), detail
    if cls and cls not in REAL_GAUGE_CLASSES:
        return False, (f"传感器类型 {cls} 不在已知真空计名单里，不能用它的读数授权粗动。"
                       "新增真空计驱动时要做两件事：把类名加进 "
                       "vacuum_interlock.REAL_GAUGE_CLASSES，"
                       "并在设置里填这只规的**量程上下限**"
                       "（vacuum_gauge_min_pa / vacuum_gauge_full_scale_pa）——"
                       "判据本身与规的型号无关，型号相关的只有这两个数。"), detail

    # 2. Status. `unavailable` deliberately does NOT trip the alarm path; here it
    #    must, because "the gauge cannot see" is precisely the case we refuse.
    status = (sample.status or "").strip().lower()
    if status != "ok":
        human = {"unavailable": "读不到（串口掉线/未接）", "error": "帧校验失败",
                 "warning": "告警", "alarm": "告警"}.get(status, status or "未知")
        return False, (f"真空计状态为「{human}」，不是有效读数 —— 拒绝粗动。"
                       "读不到 ≠ 真空好。"), detail

    # 3. Unit.
    pa = to_pascal(sample.value, sample.unit)
    if pa is None:
        return False, (f"真空计读数无法换算成 Pa（值 {sample.value!r} 单位 "
                       f"{sample.unit!r}）—— 拒绝粗动。Pa 与 mbar 差 100 倍，"
                       "猜错方向正好落进放电区。"), detail
    detail["pressure_pa"] = pa

    # 4. Age.
    age = sample.age_s(now)
    detail["age_s"] = age
    if age is None:
        return False, "真空计读数没有时间戳，无法判断新旧 —— 拒绝粗动。", detail
    if age > max_age_s:
        return False, (f"真空计读数已经是 {age:.0f} 秒前的了（上限 {max_age_s:.0f} 秒）"
                       "—— 抽气/放气时压强变化很快，旧读数不能给现在授权。"), detail

    # 5a. Over range. At/near full scale the gauge cannot distinguish "just over"
    #     from "atmosphere" — and the DL-7 frame parser does not validate the
    #     exponent, so an over-range condition can decode to a plausible small
    #     number.
    if full_scale_pa > 0 and pa >= full_scale_pa * _OVERRANGE_FRACTION:
        detail["over_range"] = True
        return False, (f"真空计读数 {pa:.3g} Pa 已到量程上限附近"
                       f"（满量程 {full_scale_pa:.3g} Pa）—— 这意味着「超量程、判断不了」，"
                       "不是「刚好卡在阈值下」。拒绝粗动。"), detail

    # 5b. UNDER range — the half that was missing, and the one a non-DL-7 gauge
    #     exposes. A gauge below its floor is not reporting a low pressure, it is
    #     bottoming out, and the number it emits (its floor, zero, or noise) looks
    #     exactly like an excellent vacuum. Same failure shape as the placeholder's
    #     0.0, arrived at from the other direction.
    #
    #     Whether that is safe depends entirely on WHERE the floor is, which is a
    #     property of the gauge, not of this code:
    #
    #       * DL-7 / ion gauge, floor 5e-8 Pa — far below the permit limit. Under
    #         range means "even lower than 5e-8", which is emphatically safe.
    #         Refusing a UHV chamber for being too clean would be absurd.
    #       * Pirani / capacitance manometer, floor ~0.5–1 Pa — ABOVE the permit
    #         limit and inside the discharge band. Under range means "somewhere
    #         below 1 Pa", which includes 0.5 Pa, which is in the band. That
    #         reading cannot authorise anything.
    #
    #     So the test is not "is it under range" but "is this gauge's floor
    #     already below the limit". Rough-vacuum-only gauges fail it permanently,
    #     and should — see the配置 sanity check below, which says so up front
    #     rather than letting it be discovered as a mysterious standing refusal.
    if min_pa > 0 and pa <= min_pa * _UNDERRANGE_FACTOR:
        detail["under_range"] = True
        if min_pa >= max_pa:
            return False, (
                f"真空计读数 {pa:.3g} Pa 已到量程**下限**附近"
                f"（下限 {min_pa:.3g} Pa）—— 这是规在触底，不是「真空非常好」。"
                f"而这只规的下限本身就高于粗动允许上限 {max_pa:.3g} Pa，"
                "所以它**永远无法证明**压强足够低。"
                "需要一只量程更低的规（冷阴极/电离规），或由用户签署。"), detail
        # Floor is below the limit ⇒ "even at my floor we are already safe".
        detail["under_range_safe"] = True
        return True, (f"真空计已到量程下限（{min_pa:.3g} Pa）以下 —— 真实压强比它更低，"
                      f"而下限本身已经远低于允许上限 {max_pa:.3g} Pa，"
                      f"距放电区下沿 {CORONA_ZONE_PA[0]:g} Pa 还有数个数量级。"), detail

    if pa > max_pa:
        band = ("，已经进入放电区间" if pa >= CORONA_ZONE_PA[0] else "")
        return False, (f"当前压强 {pa:.3g} Pa 高于粗动允许上限 {max_pa:.3g} Pa{band}"
                       f"（放电区 {CORONA_ZONE_PA[0]:g}–{CORONA_ZONE_PA[1]:g} Pa）"
                       "—— 拒绝粗动，等抽到更低再动。"), detail

    return True, (f"当前压强 {pa:.3g} Pa ≤ 允许上限 {max_pa:.3g} Pa，"
                  f"远低于放电区下沿 {CORONA_ZONE_PA[0]:g} Pa（读数 {age:.0f} 秒前）。"), detail


def assess(
    sample: PressureSample | None,
    *,
    mode: str = "gauge_or_attest",
    max_pa: float = DEFAULT_MAX_PRESSURE_PA,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    full_scale_pa: float = DEFAULT_GAUGE_FULL_SCALE_PA,
    min_pa: float = DEFAULT_GAUGE_MIN_PA,
    attestation: Attestation | None = None,
    now: float | None = None,
) -> Verdict:
    """The whole interlock, as a pure function. Nothing here touches hardware."""
    mode = mode if mode in MODES else "gauge_or_attest"
    ok, reason, detail = gauge_verdict(
        sample, max_pa=max_pa, max_age_s=max_age_s,
        full_scale_pa=full_scale_pa, min_pa=min_pa, now=now)
    # A gauge that CANNOT prove what we need proved is a configuration fact, not
    # a reading. Surface it in the reason so a standing refusal explains itself
    # the first time instead of the tenth.
    problem = gauge_config_problem(min_pa=min_pa, full_scale_pa=full_scale_pa,
                                   max_pa=max_pa)
    if problem and not ok:
        reason = f"{reason} {problem}"
        detail["gauge_config_problem"] = problem
    pa = detail.get("pressure_pa")
    age = detail.get("age_s")
    over = bool(detail.get("over_range"))

    if ok:
        return Verdict(allow=True, reason=reason, source="gauge", pressure_pa=pa,
                       age_s=age, mode=mode, gauge_ok=True, detail=detail)

    live_att = (attestation if attestation is not None
                and not attestation.expired(now) else None)

    if mode == "gauge_only":
        tail = ""
        if attestation is not None:
            tail = "（本机模式为「只认真空计」，用户签署在此模式下无效。）"
        return Verdict(allow=False, reason=reason + tail, source="none",
                       pressure_pa=pa, age_s=age, mode=mode, over_range=over,
                       detail=detail)

    if mode == "off":
        return Verdict(
            allow=True,
            reason=("【真空互锁已被关闭】不阻断，但判据如下：" + reason
                    + " —— 关闭互锁不代表没有风险，只代表这次没人检查。"),
            source="disabled", pressure_pa=pa, age_s=age, mode=mode,
            over_range=over, detail=detail)

    if live_att is not None:
        return Verdict(
            allow=True,
            reason=(f"真空计不可用：{reason} 但用户已签署「{live_att.label()}」"
                    f"（{live_att.signed_by or '未署名'}，还剩 "
                    f"{live_att.remaining_s(now) / 3600.0:.1f} 小时）—— 按签署放行。"),
            source="attestation", pressure_pa=pa, age_s=age, mode=mode,
            over_range=over, attested=True,
            attestation_remaining_s=live_att.remaining_s(now), detail=detail)

    expired_note = ""
    if attestation is not None:
        expired_note = (f"（用户此前签署过「{attestation.label()}」，"
                        f"但已在 {attestation.age_s(now) / 3600.0:.1f} 小时前过期。）")
    return Verdict(
        allow=False,
        reason=(reason + expired_note
                + " 若确认当前气压安全（例如腔体已通大气），"
                  "请用户在界面上签署一次「当前气压安全」再重试。"),
        source="none", pressure_pa=pa, age_s=age, mode=mode, over_range=over,
        detail=detail)


# ── Process-level wiring ─────────────────────────────────────────────────────

_lock = threading.RLock()
_source: Callable[[], PressureSample | None] | None = None
_attestation: Attestation | None = None
_audit_sink: Callable[[str, dict], None] | None = None


def set_pressure_source(source: Callable[[], PressureSample | None] | None) -> None:
    """Install the reader the runtime wires to the EnvironmentMonitor.

    Injected rather than imported so this module stays free of any hardware
    dependency and can be unit-tested as the pure function it mostly is."""
    global _source
    with _lock:
        _source = source


def current_sample() -> PressureSample | None:
    src = _source
    if src is None:
        return None
    try:
        return src()
    except Exception as exc:  # noqa: BLE001 — a broken source is "no reading"
        logger.debug("vacuum interlock: pressure source failed: %s", exc)
        return None


def set_audit_sink(sink: Callable[[str, dict], None] | None) -> None:
    global _audit_sink
    _audit_sink = sink


def _audit(event: str, payload: dict) -> None:
    sink = _audit_sink
    if callable(sink):
        try:
            sink(event, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vacuum interlock audit failed: %s", exc)


def attest(reason: str, *, signed_by: str = "",
           ttl_s: float = DEFAULT_ATTESTATION_TTL_S,
           note: str = "") -> Attestation:
    """Record the operator's signed statement that the pressure is safe.

    Deliberately process-local: it is gone after a restart, and a restart is a
    change of scene. Re-signing costs ten seconds; an authorisation that
    outlives the conditions it was signed under costs a piezo stack."""
    global _attestation
    if reason not in ATTESTATION_REASONS:
        raise ValueError(f"unknown attestation reason: {reason!r} "
                         f"(expected one of {sorted(ATTESTATION_REASONS)})")
    att = Attestation(reason=reason, signed_by=signed_by or "",
                      signed_at=time.time(), ttl_s=float(ttl_s), note=note or "")
    with _lock:
        _attestation = att
    logger.warning("真空互锁：用户签署「%s」(%s)，有效 %.1f 小时",
                   att.label(), att.signed_by or "未署名", att.ttl_s / 3600.0)
    _audit("attest", {"reason": reason, "signed_by": signed_by,
                      "ttl_s": ttl_s, "note": note})
    return att


def revoke_attestation() -> None:
    global _attestation
    with _lock:
        had = _attestation
        _attestation = None
    if had is not None:
        logger.warning("真空互锁：用户签署已撤销")
        _audit("revoke", {"reason": had.reason})


def get_attestation() -> Attestation | None:
    with _lock:
        att = _attestation
    return att


def _config() -> dict[str, Any]:
    """Thresholds from instrument_profile, falling back to module defaults."""
    try:
        from mast.core import instrument_profile as ip

        mode = ip.get_config("vacuum_interlock_mode", None)
        return {
            "mode": mode if mode in MODES else "gauge_or_attest",
            "max_pa": float(ip.get_config("coarse_motion_max_pressure_pa",
                                          DEFAULT_MAX_PRESSURE_PA)
                            or DEFAULT_MAX_PRESSURE_PA),
            "max_age_s": float(ip.get_config("vacuum_reading_max_age_s",
                                             DEFAULT_MAX_AGE_S)
                               or DEFAULT_MAX_AGE_S),
            "full_scale_pa": float(ip.get_config("vacuum_gauge_full_scale_pa",
                                                 DEFAULT_GAUGE_FULL_SCALE_PA)
                                   or DEFAULT_GAUGE_FULL_SCALE_PA),
            "min_pa": float(ip.get_config("vacuum_gauge_min_pa",
                                          DEFAULT_GAUGE_MIN_PA)
                            or DEFAULT_GAUGE_MIN_PA),
        }
    except Exception as exc:  # noqa: BLE001 — defaults are the safe direction
        logger.debug("vacuum interlock: profile unreadable, using defaults: %s", exc)
        return {"mode": "gauge_or_attest", "max_pa": DEFAULT_MAX_PRESSURE_PA,
                "max_age_s": DEFAULT_MAX_AGE_S,
                "full_scale_pa": DEFAULT_GAUGE_FULL_SCALE_PA,
                "min_pa": DEFAULT_GAUGE_MIN_PA}


def check() -> Verdict:
    """The verdict right now — the call every coarse motion makes."""
    cfg = _config()
    return assess(current_sample(), attestation=get_attestation(), **cfg)


def format_block() -> str:
    """One short block for the system prompt. Never empty."""
    v = check()
    head = "可以粗动" if v.allow else "**禁止粗动**"
    tail = ""
    if not v.allow:
        tail = ("（这不是建议，是硬闸门：粗动/换区技能会直接拒绝执行。"
                "在中间真空区给粗动压电加几百伏会打火击穿。）")
    return f"【真空互锁】{head} —— {v.reason}{tail}"


__all__ = [
    "CORONA_ZONE_PA",
    "DEFAULT_MAX_PRESSURE_PA",
    "DEFAULT_GAUGE_FULL_SCALE_PA",
    "DEFAULT_GAUGE_MIN_PA",
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_ATTESTATION_TTL_S",
    "MODES",
    "ATTESTATION_REASONS",
    "REAL_GAUGE_CLASSES",
    "PLACEHOLDER_CLASSES",
    "PressureSample",
    "Attestation",
    "Verdict",
    "to_pascal",
    "gauge_verdict",
    "gauge_config_problem",
    "assess",
    "set_pressure_source",
    "current_sample",
    "set_audit_sink",
    "attest",
    "revoke_attestation",
    "get_attestation",
    "check",
    "format_block",
]
