"""Second-opinion channels: they must add, never subtract, and never say "fine".

The qPlus amplitude and the lock-in dI/dV are the two witnesses that are not
downstream of the same measurement as everything else MAST checks. That makes
them valuable and dangerous in the same way: valuable because a tip ploughed into
the surface still reads "at setpoint" to a current-only check, dangerous because
if their failure mode is silence, the extra confidence is imaginary.

So the invariants under test are:
  * an absent / unreadable channel yields NOTHING, never a reassuring value;
  * a positive crash verdict is never produced from a missing baseline;
  * the dI/dV channel reports and does not gate;
  * the baseline is captured only where the tip is verified clear.
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import _tip_evidence as ev


class _Ctx:
    """Minimal ExecutionContext stand-in with scripted Nanonis replies."""

    #: 默认这台机器的**音叉是被驱动着的**(PLL 输出开、激励 0.2 V)。
    #: ⑰ 之后这不是可省略的细节:激励关着时振幅通道没有判据能力,撞针判据一律
    #: 返回 unavailable。这些用例问的是「振幅塌了算不算撞针」,前提本来就是它在振 ——
    #: 现在把前提**写出来**,而不是靠假机器对未知动词返回 None 碰巧通过。
    _DRIVEN = {"PLL_OutOnOffGet": ("", b"", [1]),
               "PLL_ExcitationGet": ("", b"", [0.2])}

    def __init__(self, replies: dict[str, object] | None = None,
                 errors: set[str] | None = None):
        self.replies = dict(self._DRIVEN)
        self.replies.update(replies or {})
        self.errors = errors or set()
        self.calls: list[tuple] = []

    def safe_call(self, verb, *args):
        self.calls.append((verb, args))

        class _R:
            pass

        r = _R()
        r.error = "boom" if verb in self.errors else ""
        r.return_value = self.replies.get(verb)
        return r


@pytest.fixture(autouse=True)
def _clean_profile():
    from mast.core import instrument_profile as ip

    before = ip.get_profile()
    ip.set_profile({})
    yield
    ip.set_profile(before)


# ── qPlus: absence is silence, not reassurance ──────────────────────────────

def test_no_amplitude_channel_yields_no_claim():
    """An STM without a qPlus sensor must contribute nothing at all.

    Not "no crash" — nothing. The caller's message must not gain a sentence
    implying a check happened."""
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["Current", "Z", "Bias"])})
    fields = ev.qplus_fields(ctx)
    assert fields.get("qplus_status") == "unavailable"
    assert fields.get("qplus_crash") is None
    assert ev.qplus_says_crashed(fields) is False


def test_a_read_failure_is_not_a_clean_bill_of_health():
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"])},
               errors={"Signals_ValGet"})
    fields = ev.qplus_fields(ctx)
    assert fields.get("qplus_crash") is None
    assert ev.qplus_says_crashed(fields) is False


def test_no_baseline_is_cannot_tell_not_no_crash():
    """Zero amplitude means nothing without a free-oscillation reference — the
    oscillator may simply not be running."""
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [0.0])})
    fields = ev.qplus_fields(ctx)
    assert fields.get("qplus_status") == "no_baseline"
    assert fields.get("qplus_crash") is None
    assert ev.qplus_says_crashed(fields) is False


def test_a_collapsed_amplitude_against_a_baseline_is_a_crash():
    from mast.core import instrument_profile as ip

    ip.set_profile({"qplus_amplitude_baseline": 1.0})
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [0.02])})
    fields = ev.qplus_fields(ctx)
    assert fields["qplus_status"] == "crash"
    assert ev.qplus_says_crashed(fields) is True


def test_a_healthy_amplitude_is_ok_but_does_not_overrule_the_current():
    from mast.core import instrument_profile as ip

    ip.set_profile({"qplus_amplitude_baseline": 1.0})
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [0.95])})
    fields = ev.qplus_fields(ctx)
    assert fields["qplus_status"] == "ok"
    assert ev.qplus_says_crashed(fields) is False


# ── Recovery check used to CONFIRM a retract ────────────────────────────────

def test_recovery_verdict_is_none_when_the_channel_cannot_answer():
    """None must not be read as True by a caller confirming clearance."""
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["Current"])})
    verdict, why = ev.qplus_recovered(ctx)
    assert verdict is None
    assert why


def test_recovery_verdict_is_three_state_not_two():
    """振幅恢复判据保留三态：归零、确认恢复和无法判断。中间区间不得当作仍在接触；其他独立保护仍须核对。"""
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.qplus_amplitude import _CRASH_FRACTION

    ip.set_profile({"qplus_amplitude_baseline": 1.0})

    # ① 恢复到自由值 ⇒ True
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [0.9])})
    assert ev.qplus_recovered(ctx)[0] is True

    # ② 振幅被压死(< 10%)⇒ False,而且要说「不要横向移动」
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [_CRASH_FRACTION / 2])})
    verdict, why = ev.qplus_recovered(ctx)
    assert verdict is False
    assert "不要横向移动" in why

    # ③ ⭐ 中间段 ⇒ **None(弃权)**,不是 False
    for frac in (0.3, 0.5, 0.79):
        ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                    "Signals_ValGet": ("h", "b", [frac])})
        verdict, why = ev.qplus_recovered(ctx)
        assert verdict is None, (
            f"振幅 {frac:.0%} 被判成 {verdict!r} —— 它既没被压死(≥"
            f"{_CRASH_FRACTION:.0%})、也不够格确认自由(<"
            f"{ev.RECOVERED_FRACTION:.0%})。这一段这个通道答不了,"
            "而判 False 会挡住一次本该放行的换位。")
        assert "答不了" in why or "弃权" in why, why


def test_the_two_amplitude_thresholds_do_not_overlap():
    """压死线必须严格低于确认自由线 —— 否则中间那个弃权带不存在,三态塌回两态。"""
    from mast.skills.builtins.qplus_amplitude import _CRASH_FRACTION

    assert 0.0 < _CRASH_FRACTION < ev.RECOVERED_FRACTION < 1.0


def test_the_recovery_threshold_is_below_one():
    """After a retract the amplitude returns to free, but not instantly and not
    exactly. Demanding equality would call a perfectly good retract a failure."""
    assert 0.0 < ev.RECOVERED_FRACTION < 1.0


# ── Baseline capture ────────────────────────────────────────────────────────

def test_baseline_capture_reports_failure_instead_of_pretending():
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["Current"])})
    out = ev.capture_qplus_baseline(ctx)
    assert out["qplus_baseline_captured"] is False
    assert out.get("qplus_baseline") is None


def test_baseline_capture_persists_into_the_profile():
    from mast.core import instrument_profile as ip

    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"]),
                "Signals_ValGet": ("h", "b", [0.77])})
    out = ev.capture_qplus_baseline(ctx, note="test")
    assert out["qplus_baseline_captured"] is True
    assert ip.get_config("qplus_amplitude_baseline", None) == pytest.approx(0.77)


def test_a_remembered_index_short_circuits_the_name_scan():
    """The index key was written and never read, so every call re-fetched the
    whole signal table to re-derive an answer already on file — and an operator
    could not override a name match that did not fit their rig."""
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.qplus_amplitude import find_amplitude_signal

    ip.set_profile({"qplus_amplitude_signal_index": 14})
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["nothing", "matching"])})
    found = find_amplitude_signal(ctx)
    assert found is not None and found[0] == 14
    assert not any(c[0] == "Signals_NamesGet" for c in ctx.calls), (
        "the remembered index should make the table fetch unnecessary"
    )


def test_the_auto_sentinel_is_not_used_as_an_index():
    """-1 means "discover by name". Passing it to Signals_ValGet would read
    whatever the controller does with a negative channel."""
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.qplus_amplitude import find_amplitude_signal

    ip.set_profile({"qplus_amplitude_signal_index": -1})
    ctx = _Ctx({"Signals_NamesGet": ("h", "b", ["OCD1 Amplitude"])})
    found = find_amplitude_signal(ctx)
    assert found is not None and found[0] == 0, "fell back to the name scan"


# ── dI/dV: reports, does not gate ───────────────────────────────────────────

def test_didv_is_silent_without_a_configured_signal_index():
    ctx = _Ctx()
    assert ev.didv_trend_fields(ctx) == {}


def test_didv_reports_absolute_value_before_any_calibration_exists():
    from mast.core import instrument_profile as ip

    ip.set_profile({"lockin_signal_index": 8})
    ctx = _Ctx({"Signals_ValGet": ("h", "b", [1.5e-9])})
    out = ev.didv_trend_fields(ctx)
    assert out["didv_v"] == pytest.approx(1.5e-9)
    assert "didv_frac_of_contact" not in out
    assert "尚无" in out["didv_note"]


def test_didv_reports_a_ratio_once_calibrated():
    from mast.core import instrument_profile as ip

    ip.set_profile({"lockin_signal_index": 8})
    ip.set_calibration(1.0e-9, bias_v=0.05, setpoint_a=100e-12, mod_amp_v=0.02)
    ctx = _Ctx({"Signals_ValGet": ("h", "b", [8.0e-10])})
    out = ev.didv_trend_fields(ctx)
    assert out["didv_frac_of_contact"] == pytest.approx(0.8, rel=1e-3)
    assert "接触在望" in out["didv_note"]


def test_didv_never_returns_a_gate_decision():
    """Reporting only, on purpose: the reference value is bootstrapped from the
    first successful approach, so a hard threshold applied before the
    calibration is trustworthy would refuse real approaches. Refusing an
    approach costs a night; printing a ratio costs a line."""
    from mast.core import instrument_profile as ip

    ip.set_profile({"lockin_signal_index": 8})
    ip.set_calibration(1.0e-9, bias_v=0.05, setpoint_a=100e-12, mod_amp_v=0.02)
    ctx = _Ctx({"Signals_ValGet": ("h", "b", [1e-15])})   # nowhere near contact
    out = ev.didv_trend_fields(ctx)
    # ``didv_readout_form`` (2026-08-04) 是**这个读数叫什么**（X/Y 还是 R/Φ），
    # 不是一个决定 —— 它加进白名单不削弱这条测试的意图：下面那行仍然钉住
    # 「不出现任何裁决字段」。
    assert set(out) <= {"didv_v", "didv_frac_of_contact",
                        "didv_engage_frac_threshold", "didv_note",
                        "didv_readout_form"}
    assert not any(k in out for k in ("allow", "blocked", "refuse", "ok"))
    assert "趋势播报" in out["didv_note"]


def test_the_previously_dead_config_key_is_now_the_narration_threshold():
    """``approach_didv_engage_frac`` was declared, documented, shown in the UI —
    and read by nothing at all."""
    from mast.core import instrument_profile as ip

    ip.set_profile({"lockin_signal_index": 8, "approach_didv_engage_frac": 0.5})
    ip.set_calibration(1.0e-9, bias_v=0.05, setpoint_a=100e-12, mod_amp_v=0.02)
    ctx = _Ctx({"Signals_ValGet": ("h", "b", [6.0e-10])})
    out = ev.didv_trend_fields(ctx)
    assert out["didv_engage_frac_threshold"] == pytest.approx(0.5)
    assert "接触在望" in out["didv_note"], "0.6 clears a 0.5 threshold"


# ── The approach wiring ─────────────────────────────────────────────────────

def test_a_contradicted_approach_keeps_its_success_but_says_so():
    """A collapsed oscillation next to an at-setpoint current is what a buried
    tip looks like. Flipping success would let a mis-scaled amplitude channel
    fail good approaches, so the contradiction is surfaced, not enforced."""
    from mast.skills.builtins.approach import _evidence_warning

    warn = _evidence_warning({"qplus_status": "crash", "qplus_crash": True,
                              "qplus_fraction": 0.03})
    assert "扎进表面" in warn
    assert "先退针复查" in warn


def test_no_warning_when_there_is_nothing_to_warn_about():
    from mast.skills.builtins.approach import _evidence_warning

    assert _evidence_warning({}) == ""
    assert _evidence_warning({"qplus_status": "unavailable",
                              "qplus_crash": None}) == ""
    assert _evidence_warning({"qplus_status": "no_baseline",
                              "qplus_crash": None}) == "", (
        "no baseline means CANNOT TELL — it must not produce a crash warning "
        "any more than it produces an all-clear"
    )
