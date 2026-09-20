"""The commissioning survey must be safe to run at any moment, and must not lie.

Two properties carry the weight:

**Strictly read-only.** It is advertised as safe with the tip engaged, mid-scan,
at any pressure — that is the reason it can be the first thing anybody runs. If
it ever issues a setter, someone will run it during a scan on the operator's word
and it will be MAST's fault.

**``ready`` is conservative.** A green light that is wrong is worse than no light,
because the whole point is to be the thing you check before touching a rig
remotely. So ``ready`` is True only when nothing is blocking, and every blocking
condition names what to do about it.
"""
from __future__ import annotations

import pytest

from mast.skills.builtins.coarse_selfcheck import CoarseMotionSelfCheck

#: Every Nanonis verb that CHANGES something. None may appear.
WRITE_VERBS = (
    "Motor_StartMove", "Motor_StartClosedLoop", "Motor_FreqAmpSet",
    "ZCtrl_OnOffSet", "ZCtrl_Withdraw", "ZCtrl_SetpntSet", "Bias_Set",
    "Scan_Action", "AutoApproach_OnOffSet", "TipShaper_Start",
    "FolMe_XYPosSet", "Motor_StopMove",
)


class _Rec:
    def __init__(self, error: str = "", return_value=None):
        self.error = error
        self.return_value = return_value


class _Ctx:
    def __init__(self, replies=None, errors=None):
        self.replies = replies or {}
        self.errors = errors or {}
        self.calls: list[tuple] = []

    def safe_call(self, verb, *args):
        self.calls.append((verb, args))
        return _Rec(self.errors.get(verb, ""), self.replies.get(verb))

    def verbs(self):
        return [c[0] for c in self.calls]


def _healthy(**kw):
    replies = {
        "Motor_FreqAmpGet": ("h", "b", [1000.0, 120.0]),
        "Motor_StepCounterGet": ("h", "b", [0, 0, 0]),
        "Motor_PosGet": ("h", "b", [0.0, 0.0, 0.0]),
        "Signals_NamesGet": ("h", "b", ["Current", "Z", "OCD1 Amplitude"]),
        "Signals_ValGet": ("h", "b", [0.8]),
    }
    replies.update(kw.pop("replies", {}))
    return _Ctx(replies=replies, **kw)


@pytest.fixture(autouse=True)
def _rig():
    import datetime as dt

    from mast.core import coarse_drive, coarse_map_provider
    from mast.core import instrument_profile as ip
    from mast.core import vacuum_interlock as vac

    prof, decl = ip.get_profile(), coarse_drive.get_declaration()
    coarse_drive.set_persist_sink(None)
    ip.set_profile({})
    coarse_drive.declare(200.0)
    vac.set_pressure_source(lambda: vac.PressureSample(
        value=1e-8, unit="Pa", status="ok",
        timestamp=dt.datetime.now().isoformat(),
        sensor_name="vacuum", sensor_class="DL7VacuumSensor"))
    vac.revoke_attestation()
    coarse_map_provider.set_marker_source(None)
    coarse_map_provider.set_temperature_source(lambda: 4.2)
    yield
    ip.set_profile(prof)
    coarse_drive.set_declaration(decl)
    vac.set_pressure_source(None)
    coarse_map_provider.set_marker_source(None)
    coarse_map_provider.set_temperature_source(None)


# ── The safety claim ────────────────────────────────────────────────────────

def test_it_issues_no_write_command_at_all():
    """The claim that makes it runnable during a scan."""
    ctx = _healthy()
    CoarseMotionSelfCheck().execute(ctx, {})
    offenders = [v for v in ctx.verbs() if v in WRITE_VERBS]
    assert not offenders, f"self-check issued write verbs: {offenders}"


def test_every_verb_it_uses_is_a_getter():
    """Belt and braces: not just "not on the deny-list" but "looks like a read"."""
    ctx = _healthy()
    CoarseMotionSelfCheck().execute(ctx, {})
    for verb in ctx.verbs():
        assert "Get" in verb, f"{verb} does not look like a getter"


def test_it_is_auto_and_read():
    from mast.core.types import SafetyLevel, SkillCategory

    meta = CoarseMotionSelfCheck().metadata()
    assert meta.category == SkillCategory.READ
    assert meta.safety_level == SafetyLevel.AUTO
    assert meta.preconditions == [], (
        "a diagnostic that refuses to run when the rig is in a bad state is "
        "useless precisely when it is needed"
    )


def test_a_dead_instrument_does_not_break_it():
    """It has to survive the case it exists to diagnose."""
    ctx = _Ctx(errors={v: "no connection" for v in (
        "Motor_FreqAmpGet", "Motor_StepCounterGet", "Motor_PosGet",
        "Signals_NamesGet", "Signals_ValGet")})
    res = CoarseMotionSelfCheck().execute(ctx, {})
    assert res.success is True
    assert res.data["ready"] is False
    assert res.data["blocking"]


# ── The verdict ─────────────────────────────────────────────────────────────

def test_ready_only_when_nothing_blocks():
    res = CoarseMotionSelfCheck().execute(_healthy(), {})
    d = res.data
    assert d["ready"] is bool(not d["blocking"])


def test_an_undeclared_drive_blocks_and_says_why():
    from mast.core import coarse_drive

    coarse_drive.set_declaration({})
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["ready"] is False
    assert any("未声明" in b for b in d["blocking"])
    assert any("admin PIN" in b for b in d["blocking"]), (
        "a blocking item that does not say how to clear it just stalls the run"
    )


def test_a_placeholder_gauge_blocks():
    from mast.core import vacuum_interlock as vac

    vac.set_pressure_source(lambda: vac.PressureSample(
        value=0.0, unit="mbar", status="unavailable",
        sensor_name="vacuum", sensor_class="VacuumSensor"))
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["ready"] is False
    assert any("占位" in b or "真空" in b for b in d["blocking"])


def test_a_rough_gauge_is_reported_as_a_configuration_fact():
    import datetime as dt

    from mast.core import instrument_profile as ip
    from mast.core import vacuum_interlock as vac

    ip.set_profile({"vacuum_gauge_min_pa": 1.0,
                    "vacuum_gauge_full_scale_pa": 1e5})
    vac.set_pressure_source(lambda: vac.PressureSample(
        value=1e-3, unit="Pa", status="ok",
        timestamp=dt.datetime.now().isoformat(),
        sensor_name="vacuum", sensor_class="DL7VacuumSensor"))
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["vacuum"].get("config_problem")
    assert any("量程" in b for b in d["blocking"])


# ── The unverified guess, surfaced ──────────────────────────────────────────

def test_the_freq_amp_order_guess_is_declared_every_time():
    """It is a GUESS from the setter's signature, and it is safety-relevant.

    An unverified assumption that nobody is told about is indistinguishable from
    a verified one — right up until it matters."""
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    drive = d["coarse_drive"]
    assert drive["order_unverified"] is True
    assert drive["parsed_as"] == {"frequency_hz": 1000.0, "amplitude_v": 120.0}
    assert "Hz" in drive["order_hint"] and "V" in drive["order_hint"], (
        "the hint must give the operator a way to TELL, not just a warning"
    )
    assert any("回包顺序" in t for t in d["todo"])


def test_the_raw_reply_is_included_so_the_order_can_be_settled_remotely():
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["coarse_drive"]["raw_reply"]


# ── Honest "not supported" reporting ────────────────────────────────────────

def test_an_absent_step_counter_is_information_not_failure():
    ctx = _healthy(errors={"Motor_StepCounterGet": "not supported"})
    d = CoarseMotionSelfCheck().execute(ctx, {}).data
    assert d["step_counter"]["supported"] is False
    assert "ANC150" in d["step_counter"]["note"]
    assert not any("step" in b.lower() for b in d["blocking"]), (
        "no counter is a normal controller, not a blocker"
    )


def test_an_stm_without_qplus_is_a_normal_configuration():
    ctx = _healthy(replies={"Signals_NamesGet": ("h", "b", ["Current", "Z"])})
    d = CoarseMotionSelfCheck().execute(ctx, {}).data
    assert d["qplus"]["channel"] is None
    assert "不是故障" in d["qplus"]["note"]
    assert d["ready"] is True, "no qPlus must not block commissioning"


def test_a_qplus_rig_without_a_baseline_gets_a_todo_not_a_block():
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["qplus"]["channel"] is not None
    assert any("基线" in t for t in d["todo"])
    assert d["ready"] is True


def test_the_step_counter_probe_can_be_skipped():
    ctx = _healthy()
    CoarseMotionSelfCheck().execute(ctx, {"probe_step_counter": False})
    assert "Motor_StepCounterGet" not in ctx.verbs()


# ── The calibration reminders ───────────────────────────────────────────────

def test_it_always_names_the_two_things_only_the_rig_can_settle():
    """Direction codes and the two step counts. Neither is knowable off-rig, and
    both are easy to assume were handled."""
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    joined = "\n".join(d["todo"])
    assert "方向码" in joined
    assert "xy_prewithdraw_steps" in joined and "xy_site_spacing_steps" in joined


def test_the_profile_note_explains_how_to_calibrate_not_just_that_to():
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    note = d["instrument_profile"]["_note"]
    assert "扫一张图" in note and "完全" in note


def test_registry_check_catches_a_frozen_build_gap():
    """The failure mode is silent absence, and it has happened at 141-skill scale."""
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data
    assert d["registry"]["ok"] is True, d["registry"].get("missing")
    assert d["registry"]["checked"] >= 8


# 已声明的驱动频率参与回包字段核对，不能只按数值大小猜频率与幅度。

def test_declared_frequency_confirms_the_reply_order():
    from mast.core import coarse_drive

    coarse_drive.declare(200.0, expected_frequency_hz=1000.0)   # 与 fixture 回包一致
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data["coarse_drive"]
    assert d["parsed_as"] == {"frequency_hz": 1000.0, "amplitude_v": 120.0}
    assert d["order_unverified"] is False
    assert "已验证" in d["order_hint"]


def test_a_swapped_reply_order_is_called_out_as_positive_evidence():
    """反过来才对上 = 顺序反了的**证据**，不是猜测 —— 而且必须警告别信 readback_ok，
    它比的是错的那个数。"""
    from mast.core import coarse_drive

    # 面板上频率其实是 120 Hz：那就说明回包第二个数才是频率，取值顺序反了。
    coarse_drive.declare(200.0, expected_frequency_hz=120.0)
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data["coarse_drive"]
    assert d["order_unverified"] is True
    assert "反了" in d["order_hint"]
    assert "readback_ok" in d["order_hint"], "没警告就等于让人接着信一个错的比较"


def test_without_a_declared_frequency_it_falls_back_to_the_eyeball_hint():
    """没声明就退回原来的人工判据，但要告诉用户怎么让它自动化。"""
    from mast.core import coarse_drive

    coarse_drive.declare(200.0)          # 只声明上限，不声明频率
    d = CoarseMotionSelfCheck().execute(_healthy(), {}).data["coarse_drive"]
    assert d["order_unverified"] is True
    assert "expected_frequency_hz" in d["order_hint"]
