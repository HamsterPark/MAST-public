"""RelocateCoarseXY verifies clearance before moving the sample stage.

Synthetic tests exercise pressure, drive settings, direction, motion monitoring
and completed-movement bookkeeping. The model ramps Z over successive polls
and gives the motor a response independent of the configured direction.
Reversing configuration must reverse the verdict with equal displacement
magnitude; waiting a fixed time cannot substitute for convergence."""
from __future__ import annotations

import pytest

from mast.core import coarse_drive, coarse_map_provider, vacuum_interlock as vac
from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY

#: Nanonis coarse-motor direction codes for the Z axis.
_Z_PLUS, _Z_MINUS = 4, 5
_LATERAL = (0, 1, 2, 3)


class _Rec:
    def __init__(self, error: str = "", return_value=None):
        self.error = error
        self.return_value = return_value


class _Rig:
    """A rig where the Z loop takes TIME, and the motor has a real direction.

    Piezo coordinates: ``Z_MIN`` is fully withdrawn, ``Z_MAX`` the extend limit.
    ``gap_m`` is the piezo position at which the tip would touch the surface, so
    moving the tip AWAY raises it (the loop must extend further to reach). When
    ``gap_m`` exceeds ``Z_MAX`` there is nothing within reach: the loop runs to
    the limit and sits there with no current — the rail case, which is the
    NORMAL outcome of a clearance ladder past the first rung or two.
    """

    Z_MIN = -750e-9
    Z_MAX = +750e-9
    SETPOINT = 100e-12
    NOISE_A = 1e-14

    def __init__(self, *, gap_m: float = 0.0, away_code: int = _Z_PLUS,
                 travel_per_poll: float = 120e-9, step_m: float = 200e-9,
                 drifts_forever: bool = False,
                 drifts_after_z_moves: "int | None" = None,
                 motor_dead: bool = False,
                 replies=None, errors=None, sub_results=None):
        #: The coarse motor accepts every command and moves nothing — the
        #: coarse-approach HV switched off, drive below the slip threshold, end
        #: of travel, a jam. Motor_StartMove still returns success.
        self.motor_dead = motor_dead
        self.gap_m = gap_m
        self.away_code = away_code
        self.travel = travel_per_poll
        self.step_m = step_m
        self.drifts_forever = drifts_forever
        #: Settle normally until this many coarse Z steps have been taken, then
        #: never converge again — the only way to reach the PER-RUNG timeout,
        #: since a rig that drifts from the start fails at the baseline instead.
        self.drifts_after_z_moves = drifts_after_z_moves
        self.z = self.Z_MIN
        self.loop_on = False
        self.lateral_moves = 0
        self.z_moves = 0
        #: Set to a callable returning a current (or None to defer to physics).
        self.current_hook = None
        self.replies = dict(replies or {})
        self.errors = dict(errors or {})
        self.sub_results = dict(sub_results or {})
        self.calls: list[tuple] = []
        self.ran: list[tuple] = []

    # ── physics ──────────────────────────────────────────────────────────
    @property
    def target(self) -> float:
        """Where the loop comes to rest: the surface, or the extend limit."""
        return min(self.gap_m, self.Z_MAX)

    def _advance(self) -> None:
        if not self.loop_on:
            return
        if self.drifts_forever:
            self.z += self.travel
            return
        tgt = self.target
        if abs(self.z - tgt) <= self.travel:
            self.z = tgt
        else:
            self.z += self.travel if tgt > self.z else -self.travel

    def _engaged(self) -> bool:
        """Is the loop actually holding a junction (as opposed to railed)?"""
        return (self.loop_on and not self.drifts_forever
                and self.gap_m <= self.Z_MAX and self.z == self.target)

    def _current(self) -> float:
        if self.current_hook is not None:
            forced = self.current_hook()
            if forced is not None:
                return forced
        return self.SETPOINT if self._engaged() else self.NOISE_A

    # ── the wire ─────────────────────────────────────────────────────────
    _STATIC = {
        "Motor_FreqAmpGet": ("h", "b", [1000.0, 120.0]),
        "Signals_NamesGet": ("h", "b", ["Current", "Z"]),  # no qPlus on the instrument
        "Motor_StepCounterGet": ("h", "b", [0, 0, 0]),
    }

    def safe_call(self, verb, *args, **kw):
        self.calls.append((verb, args))
        if verb in self.errors:
            return _Rec(self.errors[verb], None)
        if verb in self.replies:                      # explicit test override
            return _Rec("", self.replies[verb])

        if verb == "ZCtrl_Withdraw":
            self.z, self.loop_on = self.Z_MIN, False
            return _Rec("", None)
        if verb == "ZCtrl_OnOffSet":
            self.loop_on = bool(args[0]) if args else False
            return _Rec("", None)
        if verb == "ZCtrl_OnOffGet":
            return _Rec("", ("h", "b", [1 if self.loop_on else 0]))
        if verb == "ZCtrl_ZPosGet":
            self._advance()
            return _Rec("", ("h", "b", [self.z]))
        if verb == "Current_Get":
            return _Rec("", ("h", "b", [self._current()]))
        if verb == "ZCtrl_SetpntGet":
            return _Rec("", ("h", "b", [self.SETPOINT]))
        if verb == "Motor_StartMove":
            code, steps = int(args[0]), int(args[1])
            if code in (_Z_PLUS, _Z_MINUS):
                # The rig's wiring decides which way the tip actually goes —
                # unless the stage is not moving at all, in which case the
                # command is still accepted and the gap is unchanged.
                away = code == self.away_code
                if not self.motor_dead:
                    self.gap_m += (self.step_m * steps) * (1 if away else -1)
                self.z_moves += 1
                if (self.drifts_after_z_moves is not None
                        and self.z_moves >= self.drifts_after_z_moves):
                    self.drifts_forever = True
            elif code in _LATERAL:
                self.lateral_moves += 1
            return _Rec("", None)
        return _Rec("", self._STATIC.get(verb))

    def run(self, skill_name, params):
        self.ran.append((skill_name, params))
        from mast.core.types import SkillResult
        return self.sub_results.get(
            skill_name, SkillResult(skill_name=skill_name, success=True, data={}))

    def check_abort(self):
        return False

    # ── inspection ───────────────────────────────────────────────────────
    def verbs(self) -> list[str]:
        return [c[0] for c in self.calls]

    def moves(self) -> list[tuple]:
        return [c for c in self.calls if c[0] == "Motor_StartMove"]

    def lateral(self) -> list[tuple]:
        return [c for c in self.moves() if c[1][0] in _LATERAL]


def _healthy_rig(**kw) -> _Rig:
    """Good vacuum, sane drive, surface within reach, config matching the wiring."""
    return _Rig(**kw)


@pytest.fixture(autouse=True)
def _rig(monkeypatch):
    """A declared drive, a good gauge, and an empty coarse map."""
    import datetime as dt

    from mast.core import instrument_profile as ip

    # Poll as fast as the fake answers — the ramp in these tests is driven by
    # POLL COUNT, not by wall clock, so the settle logic is exercised in full
    # without any sleeping. The timeout is a wall-clock backstop that a
    # converging rig never reaches; tests that WANT a timeout shrink it locally.
    #
    # Note what is deliberately NOT shrunk to zero: there is no "settle window"
    # left to shrink. That knob was the bug.
    monkeypatch.setattr(RelocateCoarseXY, "_poll_interval_s", 0.0)
    monkeypatch.setattr(RelocateCoarseXY, "_settle_timeout_s", 5.0)

    prof_before = ip.get_profile()
    decl_before = coarse_drive.get_declaration()
    ip.set_profile({"xy_prewithdraw_steps": 100, "xy_move_chunk_steps": 50,
                    "xy_site_spacing_steps": 200, "xy_axis_step_budget": 5000,
                    "z_recede_min_nm": 1.0, "retract_motor_dir": "z+",
                    "z_extend_sign": "+1"})
    coarse_drive.set_persist_sink(None)
    coarse_drive.declare(200.0)
    vac.set_pressure_source(lambda: vac.PressureSample(
        value=1e-8, unit="Pa", status="ok",
        timestamp=dt.datetime.now().isoformat(),
        sensor_name="vacuum", sensor_class="DL7VacuumSensor"))
    vac.revoke_attestation()
    coarse_map_provider.set_marker_source(None)
    coarse_map_provider.set_temperature_source(lambda: 4.2)
    yield
    ip.set_profile(prof_before)
    coarse_drive.set_declaration(decl_before)
    vac.set_pressure_source(None)
    coarse_map_provider.set_marker_source(None)
    coarse_map_provider.set_temperature_source(None)


def _params(**kw):
    out = {"axis": "x", "direction": "+", "steps": 120, "reapproach": False}
    out.update(kw)
    return out


def _rungs(res) -> list[dict]:
    """The clearance self-check's verdicts, on success as well as on failure."""
    return list(res.data.get("clearance_rungs") or [])


# ── Shape ───────────────────────────────────────────────────────────────────

def test_it_is_confirm_so_it_can_run_unattended():
    """The whole incentive design rests on this.

    The bare lateral MotorMove is the one gated to a human; if the guarded
    composite were ALSO gated, the model would simply keep calling the primitive
    that checks nothing."""
    from mast.core.types import SafetyLevel

    meta = RelocateCoarseXY().metadata()
    assert meta.safety_level == SafetyLevel.CONFIRM
    assert "vacuum_ok_for_coarse" in meta.preconditions
    assert "scan_not_running" in meta.preconditions


def test_the_plan_clears_before_it_moves_and_chunks_the_move():
    plan = RelocateCoarseXY().plan(_params(steps=120))
    ids = [s.step_id for s in plan]
    assert ids[0] == "preflight" and ids[1] == "clear"
    assert ids.index("clear") < min(i for i, s in enumerate(ids) if s.startswith("move_"))
    assert len([s for s in ids if s.startswith("move_")]) == 3, "50+50+20"


def test_reapproach_is_only_planned_when_asked_for():
    assert "reapproach" not in [s.step_id for s in
                                RelocateCoarseXY().plan(_params(reapproach=False))]
    assert "reapproach" in [s.step_id for s in
                            RelocateCoarseXY().plan(_params(reapproach=True))]


# ── Preflight refusals ──────────────────────────────────────────────────────

def test_a_bad_vacuum_stops_it_before_anything_moves():
    vac.set_pressure_source(None)          # no gauge → fail-closed
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "真空" in (res.error or "")
    assert not ctx.moves(), "no motor command may be issued after a vacuum refusal"


def test_an_undeclared_drive_stops_it_before_anything_moves():
    coarse_drive.set_declaration({})
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "驱动" in (res.error or "")
    assert not ctx.moves()


def test_an_unreadable_drive_stops_it():
    """Unreadable is a refusal, not a pass. The declaration bounds what MAST
    writes; it says nothing about what somebody set in the Nanonis UI."""
    ctx = _healthy_rig(errors={"Motor_FreqAmpGet": "timeout"})
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert not ctx.moves()


def test_a_drive_left_too_high_stops_it():
    ctx = _healthy_rig(replies={"Motor_FreqAmpGet": ("h", "b", [1000.0, 380.0])})
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert not ctx.moves()


def test_a_destination_on_a_visited_site_is_refused_with_a_way_forward():
    """"Don't go back" enforced as geometry, not as a remembered rule."""
    from mast.io.coarse_map import CoarseMapConfig

    rows = [{"kind": "scan", "coord_epoch": 0, "timestamp": "t"},
            {"kind": "coarse_move", "coord_epoch": 0, "timestamp": "t",
             "meta": {"direction": "x+", "steps": 400}},
            {"kind": "scan", "coord_epoch": 1, "timestamp": "t"}]
    coarse_map_provider.set_marker_source(
        lambda: (rows, CoarseMapConfig(site_spacing_steps=200)))
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(direction="-", steps=400))
    assert res.success is False
    assert "已经去过" in (res.error or "")
    assert "get_coarse_map" in (res.error or ""), (
        "a refusal that does not say how to get an acceptable target just stalls"
    )
    assert not ctx.moves()


def test_an_unknown_odometer_permits_only_the_same_direction():
    from mast.io.coarse_map import CoarseMapConfig

    rows = [{"kind": "coarse_move", "coord_epoch": 0, "timestamp": "t",
             "meta": {"direction": "x+", "steps": 400}},
            {"kind": "coarse_move", "coord_epoch": 1, "timestamp": "t",
             "meta": {}}]           # backfilled with no direction/steps
    coarse_map_provider.set_marker_source(lambda: (rows, CoarseMapConfig()))
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(direction="-"))
    assert res.success is False
    assert "只允许沿上一次的方向" in (res.error or "")


def test_an_unreadable_map_does_not_strand_the_run():
    """Where the stage has been is a surface-budget question, not a safety one.
    Refusing to relocate because the database is unreachable would turn a
    bookkeeping outage into a stopped experiment."""
    coarse_map_provider.set_marker_source(lambda: None)
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is True


# ── Clearance ───────────────────────────────────────────────────────────────

def test_it_steps_the_coarse_z_motor_back_before_moving_laterally():
    """The gap the operator named: a piezo withdraw is ~1 µm, and a lateral
    move needs tens."""
    ctx = _healthy_rig()
    RelocateCoarseXY().run_composite(ctx, _params())
    retracts = [c for c in ctx.moves() if c[1][0] in (_Z_PLUS, _Z_MINUS)]
    assert retracts, "no coarse Z retract was issued before the lateral move"
    assert sum(c[1][1] for c in retracts) == 100, "xy_prewithdraw_steps"
    first_lateral = next(i for i, c in enumerate(ctx.moves()) if c[1][0] in _LATERAL)
    assert all(ctx.moves()[i][1][0] in (_Z_PLUS, _Z_MINUS)
               for i in range(first_lateral))


def test_the_clearance_ladder_starts_with_a_single_step():
    """The minimum-risk probe: one coarse step is much smaller than the piezo
    range, so even a backwards configured direction is absorbed and revealed for
    the price of one step."""
    assert RelocateCoarseXY._clearance_ladder(100)[0] == 1
    assert RelocateCoarseXY._clearance_ladder(100)[:2] == [1, 10]
    assert sum(RelocateCoarseXY._clearance_ladder(100)) == 100
    assert RelocateCoarseXY._clearance_ladder(0) == []


def test_the_piezo_is_withdrawn_before_every_coarse_step():
    """A coarse step with the piezo extended is the crash the ladder is for.

    The settle leaves the loop tracking the surface — that is what makes the
    reading a measurement — so the withdraw before the NEXT step is load-bearing,
    and so is the one before the lateral move."""
    ctx = _healthy_rig()
    RelocateCoarseXY().run_composite(ctx, _params())
    for i, call in enumerate(ctx.calls):
        if call[0] != "Motor_StartMove":
            continue
        before = ctx.calls[:i]
        last_withdraw = max((j for j, c in enumerate(before)
                             if c[0] == "ZCtrl_Withdraw"), default=-1)
        last_closed = max((j for j, c in enumerate(before)
                           if c[0] == "ZCtrl_OnOffSet" and c[1] == (1,)), default=-1)
        assert last_withdraw > last_closed, (
            f"coarse step {call[1]} was issued after the loop was re-closed "
            "with no withdraw since — the piezo could be extended at the surface"
        )


# ── The self-check that has to be able to answer BOTH ways ──────────────────

def test_reversing_the_configured_direction_reverses_the_verdict():
    """On a fixed synthetic model, reversing retract_motor_dir must reverse the verdict while preserving displacement magnitude."""
    from mast.core import instrument_profile as ip

    verdicts, dz = {}, {}
    for cfg, expected in (("z+", "receding"), ("z-", "approaching")):
        # ``z_extend_sign`` 必须一起声明:``set_profile`` 是**整体替换**,不写就等于
        # 「这台机器没答过这一问」,而自检从 2026-08-11 起对没答过的机器拒判
        # (``no_sign``)。``_Rig`` 的物理是「退针 ⇒ z 升」⇒ 伸长时 Z 增大 ⇒ ``+1``。
        ip.set_profile({"retract_motor_dir": cfg, "z_extend_sign": "+1"})
        # away_code is the RIG's wiring and never changes: z+ really is away.
        ctx = _healthy_rig(away_code=_Z_PLUS)
        res = RelocateCoarseXY().run_composite(ctx, _params())
        rungs = _rungs(res)
        assert rungs, f"no rung was judged with retract_motor_dir={cfg}"
        verdicts[cfg] = rungs[0]["verdict"]
        assert verdicts[cfg] == expected, (
            f"retract_motor_dir={cfg} judged {verdicts[cfg]!r} "
            f"({rungs[0]['reason']}), expected {expected!r}")
        dz[cfg] = rungs[0]["z_after_m"] - res.data["clearance_baseline"]["z_m"]

    assert verdicts["z+"] != verdicts["z-"], (
        "the verdict did not change when the direction was reversed — "
        "whatever this is measuring, it is not the direction"
    )

    # THE SIGNATURE, and it is stronger than "the two answers differ".
    #
    #   same magnitude, opposite sign  → the check is measuring a DISPLACEMENT
    #   different magnitude, no flip   → it is measuring the RAMP
    #
    # Equal synthetic steps in opposite directions must have equal magnitudes.
    assert dz["z+"] == pytest.approx(-dz["z-"], rel=1e-9), (
        f"|dz| differs between directions ({dz['z+']*1e9:.1f} nm vs "
        f"{dz['z-']*1e9:.1f} nm). One coarse step is the same distance either "
        "way; unequal magnitudes mean the number is coming from something other "
        "than the completed step, such as an unfinished ramp."
    )


def test_a_backwards_retract_direction_aborts_after_one_step():
    """Config is an intent; the runtime Z self-check is the guard.

    The rig is wired so z+ is away; the profile claims z-. One coarse step is
    much smaller than the piezo range, so the mistake is absorbed and revealed
    for the price of that one step."""
    from mast.core import instrument_profile as ip

    ip.set_profile({"retract_motor_dir": "z-", "z_extend_sign": "+1"})
    ctx = _healthy_rig(away_code=_Z_PLUS)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "逼近" in (res.error or "")
    assert "方向很可能配反" in (res.error or "")
    assert not ctx.lateral(), "must not slide sideways after a failed clearance"
    assert "Motor_StopMove" in ctx.verbs() and "ZCtrl_Withdraw" in ctx.verbs()
    z_steps = [c[1][1] for c in ctx.moves() if c[1][0] in (_Z_PLUS, _Z_MINUS)]
    assert z_steps == [1], "the probe rung is one step; it must not take the next"


def test_undeclared_sign_stops_the_ladder_and_leaves_no_dz_in_the_ledger():
    """没声明 ``z_extend_sign`` 的机器上,清障梯子必须在**第一级**停住。

    出厂默认 ``"+1"`` 在这一项上不是保守值,是两个互斥答案里的一个 —— 本机
    08-05 与 08-08 的原始读数隐含的符号是**相反**的(降温后 Z 量程减半、符号同时
    翻),所以「猜一个」和「猜对了」长得一模一样。见
    ``instrument_profile.z_extend_sign_or_none``。

    同时钉住台账:``dz_m`` 必须是 ``None`` 而不是「拿出厂符号算出来的数」。
    一个方向可能反了的位移写进 rungs,事后没人分得出它是哪个方向 ——
    而 ``_no_displacement`` 只用 ``abs(dz)``,本来就不需要这个符号。
    """
    from mast.core import instrument_profile as ip

    # 其余键照抄 fixture,唯独**不写** z_extend_sign。
    ip.set_profile({"xy_prewithdraw_steps": 100, "xy_move_chunk_steps": 50,
                    "xy_site_spacing_steps": 200, "xy_axis_step_budget": 5000,
                    "z_recede_min_nm": 1.0, "retract_motor_dir": "z+"})
    ctx = _healthy_rig(away_code=_Z_PLUS)
    res = RelocateCoarseXY().run_composite(ctx, _params())

    assert res.success is False
    rungs = _rungs(res)
    assert rungs and rungs[0]["verdict"] == "no_sign"
    assert rungs[0]["dz_m"] is None, "没有符号却还是把一个有向位移记进了台账"
    # 处方指向没填的那一项,**不是**指向 z_settle_timeout_s(那是 unsettled 的方子)。
    assert "z_extend_sign" in (res.error or "")
    assert "z_settle_timeout_s" not in (res.error or "")
    # 停在第一级,而且没有横移。
    z_steps = [c[1][1] for c in ctx.moves() if c[1][0] in (_Z_PLUS, _Z_MINUS)]
    assert z_steps == [1]
    assert not ctx.lateral()


def test_the_verdict_does_not_depend_on_how_fast_the_loop_is():
    """The defect in one sentence: the answer was a function of the wait.

    Three rigs, identical except for how fast the piezo ramps — i.e. how many
    polls the loop needs to find the surface, which on real hardware moves with
    temperature, gain and tip. The verdict AND the Z it was computed from must
    be identical, because none of that changes where the surface is."""
    seen = set()
    for travel in (400e-9, 120e-9, 35e-9):
        ctx = _healthy_rig(travel_per_poll=travel)
        res = RelocateCoarseXY().run_composite(ctx, _params())
        rungs = _rungs(res)
        assert rungs
        seen.add((rungs[0]["verdict"], round(rungs[0]["z_after_m"], 12)))
    assert len(seen) == 1, (
        f"loop speed changed the answer: {seen} — the check is timing the ramp, "
        "not measuring the gap"
    )


def test_a_mid_ramp_reading_is_never_what_gets_judged():
    """The number that must never reach the verdict.

    A slow rig is one where a fixed short wait lands in transit. The judged Z has
    to be where the loop CAME TO REST, not a point it happened to be passing —
    and the settle record has to show it waited longer than one window to say so.
    """
    ctx = _healthy_rig(travel_per_poll=25e-9)   # ~30 polls of travel per settle
    res = RelocateCoarseXY().run_composite(ctx, _params())
    rungs = _rungs(res)
    assert rungs
    first = rungs[0]
    # After one step of 200 nm away, the surface sits 200 nm further out.
    assert first["z_after_m"] == pytest.approx(200e-9, abs=1e-12)
    settle = first["settle"]
    assert settle["settled"] is True and settle["state"] == "tracking"
    assert settle["samples"] > 5, (
        "converged inside a single window — the ramp was not actually observed"
    )


def test_a_loop_that_never_settles_is_unsettled_not_approaching(monkeypatch):
    """Timing out must get its own answer.

    Reporting "approaching" for a reading we could not take is what made the
    broken judge indistinguishable from a backwards cable — it is also how a
    fail-closed check ends up permanently closed. It must say so, and stop."""
    monkeypatch.setattr(RelocateCoarseXY, "_poll_interval_s", 0.005)
    monkeypatch.setattr(RelocateCoarseXY, "_settle_timeout_s", 0.08)
    # Baseline settles fine; the loop stops converging once the ladder starts,
    # so the timeout lands on a RUNG rather than on the baseline.
    ctx = _healthy_rig(drifts_after_z_moves=1)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "没有得出结论" in (res.error or "")
    assert [r["verdict"] for r in _rungs(res)][-1] == "unsettled", (
        "a reading that could not be taken must not wear the 'approaching' label"
    )
    assert "不是判定为逼近" in (res.error or ""), (
        "the message has to disown the approach verdict explicitly — 'the check "
        "could not run' and 'the tip is closing in' being one word is the defect"
    )
    assert "z_settle_timeout_s" in (res.error or ""), (
        "a timeout that does not name the budget leaves the operator guessing"
    )
    assert not ctx.lateral(), "must not slide sideways on an inconclusive check"
    assert "Motor_StopMove" in ctx.verbs()


def test_an_unusable_baseline_refuses_instead_of_retracting_blind(monkeypatch):
    """No baseline ⇒ no self-check ⇒ no ladder.

    Every rung is a subtraction against the baseline. Running the ladder without
    one is the bare MotorMove this composite exists to replace, so it refuses —
    loudly, before a single coarse step."""
    monkeypatch.setattr(RelocateCoarseXY, "_poll_interval_s", 0.005)
    monkeypatch.setattr(RelocateCoarseXY, "_settle_timeout_s", 0.08)
    ctx = _healthy_rig(drifts_forever=True)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert not [c for c in ctx.moves() if c[1][0] in (_Z_PLUS, _Z_MINUS)], (
        "not one coarse step may be taken before the baseline is established"
    )


def test_going_out_of_piezo_range_is_receding_but_reported_as_a_bound():
    """The normal outcome once the ladder has done its job.

    The piezo runs to its limit and finds nothing: the tip is farther than it can
    reach. That is receding — and the number is a lower bound, not a distance, so
    it must not be printed as though it were a measurement."""
    # Surface already near the extend limit: one 200 nm step puts it out of range.
    ctx = _healthy_rig(gap_m=700e-9)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    rungs = _rungs(res)
    assert rungs and rungs[0]["verdict"] == "receding"
    assert rungs[0]["settle"]["state"] == "out_of_range"
    assert "极限" in rungs[0]["reason"] and "实际更远" in rungs[0]["reason"]


def test_two_rail_readings_are_ambiguous_not_receding():
    """A limit is the same number whichever way the stage went.

    Calling that "receding" would be a guard that isn't: it is equally true of a
    tip going away, a tip coming closer but still out of range, and a dead
    preamp. It reports ambiguous — and the approaching half stays live, because a
    tip that closed in would pull the piezo off the rail."""
    ctx = _healthy_rig(gap_m=5e-6)      # far beyond reach from the start
    res = RelocateCoarseXY().run_composite(ctx, _params())
    rungs = _rungs(res)
    assert rungs
    assert rungs[0]["verdict"] == "ambiguous"
    assert "证明不了" in rungs[0]["reason"]
    assert res.success is True, "an honest 'cannot prove' must not strand the run"


def test_a_tip_that_comes_back_into_range_is_approaching():
    """A railed baseline still catches the dangerous direction.

    Out of reach before the step, holding a junction after it, means the tip got
    CLOSER — the one verdict that must survive a baseline that could not measure
    a distance."""
    from mast.core import instrument_profile as ip

    # ``z_extend_sign`` 跟着一起声明:``set_profile`` 是整体替换,不写就等于
    # 「这台机器没答过」⇒ 自检拒判(no_sign)。``_Rig`` 是「退针 ⇒ z 升」的机器。
    ip.set_profile({"retract_motor_dir": "z-",     # config disagrees with wiring
                    "z_extend_sign": "+1"})
    ctx = _healthy_rig(gap_m=850e-9, away_code=_Z_PLUS)   # starts out of range
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "逼近" in (res.error or "")
    rungs = _rungs(res)
    assert rungs and rungs[0]["verdict"] == "approaching"
    assert "靠近" in rungs[0]["reason"]


# ── The stage that never moved (KNOWN_ISSUES 2.28) ──────────────────────────

def test_a_motor_that_accepts_commands_and_moves_nothing_is_refused():
    """The coarse-approach HV is off: every command succeeds, the stage sits still.

    Nothing else in the composite notices — the step counter is unsupported and
    the current watch only says "nothing is touching the tip", which is just as
    true of a stage that never moved."""
    ctx = _healthy_rig(motor_dead=True)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert not ctx.lateral(), "must not slide a stage we just proved is not moving"
    assert "没有产生位移" in (res.error or "")


def test_the_refusal_says_measured_zero_not_could_not_measure():
    """Two different facts, and collapsing them is the whole family of defects.

    It also has to name what to go and check — a refusal that does not say
    "look at the coarse-approach HV" just stalls the operator."""
    ctx = _healthy_rig(motor_dead=True)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    err = res.error or ""
    assert "测出来是零" in err and "不是「测不出来」" in err
    assert "粗逼近高压" in err and "驱动幅度" in err and "机械卡死" in err


def test_the_refused_move_never_reaches_the_odometer():
    """The half that matters more than the refusal.

    A relocation that did not happen must not advance ``coord_epoch``, must not
    add a coarse-map site, and must not feed ``lateral_coarse_move_info`` — those
    write PERSISTENT state, and a phantom displacement in them cannot be
    reconstructed after the fact. Every later "have we scanned here?" would rest
    on a coordinate frame the stage never entered."""
    from mast.core.runtime import lateral_coarse_move_info

    ctx = _healthy_rig(motor_dead=True)
    res = RelocateCoarseXY().run_composite(ctx, _params(steps=120))
    assert res.success is False
    payload = {"skill": "RelocateCoarseXY", "success": res.success,
               "data": res.data, "params": _params(steps=120)}
    assert lateral_coarse_move_info(payload) is None, (
        "the odometer accepted a move the stage never made"
    )
    # And the fallback path cannot resurrect it either: the reader falls back to
    # params for direction/steps, so success must be the gate that stops it.
    assert res.data["steps"] == 0, "no lateral step was actually taken"


def test_a_working_rig_is_not_accused_of_being_stuck():
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is True
    assert "没有产生位移" not in (res.error or "")


def test_without_a_ruler_the_stuck_check_abstains_rather_than_guessing():
    """The counter-example that killed the simpler "all rungs ambiguous" rule.

    A tip starting BEYOND piezo range gives a railed baseline and railed rungs —
    ambiguous every time, legitimately — while the motor may be working fine.
    Refusing here would block relocation after every sample-change retract,
    failed approach, or manual withdraw."""
    ctx = _healthy_rig(gap_m=5e-6, motor_dead=True)   # out of reach AND stuck
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is True, (
        "with no gap reading at either end there is no evidence either way, and "
        "a guard that fires on absent evidence is the rule we rejected"
    )
    assert all(r["has_ruler"] is False for r in _rungs(res))


def test_the_stuck_check_is_cumulative_not_per_rung():
    """One coarse step may legitimately be under the threshold.

    Stick-slip travel shrinks sharply at low temperature, so vetoing on the
    single-step probe rung would false-positive on a cold rig that is moving.
    The ladder's later rungs are 10 and 89 steps."""
    from mast.core import instrument_profile as ip

    # 0.4 nm per step against a 1 nm threshold: rung 1 (1 step) is under it,
    # but the cumulative 100 steps are far over.
    ip.set_profile({"z_recede_min_nm": 1.0, "z_extend_sign": "+1"})
    ctx = _healthy_rig(step_m=0.4e-9)
    res = RelocateCoarseXY().run_composite(ctx, _params())
    rungs = _rungs(res)
    assert rungs[0]["verdict"] == "ambiguous", "the 1-step probe is under threshold"
    assert res.success is True, (
        "a per-rung veto would have refused a rig that is moving perfectly well"
    )
    assert abs(rungs[-1]["dz_m"]) == pytest.approx(40e-9, abs=1e-12), "100 × 0.4 nm"


def test_a_current_still_flowing_after_the_retract_blocks_the_move():
    ctx = _healthy_rig()
    ctx.current_hook = lambda: 5e-10
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert "不要横向移动" in (res.error or "") or "逼近" in (res.error or "")
    assert not ctx.lateral()


def test_feedback_is_turned_off_before_the_lateral_move():
    """With feedback on, the piezo chases the surface sliding underneath and
    drives the tip straight back down."""
    ctx = _healthy_rig()
    RelocateCoarseXY().run_composite(ctx, _params())
    off_idx = max(i for i, c in enumerate(ctx.calls)
                  if c[0] == "ZCtrl_OnOffSet" and c[1] == (0,))
    first_lateral = min(i for i, c in enumerate(ctx.calls)
                        if c[0] == "Motor_StartMove" and c[1][0] in _LATERAL)
    assert off_idx < first_lateral


# ── The move itself ─────────────────────────────────────────────────────────

def test_the_move_is_chunked_so_something_can_look_in_between():
    ctx = _healthy_rig()
    RelocateCoarseXY().run_composite(ctx, _params(steps=120))
    assert [c[1][1] for c in ctx.lateral()] == [50, 50, 20]
    assert all(c[1][0] == 0 for c in ctx.lateral()), "x+ is direction code 0"


def test_current_appearing_mid_move_stops_it_immediately():
    """Nothing should be electrically connected while the stage slides. Any
    current means the clearance was not what we proved it was."""
    ctx = _healthy_rig()
    # Keyed on "has a LATERAL move happened yet", not on a call count: the
    # clearance ladder's length depends on config, so a count would silently
    # start testing the wrong phase the moment a default changed.
    ctx.current_hook = lambda: 5e-10 if ctx.lateral_moves else None
    res = RelocateCoarseXY().run_composite(ctx, _params(steps=200))
    assert res.success is False
    assert "横向移动中检测到电流" in (res.error or "")
    assert len(ctx.lateral()) < 4, "it must stop mid-move, not finish the plan"
    assert "Motor_StopMove" in ctx.verbs()


def test_it_reports_the_steps_actually_moved_not_the_steps_requested():
    """The odometer is fed from this number. An odometer fed the REQUEST instead
    of the OUTCOME is fiction — and a half-finished move still moved the stage."""
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(steps=120))
    assert res.data["steps"] == 120
    assert res.data["requested_steps"] == 120


# ── The contract with the recorder ──────────────────────────────────────────

def test_the_result_carries_what_the_recorder_reads():
    """This composite issues Motor_StartMove through safe_call directly, so the
    recorder never sees a MotorMove payload for it. Without these two fields the
    stage moves, every recorded coordinate silently becomes meaningless, and
    coord_epoch never advances to say so."""
    from mast.core.runtime import lateral_coarse_move_info

    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(steps=120))
    assert res.success is True
    info = lateral_coarse_move_info({"skill": "RelocateCoarseXY", "success": True,
                                     "data": res.data, "params": {}})
    assert info == {"direction": "x+", "steps": 120, "partial": False}


def test_a_failed_relocation_does_not_advance_the_generation():
    from mast.core.runtime import lateral_coarse_move_info

    vac.set_pressure_source(None)
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is False
    assert lateral_coarse_move_info(
        {"skill": "RelocateCoarseXY", "success": False, "data": res.data}) is None


# ── Verification and re-approach ────────────────────────────────────────────

def test_an_absent_step_counter_is_reported_as_unverified():
    """Only Attocube ANC150 exposes one. Printing a reassuring tick for a check
    that never ran is how "verified" stops meaning anything."""
    ctx = _healthy_rig(errors={"Motor_StepCounterGet": "not supported"})
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.success is True
    assert "verify" in str(res.data.get("_progress", ""))


def test_reapproach_goes_through_approach_tip_never_a_coarse_z_step():
    """AutoApproach stops on current feedback. An open-loop coarse Z step toward
    the sample has no stop and is the one action gated to a human."""
    ctx = _healthy_rig()
    RelocateCoarseXY().run_composite(ctx, _params(reapproach=True))
    assert ("ApproachTip", {}) in ctx.ran
    assert not [c for c in ctx.moves() if c[1][0] == _Z_MINUS], "no coarse Z-approach"


def _failed_reapproach(**kw):
    from mast.core.types import SkillResult

    return _healthy_rig(sub_results={
        "ApproachTip": SkillResult(skill_name="ApproachTip", success=False,
                                   error="coarse range exhausted")}, **kw)


def test_a_failed_reapproach_says_the_move_already_happened():
    """The worst possible follow-up would be relocating again because the
    re-approach failed — the stage has already slid to a new patch."""
    res = RelocateCoarseXY().run_composite(_failed_reapproach(),
                                           _params(reapproach=True))
    assert res.success is False
    assert "横向移动本身已经完成" in (res.error or "")
    assert "不要再移动一次" in (res.error or "")


def test_the_failure_message_does_not_narrate_the_bookkeeping():
    """The phase reports completed movement, not a ledger update performed later by a separate component."""
    res = RelocateCoarseXY().run_composite(_failed_reapproach(),
                                           _params(reapproach=True, steps=240))
    err = res.error or ""
    assert "坐标代次已经推进" not in err
    assert "240 步" in err, "说了「移动完成」就要说清楚移动了多少"
    assert "get_coarse_map" in err, "要给一个能查证的去处,而不是让人信这句话"


# ── the ledger survives a failure that came AFTER the move ──────────────────

def test_a_failed_reapproach_does_not_erase_the_lateral_move_from_the_ledger():
    """A failed re-approach must still pass the completed synthetic lateral displacement to the recorder."""
    from mast.core.runtime import lateral_coarse_move_info

    params = _params(reapproach=True, steps=240)
    res = RelocateCoarseXY().run_composite(_failed_reapproach(), params)
    assert res.success is False
    info = lateral_coarse_move_info(
        {"skill": "RelocateCoarseXY", "success": res.success,
         "data": res.data, "params": params})
    assert info is not None, "换区实际发生了,账本却收不到"
    assert info["steps"] == 240
    assert info["partial"] is True, "记下来了,但不能记成一次干净的换区"


def test_the_ledger_is_fed_the_outcome_not_the_request():
    """Watchdog abort mid-move: the odometer must get the steps actually taken.

    ``xy_move_chunk_steps`` is 50 by default, so a rig that trips the current
    watch after the first chunk moved 50 of the 240 asked for."""
    from mast.core.runtime import lateral_coarse_move_info

    ctx = _healthy_rig()
    # Anything above _MOVE_DANGER_CURRENT_A during the lateral move aborts it.
    calls = {"n": 0}

    def _hook():
        calls["n"] += 1
        return 1e-9 if ctx.lateral_moves >= 1 else None

    ctx.current_hook = _hook
    params = _params(reapproach=True, steps=240)
    res = RelocateCoarseXY().run_composite(ctx, params)
    assert res.success is False
    taken = res.data["lateral_steps_taken"]
    assert 0 < taken < 240, f"expected a partial move, got {taken}"
    info = lateral_coarse_move_info(
        {"skill": "RelocateCoarseXY", "success": False,
         "data": res.data, "params": params})
    assert info["steps"] == taken


def test_a_refusal_before_any_lateral_step_still_reaches_nothing():
    """The half that must NOT change. §2.28: the motor takes commands and moves
    nothing; the composite refuses in the clearance ladder, before a single
    lateral step. Zero taken ⇒ nothing recorded, even though the params still
    carry the full request."""
    from mast.core.runtime import lateral_coarse_move_info

    params = _params(steps=120)
    res = RelocateCoarseXY().run_composite(_healthy_rig(motor_dead=True), params)
    assert res.success is False
    assert res.data["lateral_steps_taken"] == 0
    assert lateral_coarse_move_info(
        {"skill": "RelocateCoarseXY", "success": False,
         "data": res.data, "params": params}) is None


# ── dry_run ─────────────────────────────────────────────────────────────────

def test_dry_run_exercises_every_phase_without_commanding_a_lateral_move():
    """For acceptance without hardware: the preflight, the clearance self-check
    and the bookkeeping all really run."""
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(dry_run=True, steps=120))
    assert res.success is True
    assert res.data["dry_run"] is True
    assert not ctx.lateral(), "dry_run must not slide the stage"
    assert [c for c in ctx.moves() if c[1][0] in (_Z_PLUS, _Z_MINUS)], (
        "but the clearance retract is real — that is what is being rehearsed"
    )
    assert res.data["checks"]["vacuum"]["allow"] is True


def test_dry_run_never_ages_a_single_coordinate():
    """Intended dry-run steps must not advance the coordinate epoch or create a coarse-map position."""
    from mast.core.runtime import lateral_coarse_move_info

    params = _params(dry_run=True, steps=120)
    res = RelocateCoarseXY().run_composite(_healthy_rig(), params)
    assert res.success is True
    assert res.data["lateral_steps_taken"] == 0
    assert lateral_coarse_move_info(
        {"skill": "RelocateCoarseXY", "success": True,
         "data": res.data, "params": params}) is None


def test_temperature_is_recorded_with_the_move():
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params())
    assert res.data.get("temperature_k") == pytest.approx(4.2)


# ── Argument validation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [{"axis": "z"}, {"direction": "up"},
                                 {"axis": ""}, {"direction": ""}])
def test_an_invalid_axis_or_direction_is_refused_loudly(bad):
    ctx = _healthy_rig()
    res = RelocateCoarseXY().run_composite(ctx, _params(**bad))
    assert res.success is False
    assert not ctx.calls, "must not touch the instrument on a malformed request"
