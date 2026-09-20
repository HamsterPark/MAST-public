"""不使用 lock-in 信号的流程应通过统一技能入口关闭调制，避免设计内纹波干扰电流形态判据；显式使用调制的流程仍保留监控上下文。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.skills.composite._preflight import (  # noqa: E402
    close_modulation,
    ensure_modulation_off,
    uses_lockin,
    wants_modulation_off,
)


class _Ctx:
    """``mod_on`` 三态:True / False / None(读不到)。"""

    def __init__(self, mod_on=True, *, set_fails=False, tuple_shape=False):
        self.mod_on = mod_on
        self.set_fails = set_fails
        self.tuple_shape = tuple_shape
        self.calls: list[tuple] = []

    def safe_call(self, method, *args, **kw):
        self.calls.append((method, args))
        err = ""
        rv = ("", b"", [])
        if method == "LockIn_ModOnOffGet":
            if self.mod_on is None:
                err = "no reply"
            else:
                v = 1 if self.mod_on else 0
                rv = ("", b"", [(v,)] if self.tuple_shape else [v])
        elif method == "LockIn_ModOnOffSet":
            if self.set_fails:
                err = "Access Denied"
            elif self.mod_on is not None:
                # 写要生效,否则「关上了没有」这条钉子是空的。
                # ``mod_on is None`` 是「读不回来」的机器 —— 它不会因为被写了一次
                # 就忽然读得回来了,所以那种机器上状态保持未知。
                self.mod_on = bool(int(args[1]))

        class _R:
            error = err
            return_value = rv
            method = ""
            args = ()
        return _R()

    def verbs(self):
        return [m for m, _ in self.calls]


# ══════════════════════════════════════════════════════════════════════
# 判据:谁该被关,谁不该被碰
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def registry():
    r = SkillRegistry()
    r.discover("mast.skills.builtins", "mast.skills.composite")
    return {m.name: m for m in r.list_skills()}


@pytest.mark.parametrize("name", [
    "ScanAt", "FullScan", "TipShape", "BiasPulse", "ForgeAuTip",
    "RelocateCoarseXY", "AutoApproach", "PrepareNobleTip",
])
def test_physical_action_skills_want_the_modulation_off(registry, name):
    """交底点名的那几族:它们不看 lock-in 的解调输出,只会被纹波污染。"""
    assert wants_modulation_off(registry[name]), name


@pytest.mark.parametrize("name", [
    "AutoPhase", "ApplyLockInPreset", "ConfigureLockIn", "ListLockInPresets",
])
def test_lockin_users_are_never_touched(registry, name):
    """**用 lock-in 的流程自己开自己关** —— 替它们关掉就是把它们弄坏。"""
    assert not wants_modulation_off(registry[name]), name


@pytest.mark.parametrize("name", ["GetBias", "GetCurrent", "GetLockInConfig"])
def test_read_skills_never_touch_the_operators_modulation(registry, name):
    """读类技能不改仪器状态。凭什么替用户关他的调制。"""
    assert not wants_modulation_off(registry[name]), name


def test_uses_lockin_matches_by_name_and_by_tag():
    class _M:
        tags = ("scan",)
    assert uses_lockin("MeasureDidvSomething", _M())
    assert uses_lockin("AutoPhase", _M())

    class _Tagged:
        tags = ("lockin", "write")
    assert uses_lockin("SomethingElse", _Tagged())


# ══════════════════════════════════════════════════════════════════════
# 动作:开着就关,并留痕
# ══════════════════════════════════════════════════════════════════════

def test_modulation_on_is_turned_off_and_recorded():
    ctx = _Ctx(mod_on=True)
    note = ensure_modulation_off(ctx, skill_name="ScanAt")
    assert ("LockIn_ModOnOffSet", (1, 0)) in ctx.calls
    assert note["modulation_was_on_turned_off"] is True
    assert "ScanAt" in note["modulation_note"]


def test_modulation_already_off_is_left_alone():
    ctx = _Ctx(mod_on=False)
    assert ensure_modulation_off(ctx, skill_name="ScanAt") is None
    assert "LockIn_ModOnOffSet" not in ctx.verbs(), "本来就关着还写了一次"


def test_an_unreadable_state_does_not_blind_fire_a_write():
    """**读不到 ≠ 开着。**

    在一个读不回状态的机器上每次都盲发一条关调制命令,是拿一个未知去换一个写操作;
    极性与 ``ctx_lockin_on`` 一致 —— 只有显式 True 才行动。
    """
    ctx = _Ctx(mod_on=None)
    assert ensure_modulation_off(ctx, skill_name="ScanAt") is None
    assert "LockIn_ModOnOffSet" not in ctx.verbs()


def test_the_tuple_wrapped_reply_shape_is_understood():
    """1-元组形态的回包也要认(§2.21/§2.31)。"""
    ctx = _Ctx(mod_on=True, tuple_shape=True)
    assert ensure_modulation_off(ctx, skill_name="ScanAt") is not None


def test_a_failed_close_is_reported_not_swallowed():
    """关不掉要说出来 —— 静默失败会让用户以为调制已经关了。"""
    ctx = _Ctx(mod_on=True, set_fails=True)
    note = ensure_modulation_off(ctx, skill_name="ScanAt")
    assert note["modulation_was_on_turned_off"] is False
    assert "关不掉" in note["modulation_note"]


def test_the_hygiene_action_never_raises():
    """卫生动作不是这个技能的目的,它失败不该把技能带走。"""
    class _Broken:
        def safe_call(self, *a, **k):
            raise RuntimeError("link down")

    assert ensure_modulation_off(_Broken(), skill_name="ScanAt") is None


# ══════════════════════════════════════════════════════════════════════
# 括号的右半边:用完把它关回去(缺陷⑪,2026-08-05 实机)
# ══════════════════════════════════════════════════════════════════════

def test_close_turns_it_off_and_confirms_by_readback():
    ctx = _Ctx(mod_on=True)
    note = close_modulation(ctx, skill_name="AutoPhase")
    assert ("LockIn_ModOnOffSet", (1, 0)) in ctx.calls
    assert note["modulation_was_on_before"] is True
    assert note["modulation_off_after"] is True
    assert note["modulation_closed_by"] == "AutoPhase"


def test_the_two_halves_have_opposite_polarity_on_purpose():
    """**别把这两个函数合并。** 读不到状态时它们必须做相反的事。

    * 开跑前的 ``ensure_modulation_off`` 要声称「我发现它开着,替你关了」——
      这句话需要知道之前的状态,所以读不到就**不动手**(与 ``ctx_lockin_on``
      只认显式 True 同极性)。
    * 用完之后的 ``close_modulation`` 说的是「我把它关了」—— 这句话**不需要**
      之前的状态,目标状态是确定的。读不到就不关的话,「读不回状态」会变成
      「调制留在开着」,而那正是缺陷⑪。

    合并成一个带 flag 的函数只会让下一个人挑错分支。**判据不同就是两个函数。**
    """
    before = _Ctx(mod_on=None)
    assert ensure_modulation_off(before, skill_name="ScanAt") is None
    assert "LockIn_ModOnOffSet" not in before.verbs()

    after = _Ctx(mod_on=None)
    note = close_modulation(after, skill_name="AutoPhase")
    assert ("LockIn_ModOnOffSet", (1, 0)) in after.calls
    # 关了,但**没确认**关上了 —— 三态里的第三态,不许塌成 True 或 False。
    assert note["modulation_off_after"] is None
    assert "没确认" in note["modulation_note"]


def test_close_never_writes_the_amplitude():
    """关它不需要重写任何值。缺陷⑧就是关的时候顺手带了个省略的幅度。"""
    ctx = _Ctx(mod_on=True)
    close_modulation(ctx, skill_name="AutoPhase")
    for verb in ("LockIn_ModAmpSet", "LockIn_ModPhasFreqSet", "LockIn_ModPhasSet"):
        assert verb not in ctx.verbs(), verb


def test_a_close_that_does_not_take_is_reported_as_still_on():
    """发了命令 ≠ 它关上了。回读说还开着就要说「不要按已关闭继续」。"""
    class _Ignores(_Ctx):
        def safe_call(self, method, *args, **kw):
            if method == "LockIn_ModOnOffSet":
                args = (args[0], 1)          # 硬件不理会,继续开着
            return super().safe_call(method, *args, **kw)

    note = close_modulation(_Ignores(mod_on=True), skill_name="AutoPhase")
    assert note["modulation_off_after"] is False
    assert "不要按已关闭继续" in note["modulation_note"]


def test_a_failed_close_command_is_reported_not_swallowed():
    ctx = _Ctx(mod_on=True, set_fails=True)
    note = close_modulation(ctx, skill_name="AutoPhase")
    assert note["modulation_off_after"] is False
    assert "失败" in note["modulation_note"]


def test_the_close_never_raises():
    """收尾动作失败不该把技能带走 —— 相位已经对好了。"""
    class _Broken:
        def safe_call(self, *a, **k):
            raise RuntimeError("link down")

    note = close_modulation(_Broken(), skill_name="AutoPhase")
    assert note["modulation_off_after"] is None


def test_the_close_note_tells_you_how_to_get_the_modulation_back():
    """代价说清楚:调制关着时 dI/dV 曲线照样画得出来,而它是零信号。"""
    note = close_modulation(_Ctx(mod_on=True), skill_name="AutoPhase")
    assert "ApplyLockInPreset" in note["modulation_note"]


# ══════════════════════════════════════════════════════════════════════
# 接线:留痕真的到得了调用方
# ══════════════════════════════════════════════════════════════════════

def test_the_note_reaches_the_caller_through_the_adapter():
    """断言落在「调用方拿到了什么」,不是「函数返回了什么」。

    留痕只写进日志的话,用户事后查不到是谁动了他的调制。
    """
    from mast.agents._shared.skill_adapter import wrap_skill
    from mast.core.types import (
        SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
    )
    from mast.skills.base import BaseSkill

    class _FakeScan(BaseSkill):
        def metadata(self):
            return SkillMetadata(
                name="FakeScanForHygiene", version="1.0.0",
                category=SkillCategory.WRITE, safety_level=SafetyLevel.AUTO,
                description="x", parameters=[], tags=["scan"],
                composition_level=0)

        def execute(self, context, params):
            return SkillResult(skill_name="FakeScanForHygiene", success=True,
                               data={"ok": True})

    ctx = _Ctx(mod_on=True)
    tool = wrap_skill(_FakeScan, lambda: ctx)
    tool.func(tool_call_id="t-1", state={})
    assert ("LockIn_ModOnOffSet", (1, 0)) in ctx.calls, (
        f"适配器没有在开跑前关调制:{ctx.verbs()}")


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ══════════════════════════════════════════════════════════════════════
# 交互:自动关调制之后,dI/dV 标定必须**拒绝记录**
# ══════════════════════════════════════════════════════════════════════
#
# 「用完即关」让进针类流程在开跑前自动关调制,于是 ApproachTip 的 dI/dV 标定走到
# 时调制**通常是关的**。不加闸的话,这次改动会把一个「没有被调制驱动的通道读数」
# 当成 dI/dV 写进持久标定库 —— 而它之后每次被引用都不会自己声明是假的。
#
# 这条是**作者这次改动引入的回归**,所以钉在这里而不是别处。


def _approach_ctx(mod_on: bool):
    written: list = []

    class _C:
        def safe_call(self, method, *args, **kw):
            rv = ("", b"", [])
            if method == "LockIn_ModOnOffGet":
                rv = ("", b"", [1 if mod_on else 0])
            elif method == "Signals_ValGet":
                rv = ("", b"", [3.3e-9])
            elif method == "Bias_Get":
                rv = ("", b"", [1.0])
            elif method == "ZCtrl_SetpntGet":
                rv = ("", b"", [1e-10])

            class _R:
                error = ""
                return_value = rv
                method = ""
                args = ()
            return _R()

    return _C(), written


@pytest.mark.parametrize("mod_on,expect_recorded", [(True, True), (False, False)])
def test_didv_calibration_only_records_with_the_modulation_on(
        monkeypatch, mod_on, expect_recorded):
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.approach import ApproachTip

    recorded: list = []
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: 8 if k == "lockin_signal_index" else d,
                        raising=False)
    monkeypatch.setattr(ip, "set_calibration",
                        lambda *a, **k: recorded.append((a, k)), raising=False)

    ctx, _ = _approach_ctx(mod_on)
    ApproachTip()._record_didv_calibration(ctx, setpoint_a=1e-10)
    assert bool(recorded) is expect_recorded, (
        "调制关着时把一个非 dI/dV 的读数写进了标定库"
        if recorded else "调制开着时该记的标定没记")


def test_an_unreadable_modulation_state_also_blocks_the_calibration(monkeypatch):
    """「没读到」不是「开着」—— 三态里只有第一种才准写。"""
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.approach import ApproachTip

    recorded: list = []
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: 8 if k == "lockin_signal_index" else d,
                        raising=False)
    monkeypatch.setattr(ip, "set_calibration",
                        lambda *a, **k: recorded.append(a), raising=False)

    class _Dead:
        def safe_call(self, method, *args, **kw):
            class _R:
                error = "no reply" if method == "LockIn_ModOnOffGet" else ""
                return_value = ("", b"", [3.3e-9])
                method = ""
                args = ()
            return _R()

    ApproachTip()._record_didv_calibration(_Dead(), setpoint_a=1e-10)
    assert not recorded


# ══════════════════════════════════════════════════════════════════════
# 退针保留调制(2026-08-06 拍板)—— 判据不用 lock-in,但人要用它导航
# ══════════════════════════════════════════════════════════════════════

def test_retract_keeps_its_modulation(registry):
    """RetractForSampleChange 保留调制开关状态。
    退针流程由对应上下文抑制策略处理调制影响，不在入口无条件关闭调制。
    """
    assert not wants_modulation_off(registry["RetractForSampleChange"])


@pytest.mark.parametrize("name", ["ScanAt", "FullScan", "AutoApproach",
                                  "ApproachTip", "ForgeAuTip"])
def test_the_exemption_did_not_leak_to_anyone_else(registry, name):
    """**只摘退针一个。** 进针/扫图/修针照旧自动关。

    豁免用**技能名**而不是去动 MODULATION_OFF_TAGS —— 动标签会连带放行一大片,
    而那一片里的进针和扫图正是纹波污染判据代价最大的地方。
    """
    assert wants_modulation_off(registry[name]), name


def test_keep_and_user_are_two_different_reasons():
    """两张表刻意分开:一张是「它拿 lock-in 当判据」,一张是「人要用它导航」。

    合成一张就是把两个不同的理由塞进一个名字,下一个人会按错的那个理由增删条目。
    """
    from mast.skills.composite._preflight import (
        MODULATION_KEEP_PATTERNS,
        MODULATION_USER_PATTERNS,
        keeps_modulation,
        uses_lockin,
    )

    assert not (MODULATION_KEEP_PATTERNS & MODULATION_USER_PATTERNS)
    assert keeps_modulation("RetractForSampleChange") is True
    assert uses_lockin("RetractForSampleChange") is False
