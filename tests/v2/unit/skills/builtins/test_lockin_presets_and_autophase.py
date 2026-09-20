"""Lock-in 参数与 AutoPhase 的合成协议测试。

检查命令序列、不写调制相位、低信号时拒绝写入，以及退出时关闭调制。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import struct  # noqa: E402

import pytest  # noqa: E402

from mast.core import lockin_presets as lp  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.builtins import lockin_presets_skills as lps  # noqa: E402
from mast.skills.builtins.lockin import (  # noqa: E402
    ConfigureLockIn,
    GetLockInConfig,
)
from mast.skills.builtins.lockin_presets_skills import (  # noqa: E402
    ApplyLockInPreset,
    AutoPhase,
    ListLockInPresets,
)


def _f32(x: float) -> float:
    """float32 量化 —— 硬件回读长的就是这个样子(0.02 → 0.019999999552965164)。"""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


@pytest.fixture()
def profile(monkeypatch):
    """可控的仪器档案。"""
    store: dict = {}
    from mast.core import instrument_profile as ip
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: store.get(k, d), raising=False)
    monkeypatch.setattr(lp._iprof, "get_config",
                        lambda k, d=None: store.get(k, d), raising=False)
    return store


class _Rec:
    """一条 Nanonis 调用记录 —— 带方法名,好让钉子断言「那条命令出现过」。"""

    def __init__(self, method, args, return_value, error=""):
        self.method = method
        self.args = args
        self.return_value = return_value
        self.error = error


class _Ctx:
    """浮点回包由独立替身生成，并经真实配置读取逻辑解析。"""

    #: 派发给真技能的名字。写死一张小表而不是查注册表 —— 单元测试不该依赖注册顺序。
    _REAL = {"ConfigureLockIn": ConfigureLockIn, "GetLockInConfig": GetLockInConfig}

    def __init__(self, *, xy=(1.0, 1.0), phase=0.0, sub=None,
                 amp=0.0, freq=0.0, mod_on=False, aborted=False,
                 mod_on_readable=True):
        self.calls: list[str] = []
        self.records: list = []
        self.ran: list[tuple] = []
        self.xy = xy
        self.phase = phase
        self.sub = sub or {}
        self.amp = amp
        self.freq = freq
        self.mod_on = mod_on
        self.aborted = aborted
        self.mod_on_readable = mod_on_readable

    def safe_call(self, method, *args, **kw):
        self.calls.append(method)
        rv, err = ("", b"", []), ""
        if method == "Signals_ValsGet":
            rv = ("", b"", [2, [self.xy[0], self.xy[1]]])
        elif method == "LockIn_DemodPhasGet":
            rv = ("", b"", [self.phase])
        elif method == "LockIn_DemodPhasSet":
            self.phase = float(args[1])
        elif method == "LockIn_ModAmpSet":
            self.amp = _f32(args[1])          # 硬件存的是 float32
        elif method == "LockIn_ModPhasFreqSet":
            self.freq = _f32(args[1])
        elif method == "LockIn_ModOnOffSet":
            self.mod_on = bool(int(args[1]))
        elif method == "LockIn_ModAmpGet":
            rv = ("", b"", [self.amp])
        elif method == "LockIn_ModPhasFreqGet":
            rv = ("", b"", [self.freq])
        elif method == "LockIn_ModOnOffGet":
            if not self.mod_on_readable:
                err = "read failed"
            else:
                rv = ("", b"", [int(self.mod_on)])
        elif method == "LockIn_ModPhasGet":
            rv = ("", b"", [0.0])
        elif method == "LockIn_ModPhasSet":
            # 模拟命令锁定。
            err = "Parameter is Locked"
        rec = _Rec(method, args, rv, err)
        self.records.append(rec)
        return rec

    def run(self, name, params):
        self.ran.append((name, dict(params)))
        if name in self.sub:
            return self.sub[name]
        real = self._REAL.get(name)
        if real is not None:
            return real().execute(self, dict(params))
        return SkillResult(skill_name=name, success=True, data={})

    def check_abort(self):
        return self.aborted


# ══════════════════════════════════════════════════════════════════════
# 参数组:档案没填的键不下发;调制侧 phase 永不出现
# ══════════════════════════════════════════════════════════════════════

def test_the_preset_uses_the_keys_that_already_exist(profile):
    """用的是仓里**已有**的 lockin_mod_freq_hz / lockin_mod_amp_v。

    交底里建议的是另一对名字。新建同义键 = 同一个物理量两个来源,迟早各自漂移,
    而且会白白触发「加用户可编辑键是双边动作」那条陷阱。"""
    assert set(lp._PROFILE_KEYS) == {"lockin_mod_freq_hz", "lockin_mod_amp_v"}


def test_unconfigured_keys_are_not_sent(profile):
    profile["lockin_mod_freq_hz"] = 973.0        # 只填频率
    p = lp.resolve()
    assert p.values == {"frequency_hz": 973.0}
    assert "lockin_mod_amp_v" in p.unset
    assert "amplitude_v" not in p.skill_params()


def test_an_empty_profile_refuses_instead_of_silently_doing_nothing(profile):
    p = lp.resolve()
    assert p.usable is False
    res = ApplyLockInPreset().execute(_Ctx(), {})
    assert not res.success
    assert "没有配置" in (res.error or "")


def test_the_preset_never_sends_a_modulator_phase(profile):
    """方案未包含相位键时不得写入相位。"""
    profile.update({"lockin_mod_freq_hz": 973.0, "lockin_mod_amp_v": 0.02})
    args = lp.resolve().skill_params()
    assert "phase_deg" not in args
    ctx = _Ctx()
    res = ApplyLockInPreset().execute(ctx, {})
    assert res.success, res.error
    cfg = next(p for n, p in ctx.ran if n == "ConfigureLockIn")
    assert "phase_deg" not in cfg, f"参数组下发了调制侧相位:{cfg}"
    assert "LockIn_ModPhasSet" not in ctx.calls


def test_a_readback_mismatch_fails_loudly(profile):
    """「写进去了」和「我们看见它在里面」是两句话。"""
    profile.update({"lockin_mod_freq_hz": 973.0, "lockin_mod_amp_v": 0.02})

    class _StubbornFreq(_Ctx):
        """频率写进去不生效的机器 —— 回读比对就是为这种情况存在的。"""

        def safe_call(self, method, *args, **kw):
            if method == "LockIn_ModPhasFreqSet":
                args = (args[0], 500.0)          # 硬件按自己的值来
            return super().safe_call(method, *args, **kw)

    res = ApplyLockInPreset().execute(_StubbornFreq(), {})
    assert not res.success and "回读不一致" in (res.error or "")
    assert "frequency_hz" in (res.error or "")


# ── 缺陷⑩:回读比对自己坏了,把成功报成失败 ─────────────────────────────

def test_the_readback_map_names_keys_the_real_getlockinconfig_returns(profile):
    """回读键使用 GetLockInConfig 的真实字段名，保持 amplitude 与 amplitude_v 的映射。"""
    data = GetLockInConfig().execute(_Ctx(amp=0.02, freq=973.0), {}).data
    for param, key in lps._READBACK_KEYS.items():
        assert key in data, (
            f"{param} 的回读键 {key!r} 不在 GetLockInConfig 的结果里:{sorted(data)}")
    # 探针有效性:这条测试真的分得出对错 —— 老键名在那边是查不到的。
    assert "amplitude_v" not in data


def test_a_float32_readback_of_the_requested_value_counts_as_a_match(profile):
    """请求 0.02,硬件回读 0.019999999552965164 —— 这是**一致**。

    相等比较会把每一次成功写入都判成失败。容差相对 1e-3,比 float32 的相对精度
    (~1.2e-7)宽四个数量级。
    """
    profile.update({"lockin_mod_freq_hz": 973.0, "lockin_mod_amp_v": 0.02})
    rig = _Ctx()
    res = ApplyLockInPreset().execute(rig, {})
    # 探针有效性:假机器真的量化过了,否则这条钉子是空的。
    assert rig.amp == _f32(0.02) and rig.amp != 0.02
    assert res.success, res.error
    assert res.data["readback_verified"] is True
    assert rig.mod_on is True                     # mod_on=true 的语义保留


def test_the_comparison_rule_is_the_shared_one_not_a_local_copy(profile):
    """比较规则只有一处:``skills.verify.values_match``(rel_tol 1e-3)。

    ``ApplyZCtrlPreset`` 里那句注释说得对 —— 第二处比较规则会漂。而这次的教训是它
    不只会漂:**一处自己写的校验会把成功报成失败**。两个断言:用的是同一个函数
    (身份),以及用户看到的报文带着它特有的「相差 N 倍」(自己重新手写一份就会
    换掉这句话,于是这条钉子红)。
    """
    from mast.skills import verify

    assert lps.values_match is verify.values_match

    profile.update({"lockin_mod_freq_hz": 973.0, "lockin_mod_amp_v": 0.02})

    class _HalfFreq(_Ctx):
        def safe_call(self, method, *args, **kw):
            if method == "LockIn_ModPhasFreqSet":
                args = (args[0], float(args[1]) / 2.0)
            return super().safe_call(method, *args, **kw)

    res = ApplyLockInPreset().execute(_HalfFreq(), {})
    assert not res.success
    assert "相差" in (res.error or "") and "倍" in (res.error or "")


def test_the_readback_says_which_key_was_missing(profile):
    """读不回来要说是**哪个键**没有。

    「amplitude_v: 读不回来」当初把人送去查硬件,而坏的是这边的键名 —— 症状措辞
    决定了下一个人往哪儿找。
    """
    profile.update({"lockin_mod_amp_v": 0.02})
    ctx = _Ctx(sub={"GetLockInConfig": SkillResult(
        skill_name="GetLockInConfig", success=True, data={})})
    res = ApplyLockInPreset().execute(ctx, {})
    assert not res.success
    assert "回包里没有 amplitude" in (res.error or "")


def test_a_failed_readback_command_is_not_reported_as_a_wrong_value(profile):
    """「回读命令没跑成」和「硬件里的值不对」是两件事,别混成一句话。"""
    profile.update({"lockin_mod_amp_v": 0.02})
    ctx = _Ctx(sub={"GetLockInConfig": SkillResult(
        skill_name="GetLockInConfig", success=False, error="link down", data={})})
    res = ApplyLockInPreset().execute(ctx, {})
    assert not res.success
    assert "回读命令失败" in (res.error or "") and "link down" in (res.error or "")


def test_list_presets_says_which_keys_are_missing(profile):
    profile["lockin_mod_amp_v"] = 0.02
    out = ListLockInPresets().execute(_Ctx(), {}).data["presets"]
    assert out[0]["unset"] == ["lockin_mod_freq_hz"]
    assert out[0]["phase_deg"] is None


# ══════════════════════════════════════════════════════════════════════
# AutoPhase
# ══════════════════════════════════════════════════════════════════════

def _auto(**over):
    s = AutoPhase()
    s._poll_interval_s = 0.0
    p = {"window_s": 0.02, "x_signal_index": 24, "y_signal_index": 25}
    p.update(over)
    return s, p


def test_autophase_refuses_when_the_xy_indices_are_unknown(profile):
    """**不猜索引。** 猜错读到的是另一路信号,而算出来的角度看上去一样合理。"""
    s, p = _auto()
    p.pop("x_signal_index"); p.pop("y_signal_index")
    res = s.execute(_Ctx(), p)
    assert not res.success
    assert "不知道解调" in (res.error or "") and "拒绝" in (res.error or "")


def test_autophase_refuses_on_a_noise_floor_signal(profile):
    """X/Y 都在噪声底 ⇒ 无可用信号 ⇒ **不给相位角**(不是给 0°)。"""
    s, p = _auto()
    res = s.execute(_Ctx(xy=(1e-15, 1e-15)), p)
    assert not res.success
    assert "噪声底" in (res.error or "")
    assert "LockIn_DemodPhasSet" not in _Ctx().calls


def test_autophase_never_writes_the_modulator_phase(profile):
    """写的必须是解调侧。调制侧那条命令一次都不许出现。"""
    s, p = _auto()
    ctx = _Ctx(xy=(1.0, 1.0))
    s.execute(ctx, p)
    assert "LockIn_DemodPhasSet" in ctx.calls
    assert "LockIn_ModPhasSet" not in ctx.calls


def test_autophase_signal_to_x_rotates_by_the_measured_angle(profile):
    """X=Y ⇒ 偏角 45°,目标 = 当前 + 45。"""
    s, p = _auto()
    ctx = _Ctx(xy=(1.0, 1.0), phase=10.0)
    res = s.execute(ctx, p)
    assert res.data["delta_deg"] == pytest.approx(45.0)
    assert res.data["target_phase_deg"] == pytest.approx(55.0)


def test_crosstalk_mode_puts_the_crosstalk_on_y(profile):
    """退针态:把串扰转到 Y ⇒ 比 signal_to_x 多转 −90°。"""
    s, p = _auto(mode="crosstalk_to_y")
    res = s.execute(_Ctx(xy=(1.0, 1.0), phase=0.0), p)
    assert res.data["delta_deg"] == pytest.approx(-45.0)


def test_autophase_needs_the_current_phase_as_a_starting_point(profile):
    """算出来的是**增量** —— 读不到起点就不写。假起点会把相位转到谁也没要的地方。"""
    s, p = _auto()

    class _NoPhase(_Ctx):
        def safe_call(self, method, *args, **kw):
            r = super().safe_call(method, *args, **kw)
            if method == "LockIn_DemodPhasGet":
                r.return_value = ("", b"", [])
            return r

    res = s.execute(_NoPhase(xy=(1.0, 1.0)), p)
    assert not res.success and "读不到当前解调相位" in (res.error or "")


def test_autophase_reports_its_evidence(profile):
    """依据留痕:角度、窗口内均值与波动、模式。"""
    s, p = _auto()
    res = s.execute(_Ctx(xy=(1.0, 1.0)), p)
    for k in ("mode", "samples", "x_mean", "y_mean", "x_sd", "y_sd",
              "delta_deg", "target_phase_deg", "readback_verified"):
        assert k in res.data, k


def test_xy_are_read_from_rt_signals_not_from_the_demod_signal_index(profile):
    """``LockIn_DemodSignalGet`` 返回的是**信号索引**,不是 X/Y。

    本文件第一版就是拿它当 X/Y 的 —— 那样 atan2 算的是两个通道号的夹角,而结果
    看上去和真的一样合理。这条钉住取数的来源。
    """
    s, p = _auto()
    ctx = _Ctx(xy=(1.0, 1.0))
    s.execute(ctx, p)
    assert "Signals_ValsGet" in ctx.calls
    assert "LockIn_DemodSignalGet" not in ctx.calls


# ══════════════════════════════════════════════════════════════════════
# 缺陷⑪:用完即关 —— 「谁开谁关」在链式调用下断了
# ══════════════════════════════════════════════════════════════════════

def test_autophase_closes_the_modulation_when_it_is_done(profile):
    """AutoPhase 消费已开启的调制，完成后必须将其关闭。"""
    s, p = _auto()
    rig = _Ctx(xy=(1.0, 1.0), mod_on=True, amp=_f32(0.02), freq=_f32(973.0))
    res = s.execute(rig, p)
    assert res.success, res.error
    assert rig.mod_on is False
    assert res.data["modulation_off_after"] is True
    assert res.data["modulation_was_on_before"] is True


def test_the_close_keeps_the_amplitude(profile):
    """关调制**不许**顺手写幅度。

    缺陷⑧就是关的时候带了个省略的幅度,把 0.02 V 清成 0 —— 下一次开调制,信号没了
    而一切看上去正常。关它不需要重写任何值。
    """
    s, p = _auto()
    rig = _Ctx(xy=(1.0, 1.0), mod_on=True, amp=_f32(0.02))
    s.execute(rig, p)
    assert rig.amp == _f32(0.02)
    assert "LockIn_ModAmpSet" not in rig.calls


def test_autophase_closes_the_modulation_even_when_the_alignment_failed(profile):
    """失败的对齐留下的调制,和成功的一样会污染后面每一条电流判据。"""
    s, p = _auto()
    rig = _Ctx(xy=(1e-15, 1e-15), mod_on=True)
    res = s.execute(rig, p)
    assert not res.success and "噪声底" in (res.error or "")
    assert rig.mod_on is False


def test_an_input_refusal_writes_nothing_at_all(profile):
    """索引不知道 ⇒ 一次硬件调用都没发过 ⇒ 也不发关调制那一条。

    一个什么都没做的拒绝不该留下写操作 —— 它没消费任何东西,也就没有括号要收。
    """
    s, p = _auto()
    p.pop("x_signal_index")
    p.pop("y_signal_index")
    rig = _Ctx(mod_on=True)
    res = s.execute(rig, p)
    assert not res.success
    assert rig.calls == []
    assert rig.mod_on is True


def test_an_abort_leaves_the_modulation_alone(profile):
    """中止后保留仪器当前状态。"""
    s, p = _auto()
    rig = _Ctx(xy=(1.0, 1.0), mod_on=True, aborted=True)
    res = s.execute(rig, p)
    assert not res.success and res.data.get("aborted") is True
    assert "LockIn_ModOnOffSet" not in rig.calls
    assert rig.mod_on is True
    assert "调制也未改动" in (res.error or "")


def test_autophase_has_no_parameter_that_keeps_the_modulation_on(profile):
    """**被否掉的方案钉在这里**:不给「跑完别关」的开关。

    加一个 bool 看起来显然对(「万一还要接着测呢」),而缺陷⑪的成因**正是**没人
    负责关 —— 多一条「保持开着」的路就是把它原样放回来,而且这次是模型来选。
    要推翻这条,先回答:那时候谁负责关?
    (接着测 dI/dV 的正解是重新 ApplyLockInPreset,一条命令的事。)
    """
    names = {ps.name for ps in AutoPhase().metadata().parameters}
    assert not (names & {"keep_modulation_on", "leave_modulation_on",
                         "close_modulation", "mod_on"}), names


def test_the_close_is_in_the_call_record(profile):
    """结果里说「我关掉了」,调用记录里就得找得到那条命令 —— 不许空口声明。"""
    s, p = _auto()
    rig = _Ctx(xy=(1.0, 1.0), mod_on=True)
    res = s.execute(rig, p)
    assert any(getattr(r, "method", "") == "LockIn_ModOnOffSet"
               for r in res.nanonis_calls)


def test_the_description_says_didv_needs_the_modulation_turned_back_on(profile):
    """代价要说清楚:调制关着时 dI/dV 曲线照样画得出来,而它是零信号。

    静默的陷阱换成一条写下来的契约 —— 这是这个设计能成立的前提。
    """
    d = AutoPhase().metadata().description
    assert "ApplyLockInPreset" in d and "关回 OFF" in d


def test_a_phase_that_does_not_stick_fails_loudly(profile):
    """写了但回读不上 ⇒ 失败。「写进去了」和「我们看见它在里面」是两句话。"""
    s, p = _auto()

    class _Stubborn(_Ctx):
        def safe_call(self, method, *args, **kw):
            if method == "LockIn_DemodPhasSet":
                args = (args[0], self.phase)      # 硬件不动
            return super().safe_call(method, *args, **kw)

    res = s.execute(_Stubborn(xy=(1.0, 1.0), phase=10.0, mod_on=True), p)
    assert not res.success and "回读" in (res.error or "")


def test_the_tuple_wrapped_reply_shape_is_also_accepted(profile):
    """数组元素被裹成 1-元组的那种回包也要接(§2.21/§2.31)。"""
    from mast.skills.builtins.lockin_presets_skills import _values_pair

    assert _values_pair(("", b"", [2, [(1.5,), (2.5,)]])) == (1.5, 2.5)
    assert _values_pair(("", b"", [2, [1.5, 2.5]])) == (1.5, 2.5)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
