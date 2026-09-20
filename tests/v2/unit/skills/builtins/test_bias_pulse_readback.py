"""BiasPulseWithReadback —— 打一发脉冲,看 Z 跳没跳。

合成 ctx 注入电流/Z 回包序列(无硬件)。验证:异步开火(wait=0)、双通道同时流、
前后稳定值判定的三个方向、脉冲没发出去时**绝不**报成功、以及固件忽略 wait=0
时的诚实标注。
"""
from __future__ import annotations

import json
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

import pytest  # noqa: E402

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillCategory  # noqa: E402
from mast.skills.builtins.bias_pulse_readback import BiasPulseWithReadback  # noqa: E402


def _rec(method, args, return_value=None, error=""):
    return NanonisCallRecord(method=method, args=args,
                             return_value=return_value, error=error)


class _SeqCtx:
    """safe_call 顺序吐出电流/Z 序列;耗尽后停在最后一个值。

    ``z_after`` 一旦给出,Bias_Pulse 之后的 Z 读数整体加上它 —— 这就是「脉冲把
    针尖改了」在数据上的样子。"""

    def __init__(self, currents=None, zs=None, *, z_after=0.0, fire_error="",
                 folme=(1.0e-9, -2.0e-9)):
        self._c = list(currents or [1e-10] * 400)
        self._z = list(zs or [0.0] * 400)
        self._ci = self._zi = 0
        self._fired = False
        self._z_after = z_after
        self._fire_error = fire_error
        self._folme = folme
        self.calls = []

    def _next(self, seq, idx):
        v = seq[min(idx, len(seq) - 1)]
        return v

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        if method == "Current_Get":
            v = self._next(self._c, self._ci)
            self._ci += 1
            return _rec(method, args, return_value=("", b"", [v]))
        if method == "ZCtrl_ZPosGet":
            v = self._next(self._z, self._zi)
            self._zi += 1
            if self._fired:
                v += self._z_after
            return _rec(method, args, return_value=("", b"", [v]))
        if method == "FolMe_XYPosGet":
            return _rec(method, args, return_value=("", b"", list(self._folme)))
        if method == "Bias_Pulse":
            self._fired = True
            return _rec(method, args, error=self._fire_error)
        return _rec(method, args, return_value=None)

    def check_abort(self):
        return False


@pytest.fixture(autouse=True)
def _deterministic_capture(readback_clock):
    """本文件每一条都跑真实的采集循环,而那个循环是墙钟驱动的 —— 采几个点、
    每个点的时间戳是多少,都由这台机器当时的真实时间决定,于是判定会间歇性
    翻成 ``insufficient_data``。夹具、成因的三条结论、以及**它换走了什么**,
    见同目录 ``conftest.py``。"""
    return readback_clock


#: 快跑参数:总窗口约 0.1+0.06+0.1 = 0.26 s。
_FAST = {"bias_v": 10.0, "width_s": 0.04, "pre_roll_s": 0.08,
         "post_roll_s": 0.08, "max_capture_s": 0.6, "poll_hz": 2000.0}


def _run(ctx, **over):
    p = dict(_FAST)
    p.update(over)
    return BiasPulseWithReadback().execute(ctx, p)


# ── 开火方式 ────────────────────────────────────────────────────────────────

def test_fires_async_and_streams_both_channels():
    ctx = _SeqCtx()
    res = _run(ctx)
    assert res.success, res.error
    pulse = next(c for c in ctx.calls if c[0] == "Bias_Pulse")
    # Bias_Pulse(Wait_until_done, width_s, bias_v, z_hold, abs_rel)
    assert pulse[1][0] == 0, "必须传 wait=0,否则脉冲期间一个采样也拿不到"
    assert pulse[1][1] == pytest.approx(0.04)
    assert pulse[1][2] == pytest.approx(10.0)
    assert pulse[1][3] == 1, "默认必须 hold Z(反馈追电流暴冲会把针撅进表面)"
    assert res.data["timing"]["n_current"] > 0 and res.data["timing"]["n_z"] > 0
    json.dumps(res.data)                       # 可序列化,不带 tensor


def test_baseline_is_sampled_before_the_pulse():
    """z1 得有东西可比 —— 开火前必须已经采到样本。"""
    ctx = _SeqCtx()
    res = _run(ctx)
    idx_pulse = next(i for i, c in enumerate(ctx.calls) if c[0] == "Bias_Pulse")
    pre = [c for c in ctx.calls[:idx_pulse] if c[0] == "ZCtrl_ZPosGet"]
    assert len(pre) >= 2
    assert res.data["step"]["n_pre"] >= 2


def test_records_where_the_pulse_landed():
    """脉冲改造表面 —— 扫描地图要的是开火那一刻的真实坐标。"""
    res = _run(_SeqCtx(folme=(3.5e-9, -7.25e-9)))
    assert res.data["x_m"] == pytest.approx(3.5e-9)
    assert res.data["y_m"] == pytest.approx(-7.25e-9)


# ── 判定 ────────────────────────────────────────────────────────────────────

def test_upward_step_is_reported_as_up():
    """向上几十 nm = 用户判据里「这一发有效」的样子。"""
    res = _run(_SeqCtx(z_after=25e-9))
    assert res.data["step"]["direction"] == "up"
    assert res.data["step"]["delta_m"] == pytest.approx(25e-9, rel=0.05)
    assert "+25" in res.summary


def test_downward_step_is_reported_as_down():
    res = _run(_SeqCtx(z_after=-12e-9))
    assert res.data["step"]["direction"] == "down"
    assert res.data["step"]["delta_m"] < 0


def test_no_step_is_reported_as_none():
    res = _run(_SeqCtx(z_after=0.0))
    assert res.data["step"]["direction"] == "none"


# ── 这条路不许再依赖墙钟（2026-08-05）──────────────────────────────────────
#
# 上面每一条判定测试都跑真实的采集循环,而那个循环按墙钟采样;点数与时间戳都由
# 真实时间决定,判定于是会间歇性翻成 `insufficient_data`。
# 下面两条钉住「已经不再依赖墙钟」这件事本身,否则夹具哪天被人拿掉,
# 这个假红会**悄悄**回来(在开发机上照样绿)。
#
# ⚠️ 它们守的是「夹具还在」,**不是**「真机采得够不够」——后者只能在实机上问,
# 见 conftest.py 那段「这个夹具换走了什么」。


def test_the_capture_has_a_wide_margin_over_the_verdict_minimum():
    """`step_verdict` 的下限是:总点数 ≥ 4、前后窗各 ≥ 2。
    余量小的时候「测不出来」就会伪装成一次判定失败。"""
    d = _run(_SeqCtx(z_after=25e-9)).data
    assert d["timing"]["n_z"] >= 100, d["timing"]
    assert d["timing"]["n_current"] >= 100, d["timing"]
    assert d["step"]["n_pre"] >= 20, d["step"]
    assert d["step"]["n_post"] >= 20, d["step"]


def test_the_sample_count_is_reproducible():
    """确定性时钟应使相同输入产生相同采样点数，避免墙钟调度造成不稳定测试。"""
    a = _run(_SeqCtx(z_after=25e-9)).data["timing"]
    b = _run(_SeqCtx(z_after=25e-9)).data["timing"]
    assert (a["n_z"], a["n_current"]) == (b["n_z"], b["n_current"])


def test_tiny_step_below_the_dead_band_is_not_a_jump():
    """0.1 nm 的挪动在 0.5 nm 死区之下 —— 脉冲要看的是几十 nm 的量级。"""
    res = _run(_SeqCtx(z_after=0.1e-9), step_tol_nm=0.5)
    assert res.data["step"]["direction"] == "none"


def test_direction_is_not_given_a_physical_reading():
    """这一层只说 Z 往哪跳。「针尖变好了」是策略层的判断,不能在这里预先下结论。"""
    d = _run(_SeqCtx(z_after=25e-9)).data
    assert d["step"]["direction"] in ("up", "down", "none")
    assert "cluster" not in json.dumps(d)


# ── 出错与诚实 ──────────────────────────────────────────────────────────────

def test_pulse_error_fails_the_skill():
    res = _run(_SeqCtx(fire_error="pulse rejected"))
    assert res.success is False and "pulse rejected" in res.error


def test_abort_before_firing_reports_no_pulse_applied():
    """脉冲没打出去时必须说清楚 —— 「没打」和「打了但没测到」是两回事。"""
    class _AbortCtx(_SeqCtx):
        def check_abort(self):
            return True

    res = _run(_AbortCtx())
    assert res.success is False
    assert "no pulse was applied" in res.error
    assert not any(c[0] == "Bias_Pulse" for c in _AbortCtx().calls)


def test_blocked_start_is_flagged_not_hidden(readback_clock):
    """固件若忽略 wait=0,曲线就只是事后状态 —— 必须说出来。

    原来这里烧一次真的 ``time.sleep(0.05)``,再去断言一个由 ``perf_counter``
    算出来的量。改成显式推进假时钟:同一个意图(「这次调用阻塞了约一个脉冲时长」),
    但不再取决于真实时间,也不再白等 50 ms。
    ``start_blocked`` 的阈值是 ``max(0.5*width_s, 0.02)`` = 0.02 s,0.05 稳过。"""

    class _SlowCtx(_SeqCtx):
        def safe_call(self, method, *args, role="main"):
            if method == "Bias_Pulse":
                readback_clock.advance(0.05)      # ≈ 整个脉冲时长
            return super().safe_call(method, *args, role=role)

    res = _run(_SlowCtx(z_after=25e-9), width_s=0.04)
    assert res.success
    assert res.data["timing"]["start_blocked"] is True
    assert "warning" in res.data
    # 判定本身仍然成立:比的本来就是前后稳定值。
    assert res.data["step"]["direction"] == "up"


# ── 门控与接线 ──────────────────────────────────────────────────────────────

def test_metadata_declares_bias_pulse_capability():
    """capabilities 才是三档操作模式的门控真源;漏了它 SAFE 模式看不见这个技能。"""
    m = BiasPulseWithReadback().metadata()
    assert "bias_pulse" in m.capabilities
    assert m.safety_level == SafetyLevel.AUTO
    assert m.category == SkillCategory.WRITE


def test_bias_parameter_is_named_so_the_global_cap_sees_it():
    """全局 ±10 V 帽按参数名子串匹配 —— 叫别的名字就悄悄逃出去了。"""
    spec = {p.name: p for p in BiasPulseWithReadback().metadata().parameters}
    assert "bias_v" in spec
    assert spec["bias_v"].min_value == -10.0 and spec["bias_v"].max_value == 10.0


def test_name_puts_it_on_the_scan_map_as_a_pulse():
    """技能名含 'pulse' → classify_skill 自动归为 pulse,自动生成 150 nm 避让圆。"""
    from mast.io.exp_map import classify_skill
    assert classify_skill("BiasPulseWithReadback") == "pulse"


def test_wraps_as_an_agent_tool():
    from mast.agents._shared.skill_adapter import wrap_skill
    tool = wrap_skill(BiasPulseWithReadback, lambda: _SeqCtx())
    assert tool.name == "BiasPulseWithReadback"
    assert tool.metadata.get("danger_level") == "AUTO"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── 针尖安全包络（既有方案表）─────────────────────────────────────────────────

def test_the_pulse_envelope_no_longer_bites_anywhere(monkeypatch):
    """⚠️ **脉冲这一维的包络门 2026-08-12 起结构上不可达** —— 钉住这个事实。

    现场把所有档的 `max_abs_pulse_v` 统一拉满到 **10 V**,而 `bias_v` 字段
    自己的上限也是 ±10 V(Nanonis bias 量程 / TipPulse spec)。两条线重合 ⇒
    **任何能通过字段校验的值都在包络内**,包络永远不会说话。

    这条测试原本叫「软针尖上的 10 V 必须被拒绝」(铂铱包络 8 V)。那件事不再发生。
    留着它是为了让下一个人**知道这道门现在是空的**,而不是以为它还在守着 ——
    「看着在防护、其实没有」是本仓最贵的那一类 bug,这次是我们自己主动造的,
    那就至少让它写在明处。

    **要恢复这道门,需要**:把某一档的 `max_abs_pulse_v` 调回 10 V 以下。
    """
    import mast.core.tip_state as tip_state
    for tip in ({"id": 1, "name": "PtIr-cut", "material": "PtIr",
                 "fabrication": "cut", "form": "stm_wire"},
                {"id": 2, "name": "W-qPlus", "material": "W",
                 "fabrication": "unknown", "form": "qplus"}):
        monkeypatch.setattr(tip_state, "get_current_tip", lambda t=tip: t)
        errs = BiasPulseWithReadback().validate_params(
            {"bias_v": 10.0, "width_s": 0.5})
        assert errs == [], (
            f"{tip['name']}: 包络已拉满到 10 V,10 V 不该再被拒绝: {errs}")


def test_envelope_allows_ten_volts_on_a_tungsten_tip(monkeypatch):
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 2, "name": "W-etched", "material": "W",
        "fabrication": "etched", "form": "stm_wire"})
    assert BiasPulseWithReadback().validate_params(
        {"bias_v": 10.0, "width_s": 0.5}) == []


def test_an_unregistered_tip_is_no_longer_limited_by_the_envelope(monkeypatch):
    """针尖没登记时通用档只给到 ±6 V —— 修针流程的 ±10 V 会被挡下。

    这是既有方案表的有意设计（「不知道针是什么的时候，宁可处理不够也不要一发把
    针打没」），不是本流程的缺陷。后果是**跑完整修针流程之前必须先登记针尖**，
    所以 TipConditioningSelfCheck 把它列为开工前必查项，拒绝信息里也直接写明。"""
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: None)
    # 2026-08-12:通用档也拉满到 10 V ⇒ **「跑修针流程前必须先登记针尖」这个
    # 后果消失了**。原来它是包络的副产品(通用档 6 V 挡住流程的 10 V),
    # 而不是有人专门设计的前置。拉满之后它就没了 —— 记在这里,免得有人继续
    # 以为「不登记就打不出脉冲」还成立。
    assert BiasPulseWithReadback().validate_params(
        {"bias_v": 10.0, "width_s": 0.5}) == []
    assert BiasPulseWithReadback().validate_params(
        {"bias_v": 5.0, "width_s": 0.5}) == []
