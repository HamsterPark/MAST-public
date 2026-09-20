"""偏压随机扰动原语(mast.skills.builtins.bias_wiggle)。

这个技能整个工作区间都落在 ``bias_settle`` 的低偏压死区之内 —— 那是**刻意的
例外**,所以它自带的四道护栏必须逐条钉住:目标有下限、穿零不停留、每步看电流、
突发有硬上限。任何一条松了,一次「轻微改性针尖」就会变成一次撞针。

硬帽那几条测的是「**拒绝而不是夹紧**」:悄悄降下来的幅度是一个被报告成成功的
错误动作(与粗动电压四重锁同一条论证)。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4] / "MASTv2"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
for _m in [m for m in list(sys.modules) if m == "mast" or m.startswith("mast.")]:
    if "MASTv2" not in (getattr(sys.modules[_m], "__file__", "") or "").replace("\\", "/"):
        del sys.modules[_m]

import pytest  # noqa: E402

from mast.skills.builtins import bias_wiggle as BW  # noqa: E402


# ── fake context ────────────────────────────────────────────────────────────

class _Rec:
    """假的 NanonisCallRecord —— ``return_value`` 是 (header, body, [值])。"""

    def __init__(self, value=None, error=""):
        self.error = error
        self.return_value = ("", b"", [value]) if value is not None else None


class Ctx:
    def __init__(self, *, bias=0.02, feedback=1, current=1e-10,
                 bias_get_error="", abort_at=None, current_seq=None):
        self.calls: list[tuple] = []
        self.bias = bias
        self.feedback = feedback
        self.current = current
        self.current_seq = list(current_seq or [])
        self.bias_get_error = bias_get_error
        self.abort_at = abort_at            # 第 N 次 check_abort 起返回 True
        self._abort_calls = 0

    def safe_call(self, method, *args, role="main", allow_on_abort=False):
        self.calls.append((method, args, allow_on_abort))
        if method == "Bias_Get":
            if self.bias_get_error:
                return _Rec(error=self.bias_get_error)
            return _Rec(self.bias)
        if method == "ZCtrl_OnOffGet":
            if self.feedback is None:
                return _Rec(error="no reply")
            return _Rec(float(self.feedback))
        if method == "Current_Get":
            if self.current_seq:
                return _Rec(self.current_seq.pop(0))
            return _Rec(self.current)
        if method == "Bias_Set":
            self.bias = float(args[0])
            return _Rec(0.0)
        return _Rec(0.0)

    def check_abort(self):
        self._abort_calls += 1
        return self.abort_at is not None and self._abort_calls >= self.abort_at

    # 断言辅助
    def sets(self):
        return [a[0] for m, a, _ in self.calls if m == "Bias_Set"]

    def count(self, method):
        return sum(1 for m, _a, _f in self.calls if m == method)


@pytest.fixture(autouse=True)
def _fast_clock(monkeypatch):
    """假时钟:``sleep`` 只推进时间,不真的等 —— 否则一次 5 s 的突发要跑 5 s。"""
    state = {"t": 0.0}
    monkeypatch.setattr(BW.time, "monotonic", lambda: state["t"])
    monkeypatch.setattr(BW.time, "sleep",
                        lambda s: state.__setitem__("t", state["t"] + float(s)))
    return state


def _run(ctx, **params):
    p = {"base_bias_v": 0.02, "burst_s": 0.5, "dwell_min_s": 0.02,
         "dwell_max_s": 0.04, "seed": 7}
    p.update(params)
    return BW.BiasWiggle().execute(ctx, p)


# ── 1. 硬帽:拒绝,不夹紧 ────────────────────────────────────────────────────

def test_amplitude_over_the_hard_cap_is_refused_not_clamped():
    ctx = Ctx()
    res = _run(ctx, wiggle_upper_v=0.5)
    assert not res.success
    assert "拒绝" in res.error
    assert ctx.count("Bias_Set") == 0, "被拒绝的调用不许下发任何硬件命令"


def test_burst_over_the_hard_cap_is_refused():
    ctx = Ctx()
    res = _run(ctx, burst_s=30.0)
    assert not res.success
    assert ctx.count("Bias_Set") == 0


def test_slew_over_the_hard_cap_is_refused():
    ctx = Ctx()
    res = _run(ctx, slew_rate_v_per_s=50.0)
    assert not res.success
    assert ctx.count("Bias_Set") == 0


def test_empty_window_is_refused():
    """下限 ≥ 上限时区间是空的 —— 不许「自动交换」蒙混过去。"""
    ctx = Ctx()
    res = _run(ctx, wiggle_lower_v=0.02, wiggle_upper_v=0.004)
    assert not res.success
    assert ctx.count("Bias_Set") == 0


def test_base_bias_far_from_the_window_is_refused():
    """从 1 V 跳进 ±20 mV 的扰动区,那一下不是扰动是一次大跳变。"""
    ctx = Ctx(bias=1.0)
    res = _run(ctx, base_bias_v=1.0)
    assert not res.success
    assert "BiasSettleChange" in res.error
    assert ctx.count("Bias_Set") == 0


# ── 2. 前置条件 ─────────────────────────────────────────────────────────────

def test_feedback_off_is_refused_by_default():
    """反馈关着时这个机制根本不存在(z 不动)——不该假装做了什么。"""
    ctx = Ctx(feedback=0)
    res = _run(ctx)
    assert not res.success
    assert "反馈" in res.error
    assert ctx.count("Bias_Set") == 0


def test_unreadable_feedback_state_is_refused():
    """读不到状态 ≠ 可以跑。这道门的失败模式必须是拒绝。"""
    ctx = Ctx(feedback=None)
    res = _run(ctx)
    assert not res.success
    assert ctx.count("Bias_Set") == 0


def test_feedback_off_can_be_explicitly_allowed():
    ctx = Ctx(feedback=0)
    res = _run(ctx, allow_feedback_off=True)
    assert res.success, res.error
    assert ctx.count("ZCtrl_OnOffGet") == 0


def test_unreadable_start_bias_is_refused():
    """与 SetBias 的斜坡同一条:不知道起点就不能受控地改变偏压。"""
    ctx = Ctx(bias_get_error="timeout")
    res = _run(ctx)
    assert not res.success
    assert "起点" in res.error
    assert ctx.count("Bias_Set") == 0


# ── 3. 核心安全性质 ─────────────────────────────────────────────────────────

def test_no_target_ever_lands_near_zero():
    """**最重要的一条**:恒流反馈下停在零附近就是把针尖往表面推。

    检查的是每一次「停留」所在的偏压 —— 也就是 log 里记下的到达值。
    """
    for seed in range(1, 25):
        ctx = Ctx()
        res = _run(ctx, seed=seed, wiggle_lower_v=0.004, wiggle_upper_v=0.020)
        assert res.success, res.error
        for entry in res.data["log"]:
            assert abs(entry["reached_v"]) >= 0.004 - 1e-9, (
                f"停留点 {entry['reached_v']:.4f} V 落进了零附近（seed={seed}）")
            assert abs(entry["reached_v"]) <= 0.020 + 1e-9


def test_zero_crossings_are_single_step_no_dwell_in_between():
    """穿零段不许出现中间停留点。

    实现上靠「穿零时用绝对上限斜率」把它压成一步。这条测试直接检查:每次穿零
    前后,下发的偏压序列里没有落在死区里的中间值。
    """
    ctx = Ctx()
    res = _run(ctx, seed=3, slew_rate_v_per_s=0.005)   # 极慢斜率，非穿零段会分很多步
    assert res.success, res.error
    assert any(e["crossed_zero"] for e in res.data["log"]), "这个种子没产生穿零，测试没测到东西"
    lower = 0.004
    # 允许恰好等于 0（不可能出现）与端点；只要没有 0 < |v| < lower 的下发值即可。
    strays = [v for v in ctx.sets() if 0.0 < abs(v) < lower - 1e-9]
    assert not strays, f"穿零时在死区里停了这些值：{strays[:5]}"


def test_current_excursion_aborts_and_restores_the_bias():
    ctx = Ctx(current_seq=[1e-10, 1e-10, 5e-6])       # 第三次读到 5 µA
    res = _run(ctx, abort_current_a=5e-9)
    assert not res.success
    assert "电流" in res.error
    assert res.data["bias_restored"] is True
    assert ctx.sets()[-1] == pytest.approx(0.02), "中止后偏压必须回到基准值"


def test_operator_abort_stops_and_still_restores_the_bias():
    """abort 之后 safe_call 拒绝一切非白名单写,而 Bias_Set 不在白名单里 ——
    所以收尾必须显式带 allow_on_abort=True,否则偏压留在一个随机的扰动值上。"""
    ctx = Ctx(abort_at=4)
    res = _run(ctx)
    assert not res.success
    restore_calls = [(m, a, f) for m, a, f in ctx.calls
                     if m == "Bias_Set" and f is True]
    assert restore_calls, "收尾的 Bias_Set 没有带 allow_on_abort=True"
    assert restore_calls[-1][1][0] == pytest.approx(0.02)


def test_current_is_checked_after_every_bias_step():
    """每一步之后都要看电流 —— 少看一步就可能在超阈状态下再走一步。"""
    ctx = Ctx()
    res = _run(ctx, burst_s=0.3)
    assert res.success, res.error
    # 扰动期间的 Bias_Set 与 Current_Get 一一对应（收尾那次不看电流）。
    wiggle_sets = [c for c in ctx.calls if c[0] == "Bias_Set" and c[2] is False]
    assert ctx.count("Current_Get") == len(wiggle_sets)


# ── 4. 正常路径 ─────────────────────────────────────────────────────────────

def test_happy_path_restores_base_bias_and_reports_the_burst():
    ctx = Ctx()
    res = _run(ctx, burst_s=0.4)
    assert res.success, res.error
    assert res.data["flips_executed"] >= 1
    assert res.data["bias_restored"] is True
    assert ctx.sets()[-1] == pytest.approx(0.02)
    assert res.data["burst_s"] <= 0.4 + 0.05


def test_burst_duration_is_respected():
    ctx = Ctx()
    res = _run(ctx, burst_s=0.2, dwell_min_s=0.05, dwell_max_s=0.05)
    assert res.success, res.error
    assert res.data["burst_s"] <= 0.25


def test_seed_makes_the_sequence_reproducible():
    a = _run(Ctx(), seed=42)
    b = _run(Ctx(), seed=42)
    assert [e["target_v"] for e in a.data["log"]] == \
           [e["target_v"] for e in b.data["log"]]
    c = _run(Ctx(), seed=43)
    assert [e["target_v"] for e in c.data["log"]] != \
           [e["target_v"] for e in a.data["log"]]


# ── 5. 元数据契约 ───────────────────────────────────────────────────────────

def test_metadata_declares_tip_shaping_and_allows_running_during_a_scan():
    meta = BW.BiasWiggle().metadata()
    # tip_shaping 让它自动落进 SAFE 模式的硬拒名单，并在 CRITICAL 关闸时仍可
    # 作为补救手段被放行。
    assert "tip_shaping" in meta.capabilities
    # 配方就是要在一张牺牲帧里打扰动 —— 加了 scan_not_running 会把配方 2 废掉。
    assert "scan_not_running" not in (meta.preconditions or [])


def test_parameter_names_land_in_the_global_bias_envelope():
    """参数名要能被 SafetyGate 的子串+单位匹配捞到 —— 这是全局 ±10 V 的兜底。"""
    from mast.core.safety import _GLOBAL_CHECKS

    names = {p.name: p.unit for p in BW.BiasWiggle().metadata().parameters}
    for pname in ("base_bias_v", "wiggle_lower_v", "wiggle_upper_v"):
        assert names[pname] == "V"
        assert any(sub in pname for sub, unit, *_ in
                   [(c[0], c[1]) for c in _GLOBAL_CHECKS] if unit == "v"), (
            f"{pname} 落不进全局偏压包络")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
