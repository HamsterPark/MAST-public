"""TipShapeWithReadback — shaper + concurrent current/Z stream + jump detection.

Synthetic ctx injects current/Z reply sequences (no hardware). Verifies the
PropsSet+Start(wait=0) sequence, that BOTH channels stream during the procedure,
robust jump detection (a current step is flagged, a smooth Z ramp is not), the
AUTO safety level, and that it wraps as an agent tool.
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

import json  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillCategory  # noqa: E402
from mast.skills.builtins.tip_shaper_readback import (  # noqa: E402
    TipShapeWithReadback,
    _detect_jumps,
)


def _rec(method, args, return_value=None, error=""):
    return NanonisCallRecord(method=method, args=args,
                             return_value=return_value, error=error)


class _SeqCtx:
    """safe_call returns the next value of a current/Z reply sequence; shaper
    calls succeed. Sequences fall back to their last value when exhausted."""

    #: 这台替身"当前"的成像偏压 —— 未指定 bias_v 时技能应当下发的就是它。
    bias_v = 0.35

    def __init__(self, currents, zs):
        self._c = list(currents)
        self._z = list(zs)
        self._ci = 0
        self._zi = 0
        self.calls = []

    def _next(self, which):
        if which == "c":
            v = self._c[min(self._ci, len(self._c) - 1)]
            self._ci += 1
        else:
            v = self._z[min(self._zi, len(self._z) - 1)]
            self._zi += 1
        return v

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        if method == "Current_Get":
            return _rec(method, args, return_value=("", b"", [self._next("c")]))
        if method == "ZCtrl_ZPosGet":
            return _rec(method, args, return_value=("", b"", [self._next("z")]))
        if method == "Bias_Get":
            # ``bias_v`` 的缺省是**当前成像偏压**,而不是锁死的固定值,所以替身
            # 必须答得上这一发 —— 答不上时技能会**如实拒绝**,那是正确行为。
            return _rec(method, args, return_value=("", b"", [self.bias_v]))
        return _rec(method, args, return_value=None)  # PropsSet / Start → ok

    def check_abort(self):
        return False


@pytest.fixture(autouse=True)
def _deterministic_capture(readback_clock):
    """走的是同一个墙钟采集循环(``_readback_stream.stream_with_action``),
    所以有同一个毛病。**这个文件今晚没翻红,不代表它不脆弱** —— 它的窗口比
    BiasPulse 还紧(0.15 s 总窗、0.02 s 前窗),只是还没轮到它。
    夹具与理由见同目录 ``conftest.py``。"""
    return readback_clock


_FAST = {"pre_roll_s": 0.02, "post_roll_s": 0.02, "switch_off_delay_s": 0.0,
         "lift_time_1_s": 0.0, "bias_settling_s": 0.0, "lift_time_2_s": 0.0,
         "end_wait_s": 0.0, "max_capture_s": 0.15, "poll_hz": 2000.0, "jump_k": 4.0}


def test_streams_both_channels_and_starts_async():
    currents = [1e-9] * 4 + [5e-9] * 400   # a step (apex change) at index 4
    zs = [i * 1e-12 for i in range(420)]   # smooth ramp — no jumps
    ctx = _SeqCtx(currents, zs)
    res = TipShapeWithReadback().execute(ctx, dict(_FAST))
    assert res.success, res.error
    d = res.data
    assert d["timing"]["n_current"] > 0 and d["timing"]["n_z"] > 0

    methods = [c[0] for c in ctx.calls]
    assert "TipShaper_PropsSet" in methods
    props = next(c for c in ctx.calls if c[0] == "TipShaper_PropsSet")
    assert len(props[1]) == 11                     # exact PropsSet arity
    start = next(c for c in ctx.calls if c[0] == "TipShaper_Start")
    assert start[1][0] == 0                         # Wait_until_finished=0 (async)

    # the current step is flagged; the smooth Z ramp is not
    assert d["jumps"]["current"]["count"] >= 1
    assert d["jumps"]["z"]["count"] == 0
    json.dumps(res.data)                            # JSON-safe (no tensors)


def test_detect_jumps_robust():
    # constant ramp → no jumps; a single step → exactly one jump
    flat_ramp = [i * 1.0 for i in range(20)]
    out = _detect_jumps(flat_ramp, [i * 0.01 for i in range(20)], k=4.0)
    assert out["count"] == 0
    stepped = [1.0] * 10 + [9.0] * 10
    out2 = _detect_jumps(stepped, [i * 0.01 for i in range(20)], k=4.0)
    assert out2["count"] == 1 and out2["max_abs_delta"] == pytest.approx(8.0)


def test_three_step_verdict_cluster():
    from mast.skills.builtins.tip_shaper_readback import _three_step_verdict
    # z1 baseline ~1.0 → plunge down → settle HIGHER (1.30) = surface cluster
    z = [1.000, 1.001, 0.999, 1.000] + [0.6, 0.3, 0.2, 0.5, 0.9] + [1.300, 1.301, 1.299, 1.300]
    t = [i * 0.05 for i in range(len(z))]
    out = _three_step_verdict(z, t, shaper_start_t=0.18, post_roll_s=0.2, tol_k=4.0)
    assert out["verdict"] == "cluster" and out["delta_m"] > 0


def test_diagnostics_survive_an_insufficient_data_verdict():
    """这一支以前是 ``return {"verdict": "insufficient_data"}`` —— 把整个 dict 丢掉。
    于是诊断字段**恰好在不需要它们的时候幸存、在需要它们的时候消失**:
    判不出来正是最想知道「采了几个点、最长往返多久」的那一刻。

    (同一个形状在隔壁 ``bias_pulse_readback`` 里被一句注释点过名:
    「守卫在,却只在不需要它的地方管用」。)
    """
    from mast.skills.builtins.tip_shaper_readback import _three_step_verdict

    # 事件几乎在采集一开始 → 前窗真的不够 → 判不了。这时诊断必须还在。
    z = [1.0] + [0.7] * 9
    t = [i * 0.05 for i in range(len(z))]
    out = _three_step_verdict(z, t, shaper_start_t=0.02, post_roll_s=0.2, tol_k=4.0)

    assert out["verdict"] == "insufficient_data"
    assert "max_gap_s" in out, "判不了的时候恰恰最需要它"
    assert out["max_gap_s"] == pytest.approx(0.05, rel=0.05)


def test_three_step_verdict_no_change():
    from mast.skills.builtins.tip_shaper_readback import _three_step_verdict
    # settles back to ~baseline → didn't bite
    z = [1.000, 1.001, 0.999, 1.000] + [0.3, 0.2, 0.4] + [1.0005, 0.9998, 1.0001, 1.000]
    t = [i * 0.05 for i in range(len(z))]
    out = _three_step_verdict(z, t, 0.18, post_roll_s=0.2, tol_k=4.0)
    assert out["verdict"] == "no_change"


def test_three_step_verdict_pit_or_tip():
    from mast.skills.builtins.tip_shaper_readback import _three_step_verdict
    # settles LOWER (0.70) than baseline → tip changed / pit
    z = [1.000, 1.001, 0.999, 1.000] + [0.3, 0.2] + [0.700, 0.701, 0.699, 0.700]
    t = [i * 0.05 for i in range(len(z))]
    out = _three_step_verdict(z, t, 0.18, post_roll_s=0.2, tol_k=4.0)
    assert out["verdict"] == "tip_changed_or_pit" and out["delta_m"] < 0


def test_stage_boundaries_ordered():
    from mast.skills.builtins.tip_shaper_readback import _stage_boundaries
    sb = _stage_boundaries(0.3, {"switch_off_delay_s": 0.05, "lift_time_1_s": 0.15,
                                 "bias_settling_s": 0.05, "lift_time_2_s": 0.15,
                                 "end_wait_s": 0.1})
    assert [s["stage"] for s in sb] == [
        "pre_roll", "switch_off", "z_ramp_1_plunge", "bias_settle",
        "z_ramp_2_retract", "end_wait", "post_roll"]
    assert sb[1]["t_start"] == 0.3
    assert sb[-2]["t_end"] == pytest.approx(0.3 + 0.05 + 0.15 + 0.05 + 0.15 + 0.1)


def test_summary_is_set_and_concise():
    """SkillResult.summary is a one-liner (so chat isn't the full data dump).

    ⚠️ 长度只量**人读的那一半**,不量后面挂的原始曲线路径。

    2026-08-12:原来是 `len(res.summary) < 200`,而 summary 末尾带一个**绝对路径**
    (`原始曲线: <tmpdir>/readback_traces/...json`)。于是这条断言实际量的是
    **这台机器的临时目录有多深** —— 并行跑时 pytest 给每个 worker 多加一层
    `popen-gw0`,路径长了 9 个字符,203 > 200,红。

    它不是被并行"弄坏"的,是**本来就会红**,换一台临时目录深一点的机器就会;
    并行只是先撞上而已。**一个把环境长度算进预算的断言,量的不是被测的东西。**
    """
    ctx = _SeqCtx([1e-9] * 4 + [5e-9] * 400, [i * 1e-12 for i in range(420)])
    res = TipShapeWithReadback().execute(ctx, dict(_FAST))
    assert res.success and res.summary
    assert "针尖整形" in res.summary
    human_part = res.summary.split("| 原始曲线:")[0].strip()
    assert len(human_part) < 200, (
        f"人读的那一段有 {len(human_part)} 字,太长了(聊天里会刷屏):\n{human_part}")


def test_propsset_error_aborts():
    class _ErrCtx:
        def safe_call(self, method, *a, role="main"):
            if method == "Bias_Get":      # 缺省偏压要读得到,否则走不到 PropsSet
                return _rec(method, a, return_value=("", b"", [0.35]))
            err = "props failed" if method == "TipShaper_PropsSet" else ""
            return _rec(method, a, error=err)

        def check_abort(self):
            return False

    res = TipShapeWithReadback().execute(_ErrCtx(), {})
    assert res.success is False and "props failed" in res.error


def test_metadata_is_auto_write():
    m = TipShapeWithReadback().metadata()
    assert m.name == "TipShapeWithReadback"
    # AUTO: shaper ramps fine-Z + pulses bias (both Nanonis-bounded, no instrument
    # damage), so autonomous agents may run it ungated.
    assert m.safety_level == SafetyLevel.AUTO
    assert m.category == SkillCategory.WRITE
    names = {p.name for p in m.parameters}
    assert {"tip_lift_m", "lift_height_m", "poll_hz", "pre_roll_s", "jump_k"} <= names


def test_post_hook_fires_on_success_and_merges_scan_paths():
    """wrap_skill must call post_hook after a successful execute and merge its
    scan_paths into the state update (the records/buffer glue seam)."""
    from mast.agents._shared.skill_adapter import wrap_skill
    fired = []

    def _hook(name, data, ctx):
        fired.append((name, isinstance(data, dict) and "indent" in data))
        return {"scan_paths": ["/tmp/x.png"]}

    ctx = _SeqCtx([1e-9] * 4 + [5e-9] * 400, [i * 1e-12 for i in range(420)])
    tool = wrap_skill(TipShapeWithReadback, lambda: ctx, post_hook=_hook)
    res = tool.func(tool_call_id="t", state={}, **dict(_FAST))
    assert fired and fired[0] == ("TipShapeWithReadback", True)
    assert "/tmp/x.png" in res.update.get("scan_paths", [])


def test_post_hook_failure_does_not_break_skill():
    """A raising post_hook must NOT turn a successful skill into an error."""
    from mast.agents._shared.skill_adapter import wrap_skill

    def _bad_hook(name, data, ctx):
        raise RuntimeError("hook boom")

    ctx = _SeqCtx([1e-9] * 4, [0.0] * 4)
    tool = wrap_skill(TipShapeWithReadback, lambda: ctx, post_hook=_bad_hook)
    res = tool.func(tool_call_id="t", state={}, **dict(_FAST))
    assert "TipShapeWithReadback" in res.update.get("executed_skills", [])


def test_wraps_as_auto_agent_tool():
    from mast.agents._shared.skill_adapter import wrap_skill
    ctx = _SeqCtx([1e-9] * 10, [0.0] * 10)
    tool = wrap_skill(TipShapeWithReadback, lambda: ctx)
    assert tool.name == "TipShapeWithReadback"
    assert tool.metadata.get("danger_level") == "AUTO"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# 未指定偏压时使用即时回读，不能静默采用默认值。

def test_omitted_bias_is_the_bias_we_just_read_not_a_hardcoded_3v():
    """未指定偏压时，下发的值必须来自即时回读，不能被旧默认覆盖。"""
    ctx = _SeqCtx([1e-9], [1e-9])
    ctx.bias_v = 0.02                      # 用户把成像偏压降到 20 mV
    res = TipShapeWithReadback().execute(ctx, {"tip_lift_m": -5e-10})
    assert res.success, res.error

    sets = [a for m, a in ctx.calls if m == "TipShaper_PropsSet"]
    assert len(sets) == 1, ctx.calls
    assert sets[0][2] == pytest.approx(0.02), (
        f"下发的偏压是 {sets[0][2]},不是刚读到的 0.02 —— 3 V 又回来了")
    assert sets[0][2] != pytest.approx(3.0)
    # 读之前必须真的读过。
    assert "Bias_Get" in [m for m, _ in ctx.calls]
    assert res.data["bias_v_source"] == "read"


def test_an_explicit_bias_still_wins():
    """用户逐字给的值不被替换 —— 与包络那条同一条纪律。"""
    ctx = _SeqCtx([1e-9], [1e-9])
    ctx.bias_v = 0.02
    res = TipShapeWithReadback().execute(
        ctx, {"tip_lift_m": -5e-10, "bias_v": 1.25})
    sets = [a for m, a in ctx.calls if m == "TipShaper_PropsSet"]
    assert sets[0][2] == pytest.approx(1.25)
    assert res.data["bias_v_source"] == "explicit"


def test_an_unreadable_bias_refuses_instead_of_falling_back_to_3v():
    """读不到就**拒绝**,不回落到 3.0 —— ``lookup() || DEFAULT`` 正是今天数不清
    第几次的形状,而这一次的代价是在结上打一个没人要求过的 3 V。"""
    class _NoBias(_SeqCtx):
        def safe_call(self, method, *a, role="main"):
            if method == "Bias_Get":
                return _rec(method, a, error="TCP timeout")
            return super().safe_call(method, *a, role=role)

    ctx = _NoBias([1e-9], [1e-9])
    res = TipShapeWithReadback().execute(ctx, {"tip_lift_m": -5e-10})
    assert res.success is False
    assert "读不到当前偏压" in res.error and "3 V" in res.error
    assert "TipShaper_PropsSet" not in [m for m, _ in ctx.calls], "拒绝了却还是下发了"
