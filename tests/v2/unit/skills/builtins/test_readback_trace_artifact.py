"""整形过程的 Z 与电流原始曲线必须保存为可读取的产物。

净高度变化和瞬态极值表达不同信息，摘要不能替代完整轨迹。
测试验证文件存在、样本数一致、回包提供路径，以及操作参数随曲线保存。
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

from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.skills.builtins._readback_stream import (  # noqa: E402
    ReadbackCapture,
    TRACE_SCHEMA,
    save_trace,
    trace_dir,
)
from mast.skills.builtins.bias_pulse_readback import BiasPulseWithReadback  # noqa: E402
from mast.skills.builtins.tip_shaper_readback import TipShapeWithReadback  # noqa: E402


def _rec(method, args, return_value=None, error=""):
    return NanonisCallRecord(method=method, args=args,
                             return_value=return_value, error=error)


class _SeqCtx:
    """电流/Z 各走一条回复序列;其余调用一律成功。序列用尽后重复最后一个值。"""

    bias_v = 0.35

    def __init__(self, currents, zs):
        self._c, self._z = list(currents), list(zs)
        self._ci = self._zi = 0
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
            return _rec(method, args, return_value=("", b"", [self.bias_v]))
        if method == "FolMe_XYPosGet":
            # 扎针留下的是**永久痕迹**,「在哪扎的」和「扎没扎上」一样是结果的一部分。
            return _rec(method, args, return_value=("", b"", [1.5e-9, -2.5e-9]))
        return _rec(method, args, return_value=None)

    def check_abort(self):
        return False


@pytest.fixture(autouse=True)
def _deterministic_capture(readback_clock):
    """同一个墙钟采集循环 ⇒ 同一个假时钟(理由见本目录 conftest)。"""
    return readback_clock


@pytest.fixture(autouse=True)
def _sandboxed_trace_dir(tmp_path, monkeypatch):
    """曲线落进**本测试自己的** tmp,不落进用户真正的 ``experiments/traces/``。

    (整个 v2 套件已由 ``tests/v2/conftest.py`` 的 ``_isolate_readback_traces``
    统一重定向;这里再指一次是为了让下面那些断言有一个自己说得清的目录。)
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MAST_TRACES_DIR", str(tmp_path / "traces"))
    return tmp_path


_FAST = {"pre_roll_s": 0.02, "post_roll_s": 0.02, "switch_off_delay_s": 0.0,
         "lift_time_1_s": 0.0, "bias_settling_s": 0.0, "lift_time_2_s": 0.0,
         "end_wait_s": 0.0, "max_capture_s": 0.15, "poll_hz": 2000.0,
         "jump_k": 4.0}


# ── ① 机制本身 ────────────────────────────────────────────────────────────

def test_save_trace_keeps_both_channels_with_their_own_timestamps():
    """两个通道**各带各的时间戳**。

    采集循环每帧先读电流再读 Z(相隔一个 RTT),任一次读失败就只有另一条记了点
    —— 所以两条曲线的点数可以不同,把它们当同一根时间轴对齐是错的。这里用一条
    刻意不等长的采集把「不对齐」钉住。
    """
    cap = ReadbackCapture(
        current_s=[1e-9, 2e-9, 3e-9], current_t=[0.0, 0.01, 0.02],
        z_s=[5e-9, 4e-9], z_t=[0.005, 0.015],
        fired=True, fire_t_s=0.004, capture_s=0.02)
    saved = save_trace(cap, skill="X", meta={"tip_lift_m": -1.8e-9},
                       stages=[{"stage": "z_ramp_1_plunge", "t_start": 0.004}])
    assert "trace_error" not in saved, saved
    doc = json.loads(Path(saved["trace_path"]).read_text(encoding="utf-8"))

    assert doc["schema"] == TRACE_SCHEMA
    assert doc["event_t_s"] == pytest.approx(0.004)   # 阶段边界的锚点
    assert doc["meta"]["tip_lift_m"] == pytest.approx(-1.8e-9)
    assert doc["stages"][0]["stage"] == "z_ramp_1_plunge"
    z, cur = doc["channels"]["z"], doc["channels"]["current"]
    assert z["unit"] == "m" and cur["unit"] == "A"
    assert z["n"] == 2 and cur["n"] == 3
    assert len(z["t_s"]) == z["n"] and len(cur["t_s"]) == cur["n"]
    assert z["samples"] == [5e-9, 4e-9]


def test_default_home_is_under_the_project_root(tmp_path, monkeypatch):
    """没设 ``MAST_TRACES_DIR`` 时曲线去哪 —— 钉住默认位置本身。

    覆写变量存在的一半理由是测试隔离,而一个**只在覆写下被测过**的解析函数,
    正好是「守卫在,却只在不需要它的地方管用」的那个形状:真机上跑的是没覆写
    的那一支,而它从没被断言过。
    """
    monkeypatch.delenv("MAST_TRACES_DIR", raising=False)
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    assert trace_dir() == tmp_path / "experiments" / "traces"
    assert trace_dir().is_dir()


def test_two_saves_never_share_a_path():
    """连续记录应生成不同文件名，避免覆盖前一次轨迹。"""
    cap = ReadbackCapture(z_s=[1e-9] * 4, z_t=[0.0, 0.1, 0.2, 0.3])
    a = save_trace(cap, skill="X", meta={})["trace_path"]
    b = save_trace(cap, skill="X", meta={})["trace_path"]
    assert a != b and Path(a).exists() and Path(b).exists()


def test_a_write_failure_is_reported_not_swallowed(monkeypatch):
    """落盘失败**不许静默**:``trace_error`` 必须出现在回包里。

    静默失败下,「曲线丢了」和「本来就没采」长得一模一样 —— 而这个文件存在的
    全部理由就是曾经分不出这两者。
    """
    from mast.skills.builtins import _readback_stream

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(_readback_stream, "trace_dir", _boom)
    saved = save_trace(ReadbackCapture(z_s=[1e-9], z_t=[0.0]),
                       skill="X", meta={})
    assert "trace_path" not in saved
    assert "disk full" in saved["trace_error"]
    assert "disk full" in _readback_stream.trace_ref(saved)


# ── ② 端到端:扎针 ────────────────────────────────────────────────────────

def test_tip_shape_lands_a_trace_that_can_be_read_back():
    """跑一次带替身的 ``TipShapeWithReadback``:产物在盘上、读得回、点数对得上,
    **而且回包里有那条路径**。"""
    ctx = _SeqCtx([1e-9] * 4 + [5e-9] * 400, [i * 1e-12 for i in range(420)])
    res = TipShapeWithReadback().execute(ctx, dict(_FAST, tip_lift_m=-1.8e-9,
                                                   lift_height_m=2.0e-9))
    assert res.success, res.error

    # ③ 指针在回包里 —— 落了盘却没人知道路径,等于没落。
    path = res.data.get("trace_path")
    assert path, f"回包里没有曲线路径: {sorted(res.data)}"
    assert path in res.summary, "摘要里没有指针 —— agent 看到的就只有摘要"

    # ① 文件真的在盘上 + ② 读得回
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    z, cur = doc["channels"]["z"], doc["channels"]["current"]
    assert z["n"] > 0 and cur["n"] > 0
    # 样本数与记录的 n 一致(两处各算一遍就会漂)
    assert len(z["samples"]) == z["n"] == len(z["t_s"])
    assert len(cur["samples"]) == cur["n"] == len(cur["t_s"])
    # 与技能自己报的点数是同一个数
    assert z["n"] == res.data["timing"]["n_z"]
    assert cur["n"] == res.data["timing"]["n_current"]
    # 落在被重定向的 tmp 根下,不是用户真实的 experiments/
    assert Path(path).parent == trace_dir()


def test_tip_shape_trace_carries_the_parameters_that_made_it():
    """曲线元数据应保留输入深度、回抬高度、偏压和位置，支持后续复现与关联。"""
    ctx = _SeqCtx([1e-9] * 400, [i * 1e-12 for i in range(420)])
    res = TipShapeWithReadback().execute(
        ctx, dict(_FAST, tip_lift_m=-2.5e-9, lift_height_m=2.0e-9,
                  bias_v=0.02, bias_lift_v=0.5))
    meta = json.loads(
        Path(res.data["trace_path"]).read_text(encoding="utf-8"))["meta"]

    assert meta["tip_lift_m"] == pytest.approx(-2.5e-9)      # 深度
    assert meta["lift_height_m"] == pytest.approx(2.0e-9)    # 回抬
    assert meta["bias_v"] == pytest.approx(0.02)
    # ``bias_lift_v`` 是**无条件施加**的那一个;只记 bias_v 会让回执少一记电压。
    assert meta["bias_lift_v"] == pytest.approx(0.5)
    assert meta["x_m"] == pytest.approx(1.5e-9)              # 扎在哪儿
    assert meta["y_m"] == pytest.approx(-2.5e-9)
    assert meta["verdict"]                                    # 当时的判定
    assert "FolMe_XYPosGet" in [m for m, _ in ctx.calls]


def test_tip_shape_stages_are_in_the_trace_so_the_reader_can_cut_the_plunge():
    """没有阶段边界,读的人对不上「哪一段是扎入」 —— 曲线就只是一堆数。"""
    ctx = _SeqCtx([1e-9] * 400, [i * 1e-12 for i in range(420)])
    res = TipShapeWithReadback().execute(ctx, dict(_FAST, tip_lift_m=-1.8e-9))
    doc = json.loads(
        Path(res.data["trace_path"]).read_text(encoding="utf-8"))
    assert [s["stage"] for s in doc["stages"]] == [
        "pre_roll", "switch_off", "z_ramp_1_plunge", "bias_settle",
        "z_ramp_2_retract", "end_wait", "post_roll"]
    # 锚点:开火时刻本身也要在,阶段表是相对它算的
    assert doc["event_t_s"] is not None


# ── ③ 端到端:电脉冲(共用 ``_readback_stream`` ⇒ 一处改两处受益)────────

def test_bias_pulse_lands_the_same_artifact():
    """电脉冲那一路走的是同一个采集循环,所以走同一个落盘口 —— 一处改、两处受益,
    而不是复制一份然后各自漂移。"""
    ctx = _SeqCtx([1e-9] * 400, [i * 1e-12 for i in range(420)])
    res = BiasPulseWithReadback().execute(ctx, {
        "bias_v": -3.0, "width_s": 0.01, "z_hold": 1,
        "pre_roll_s": 0.02, "post_roll_s": 0.02, "max_capture_s": 0.15,
        "poll_hz": 2000.0})
    assert res.success, res.error

    path = res.data.get("trace_path")
    assert path and path in res.summary
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert doc["skill"] == "BiasPulseWithReadback"
    assert doc["meta"]["bias_v"] == pytest.approx(-3.0)
    assert doc["meta"]["width_s"] == pytest.approx(0.01)
    assert doc["channels"]["z"]["n"] == res.data["timing"]["n_z"]
    assert doc["event_t_s"] is not None       # 脉冲时刻 = 前后窗的分界


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
